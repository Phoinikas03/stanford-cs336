#!/bin/bash
# 带断点续传功能的tokenizer运行脚本示例

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置
OUTPUT_FILE="../artifacts/openwebtext_train.bin"
CHECKPOINT_FILE="${OUTPUT_FILE}.checkpoint.json"
LOG_FILE="tokenizer_$(date +%Y%m%d_%H%M%S).log"

echo "========================================="
echo "Tokenizer with Resume Support"
echo "========================================="
echo ""
echo "配置:"
echo "  输出文件: $OUTPUT_FILE"
echo "  Checkpoint: $CHECKPOINT_FILE"
echo "  日志文件: $LOG_FILE"
echo ""

# 检查是否有checkpoint
if [ -f "$CHECKPOINT_FILE" ]; then
    echo "✓ 发现checkpoint文件"
    echo ""
    
    # 显示checkpoint信息
    python manage_tokenizer_checkpoint.py view "$CHECKPOINT_FILE"
    
    echo ""
    read -p "是否从checkpoint恢复? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "✓ 将从checkpoint恢复"
        RESUME_MODE=true
    else
        echo "✗ 将从头开始"
        read -p "是否删除现有文件? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "🧹 删除checkpoint和输出文件..."
            rm -f "$CHECKPOINT_FILE"
            rm -f "$OUTPUT_FILE"
        fi
        RESUME_MODE=false
    fi
else
    echo "ℹ️  未找到checkpoint，将从头开始"
    RESUME_MODE=false
fi

echo ""
echo "========================================="
echo "启动Tokenizer"
echo "========================================="
echo ""

# 启动tokenizer并保存日志
if [ "$RESUME_MODE" = true ]; then
    echo "yes" | python tokenizer.py 2>&1 | tee "$LOG_FILE"
else
    echo "no" | python tokenizer.py 2>&1 | tee "$LOG_FILE"
fi

# 检查是否成功完成
EXIT_CODE=$?

echo ""
echo "========================================="
echo "处理结果"
echo "========================================="
echo ""

if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Tokenization完成!"
    
    if [ -f "$OUTPUT_FILE" ]; then
        OUTPUT_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "  输出文件: $OUTPUT_FILE ($OUTPUT_SIZE)"
    fi
    
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "  Checkpoint文件仍存在: $CHECKPOINT_FILE"
        echo ""
        read -p "是否删除checkpoint文件? (y/n): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -f "$CHECKPOINT_FILE"
            echo "✓ 已删除checkpoint"
        fi
    fi
else
    echo "⚠️  Tokenization未完成（退出码: $EXIT_CODE）"
    
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "✓ Checkpoint已保存，可以稍后恢复:"
        echo "  $0"
        echo ""
        python manage_tokenizer_checkpoint.py view "$CHECKPOINT_FILE"
    fi
fi

echo ""
echo "日志文件: $LOG_FILE"
echo ""
