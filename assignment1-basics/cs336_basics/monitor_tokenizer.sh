#!/bin/bash

# 实时监控tokenizer进度

OUTPUT_FILE="../artifacts/openwebtext_train.bin"

echo "========================================="
echo "  Tokenizer实时监控"
echo "========================================="
echo "按 Ctrl+C 停止监控"
echo ""

# 获取初始文件大小
if [ -f "$OUTPUT_FILE" ]; then
    PREV_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
else
    PREV_SIZE=0
fi

PREV_TIME=$(date +%s)

while true; do
    clear
    echo "========================================="
    echo "  Tokenizer进度监控"
    echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "========================================="
    echo ""
    
    # 检查进程
    MAIN_PID=$(pgrep -f "python tokenizer.py" | head -1)
    
    if [ -z "$MAIN_PID" ]; then
        echo "❌ Tokenizer进程未运行"
        echo ""
        echo "可能原因："
        echo "  - 进程已完成"
        echo "  - 进程崩溃"
        echo "  - 进程在tmux中运行（使用 tmux attach 查看）"
        break
    fi
    
    echo "✓ 主进程运行中 (PID: $MAIN_PID)"
    
    # 统计worker数量
    WORKER_COUNT=$(pgrep -P $MAIN_PID 2>/dev/null | wc -l)
    echo "✓ Worker进程: $WORKER_COUNT 个"
    echo ""
    
    # 检查文件
    if [ -f "$OUTPUT_FILE" ]; then
        CURR_SIZE=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
        CURR_TIME=$(date +%s)
        
        SIZE_MB=$(echo "scale=2; $CURR_SIZE / 1024 / 1024" | bc)
        SIZE_GB=$(echo "scale=2; $CURR_SIZE / 1024 / 1024 / 1024" | bc)
        
        echo "📁 输出文件: $OUTPUT_FILE"
        echo "   大小: ${SIZE_MB} MB (${SIZE_GB} GB)"
        
        # 计算增长速度
        TIME_DIFF=$((CURR_TIME - PREV_TIME))
        SIZE_DIFF=$((CURR_SIZE - PREV_SIZE))
        
        if [ $TIME_DIFF -gt 0 ]; then
            SPEED_MB=$(echo "scale=2; $SIZE_DIFF / $TIME_DIFF / 1024 / 1024" | bc)
            echo "   增长速度: ${SPEED_MB} MB/s"
            
            if [ $SIZE_DIFF -gt 0 ]; then
                echo "   状态: ✓ 正在写入"
            else
                echo "   状态: ⚠️  停滞（${TIME_DIFF}秒未增长）"
            fi
        fi
        
        # 估算进度（假设总目标约10GB）
        PROGRESS=$(echo "scale=1; $SIZE_GB / 10 * 100" | bc)
        echo "   估算进度: ${PROGRESS}%"
        
        PREV_SIZE=$CURR_SIZE
        PREV_TIME=$CURR_TIME
    else
        echo "❌ 输出文件不存在"
    fi
    
    echo ""
    
    # CPU使用率
    echo "💻 CPU使用率:"
    TOTAL_CPU=$(ps aux | grep "[p]ython.*tokenizer" | awk '{sum += $3} END {print sum}')
    echo "   总CPU: ${TOTAL_CPU}%"
    
    # 显示top 5 worker
    echo "   Top 5 workers:"
    ps aux | grep "[p]ython.*tokenizer" | sort -k3 -rn | head -5 | awk '{printf "     PID %s: %s%% CPU\n", $2, $3}'
    
    echo ""
    
    # 内存
    echo "💾 内存:"
    free -h | grep Mem | awk '{printf "   使用: %s / %s (可用: %s)\n", $3, $2, $7}'
    
    echo ""
    echo "----------------------------------------"
    echo "刷新间隔: 10秒"
    echo "按 Ctrl+C 停止监控"
    echo "========================================="
    
    sleep 10
done
