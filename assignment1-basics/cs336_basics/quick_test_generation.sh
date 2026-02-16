#!/bin/bash

# 快速测试文本生成功能

echo "========================================="
echo "    文本生成快速测试"
echo "========================================="
echo ""

# 检查是否提供了checkpoint路径
if [ $# -eq 0 ]; then
    echo "使用方法: $0 <checkpoint_path>"
    echo ""
    echo "例如:"
    echo "  $0 ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt"
    echo ""
    echo "可用的checkpoint:"
    ls -1t ../checkpoints/*/checkpoint_step_*.pt 2>/dev/null | head -5
    exit 1
fi

CHECKPOINT=$1

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint文件不存在: $CHECKPOINT"
    exit 1
fi

echo "✓ 使用checkpoint: $CHECKPOINT"
echo ""

# 测试1: 基本生成
echo "========================================="
echo "测试 1: 基本生成"
echo "========================================="
python text_generate.py \
    --checkpoint "$CHECKPOINT" \
    --prompt "Once upon a time" \
    --max_tokens 50 \
    --temperature 0.8

echo ""
read -p "按Enter继续下一个测试..."
echo ""

# 测试2: Top-k采样
echo "========================================="
echo "测试 2: Top-k 采样"
echo "========================================="
python text_generate.py \
    --checkpoint "$CHECKPOINT" \
    --prompt "The princess lived in" \
    --max_tokens 50 \
    --temperature 0.9 \
    --top_k 50

echo ""
read -p "按Enter继续下一个测试..."
echo ""

# 测试3: Top-p采样
echo "========================================="
echo "测试 3: Top-p 采样"
echo "========================================="
python text_generate.py \
    --checkpoint "$CHECKPOINT" \
    --prompt "In a magical forest" \
    --max_tokens 50 \
    --temperature 0.9 \
    --top_p 0.95

echo ""
read -p "按Enter继续性能测试..."
echo ""

# 测试4: 性能对比
echo "========================================="
echo "测试 4: KV Cache 性能对比"
echo "========================================="
python benchmark_generation.py \
    --checkpoint "$CHECKPOINT" \
    --max_tokens 30 \
    --num_samples 3

echo ""
echo "========================================="
echo "✓ 所有测试完成！"
echo "========================================="
echo ""
echo "下一步:"
echo "  1. 尝试交互式生成: python interactive_generate.py --checkpoint $CHECKPOINT"
echo "  2. 调整采样参数以获得更好的结果"
echo "  3. 查看完整文档: cat README_GENERATION.md"
echo ""
