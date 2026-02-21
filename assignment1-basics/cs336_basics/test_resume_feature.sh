#!/bin/bash
# 测试tokenizer断点续传功能的脚本

set -e

echo "========================================="
echo "测试 Tokenizer 断点续传功能"
echo "========================================="
echo ""

# 配置
TEST_DIR="../artifacts/test_resume"
TEST_INPUT="$TEST_DIR/test_input.txt"
TEST_OUTPUT="$TEST_DIR/test_output.bin"
TEST_CHECKPOINT="$TEST_OUTPUT.checkpoint.json"

# 创建测试目录
mkdir -p "$TEST_DIR"

# 生成测试数据（1000行）
echo "📝 生成测试数据..."
if [ ! -f "$TEST_INPUT" ]; then
    for i in {1..1000}; do
        echo "This is test line number $i with some random text to tokenize." >> "$TEST_INPUT"
    done
    echo "✓ 已生成1000行测试数据: $TEST_INPUT"
else
    echo "✓ 使用已存在的测试数据: $TEST_INPUT"
fi

# 清理之前的输出
if [ -f "$TEST_OUTPUT" ]; then
    echo "🧹 清理旧的输出文件..."
    rm -f "$TEST_OUTPUT"
fi
if [ -f "$TEST_CHECKPOINT" ]; then
    rm -f "$TEST_CHECKPOINT"
fi

echo ""
echo "========================================="
echo "步骤1: 首次运行（处理前500行后中断）"
echo "========================================="
echo ""

# 创建临时修改的tokenizer（只处理前500行）
TEMP_TOKENIZER="$TEST_DIR/temp_tokenizer.py"
cp tokenizer.py "$TEMP_TOKENIZER"

# 修改输入输出路径
sed -i "s|file_path = \"../dataset/openwebtext/owt_train.txt\"|file_path = \"$TEST_INPUT\"|" "$TEMP_TOKENIZER"
sed -i "s|output_path = \"../artifacts/openwebtext_train.bin\"|output_path = \"$TEST_OUTPUT\"|" "$TEMP_TOKENIZER"
sed -i "s|checkpoint_interval = 1000000|checkpoint_interval = 300|" "$TEMP_TOKENIZER"

echo "⚙️  配置修改完成"
echo "   输入文件: $TEST_INPUT"
echo "   输出文件: $TEST_OUTPUT"
echo "   Checkpoint间隔: 每300行"
echo ""

# 运行tokenizer（会在300行和600行保存checkpoint）
echo "🚀 启动tokenizer..."
timeout 30s python "$TEMP_TOKENIZER" 2>&1 | head -n 50 || true

echo ""
echo "========================================="
echo "步骤2: 查看checkpoint状态"
echo "========================================="
echo ""

if [ -f "$TEST_CHECKPOINT" ]; then
    echo "✓ Checkpoint已创建"
    python manage_tokenizer_checkpoint.py view "$TEST_CHECKPOINT"
else
    echo "❌ Checkpoint未创建"
    exit 1
fi

echo ""
echo "========================================="
echo "步骤3: 验证checkpoint"
echo "========================================="
echo ""

python manage_tokenizer_checkpoint.py validate "$TEST_CHECKPOINT"

echo ""
echo "========================================="
echo "步骤4: 从checkpoint恢复（模拟）"
echo "========================================="
echo ""

echo "💡 在实际使用中，你可以这样恢复:"
echo "   cd /mnt/data_x3/xiazeyu/stanford-cs336-main/assignment1-basics/cs336_basics"
echo "   python tokenizer.py"
echo "   # 然后输入 'yes' 从checkpoint恢复"
echo ""

echo "========================================="
echo "测试完成!"
echo "========================================="
echo ""
echo "📊 生成的测试文件:"
echo "   输入: $TEST_INPUT"
echo "   输出: $TEST_OUTPUT"
echo "   Checkpoint: $TEST_CHECKPOINT"
echo ""
echo "🧹 清理测试文件:"
echo "   rm -rf $TEST_DIR"
echo ""
