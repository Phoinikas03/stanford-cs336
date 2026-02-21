#!/bin/bash

# 检查tokenizer进程状态的快速脚本

echo "========================================="
echo "  Tokenizer状态检查"
echo "========================================="
echo ""

# 1. 检查进程是否存在
echo "1. 检查tokenizer进程..."
PIDS=$(pgrep -f "python.*tokenizer.py")
if [ -z "$PIDS" ]; then
    echo "❌ 没有找到tokenizer.py进程"
    echo ""
    echo "可能原因："
    echo "- 进程已经完成"
    echo "- 进程已经崩溃"
    echo "- 进程名称不匹配"
else
    echo "✓ 找到tokenizer进程:"
    ps -o pid,stat,%cpu,%mem,etime,cmd -p $PIDS
    echo ""
    
    # 检查进程状态
    for PID in $PIDS; do
        STAT=$(ps -o stat= -p $PID)
        echo "  PID $PID 状态: $STAT"
        case $STAT in
            *D*)
                echo "    ⚠️  不可中断睡眠 (D) - 可能在等待I/O"
                ;;
            *R*)
                echo "    ✓ 运行中 (R)"
                ;;
            *S*)
                echo "    ℹ️  可中断睡眠 (S) - 可能在等待"
                ;;
            *Z*)
                echo "    ❌ 僵尸进程 (Z)"
                ;;
        esac
    done
fi

echo ""

# 2. 检查输出文件
echo "2. 检查输出文件..."
OUTPUT_FILE="../artifacts/openwebtext_train.bin"
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo "✓ 输出文件存在: $OUTPUT_FILE"
    echo "  文件大小: $SIZE"
    
    # 检查文件是否在增长
    echo "  检查文件是否在增长（等待5秒）..."
    SIZE1=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
    sleep 5
    SIZE2=$(stat -c%s "$OUTPUT_FILE" 2>/dev/null || stat -f%z "$OUTPUT_FILE" 2>/dev/null)
    
    if [ "$SIZE2" -gt "$SIZE1" ]; then
        GROWTH=$((SIZE2 - SIZE1))
        echo "  ✓ 文件正在增长! (+${GROWTH} bytes in 5s)"
        echo "  ➜ 进程可能在正常工作，只是很慢"
    else
        echo "  ⚠️  文件没有增长"
        echo "  ➜ 进程可能卡住了"
    fi
else
    echo "❌ 输出文件不存在: $OUTPUT_FILE"
fi

echo ""

# 3. 检查CPU使用
echo "3. 检查Python进程CPU使用..."
PYTHON_PROCS=$(ps aux | grep "[p]ython.*tokenizer" | wc -l)
if [ "$PYTHON_PROCS" -gt 0 ]; then
    echo "找到 $PYTHON_PROCS 个相关Python进程:"
    ps aux | grep "[p]ython.*tokenizer" | awk '{printf "  PID %s: CPU %s%%, MEM %s%%\n", $2, $3, $4}'
    
    # 计算总CPU使用率
    TOTAL_CPU=$(ps aux | grep "[p]ython.*tokenizer" | awk '{sum += $3} END {print sum}')
    echo "  总CPU使用率: ${TOTAL_CPU}%"
    
    if (( $(echo "$TOTAL_CPU < 10" | bc -l) )); then
        echo "  ⚠️  CPU使用率很低 - 可能在等待I/O或卡住"
    fi
fi

echo ""

# 4. 检查内存
echo "4. 检查内存使用..."
free -h | head -2

echo ""

# 5. 检查磁盘I/O（如果有iostat）
echo "5. 检查磁盘I/O..."
if command -v iostat &> /dev/null; then
    iostat -x 1 2 | tail -n +4
else
    echo "  ℹ️  iostat未安装，跳过"
fi

echo ""

# 6. 给出建议
echo "========================================="
echo "  建议"
echo "========================================="

if [ -z "$PIDS" ]; then
    echo "❓ 进程不存在:"
    echo "   - 检查进程是否已完成"
    echo "   - 查看日志: less tokenizer.log"
    echo "   - 检查是否有错误"
elif [ -f "$OUTPUT_FILE" ] && [ "$SIZE2" -gt "$SIZE1" ]; then
    echo "✓ 进程看起来在正常工作:"
    echo "   - 耐心等待完成"
    echo "   - 可以关闭终端，进程会继续运行"
    echo "   - 使用 'watch -n 10 ls -lh $OUTPUT_FILE' 监控进度"
else
    echo "⚠️  进程可能卡住了:"
    echo "   1. 查看详细诊断: cat TOKENIZER_HANG_FIX.md"
    echo "   2. 考虑减少worker数量（tokenizer.py 第385行）"
    echo "   3. 如果需要，可以杀死进程:"
    echo "      pkill -f 'python.*tokenizer'"
    echo "   4. 修改配置后重新运行"
fi

echo ""
echo "========================================="
