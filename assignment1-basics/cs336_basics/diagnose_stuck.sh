#!/bin/bash

# 诊断卡住的tokenizer进程

echo "========================================="
echo "  诊断卡住的Tokenizer"
echo "========================================="
echo ""

# 找到主进程
MAIN_PID=$(pgrep -f "python tokenizer.py" | head -1)

if [ -z "$MAIN_PID" ]; then
    echo "❌ 未找到tokenizer进程"
    exit 1
fi

echo "主进程PID: $MAIN_PID"
echo ""

# 1. 检查主进程在做什么
echo "1. 检查主进程系统调用..."
timeout 3 strace -p $MAIN_PID 2>&1 | head -20 &
STRACE_PID=$!
sleep 3
kill $STRACE_PID 2>/dev/null
wait $STRACE_PID 2>/dev/null
echo ""

# 2. 检查worker进程
echo "2. 检查worker进程状态..."
WORKER_PIDS=$(pgrep -P $MAIN_PID)
NUM_WORKERS=$(echo "$WORKER_PIDS" | wc -w)
echo "找到 $NUM_WORKERS 个worker进程"
echo ""

# 随机检查几个worker
echo "抽样检查3个worker进程的系统调用..."
for PID in $(echo "$WORKER_PIDS" | head -3); do
    echo "  Worker PID $PID:"
    timeout 1 strace -p $PID 2>&1 | head -5 | sed 's/^/    /'
    echo ""
done

# 3. 检查进程打开的文件
echo "3. 检查主进程打开的文件..."
lsof -p $MAIN_PID 2>/dev/null | grep -E "(openwebtext|artifacts)" | head -10

echo ""

# 4. 检查是否有死锁
echo "4. 检查进程栈（可能的死锁）..."
if [ -f "/proc/$MAIN_PID/stack" ]; then
    echo "主进程栈:"
    cat /proc/$MAIN_PID/stack | sed 's/^/  /'
fi

echo ""
echo "========================================="
echo "  分析结果"
echo "========================================="
echo ""
echo "可能的问题："
echo "1. 正则表达式遇到病态输入（灾难性回溯）"
echo "2. 多进程通信队列满了，主进程在等待"
echo "3. 某些worker卡在特定的文本行"
echo "4. 内存不足导致交换"
echo ""
echo "建议操作："
echo "1. 等待5-10分钟看是否恢复"
echo "2. 如果确认卡死，杀死进程: pkill -9 -f 'python.*tokenizer'"
echo "3. 修改配置后重新运行"
echo ""
