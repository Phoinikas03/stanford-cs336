#!/bin/bash

# 安全地重启tokenizer

echo "========================================="
echo "  重启Tokenizer"
echo "========================================="
echo ""

# 1. 检查现有进程
EXISTING_PIDS=$(pgrep -f "python.*tokenizer.py")
if [ -n "$EXISTING_PIDS" ]; then
    echo "⚠️  发现现有tokenizer进程:"
    ps -o pid,%cpu,%mem,etime,cmd -p $EXISTING_PIDS | head -10
    echo ""
    read -p "是否杀死这些进程? (yes/no): " CONFIRM
    
    if [ "$CONFIRM" = "yes" ]; then
        echo "正在杀死进程..."
        pkill -9 -f "python.*tokenizer"
        sleep 2
        echo "✓ 进程已杀死"
    else
        echo "已取消"
        exit 1
    fi
else
    echo "✓ 没有发现现有tokenizer进程"
fi

echo ""

# 2. 检查输出文件
OUTPUT_FILE="../artifacts/openwebtext_train.bin"
if [ -f "$OUTPUT_FILE" ]; then
    SIZE=$(ls -lh "$OUTPUT_FILE" | awk '{print $5}')
    echo "⚠️  发现现有输出文件: $SIZE"
    echo ""
    echo "选项:"
    echo "  1. 删除并重新开始"
    echo "  2. 备份并重新开始"
    echo "  3. 取消（保留文件）"
    echo ""
    read -p "请选择 (1/2/3): " CHOICE
    
    case $CHOICE in
        1)
            echo "删除现有文件..."
            rm "$OUTPUT_FILE"
            echo "✓ 文件已删除"
            ;;
        2)
            BACKUP="${OUTPUT_FILE}.backup_$(date +%Y%m%d_%H%M%S)"
            echo "备份到: $BACKUP"
            mv "$OUTPUT_FILE" "$BACKUP"
            echo "✓ 文件已备份"
            ;;
        3)
            echo "保留现有文件，将追加写入"
            echo "⚠️  注意: 如果文件损坏可能导致问题"
            ;;
        *)
            echo "无效选项，已取消"
            exit 1
            ;;
    esac
else
    echo "✓ 输出文件不存在"
fi

echo ""

# 3. 显示当前配置
echo "========================================="
echo "  当前配置"
echo "========================================="
WORKERS=$(grep "num_workers =" tokenizer.py | head -1 | awk -F= '{print $2}' | tr -d ' ')
LINES_PER=$(grep "lines_per_task =" tokenizer.py | head -1 | awk -F= '{print $2}' | awk '{print $1}')

echo "Workers: $WORKERS"
echo "Lines per task: $LINES_PER"
echo ""

read -p "是否需要修改配置? (yes/no): " MODIFY

if [ "$MODIFY" = "yes" ]; then
    echo ""
    read -p "输入worker数量 [当前: $WORKERS]: " NEW_WORKERS
    read -p "输入lines_per_task [当前: $LINES_PER]: " NEW_LINES
    
    if [ -n "$NEW_WORKERS" ]; then
        sed -i "s/num_workers = $WORKERS/num_workers = $NEW_WORKERS/" tokenizer.py
        echo "✓ 已更新 num_workers = $NEW_WORKERS"
    fi
    
    if [ -n "$NEW_LINES" ]; then
        sed -i "s/lines_per_task = $LINES_PER/lines_per_task = $NEW_LINES/" tokenizer.py
        echo "✓ 已更新 lines_per_task = $NEW_LINES"
    fi
fi

echo ""

# 4. 启动tokenizer
echo "========================================="
echo "  启动Tokenizer"
echo "========================================="
echo ""

read -p "准备启动tokenizer，继续? (yes/no): " START

if [ "$START" = "yes" ]; then
    echo ""
    echo "启动tokenizer (后台运行)..."
    cd /mnt/data_x3/xiazeyu/stanford-cs336-main/assignment1-basics/cs336_basics
    
    # 创建日志目录
    mkdir -p logs
    LOG_FILE="logs/tokenizer_$(date +%Y%m%d_%H%M%S).log"
    
    nohup python tokenizer.py > "$LOG_FILE" 2>&1 &
    PID=$!
    
    echo "✓ Tokenizer已启动"
    echo "  PID: $PID"
    echo "  日志: $LOG_FILE"
    echo ""
    echo "监控命令:"
    echo "  ./monitor_tokenizer.sh          # 实时监控"
    echo "  tail -f $LOG_FILE               # 查看日志"
    echo "  ps -p $PID                      # 检查进程"
    echo ""
    
    sleep 2
    
    if ps -p $PID > /dev/null; then
        echo "✅ 进程运行正常"
        echo ""
        echo "建议: 打开新终端运行 ./monitor_tokenizer.sh"
    else
        echo "❌ 进程启动失败，查看日志:"
        echo "  cat $LOG_FILE"
    fi
else
    echo "已取消"
fi

echo ""
echo "========================================="
