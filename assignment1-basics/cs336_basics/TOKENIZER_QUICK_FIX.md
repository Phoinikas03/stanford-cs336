# Tokenizer卡在37% - 快速解决方案

## 当前情况

✅ **已改进**：我已经为 `tokenizer.py` 添加了超时保护和错误处理
- 每行最多处理10秒，超时则跳过
- 跳过超过100KB的异常长行
- 自动记录timeout和error数量

## 立即行动方案

### 方案A：重新运行（推荐）

如果当前进程确实卡住了：

```bash
# 1. 杀死所有tokenizer进程
pkill -9 -f "python.*tokenizer"

# 2. 清理输出文件（或备份）
mv ../artifacts/openwebtext_train.bin ../artifacts/openwebtext_train.bin.backup 2>/dev/null

# 3. 重新运行（已有超时保护）
cd /mnt/data_x3/xiazeyu/stanford-cs336-main/assignment1-basics/cs336_basics
nohup python tokenizer.py > tokenizer.log 2>&1 &

# 4. 在另一个终端监控
./monitor_tokenizer.sh
```

### 方案B：检查进程状态

如果不确定是否真的卡住：

```bash
# 运行监控脚本（会显示文件是否在增长）
./monitor_tokenizer.sh
```

观察10-20分钟：
- ✓ 如果文件大小在增长 → 进程正常，继续等待
- ❌ 如果完全不增长 → 确实卡住，使用方案A

### 方案C：进入tmux查看实际进度

```bash
# 附加到tokenizer运行的tmux会话
tmux a -t cs336

# 查看tqdm进度条是否在更新
# 按 Ctrl+B 然后 D 退出（不杀死进程）
```

## 改进后的特性

### 1. 超时保护 ✅
```python
# 每行最多10秒，避免无限卡住
signal.alarm(10)
ids = tokenizer.encode(line)
signal.alarm(0)
```

### 2. 跳过问题行 ✅
```python
# 跳过超长行（>100KB）
if len(line) > 100000:
    results.append([])
    continue
```

### 3. 错误统计 ✅
```python
# 记录并报告timeout和error数量
print(f"Batch summary: {timeout_count} timeouts, {error_count} errors")
```

### 4. 更合理的配置 ✅
- Workers: 16（原32）
- Lines/task: 50000（原200000）
- 更频繁的进度更新

## 性能预期

**改进后的性能：**
- 每个批次处理更快（50k vs 200k行）
- 进度更新更频繁（约每30秒-1分钟）
- 遇到问题行会跳过而不是卡住
- 总处理时间可能略增，但更稳定

**预计总时间：**
- 文件大小：94M行
- 速度：约20-30k行/秒
- 总时间：约50-70分钟

## 如果仍然卡住

### 进一步降低并发

编辑 `tokenizer.py` 第426-428行：

```python
num_workers = 4  # 大幅降低
lines_per_task = 10000  # 小批次
```

### 或使用单进程版本

```python
# 完全禁用多进程（调试用）
# 在 tokenizer.py 添加单进程模式
if __name__ == '__main__':
    # ... 设置 ...
    
    # 单进程模式
    use_multiprocessing = False  # 设置为False
    
    if not use_multiprocessing:
        # 单进程处理
        tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens)
        
        with open(input_path, 'r') as f_in, open(output_path, 'wb') as f_out:
            for line in tqdm(f_in, total=total_lines):
                try:
                    ids = tokenizer.encode(line)
                    buffer.extend(ids)
                    
                    if len(buffer) >= buffer_size:
                        arr = np.array(buffer, dtype=dtype)
                        f_out.write(arr.tobytes())
                        buffer = []
                except Exception as e:
                    print(f"Error: {e}")
                    continue
```

## 调试命令集合

```bash
# 1. 检查进程是否在运行
ps aux | grep tokenizer

# 2. 检查文件是否在增长
ls -lh ../artifacts/openwebtext_train.bin
sleep 60
ls -lh ../artifacts/openwebtext_train.bin

# 3. 查看CPU使用
top -p $(pgrep -d',' -f tokenizer)

# 4. 查看内存使用
ps aux | grep tokenizer | awk '{sum+=$6} END {print sum/1024 " MB"}'

# 5. 实时监控
./monitor_tokenizer.sh

# 6. 杀死卡住的进程
pkill -9 -f "python.*tokenizer"
```

## 预防措施（下次运行）

### 1. 先测试小文件

```bash
# 取前10000行测试
head -10000 ../dataset/openwebtext/owt_train.txt > test_sample.txt
# 修改tokenizer.py的input_path
# 先跑测试文件，确保没问题再跑完整文件
```

### 2. 添加定期checkpoint

修改代码，每1GB保存一次检查点：

```python
checkpoint_size = 1_000_000_000  # 1GB
if f_out.tell() > checkpoint_size and not checkpoint_saved:
    print(f"Checkpoint: {f_out.tell() / 1e9:.2f} GB written")
    checkpoint_saved = True
```

### 3. 使用更简单的tokenizer

考虑使用Hugging Face的tokenizers库（更快更稳定）：

```bash
pip install tokenizers

# 使用预训练的tokenizer
from tokenizers import Tokenizer
tokenizer = Tokenizer.from_pretrained("gpt2")
```

## 总结

**已完成的改进：**
- ✅ 添加超时保护（10秒/行）
- ✅ 跳过超长行（>100KB）
- ✅ 错误统计和报告
- ✅ 减少worker数量（32→16）
- ✅ 减少批次大小（200k→100k）

**现在可以：**
1. 重新运行tokenizer（更稳定）
2. 使用monitor_tokenizer.sh实时监控
3. 如果还卡住，进一步减少并发

**下次运行时记得：**
- 先用小文件测试
- 使用nohup后台运行
- 定期检查进度
