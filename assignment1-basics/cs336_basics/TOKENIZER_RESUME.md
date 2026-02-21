# Tokenizer 断点续传功能说明

## 功能概述

为 `tokenizer.py` 添加了完整的断点续传功能，支持在tokenization过程中断后从上次的进度继续处理，避免重复计算。

## 核心特性

### 1. Checkpoint信息保存

程序会自动保存checkpoint文件：
- **文件名**: `<输出文件名>.checkpoint.json`
- **位置**: 与输出的`.bin`文件在同一目录
- **保存频率**: 每处理100万行自动保存一次
- **保存时机**: 
  - 定期保存（每100万行）
  - 程序正常结束时
  - 可手动 Ctrl+C 中断（最近的checkpoint可用）

### 2. Checkpoint内容

checkpoint文件包含以下元信息（JSON格式）：

```json
{
  "lines_processed": 3000000,
  "total_tokens": 45678901,
  "total_lines": 8013769,
  "timestamp": "2026-02-15 14:30:45",
  "input_path": "../dataset/openwebtext/owt_train.txt",
  "output_path": "../artifacts/openwebtext_train.bin",
  "progress_percent": 37.45
}
```

### 3. 恢复流程

启动tokenizer时：

1. **自动检测checkpoint**
   ```
   ✓ 发现checkpoint文件: ../artifacts/openwebtext_train.bin.checkpoint.json
     已处理行数: 3,000,000
     已生成token数: 45,678,901
     上次保存时间: 2026-02-15 14:30:45
   
   是否从checkpoint恢复? (yes/no):
   ```

2. **选择恢复模式**
   - 输入 `yes`: 从上次中断处继续
   - 输入 `no`: 从头开始处理（会覆盖原文件）

3. **恢复处理**
   - 自动跳过已处理的行
   - 以追加模式写入输出文件
   - 进度条显示从checkpoint位置开始

## 使用示例

### 场景1：首次处理

```bash
cd /mnt/data_x3/xiazeyu/stanford-cs336-main/assignment1-basics/cs336_basics
python tokenizer.py
```

**输出**:
```
ℹ️  未找到checkpoint文件，从头开始处理
Processing ../dataset/openwebtext/owt_train.txt...
Using 16 workers, 100000 lines per task...

文件总行数: 8,013,769

Tokenizing:   0%|          | 0/8013769 [00:00<?, ?it/s]
```

处理过程中会自动保存checkpoint：
```
Tokenizing:  12%|█▎        | 1000000/8013769 [05:23<38:12]
💾 Checkpoint saved at line 1,000,000 (12.5%)
```

### 场景2：从中断恢复

假设处理到37%时被中断，重新启动：

```bash
python tokenizer.py
```

**输出**:
```
✓ 发现checkpoint文件: ../artifacts/openwebtext_train.bin.checkpoint.json
  已处理行数: 3,000,000
  已生成token数: 45,678,901
  上次保存时间: 2026-02-15 14:30:45

是否从checkpoint恢复? (yes/no): yes
✓ 将从第 3,000,001 行继续处理

Processing ../dataset/openwebtext/owt_train.txt...
Using 16 workers, 100000 lines per task...

文件总行数: 8,013,769
剩余行数: 5,013,769

Tokenizing:  37%|███▊      | 3000000/8013769 [00:00<?, ?it/s]
```

### 场景3：处理完成后

处理完成时：
```
✓ 处理完成!
  总行数: 8,013,769
  已处理行数: 8,013,769
  总token数: 123,456,789
  输出文件: ../artifacts/openwebtext_train.bin
  Checkpoint文件: ../artifacts/openwebtext_train.bin.checkpoint.json

处理已完成，是否删除checkpoint文件? (yes/no):
```

## 高级功能

### 手动检查checkpoint

```bash
cat ../artifacts/openwebtext_train.bin.checkpoint.json
```

```json
{
  "lines_processed": 3000000,
  "total_tokens": 45678901,
  "total_lines": 8013769,
  "timestamp": "2026-02-15 14:30:45",
  "input_path": "../dataset/openwebtext/owt_train.txt",
  "output_path": "../artifacts/openwebtext_train.bin",
  "progress_percent": 37.45
}
```

### 手动删除checkpoint（重新开始）

```bash
rm ../artifacts/openwebtext_train.bin.checkpoint.json
rm ../artifacts/openwebtext_train.bin
python tokenizer.py
```

### 修改checkpoint间隔

在 `tokenizer.py` 中修改：
```python
checkpoint_interval = 1000000  # 改为其他值，如 500000（50万行）
```

## 技术细节

### 跳过已处理行的性能

- **快速跳过**: 在读取文件时直接跳过，不进行tokenization
- **进度提示**: 每100万行显示一次跳过进度
- **内存效率**: 跳过的行不会加载到内存

### 追加模式写入

- **文件模式**: resume时使用 `'ab'`（追加二进制）模式
- **数据一致性**: 使用 `f_out.flush()` 强制写入磁盘
- **无重复**: 已处理的行不会重复写入

### 错误处理

如果checkpoint文件损坏：
```
⚠️  读取checkpoint失败: <错误信息>
将从头开始处理
```

## 注意事项

1. **不要手动编辑checkpoint文件**
   - checkpoint文件为JSON格式，但不建议手动修改
   - 如需重新开始，直接删除checkpoint文件

2. **磁盘空间**
   - 确保有足够空间存储输出文件
   - checkpoint文件很小（< 1KB）

3. **中断安全**
   - 每100万行保存一次checkpoint
   - 如果在两次checkpoint之间中断，最多重复处理100万行
   - 可以安全地用 Ctrl+C 中断程序

4. **多次运行**
   - 如果不想恢复，选择 `no` 会覆盖原文件
   - 建议备份重要的输出文件

## 故障排除

### Q: 恢复后进度条显示不准确？
A: 进度条会从checkpoint位置开始显示，这是正常的。总行数和剩余行数都是准确的。

### Q: 为什么跳过行需要时间？
A: 因为需要读取文件定位到正确位置。但跳过速度远快于tokenization速度。

### Q: checkpoint多久保存一次？
A: 默认每100万行保存一次。可以修改 `checkpoint_interval` 参数。

### Q: 可以在不同机器上恢复吗？
A: 可以，但需要确保文件路径一致，或者手动修改checkpoint中的路径。

## 与之前修复的兼容性

断点续传功能与之前的超时保护、长行跳过等功能完全兼容：

- ✅ 10秒超时机制仍然有效
- ✅ 跳过超长行（>100KB）仍然有效  
- ✅ 错误处理和日志记录仍然有效
- ✅ 16个worker并行处理仍然有效

## 测试建议

```bash
# 1. 开始处理
python tokenizer.py

# 2. 等待几分钟后按 Ctrl+C 中断

# 3. 重新启动并选择恢复
python tokenizer.py
# 输入: yes

# 4. 观察是否从正确位置继续
```

## 相关文件

- `tokenizer.py`: 主程序（已添加断点续传功能）
- `*.checkpoint.json`: checkpoint元信息文件
- `TOKENIZER_HANG_FIX.md`: 之前的超时修复文档
- `TOKENIZER_QUICK_FIX.md`: 快速修复指南
