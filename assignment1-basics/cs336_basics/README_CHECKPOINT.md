# Checkpoint 使用指南

本文档说明如何使用checkpoint功能保存和恢复训练状态。

## 功能概述

训练脚本会自动：
- ✅ 每1000步保存一次checkpoint
- ✅ Checkpoint包含：模型参数、优化器状态、当前训练步数
- ✅ 保存在 `../checkpoints/` 目录下
- ✅ 文件命名格式：`checkpoint_step_{步数}.pt`

## Checkpoint文件结构

每个checkpoint文件包含以下内容：

```python
{
    'model_state_dict': model.state_dict(),      # 模型参数
    'optimizer_state_dict': optimizer.state_dict(),  # 优化器状态（包含动量等）
    'iteration': step                             # 当前训练步数
}
```

## 自动保存

### 单GPU训练 (`train_loop.py`)

训练时会自动每1000步保存checkpoint：

```bash
python train_loop.py
```

输出示例：
```
Step 1000/10000 | Train Loss: 3.2134 | Eval Train Loss: 3.1854
✓ Checkpoint saved: ../checkpoints/checkpoint_step_1000.pt
Step 2000/10000 | Train Loss: 2.8912 | Eval Train Loss: 2.8654
✓ Checkpoint saved: ../checkpoints/checkpoint_step_2000.pt
```

### 多GPU训练 (`train_loop_multigpu.py`)

多GPU训练时，只有主进程（rank 0）保存checkpoint：

```bash
torchrun --nproc_per_node=4 train_loop_multigpu.py
```

## 手动加载Checkpoint

### 方法1：使用辅助脚本

```bash
python resume_training.py --checkpoint ../checkpoints/checkpoint_step_1000.pt
```

### 方法2：在代码中加载

```python
from cs336_basics.checkpoint import load_checkpoint

# 初始化模型和优化器（配置必须与训练时一致）
model = TransformerLM(...)
optimizer = AdamW(model.parameters(), ...)

# 加载checkpoint
start_step = load_checkpoint(
    src="../checkpoints/checkpoint_step_1000.pt",
    model=model,
    optimizer=optimizer
)

print(f"从第 {start_step + 1} 步继续训练")

# 继续训练
for step in range(start_step, total_steps):
    # ... 训练代码
```

## 修改Checkpoint保存间隔

### 在 `train_loop.py` 中修改

找到这一行（大约第56行）：

```python
checkpoint_interval = 1000  # 每1000步保存checkpoint
```

修改为你需要的间隔，例如：

```python
checkpoint_interval = 500   # 每500步保存
checkpoint_interval = 2000  # 每2000步保存
```

### 在 `train_loop_multigpu.py` 中修改

找到这一行（大约第109行）：

```python
checkpoint_interval = 1000  # 每1000步保存checkpoint
```

## Checkpoint目录管理

### 查看已保存的checkpoint

```bash
ls -lh ../checkpoints/
```

输出示例：
```
checkpoint_step_1000.pt    # 245 MB
checkpoint_step_2000.pt    # 245 MB
checkpoint_step_3000.pt    # 245 MB
```

### 删除旧的checkpoint（节省空间）

如果磁盘空间不足，可以删除较早的checkpoint：

```bash
# 只保留最新的3个checkpoint
cd ../checkpoints
ls -t checkpoint_*.pt | tail -n +4 | xargs rm -f
```

或创建清理脚本：

```bash
#!/bin/bash
# 只保留最新的N个checkpoint
N=5
cd ../checkpoints
ls -t checkpoint_*.pt | tail -n +$((N+1)) | xargs rm -f
echo "保留了最新的 $N 个checkpoint"
```

## 最佳实践

### 1. 定期备份重要checkpoint

```bash
# 备份特定步数的checkpoint
cp ../checkpoints/checkpoint_step_10000.pt ~/backups/
```

### 2. 监控磁盘空间

每个checkpoint约250MB（取决于模型大小）：

```bash
# 查看checkpoint目录占用的空间
du -sh ../checkpoints/
```

### 3. 在训练中断后恢复

如果训练意外中断：

1. 找到最新的checkpoint：
   ```bash
   ls -lt ../checkpoints/ | head -n 5
   ```

2. 修改训练脚本，从该checkpoint恢复：
   ```python
   # 在训练循环开始前添加
   checkpoint_path = "../checkpoints/checkpoint_step_8000.pt"
   if os.path.exists(checkpoint_path):
       start_step = load_checkpoint(checkpoint_path, model, optimizer)
       print(f"从checkpoint恢复训练，起始步数: {start_step}")
   else:
       start_step = 0
   
   # 修改训练循环
   for step in range(start_step, total_steps):
       # ... 训练代码
   ```

## 多GPU训练的特殊注意事项

在多GPU训练中：

1. **保存时使用 `model.module`**：
   ```python
   # DDP包装的模型需要使用.module来访问原始模型
   model_to_save = model.module if world_size > 1 else model
   save_checkpoint(model_to_save, optimizer, step, path)
   ```

2. **加载时先创建模型，再包装DDP**：
   ```python
   # 1. 创建模型
   model = TransformerLM(...)
   
   # 2. 加载checkpoint
   load_checkpoint(checkpoint_path, model, optimizer)
   
   # 3. 用DDP包装
   model = DDP(model, device_ids=[local_rank])
   ```

## 常见问题

### Q: Checkpoint文件太大怎么办？

A: 可以使用压缩或只保存模型参数：

```python
# 只保存模型参数（不保存优化器状态）
torch.save(model.state_dict(), "model_only.pt")
```

### Q: 如何验证checkpoint是否正确？

A: 加载并检查：

```python
checkpoint = torch.load("../checkpoints/checkpoint_step_1000.pt")
print(f"Checkpoint包含的键: {checkpoint.keys()}")
print(f"训练步数: {checkpoint['iteration']}")
```

### Q: 能否在不同的GPU数量下恢复训练？

A: 可以，但需要注意：
- 从单GPU保存的checkpoint可以在多GPU上加载
- 从多GPU保存的checkpoint可以在单GPU上加载
- 但global batch size会改变，可能影响训练效果

## 相关文件

- `checkpoint.py` - Checkpoint保存和加载函数
- `train_loop.py` - 单GPU训练脚本（带checkpoint）
- `train_loop_multigpu.py` - 多GPU训练脚本（带checkpoint）
- `resume_training.py` - Checkpoint恢复示例
