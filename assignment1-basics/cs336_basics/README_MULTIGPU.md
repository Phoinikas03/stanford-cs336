# 多GPU分布式训练指南

本文档介绍如何使用 `train_loop_multigpu.py` 进行多GPU分布式训练。

## 主要特点

- ✅ 使用 PyTorch DistributedDataParallel (DDP) 实现高效多GPU训练
- ✅ 自动检测和使用所有可用GPU
- ✅ 支持更大的batch size（总batch = batch_size_per_gpu × GPU数量）
- ✅ 自动同步梯度和参数
- ✅ 只在主进程打印日志，避免重复输出

## 运行方法

### 方法1：使用启动脚本（推荐）

```bash
cd /mnt/data_x3/xiazeyu/stanford-cs336-main/assignment1-basics/cs336_basics
./run_multigpu.sh
```

这个脚本会自动检测所有可用GPU并启动训练。

### 方法2：手动使用 torchrun

**使用所有可用GPU：**
```bash
torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) train_loop_multigpu.py
```

**指定GPU数量（例如使用4个GPU）：**
```bash
torchrun --nproc_per_node=4 train_loop_multigpu.py
```

**使用特定的GPU（例如只使用GPU 0,1,2,3）：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_loop_multigpu.py
```

### 方法3：单GPU调试模式

```bash
python train_loop_multigpu.py
```

## 配置说明

在 `train_loop_multigpu.py` 中可以修改以下参数：

```python
# 第89行：每个GPU的batch size
batch_size_per_gpu = 128

# 第90行：序列长度
context_length = 256

# 第105行：评估间隔
eval_interval = 100
```

### Batch Size 计算

- **单GPU模式**：总batch_size = batch_size_per_gpu
- **多GPU模式**：总batch_size = batch_size_per_gpu × GPU数量

例如：
- 如果有4个GPU，`batch_size_per_gpu=128`
- 则总batch_size = 128 × 4 = 512

## 性能优化建议

1. **增大batch_size_per_gpu**：如果GPU显存充足，可以增大到256或更高
2. **调整评估间隔**：可以增大`eval_interval`减少评估开销
3. **使用混合精度训练**：可以添加`torch.cuda.amp`进一步提升性能

## 监控训练

训练过程中会打印：
- 使用的GPU数量
- 每个GPU的batch size和总batch size
- 训练loss和评估loss
- 训练步数

示例输出：
```
使用 4 个GPU进行训练
当前进程 local_rank: 0
模型参数数量: 51,404,800
训练数据加载完成:
  - 文件路径: ../artifacts/tinystories_train.bin
  - Token 总数: 327,680,000
  - 数据类型: uint16
  - 内存占用: 625.00 MB (虚拟，实际按需加载)
total_steps: 2500

训练配置:
  - 训练设备: 4 x GPU
  - 每个GPU的batch_size: 128
  - 全局batch_size: 512
  - 总训练步数: 2500
Step 1/2500 | Train Loss: 9.2103 | Eval Train Loss: 9.1854
Step 10/2500 | Train Loss: 8.9234
...
```

## 常见问题

### Q: 如何查看可用的GPU数量？
```bash
nvidia-smi --list-gpus
# 或者
nvidia-smi
```

### Q: 训练时显存不足怎么办？
减小 `batch_size_per_gpu` 参数，例如从128改为64或32。

### Q: 如何保存和加载检查点？
可以在代码中添加检查点保存逻辑，只在主进程（`is_main_process=True`）保存：

```python
if is_main_process and (step + 1) % save_interval == 0:
    torch.save({
        'step': step,
        'model_state_dict': model.module.state_dict(),  # 注意使用 .module
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'checkpoint_step_{step+1}.pt')
```

### Q: 如何在后台运行训练？
```bash
nohup ./run_multigpu.sh > training.log 2>&1 &
# 查看日志
tail -f training.log
```

## 技术细节

### DDP工作原理

1. **模型复制**：每个GPU上都有完整的模型副本
2. **数据分片**：每个GPU处理不同的数据batch
3. **前向传播**：各GPU独立计算前向传播
4. **梯度同步**：反向传播后，DDP自动同步所有GPU的梯度（使用All-Reduce）
5. **参数更新**：各GPU使用相同的同步后梯度更新参数

### 为什么使用DDP而不是DataParallel？

- **效率更高**：DDP使用多进程，避免了GIL限制
- **更好的扩展性**：支持多节点训练
- **更少的通信开销**：只同步梯度，不同步数据

## 相关资源

- [PyTorch DDP官方文档](https://pytorch.org/docs/stable/notes/ddp.html)
- [torchrun使用指南](https://pytorch.org/docs/stable/elastic/run.html)
