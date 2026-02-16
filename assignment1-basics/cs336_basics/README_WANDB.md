# Wandb集成使用指南

本文档说明如何使用Weights & Biases (wandb)来跟踪和可视化训练过程。

## 什么是Wandb？

Wandb是一个强大的机器学习实验跟踪工具，可以：
- 📊 实时可视化训练指标（loss、学习率等）
- 💾 自动保存实验配置和结果
- 📈 对比多次实验的性能
- 🌐 通过网页访问实验记录
- 👥 团队协作和分享实验

## 安装和配置

### 1. 安装wandb

```bash
pip install wandb
```

### 2. 登录wandb账号

首次使用需要登录：

```bash
wandb login
```

这会打开浏览器，要求你登录wandb账号并获取API key。如果没有账号，可以免费注册：https://wandb.ai/

### 3. 或者设置API key（服务器环境推荐）

如果在没有浏览器的服务器上：

```bash
# 在 https://wandb.ai/authorize 获取你的API key
export WANDB_API_KEY=your_api_key_here

# 或者直接使用命令
wandb login your_api_key_here
```

## 已集成的功能

训练脚本已经自动集成了wandb，会记录以下信息：

### 训练指标
- `train/loss` - 每步的训练loss
- `train/learning_rate` - 实时学习率
- `train/step` - 当前训练步数

### 评估指标
- `eval/train_loss` - 评估时的平均loss（每100步）

### Checkpoint事件
- `checkpoint/step` - checkpoint保存的步数

### 配置信息
自动记录所有超参数：
- 模型配置：vocab_size, d_model, num_layers, num_heads等
- 训练配置：batch_size, learning_rate, optimizer等
- GPU信息：GPU数量、全局batch size等

## 使用方法

### 单GPU训练

```bash
python train_loop.py
```

训练开始后会看到：
```
wandb: Currently logged in as: your_username
wandb: Tracking run with wandb version 0.16.0
wandb: Run data is saved locally in ./wandb/run-xxx
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run single-gpu-training
wandb: ⭐️ View project at https://wandb.ai/your_username/cs336-transformer-lm
wandb: 🚀 View run at https://wandb.ai/your_username/cs336-transformer-lm/runs/xxx
```

### 多GPU训练

```bash
./run_multigpu.sh
```

训练运行名称会自动包含GPU数量，例如：`multi-gpu-4gpus-training`

## 查看实验结果

### 1. 网页端（推荐）

训练开始后，点击终端输出中的链接，例如：
```
🚀 View run at https://wandb.ai/your_username/cs336-transformer-lm/runs/xxx
```

在网页上可以：
- 实时查看训练曲线
- 对比多次实验
- 下载实验数据
- 分享实验链接

### 2. 本地查看

训练日志也会保存在本地：
```bash
ls ./wandb/
```

## 自定义配置

### 修改项目名称

在训练脚本中修改：

**train_loop.py (第23行)：**
```python
wandb.init(
    project="your-project-name",  # 改成你的项目名
    name="your-run-name",          # 改成你的运行名
    config={...}
)
```

**train_loop_multigpu.py (第68行)：**
```python
wandb.init(
    project="your-project-name",
    name=f"multi-gpu-{world_size}gpus-training",
    config={...}
)
```

### 添加自定义标签

```python
wandb.init(
    project="cs336-transformer-lm",
    name="experiment-1",
    tags=["baseline", "tinystories", "transformer"],  # 添加标签
    config={...}
)
```

### 添加更多指标

在训练循环中可以记录任何指标：

```python
wandb.log({
    "train/loss": loss.item(),
    "train/learning_rate": current_lr,
    "train/gradient_norm": grad_norm,  # 新增指标
    "train/tokens_per_second": tokens_per_sec,  # 新增指标
}, step=step + 1)
```

## 离线模式

如果网络不稳定或不想实时同步：

```bash
# 设置离线模式
export WANDB_MODE=offline

# 运行训练
python train_loop.py

# 训练完成后手动同步
wandb sync ./wandb/offline-run-xxx
```

## 禁用wandb

如果临时不想使用wandb：

```bash
# 方法1：设置环境变量
export WANDB_MODE=disabled
python train_loop.py

# 方法2：在代码中设置
wandb.init(mode="disabled")
```

## 高级功能

### 1. 记录模型架构

```python
# 在模型初始化后
wandb.watch(model, log="all", log_freq=100)
```

这会记录：
- 模型参数的梯度
- 模型参数的分布
- 模型架构图

### 2. 记录图片/表格

```python
# 记录图片
wandb.log({"examples": wandb.Image(img)})

# 记录表格
wandb.log({"predictions": wandb.Table(data=data, columns=columns)})
```

### 3. 保存模型到wandb

```python
# 保存checkpoint到wandb云端
artifact = wandb.Artifact('model-checkpoint', type='model')
artifact.add_file(checkpoint_path)
wandb.log_artifact(artifact)
```

### 4. 对比多次实验

在wandb网页端：
1. 进入项目页面
2. 选择多个runs
3. 点击"Compare"按钮
4. 查看并排对比的图表

## 常见问题

### Q: 如何找到我的API key？

A: 访问 https://wandb.ai/authorize 获取

### Q: wandb会占用多少存储空间？

A: 本地日志很小（几MB），主要数据在云端。免费账户有100GB存储空间。

### Q: 训练很慢，是wandb导致的吗？

A: wandb的开销很小（<1%），如果觉得慢可以：
- 减少log频率：`wandb.log(..., commit=False)` 
- 使用异步模式（默认已开启）
- 使用离线模式

### Q: 多GPU训练时所有进程都在记录日志吗？

A: 不会，代码已经配置为只在主进程（rank 0）记录日志，避免重复。

### Q: 如何删除不需要的实验？

A: 在wandb网页端，选择run，点击"Delete"按钮。

### Q: 可以设置实验为私有吗？

A: 可以，在创建项目时设置为private，或在项目设置中修改。

## 最佳实践

1. **使用有意义的运行名称**：包含实验的关键信息
   ```python
   name=f"lr-{lr}_bs-{batch_size}_layers-{num_layers}"
   ```

2. **添加标签方便筛选**：
   ```python
   tags=["baseline", "experiment", "v1"]
   ```

3. **记录重要的配置变化**：
   ```python
   wandb.config.update({"note": "使用了新的数据增强"})
   ```

4. **定期检查训练曲线**：及早发现问题

5. **保存关键checkpoint到wandb**：方便后续复现

## 相关链接

- Wandb官方文档：https://docs.wandb.ai/
- Wandb Python API：https://docs.wandb.ai/ref/python/
- Wandb社区：https://community.wandb.ai/

## 示例截图

训练开始后，在wandb网页端你会看到：
- 实时更新的loss曲线
- 学习率变化曲线
- 系统资源使用情况（GPU、内存等）
- 完整的配置和代码版本
- 训练时长和进度

现在你可以开始训练，并通过wandb跟踪整个实验过程了！🚀
