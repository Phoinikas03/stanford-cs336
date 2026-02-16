# 训练配置管理

本文档说明训练配置的保存和使用方法。

## 自动配置保存

每次训练开始时，脚本会自动：
1. ✅ 创建带时间戳的checkpoint目录
2. ✅ 在该目录中保存完整的训练配置（`config.yaml`）
3. ✅ 在该目录中保存所有checkpoint

## 目录结构

```
checkpoints/
├── run_20240216_143025/          # 带时间戳的运行目录
│   ├── config.yaml               # 训练配置文件
│   ├── checkpoint_step_1000.pt   # Checkpoint文件
│   ├── checkpoint_step_2000.pt
│   └── ...
├── run_20240216_150312/          # 另一次运行
│   ├── config.yaml
│   └── ...
└── ...
```

## 配置文件内容

### 单GPU训练 (train_loop.py)

`config.yaml` 包含以下内容：

```yaml
model:
  vocab_size: 10000
  context_length: 256
  d_model: 512
  num_layers: 4
  num_heads: 16
  d_ff: 1344
  rope_theta: 10000

training:
  batch_size: 128
  learning_rate: 0.0001
  min_learning_rate: 0.00001
  warmup_steps: 1000
  total_steps: 10000
  optimizer: AdamW
  optimizer_betas:
  - 0.9
  - 0.95
  weight_decay: 0.01
  scheduler: CosineLR

data:
  train_data_path: ../artifacts/tinystories_train.bin
  total_tokens: 327680000

checkpointing:
  eval_interval: 100
  checkpoint_interval: 1000

device: cuda
timestamp: '20240216_143025'
```

### 多GPU训练 (train_loop_multigpu.py)

多GPU训练的配置文件额外包含分布式训练信息：

```yaml
model:
  vocab_size: 10000
  context_length: 256
  d_model: 512
  num_layers: 6
  num_heads: 16
  d_ff: 1344
  rope_theta: 10000

training:
  batch_size_per_gpu: 32
  num_gpus: 4
  global_batch_size: 128
  learning_rate: 0.0005
  min_learning_rate: 0.00005
  warmup_steps: 100
  total_steps: 40000
  optimizer: AdamW
  optimizer_betas:
  - 0.9
  - 0.95
  weight_decay: 0.01
  scheduler: CosineLR

data:
  train_data_path: ../artifacts/tinystories_train.bin
  total_tokens: 327680000

checkpointing:
  eval_interval: 100
  checkpoint_interval: 1000

distributed:
  backend: nccl
  world_size: 4

timestamp: '20240216_150312'
```

## 查看配置

### 使用Python读取

```python
import yaml

# 读取配置
with open('../checkpoints/run_20240216_143025/config.yaml', 'r') as f:
    config = yaml.safe_load(f)

print(f"模型层数: {config['model']['num_layers']}")
print(f"学习率: {config['training']['learning_rate']}")
print(f"Batch size: {config['training']['batch_size']}")
```

### 使用命令行查看

```bash
# 查看配置文件
cat ../checkpoints/run_20240216_143025/config.yaml

# 或使用更美观的格式
python -c "import yaml; print(yaml.dump(yaml.safe_load(open('../checkpoints/run_20240216_143025/config.yaml'))))"
```

## 从配置文件恢复训练

创建一个脚本从配置文件恢复模型：

```python
import yaml
import torch
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.checkpoint import load_checkpoint

# 读取配置
config_path = '../checkpoints/run_20240216_143025/config.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# 使用配置初始化模型
model = TransformerLM(
    vocab_size=config['model']['vocab_size'],
    context_length=config['model']['context_length'],
    d_model=config['model']['d_model'],
    num_layers=config['model']['num_layers'],
    num_heads=config['model']['num_heads'],
    d_ff=config['model']['d_ff'],
    rope_theta=config['model']['rope_theta'],
)

# 加载checkpoint
checkpoint_path = '../checkpoints/run_20240216_143025/checkpoint_step_5000.pt'
start_step = load_checkpoint(checkpoint_path, model, optimizer)

print(f"✓ 模型和配置已恢复，从第 {start_step + 1} 步继续训练")
```

## 比较不同运行的配置

```python
import yaml
import os
from pathlib import Path

def compare_configs(run1_dir, run2_dir):
    """比较两次运行的配置差异"""
    config1_path = os.path.join(run1_dir, 'config.yaml')
    config2_path = os.path.join(run2_dir, 'config.yaml')
    
    with open(config1_path, 'r') as f:
        config1 = yaml.safe_load(f)
    with open(config2_path, 'r') as f:
        config2 = yaml.safe_load(f)
    
    print(f"比较配置:")
    print(f"Run 1: {run1_dir}")
    print(f"Run 2: {run2_dir}\n")
    
    # 比较模型配置
    print("模型配置差异:")
    for key in config1['model']:
        val1 = config1['model'][key]
        val2 = config2['model'][key]
        if val1 != val2:
            print(f"  {key}: {val1} -> {val2}")
    
    # 比较训练配置
    print("\n训练配置差异:")
    for key in config1['training']:
        val1 = config1['training'][key]
        val2 = config2['training'][key]
        if val1 != val2:
            print(f"  {key}: {val1} -> {val2}")

# 使用示例
compare_configs(
    '../checkpoints/run_20240216_143025',
    '../checkpoints/run_20240216_150312'
)
```

## 配置模板

如果你想预先定义配置并从配置文件启动训练，可以创建配置模板：

```yaml
# config_template.yaml
model:
  vocab_size: 10000
  context_length: 256
  d_model: 512
  num_layers: 4
  num_heads: 16
  d_ff: 1344

training:
  batch_size: 128
  learning_rate: 0.0001
  warmup_steps: 1000
  total_steps: 10000
```

然后创建一个从配置文件启动的训练脚本：

```python
def train_from_config(config_path):
    """从配置文件启动训练"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 使用配置初始化模型和训练
    model = TransformerLM(**config['model'])
    # ... 训练代码
```

## 查找特定配置的运行

```bash
# 查找所有checkpoint目录
ls -lt ../checkpoints/

# 查找特定日期的运行
ls -d ../checkpoints/run_20240216_*

# 查找所有配置文件
find ../checkpoints -name "config.yaml"

# 查找使用特定学习率的运行
grep -r "learning_rate: 0.0001" ../checkpoints/*/config.yaml
```

## OOM问题解决方案

如果遇到 `CUDA out of memory` 错误：

### 方法1：降低batch size

在配置中查看当前的batch size，然后在训练脚本中降低：

**单GPU：**
```python
batch_size = 64  # 原来是128
```

**多GPU：**
```python
batch_size_per_gpu = 16  # 原来是32
```

### 方法2：减少模型层数

```python
num_layers = 4  # 原来是6
```

### 方法3：减少模型维度

```python
d_model = 256  # 原来是512
d_ff = 672     # 原来是1344
```

## 最佳实践

1. **保留所有配置文件**：配置文件很小，建议全部保留以便追溯
2. **使用描述性的wandb运行名称**：与checkpoint目录时间戳对应
3. **定期备份重要运行的配置和checkpoint**
4. **在笔记本中记录实验设置和结果**
5. **使用版本控制管理配置模板**

## 配置文件的优势

- ✅ **完整记录**：每次运行的所有参数都被记录
- ✅ **可复现**：可以精确复现任何历史实验
- ✅ **易于比较**：快速对比不同实验的配置差异
- ✅ **自动化**：无需手动记录参数
- ✅ **版本追踪**：时间戳确保不会覆盖历史配置

## 相关文件

- `train_loop.py` - 单GPU训练脚本（带配置保存）
- `train_loop_multigpu.py` - 多GPU训练脚本（带配置保存）
- `checkpoint.py` - Checkpoint保存和加载函数
- `README_CHECKPOINT.md` - Checkpoint使用指南
