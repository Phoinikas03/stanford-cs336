"""
多GPU分布式训练脚本 - 使用PyTorch DistributedDataParallel (DDP)

运行方法：
---------
单GPU运行（调试模式）：
    python train_loop_multigpu.py

多GPU运行（推荐）：
    torchrun --nproc_per_node=NUM_GPUS train_loop_multigpu.py
    
    例如使用4个GPU：
    torchrun --nproc_per_node=4 train_loop_multigpu.py
    
    或者使用所有可用GPU：
    torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) train_loop_multigpu.py

特点：
-----
- 使用DDP实现高效的多GPU训练
- 每个GPU处理独立的batch，总batch_size = batch_size_per_gpu × GPU数量
- 自动同步梯度和参数
- 只在主进程（rank 0）打印日志和评估
"""

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.get_batch import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.lr_scheduler import CosineLR
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import wandb
from datetime import datetime
import yaml

def train_loop():
    # ========== 分布式训练初始化 ==========
    # 从环境变量获取分布式训练参数
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 初始化进程组（使用NCCL后端，适合GPU）
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    is_main_process = (local_rank == 0)
    
    if is_main_process:
        print(f"使用 {world_size} 个GPU进行训练")
        print(f"当前进程 local_rank: {local_rank}")
    
    # ========== 训练配置 ==========
    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 6
    num_heads = 16
    d_ff = 1344
    batch_size_per_gpu = 64  # 降低batch size以避免OOM
    init_lr = 5e-4
    
    # ========== 初始化 Wandb (只在主进程) ==========
    if is_main_process:
        wandb.init(
            project="cs336-transformer-lm",
            name=f"multi-gpu-{world_size}gpus-training",
            config={
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "batch_size_per_gpu": batch_size_per_gpu,
                "num_gpus": world_size,
                "learning_rate": init_lr,
                "warmup_steps": 1000,
                "optimizer": "AdamW",
                "scheduler": "CosineLR",
            }
        )
    
    # ========== 模型初始化 ==========
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=10000,
    )
    if is_main_process:
        print(f"模型参数数量: {model.get_num_params():,}")
        wandb.config.update({"total_params": model.get_num_params()})
    
    # ========== 使用 memmap 读取 tokenized 数据 ==========
    # 重要：dtype 必须与 tokenizer.py 中保存时使用的一致
    train_data_path = "../artifacts/tinystories_train.bin"
    
    # np.memmap 的优势：
    # 1. 内存映射：不会一次性加载整个文件到内存
    # 2. 按需加载：只有访问某个位置时才加载对应的数据块
    # 3. 适合大文件：即使文件有几GB，也能高效处理
    # 4. 零拷贝：直接从磁盘映射到虚拟内存，无需额外复制
    #
    # 参数说明：
    # - mode='r': 只读模式，不会修改文件
    # - dtype=np.uint16: 必须与 tokenizer.py 中保存时使用的 dtype 一致
    #                    如果 vocab_size > 65535，tokenizer 会使用 np.uint32
    train_data = np.memmap(train_data_path, dtype=np.uint16, mode='r')
    
    if is_main_process:
        print(f"训练数据加载完成:")
        print(f"  - 文件路径: {train_data_path}")
        print(f"  - Token 总数: {len(train_data):,}")
        print(f"  - 数据类型: {train_data.dtype}")
        print(f"  - 内存占用: {train_data.nbytes / (1024**2):.2f} MB (虚拟，实际按需加载)")
    
    # ========== 设备和batch配置 ==========
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    # 全局batch_size = batch_size_per_gpu * world_size
    global_batch_size = batch_size_per_gpu * world_size
    total_steps = 2 * 327680000 // global_batch_size // context_length
    
    if is_main_process:
        print(f"total_steps: {total_steps}")
        print(f"\n训练配置:")
        print(f"  - 训练设备: {world_size} x GPU")
        print(f"  - 每个GPU的batch_size: {batch_size_per_gpu}")
        print(f"  - 全局batch_size: {global_batch_size}")
        print(f"  - 总训练步数: {total_steps}")
        
        # 更新wandb配置
        wandb.config.update({
            "global_batch_size": global_batch_size,
            "total_steps": total_steps,
        })
    
    eval_interval = 100  # 每100步评估一次
    checkpoint_interval = 1000  # 每1000步保存checkpoint
    
    # 创建带时间戳的checkpoint目录（只在主进程）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("../checkpoints", f"run_{timestamp}")
    
    if is_main_process:
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存训练配置到yaml
        config_dict = {
            "model": {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
                "rope_theta": 10000,
            },
            "training": {
                "batch_size_per_gpu": batch_size_per_gpu,
                "num_gpus": world_size,
                "global_batch_size": global_batch_size,
                "learning_rate": init_lr,
                "min_learning_rate": init_lr * 0.1,
                "warmup_steps": 100,
                "total_steps": total_steps,
                "optimizer": "AdamW",
                "optimizer_betas": [0.9, 0.95],
                "weight_decay": 0.01,
                "scheduler": "CosineLR",
            },
            "data": {
                "train_data_path": train_data_path,
                "total_tokens": len(train_data),
            },
            "checkpointing": {
                "eval_interval": eval_interval,
                "checkpoint_interval": checkpoint_interval,
            },
            "distributed": {
                "backend": "nccl",
                "world_size": world_size,
            },
            "timestamp": timestamp,
        }
        
        config_path = os.path.join(checkpoint_dir, "config.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        
        print(f"  - 评估间隔: {eval_interval} 步")
        print(f"  - Checkpoint间隔: {checkpoint_interval} 步")
        print(f"  - Checkpoint保存目录: {checkpoint_dir}")
        print(f"  - 配置文件已保存: {config_path}")
    
    # 同步checkpoint目录路径到所有进程
    if world_size > 1:
        # 广播checkpoint_dir到所有进程
        checkpoint_dir_list = [checkpoint_dir]
        dist.broadcast_object_list(checkpoint_dir_list, src=0)
        checkpoint_dir = checkpoint_dir_list[0]
    
    # ========== 使用DDP包装模型 ==========
    model.to(device)
    
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    # 在DDP包装后创建优化器和scheduler
    optimizer = AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.95), weight_decay=0.01)
    scheduler = CosineLR(
        max_lr=init_lr, 
        min_lr=init_lr * 0.1, 
        warmup_steps=100, 
        total_steps=total_steps
    )
    
    if is_main_process:
        print(f"\n学习率调度器配置:")
        print(f"  - 初始学习率: {init_lr}")
        print(f"  - 最小学习率: {init_lr * 0.1}")
    
    model.train()
    
    for step in range(total_steps):
        # 每次随机获取一个新的训练批次
        inputs_tensor, targets_tensor = get_batch(
            x=train_data,
            batch_size=batch_size_per_gpu,
            context_length=context_length,
            device=device
        )
        
        # 训练步骤
        optimizer.zero_grad()
        logits = model(inputs_tensor)
        loss = cross_entropy(logits, targets_tensor)
        loss.backward()
        optimizer.step()
        
        # 更新学习率
        current_lr = scheduler.step(step)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        
        # 记录训练loss和学习率到wandb（只在主进程）
        if is_main_process:
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/step": step + 1,
            }, step=step + 1)
        
        # 定期评估（所有进程都执行以保持同步，但只在主进程打印）
        if (step + 1) % eval_interval == 0 or step == 0:
            # 同步所有进程，确保在评估前都完成当前步
            if world_size > 1:
                dist.barrier()
            
            model.eval()
            with torch.no_grad():
                # 在训练集上评估
                train_loss = 0
                num_eval_batches = 10
                for _ in range(num_eval_batches):
                    eval_inputs, eval_targets = get_batch(
                        x=train_data,
                        batch_size=batch_size_per_gpu,
                        context_length=context_length,
                        device=device
                    )
                    eval_logits = model(eval_inputs)
                    train_loss += cross_entropy(eval_logits, eval_targets).item()
                train_loss /= num_eval_batches
            
            # 只在主进程打印评估结果和记录到wandb
            if is_main_process:
                print(f"Step {step + 1}/{total_steps} | Train Loss: {loss.item():.4f} | Eval Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
                
                # 记录评估loss到wandb
                wandb.log({
                    "eval/train_loss": train_loss,
                }, step=step + 1)
            
            model.train()
            
            # 同步所有进程，确保在继续训练前都完成评估
            if world_size > 1:
                dist.barrier()
        elif is_main_process and (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        # 定期保存checkpoint（只在主进程）
        if is_main_process and (step + 1) % checkpoint_interval == 0:
            # 对于DDP模型，需要保存model.module而不是model
            model_to_save = model.module if world_size > 1 else model
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step + 1}.pt")
            save_checkpoint(model_to_save, optimizer, step + 1, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # 记录checkpoint保存事件到wandb
            wandb.log({
                "checkpoint/step": step + 1,
            }, step=step + 1)
    
    # 训练完成，关闭wandb（只在主进程）
    if is_main_process:
        wandb.finish()
        print("\n✓ 训练完成，wandb日志已同步")
    
    # 清理分布式进程组
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train_loop()