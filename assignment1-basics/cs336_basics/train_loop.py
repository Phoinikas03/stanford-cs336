from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.get_batch import get_batch
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.adamw import AdamW
from cs336_basics.lr_scheduler import CosineLR
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
import numpy as np
import torch
import os
import wandb
from datetime import datetime
import yaml

def train_loop():
    # ========== 训练配置 ==========
    batch_size = 128
    context_length = 256
    init_lr = 1e-4
    vocab_size = 10000
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    
    # ========== 初始化 Wandb ==========
    wandb.init(
        project="cs336-transformer-lm",  # 项目名称
        name="single-gpu-training",       # 运行名称
        config={
            "vocab_size": vocab_size,
            "context_length": context_length,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "d_ff": d_ff,
            "batch_size": batch_size,
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
    print(f"模型参数数量: {model.get_num_params():,}")
    
    # 记录模型参数数量到wandb
    wandb.config.update({"total_params": model.get_num_params()})
    
    optimizer = AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = 327680000 // batch_size // context_length
    scheduler = CosineLR(max_lr=init_lr, min_lr=init_lr * 0.1, warmup_steps=1000, total_steps=total_steps)
    
    wandb.config.update({"total_steps": total_steps})
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
    
    print(f"训练数据加载完成:")
    print(f"  - 文件路径: {train_data_path}")
    print(f"  - Token 总数: {len(train_data):,}")
    print(f"  - 数据类型: {train_data.dtype}")
    print(f"  - 内存占用: {train_data.nbytes / (1024**2):.2f} MB (虚拟，实际按需加载)")
    
    # ========== 设备和训练间隔配置 ==========
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"total_steps: {total_steps}")
    eval_interval = 100  # 每100步评估一次
    checkpoint_interval = 1000  # 每1000步保存checkpoint
    
    # 创建带时间戳的checkpoint目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = os.path.join("../checkpoints", f"run_{timestamp}")
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
            "batch_size": batch_size,
            "learning_rate": init_lr,
            "min_learning_rate": init_lr * 0.1,
            "warmup_steps": 1000,
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
        "device": device,
        "timestamp": timestamp,
    }
    
    config_path = os.path.join(checkpoint_dir, "config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n训练设备: {device}")
    print(f"总训练步数: {total_steps}")
    print(f"评估间隔: {eval_interval} 步")
    print(f"Checkpoint间隔: {checkpoint_interval} 步")
    print(f"Checkpoint保存目录: {checkpoint_dir}")
    print(f"配置文件已保存: {config_path}")
    
    # ========== 训练循环 ==========
    model.to(device)
    model.train()
    
    for step in range(total_steps):
        # 每次随机获取一个新的训练批次
        inputs_tensor, targets_tensor = get_batch(
            x=train_data,
            batch_size=batch_size,
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
        
        # 记录训练loss和学习率到wandb
        wandb.log({
            "train/loss": loss.item(),
            "train/learning_rate": current_lr,
            "train/step": step + 1,
        }, step=step + 1)
            
        # 定期评估
        if (step + 1) % eval_interval == 0 or step == 0:
            model.eval()
            with torch.no_grad():
                # 在训练集上评估
                train_loss = 0
                num_eval_batches = 10
                for _ in range(num_eval_batches):
                    eval_inputs, eval_targets = get_batch(
                        x=train_data,
                        batch_size=batch_size,
                        context_length=context_length,
                        device=device
                    )
                    eval_logits = model(eval_inputs)
                    train_loss += cross_entropy(eval_logits, eval_targets).item()
                train_loss /= num_eval_batches
            
            # 记录评估loss到wandb
            wandb.log({
                "eval/train_loss": train_loss,
            }, step=step + 1)
                
            print(f"Step {step + 1}/{total_steps} | Train Loss: {loss.item():.4f} | Eval Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")
            model.train()
        elif (step + 1) % 10 == 0:
            print(f"Step {step + 1}/{total_steps} | Train Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
        # 定期保存checkpoint
        if (step + 1) % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step + 1}.pt")
            save_checkpoint(model, optimizer, step + 1, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # 记录checkpoint保存事件到wandb
            wandb.log({
                "checkpoint/step": step + 1,
            }, step=step + 1)
    
    # 训练完成，关闭wandb
    wandb.finish()
    print("\n✓ 训练完成，wandb日志已同步")

if __name__ == "__main__":
    train_loop()