"""
从checkpoint恢复训练的示例脚本

使用方法:
    python resume_training.py --checkpoint ../checkpoints/checkpoint_step_1000.pt
"""

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.checkpoint import load_checkpoint
import torch
import argparse

def resume_from_checkpoint(checkpoint_path):
    """
    从checkpoint恢复模型和优化器
    
    Args:
        checkpoint_path: checkpoint文件路径
    
    Returns:
        model: 恢复的模型
        optimizer: 恢复的优化器
        start_step: 从哪一步继续训练
    """
    # 初始化模型（必须与训练时的配置一致）
    model = TransformerLM(
        vocab_size=10000,
        context_length=256,
        d_model=512,
        num_layers=4,
        num_heads=16,
        d_ff=1344,
        rope_theta=10000,
    )
    
    # 初始化优化器（参数需要与训练时一致）
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    
    # 加载checkpoint
    start_step = load_checkpoint(checkpoint_path, model, optimizer)
    
    print(f"✓ 成功加载checkpoint: {checkpoint_path}")
    print(f"✓ 将从第 {start_step + 1} 步继续训练")
    print(f"✓ 模型参数数量: {model.get_num_params():,}")
    
    return model, optimizer, start_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从checkpoint恢复训练")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="checkpoint文件路径，例如: ../checkpoints/checkpoint_step_1000.pt")
    args = parser.parse_args()
    
    # 恢复模型和优化器
    model, optimizer, start_step = resume_from_checkpoint(args.checkpoint)
    
    # 移动到GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    
    print(f"\n模型已准备好，可以继续训练...")
    print(f"设备: {device}")
    
    # 这里可以继续训练循环...
    # for step in range(start_step, total_steps):
    #     ...
