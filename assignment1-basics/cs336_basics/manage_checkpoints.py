#!/usr/bin/env python3
"""
Checkpoint管理工具

功能:
- 列出所有checkpoint
- 查看checkpoint详细信息
- 删除旧的checkpoint
- 只保留最新的N个checkpoint

使用方法:
    python manage_checkpoints.py list                    # 列出所有checkpoint
    python manage_checkpoints.py info checkpoint.pt      # 查看checkpoint信息
    python manage_checkpoints.py keep 5                  # 只保留最新的5个checkpoint
"""

import os
import torch
import argparse
from pathlib import Path
from datetime import datetime

CHECKPOINT_DIR = "../checkpoints"


def list_checkpoints(checkpoint_dir=CHECKPOINT_DIR):
    """列出所有checkpoint文件"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return []
    
    checkpoints = list(checkpoint_path.glob("checkpoint_step_*.pt"))
    
    if not checkpoints:
        print(f"📁 {checkpoint_dir}/ 中没有找到checkpoint文件")
        return []
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    print(f"\n📁 找到 {len(checkpoints)} 个checkpoint文件:")
    print(f"{'文件名':<35} {'大小':<12} {'修改时间':<25} {'训练步数':<10}")
    print("-" * 85)
    
    for ckpt in checkpoints:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        mtime = datetime.fromtimestamp(ckpt.stat().st_mtime)
        
        # 从文件名提取步数
        try:
            step = int(ckpt.stem.split("_")[-1])
        except:
            step = "N/A"
        
        print(f"{ckpt.name:<35} {size_mb:>8.1f} MB   {mtime.strftime('%Y-%m-%d %H:%M:%S'):<25} {step:>8}")
    
    total_size = sum(ckpt.stat().st_size for ckpt in checkpoints) / (1024 * 1024)
    print("-" * 85)
    print(f"总大小: {total_size:.1f} MB\n")
    
    return checkpoints


def show_checkpoint_info(checkpoint_path):
    """显示checkpoint详细信息"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
        return
    
    print(f"\n📄 Checkpoint信息: {checkpoint_path}")
    print("-" * 60)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        print(f"训练步数: {checkpoint.get('iteration', 'N/A')}")
        print(f"包含的键: {list(checkpoint.keys())}")
        
        if 'model_state_dict' in checkpoint:
            model_params = checkpoint['model_state_dict']
            total_params = sum(p.numel() for p in model_params.values())
            print(f"模型参数数量: {total_params:,}")
            print(f"模型层数: {len(model_params)}")
        
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            print(f"优化器状态: {opt_state.get('param_groups', [{}])[0].get('lr', 'N/A')} (学习率)")
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"文件大小: {file_size:.1f} MB")
        
        print("-" * 60 + "\n")
        
    except Exception as e:
        print(f"❌ 读取checkpoint失败: {e}\n")


def keep_latest_n(n, checkpoint_dir=CHECKPOINT_DIR):
    """只保留最新的N个checkpoint，删除其他的"""
    checkpoints = list(Path(checkpoint_dir).glob("checkpoint_step_*.pt"))
    
    if len(checkpoints) <= n:
        print(f"✓ 当前只有 {len(checkpoints)} 个checkpoint，无需删除")
        return
    
    # 按修改时间排序
    checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    
    # 保留最新的N个
    to_keep = checkpoints[:n]
    to_delete = checkpoints[n:]
    
    print(f"\n将保留最新的 {n} 个checkpoint:")
    for ckpt in to_keep:
        print(f"  ✓ {ckpt.name}")
    
    print(f"\n将删除 {len(to_delete)} 个旧checkpoint:")
    total_size = 0
    for ckpt in to_delete:
        size_mb = ckpt.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  ✗ {ckpt.name} ({size_mb:.1f} MB)")
    
    print(f"\n将释放磁盘空间: {total_size:.1f} MB")
    
    # 确认删除
    response = input("\n确认删除? (yes/no): ").strip().lower()
    
    if response == 'yes':
        for ckpt in to_delete:
            ckpt.unlink()
            print(f"  ✓ 已删除: {ckpt.name}")
        print(f"\n✓ 成功删除 {len(to_delete)} 个checkpoint")
    else:
        print("\n✗ 已取消删除操作")


def main():
    parser = argparse.ArgumentParser(
        description="Checkpoint管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python manage_checkpoints.py list
  python manage_checkpoints.py info ../checkpoints/checkpoint_step_1000.pt
  python manage_checkpoints.py keep 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='命令')
    
    # list命令
    subparsers.add_parser('list', help='列出所有checkpoint')
    
    # info命令
    info_parser = subparsers.add_parser('info', help='查看checkpoint详细信息')
    info_parser.add_argument('checkpoint', help='checkpoint文件路径')
    
    # keep命令
    keep_parser = subparsers.add_parser('keep', help='只保留最新的N个checkpoint')
    keep_parser.add_argument('n', type=int, help='保留的checkpoint数量')
    keep_parser.add_argument('--dir', default=CHECKPOINT_DIR, help='checkpoint目录')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_checkpoints()
    elif args.command == 'info':
        show_checkpoint_info(args.checkpoint)
    elif args.command == 'keep':
        keep_latest_n(args.n, args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
