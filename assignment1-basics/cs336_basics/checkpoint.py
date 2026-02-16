import torch
import os
import typing


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    """
    保存模型、优化器和迭代次数的检查点
    
    参数:
        model: PyTorch 模型
        optimizer: PyTorch 优化器
        iteration: 当前训练迭代次数
        out: 输出路径或文件对象
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    """
    从检查点加载模型和优化器状态
    
    参数:
        src: 检查点文件路径或文件对象
        model: 要恢复状态的模型
        optimizer: 要恢复状态的优化器
        
    返回:
        int: 保存时的迭代次数
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
