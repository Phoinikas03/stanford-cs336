import torch
import torch.nn as nn
from einops import einsum
import math
from cs336_basics.softmax import Softmax

class ScaledDotProductAttention(nn.Module):
    """
    缩放点积注意力机制实现。
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行缩放点积注意力计算。

        Args:
            q: 查询张量 (batch_size, ..., seq_len, d_k)
            k: 键张量 (batch_size, ..., seq_len, d_k)
            v: 值张量 (batch_size, ..., seq_len, d_v)
            mask: 可选的布尔遮罩 (seq_len, seq_len) 或 (..., seq_len, seq_len)
                  True 表示保留，False 表示遮蔽（注意力权重设为 0）

        Returns:
            torch.Tensor: 注意力输出结果 (batch_size, ..., seq_len, d_v)
        """
        # 1. 计算点积分数并缩放 (Scale)
        d_k = k.shape[-1]
        QK_T = einsum(q, k, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
        softmax = Softmax(dim=-1)
        
        # 2. 应用遮罩 (Masking, 如果提供)
        if mask is not None:
            QK_T = torch.where(mask, QK_T, float("-inf"))
        # torch.where(condition, x, y) 的含义
        # 这是一个条件选择函数，逐元素地根据条件选择值：
        # 当 condition 为 True 时，输出 x 的值
        # 当 condition 为 False 时，输出 y 的值
            
        # 3. 计算 Softmax 得到注意力权重
        attention_weights = softmax(QK_T)

        # 4. 将权重应用于值 (v)
        attention_output = einsum(
            attention_weights, v, 
            "... queries keys, ... keys d_v -> ... queries d_v"
        )

        return attention_output
