import torch
import torch.nn as nn
from einops import einsum, rearrange

class RotaryPositionalEmbeddings(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    """

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: Θ value for the RoPE frequency calculations.
            d_k: The dimension of query and key vectors (must be even).
            max_seq_len: Maximum sequence length that will be inputted to precompute cos/sin buffers.
            device: The torch device to store precomputed buffers on.
        """
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        dim_freq = theta ** -(torch.arange(0, d_k, 2, device=device) / d_k)
        token_pos = torch.arange(max_seq_len, device=device)
        freqs = einsum(token_pos, dim_freq, "token_pos, dim_freq -> token_pos dim_freq")
        cos, sin = torch.cos(freqs), torch.sin(freqs)
        self.register_buffer("cos_sin", torch.stack((cos, sin)), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor based on specified token positions.

        Args:
            x: Input tensor of shape (..., seq_len, d_k). 
               Supports an arbitrary number of batch dimensions.
            token_positions: Tensor of shape (..., seq_len) specifying the absolute 
                             integer positions of each token in x.

        Returns:
            torch.Tensor: The rotated tensor of the same shape as x (..., seq_len, d_k).
            
        Note:
            Use the token_positions to slice precomputed cos and sin tensors along 
            the sequence dimension.
        """
        # cos/sin的形状是 (seq_len, half_d)
        cos, sin = self.cos_sin[:, token_positions, :]
        # (half_d xy) 代表将最后一个维度拆成两个子维度的乘积
        # 这一解包操作会将这个 xy 维度拆开：x1 拿到了 xy=0 的部分。x2 拿到了 xy=1 的部分。
        x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)
        # x1：包含了原特征维度中所有偶数索引的元素（x0, x2, x4, ...）
        # x2：包含了原特征维度中所有奇数索引的元素（x1, x3, x5, ...）
        # x1/x2的形状是 (batch, seq_len, half_d)

        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        
        # 这样就把 x1_rot 和 x2_rot 交织拼接起来了
        # [x1_rot[0], x2_rot[0], x1_rot[1], x2_rot[1], ...]
        result = rearrange([x1_rot, x2_rot], "xy ... x_half -> ... (x_half xy)")

        return result