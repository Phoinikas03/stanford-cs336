import torch
import torch.nn as nn
from cs336_basics.linear import Linear
from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
from einops import rearrange
from cs336_basics.rope import RotaryPositionalEmbeddings

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, d_in: int, d_out: int, num_heads: int):
        """
        初始化因果多头自注意力模块。
        
        Args:
            d_model (int): Transformer 块输入的维度。
            num_heads (int): 多头注意力的头数。
        """
        super().__init__()
        
        # 1. 检查维度约束: d_model 必须能被 num_heads 整除
        # 根据图片提示: d_k = d_v = d_model / h
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads

        # TODO: 在这里定义线性投影层 (Linear Projections)
        # 通常包括: W_q, W_k, W_v 和 W_o
        self.w_q = Linear(d_in, d_model)
        self.w_k = Linear(d_in, d_model)
        self.w_v = Linear(d_in, d_model)
        self.w_o = Linear(d_model, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播函数。

        Args:
            x (torch.Tensor): 输入张量，形状通常为 (batch_size, seq_len, d_model)

        Returns:
            torch.Tensor: 输出张量，形状与输入相同 (batch_size, seq_len, d_model)
        """
        
        # TODO: 1. 对输入 x 进行线性投影得到 Q, K, V
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        # TODO: 2. 将 Q, K, V 分割成多头 (Split heads)
        # 形状变换: (batch_size, seq_len, d_model) -> (batch_size, num_heads, seq_len, d_k)
        Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=self.num_heads)

        # TODO: 3. 计算 Scaled Dot-Product Attention
        # 注意：这里必须应用 Causal Mask (因果掩码)，确保位置 i 只能关注到 j <= i 的位置
        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2], device=Q.device)).bool()
        mask = rearrange(mask, "query key -> 1 query key")
        attention_output = ScaledDotProductAttention()(Q, K, V, mask)

        # TODO: 4. 连接多头的结果 (Concat heads)
        attention_output = rearrange(attention_output, "... heads seq d_v -> ... seq (heads d_v)")

        # TODO: 5. 通过输出线性层 (W_o) 得到最终输出
        output = self.w_o(attention_output)

        return output

class MultiHeadSelfAttentionWithRope(nn.Module):
    def __init__(self, d_model: int, d_in: int, d_out: int, num_heads: int, theta: float, max_seq_len: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        self.w_q = Linear(d_in, d_model)
        self.w_k = Linear(d_in, d_model)
        self.w_v = Linear(d_in, d_model)
        self.w_o = Linear(d_model, d_out)
        self.rope = RotaryPositionalEmbeddings(theta, self.d_k, self.max_seq_len)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MultiHeadSelfAttentionWithRope module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_in)
            token_positions: Tensor of shape (batch_size, seq_len) specifying the positions of the tokens

        Returns:
            Output tensor of shape (batch_size, seq_len, d_out)
        """
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=self.num_heads)
        V = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=self.num_heads)

        Q = self.rope(Q, token_positions)
        K = self.rope(K, token_positions)
        # 添加因果掩码，不泄露后面的信息
        mask = torch.tril(torch.ones(Q.shape[-2], K.shape[-2], device=Q.device)).bool()
        mask = rearrange(mask, "query key -> 1 query key")
        attention_output = ScaledDotProductAttention()(Q, K, V, mask)
        attention_output = rearrange(attention_output, "... heads seq d_v -> ... seq (heads d_v)")
        output = self.w_o(attention_output)

        return output