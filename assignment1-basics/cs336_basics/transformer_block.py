import torch
import torch.nn as nn
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multi_head_attention import MultiHeadSelfAttentionWithRope
from cs336_basics.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttentionWithRope(
            d_model=d_model,
            d_in=d_model,
            d_out=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len
        )
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm architecture
        # 1. Attention sublayer with residual
        x = x + self.attn(self.ln1(x), token_positions)
        # 2. FFN sublayer with residual
        x = x + self.ffn(self.ln2(x))
        return x
