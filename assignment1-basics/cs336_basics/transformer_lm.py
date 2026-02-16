import torch
import torch.nn as nn
from einops import rearrange, repeat
from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len)
        batch_size, seq_len = x.shape

        # Generate token positions for RoPE
        token_positions = torch.arange(seq_len, device=x.device)
        token_positions = repeat(token_positions, "seq -> b seq", b=batch_size)

        # 1. Embeddings
        x = self.token_embeddings(x)  # (batch_size, seq_len, d_model)

        # 2. Transformer Blocks
        for layer in self.layers:
            x = layer(x, token_positions)

        # 3. Final Layer Norm
        x = self.ln_final(x)

        # 4. LM Head (logits)
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)

        return logits

    def get_num_params(self) -> int:
        """
        计算模型的总参数数量
        
        Returns
        -------
        int
            模型中所有可训练参数的总数
        """
        return sum(p.numel() for p in self.parameters())
