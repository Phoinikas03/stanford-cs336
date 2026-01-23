import math
import torch
import einops
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        sigma = math.sqrt(2 / (in_features + out_features))
        W = torch.randn(out_features, in_features, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, std=sigma, a=-3*sigma, b=3*sigma)
        self.weight = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out