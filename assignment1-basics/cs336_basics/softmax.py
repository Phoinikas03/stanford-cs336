import torch

class Softmax(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = torch.max(x, dim=self.dim, keepdim=True)[0]
        x = x - x_max
        denominator = torch.sum(torch.exp(x), dim=self.dim, keepdim=True)
        return torch.exp(x) / denominator