import torch
from collections.abc import Iterable

class GradientClipping:
    def __init__(self, max_norm=1.0, eps=1e-6):
        self.eps = eps
        self.max_norm = max_norm

    def clip(self, parameters: Iterable[torch.nn.Parameter]):
        grads = [p.grad for p in parameters if p.grad is not None]
        # 计算全局范数：将所有梯度展平并计算总范数
        norm = torch.sqrt(sum(torch.sum(g ** 2) for g in grads))
        clip_coef = min(1, self.max_norm / (norm + self.eps))
        for grad in grads:
            grad.data *= clip_coef