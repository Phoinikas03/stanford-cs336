from collections.abc import Callable
from typing import Optional
import torch
import math

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        """
        Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                # 获取迭代次数 t，初始为 0
                t = state.get("t", 0)
                
                grad = p.grad.data
                # 应用带 1/sqrt(t+1) 衰减的梯度更新
                p.data -= (lr / math.sqrt(t + 1)) * grad
                
                # 更新状态中的迭代次数
                state["t"] = t + 1
                
        return loss

if __name__ == "__main__":
    # 简单的测试示例
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), device="cuda"))
    opt = SGD([weights], lr=1e2)
    
    print("Starting optimization...")
    for t in range(10):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Step {t}, Loss: {loss.item():.6f}")
        loss.backward()
        opt.step()
