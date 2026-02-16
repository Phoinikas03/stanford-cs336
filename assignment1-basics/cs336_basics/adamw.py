import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        super(AdamW, self).__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                # 初始化状态
                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                m, v = state["m"], state["v"]
                state["t"] += 1
                t = state["t"]

                g = p.grad.data
                
                # 1. 应用权重衰减 (AdamW 的特点是衰减直接作用于参数)
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                # 2. 更新一阶和二阶动量 (原地操作)
                # m = beta1 * m + (1 - beta1) * g
                m.mul_(group["betas"][0]).add_(g, alpha=1 - group["betas"][0])
                # v = beta2 * v + (1 - beta2) * g^2
                v.mul_(group["betas"][1]).addcmul_(g, g, value=1 - group["betas"][1])

                # 3. 计算偏差修正后的学习率
                bias_correction1 = 1 - group["betas"][0] ** t
                bias_correction2 = 1 - group["betas"][1] ** t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                # 4. 更新参数 (原地操作)
                # denom = sqrt(v) + eps
                denom = v.sqrt().add_(eps)
                # p = p - step_size * m / denom
                p.data.addcdiv_(m, denom, value=-step_size)

if __name__ == "__main__":
    # 简单的测试示例
    weights = torch.nn.Parameter(5 * torch.randn((10, 10), device="cuda"))
    opt = AdamW([weights], lr=1.0, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    
    print("Starting optimization...")
    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Step {t}, Loss: {loss.item():.6f}")
        loss.backward()
        opt.step()
