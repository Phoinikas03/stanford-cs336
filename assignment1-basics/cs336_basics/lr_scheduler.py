import math

class CosineLR:
    def __init__(self, max_lr=1e-3, min_lr=1e-4, warmup_steps=1000, total_steps=10000):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def step(self, step):
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        elif step > self.total_steps:
            return self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
            return self.min_lr + coeff * (self.max_lr - self.min_lr)