import torch
import torch.nn as nn
import torch.nn.functional as F
from cs336_basics.linear import Linear  # 导入你自己的类

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(SwiGLU, self).__init__()
        
        # 1. 两个升维层：从 d_model 映射到 d_ff
        # 使用nn.Linear时要注意bias=False，用之前定义的linear类，不带bias
        # self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc1 = Linear(d_model, d_ff)
        self.fc3 = Linear(d_model, d_ff)
        # fc1作为门控，fc3作为线性变换，它们把从d_model映射到d_ff维度，之间用Hardmard积作为门控激活，再通过fc2映射回d_model维度
        # 2. 一个降维层：从 d_ff 映射回 d_model
        self.fc2 = Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)) * self.fc3(x))