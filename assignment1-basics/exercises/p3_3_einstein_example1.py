import torch
from einops import rearrange, einsum
## Basic implementation

D = torch.randn(10, 100).to("cuda")
A = torch.randn(10, 100).to("cuda")

Y = D @ A.T
print(Y.shape)
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?
## Einsum is self-documenting and robust
# D A -> Y
Y = einsum(D, A, "batch d_in, batch d_out -> batch d_out")
## Or, a batched version where D can have any leading dimensions but A is constrained.
print(Y.shape)

Y = einsum(D, A, "... d_in, batch d_out -> ... d_out")
print(Y.shape)

D = torch.randn(10, 100, 100).to("cuda")
A = torch.randn(10, 100).to("cuda")

Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
print(Y.shape)
## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
print(Y.shape)