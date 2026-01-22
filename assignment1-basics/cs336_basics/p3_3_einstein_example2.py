import torch
from einops import rearrange, einsum
# We have a batch of images, and for each image we want to generate 10 dimmed versions based on some scaling factor:
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)
## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")

dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
images, dim_by,
"batch height width channel, dim_value -> batch dim_value height width channel"
)
print(dimmed_images.shape)

print(images[0, 0, 0, 0])
for i in range(10):
    print(dimmed_images[0, i, 0, 0, 0])