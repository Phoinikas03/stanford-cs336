import torch
from einops import rearrange, einsum
import einx

channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
channels_last,
"batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
channels_first, B,
"batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
channels_first_transformed,
"batch channel (height width) -> batch height width channel",
height=height, width=width
)
# Or, if you’re feeling crazy: all in one go using einx.dot (einx equivalent of einops.einsum)
height = width = 32
channels_last_transformed = einx.dot(
"batch row_in col_in channel, (row_out col_out) (row_in col_in)"
"-> batch row_out col_out channel",
channels_last, B,
col_in=width, col_out=width
)