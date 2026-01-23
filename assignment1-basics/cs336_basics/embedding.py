import torch
import einops
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., d_model
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """

        super().__init__()
        W = torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype)
        torch.nn.init.trunc_normal_(W, std=1, a=-3, b=3)
        self.weight = nn.Parameter(W)

    def forward(self, token_ids: torch.LongTensor):
        """
        Lookup the embedding vectors for the given token IDs.

        Args:
            token_ids: torch.Tensor of shape (batch_size, sequence_length)

        Returns:
            torch.Tensor of shape (batch_size, sequence_length, embedding_dim)
        """
        output = self.weight[token_ids]
        """
        在 PyTorch 中，当你用一个 LongTensor（整数张量，这里是 token_ids）去索引另一个张量（这里是 self.weight）时，PyTorch 会自动执行Gather操作。它仅仅是把对应索引的行“拿”出来拼在一起，没有加法或乘法运算。
        维度自动广播： 这种写法会自动保留 token_ids 的所有维度结构，并在最后追加上 self.weight 被索引维度之后的维度（即 embedding_dim）
        token_ids: [B, S]
        self.weight: [V, D] (你正在索引 V 这个维度)
        结果: [B, S, D]
        """

        return output