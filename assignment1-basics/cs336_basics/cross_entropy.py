import torch
from einops import rearrange

def cross_entropy(predicted_logits, target_indices):
    """
    Compute the cross-entropy loss between predicted logits and target indices.

    Args:
        predicted_logits: Tensor of shape (batch_size, num_classes)
        target_indices: Tensor of shape (batch_size,)
    """
    max_logits = torch.max(predicted_logits, dim=-1, keepdim=True)[0]
    # softmax操作对于减去同一个值不改变结果，防止数值溢出
    cutted_logits = predicted_logits - max_logits
    log_probs = cutted_logits - torch.log(torch.sum(torch.exp(cutted_logits), dim=-1, keepdim=True))
    indices_expanded = rearrange(target_indices, "... -> ... 1")
    gathered = torch.gather(log_probs, dim=-1, index=indices_expanded)
    loss = -rearrange(gathered, "... 1 -> ...")
    return loss.mean()


if __name__ == "__main__":
    predicted_logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    target_indices = torch.tensor([0, 1])
    print(cross_entropy(predicted_logits, target_indices))