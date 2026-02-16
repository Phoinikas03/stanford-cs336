import torch
import numpy as np

def get_batch(x, batch_size, context_length, device):
    """
    If the size of x is large, it should use memmap to load the data.
    """
    # 1. 随机生成 batch_size 个起始索引
    # 索引范围必须保证后面有足够的长度取 input 和 target (也就是 context_length + 1)
    # len(x) - context_length 是为了防止数组越界
    ix = torch.randint(len(x) - context_length, (batch_size,))

    # 2. 从 x 中抓取数据块
    # 注意：这里需要根据起始索引 ix，把 numpy 数据切片并堆叠起来
    # x 是 numpy 数组 (可能是 memmap)，需要转成 torch tensor
    x_batch = torch.stack([torch.from_numpy((x[i : i + context_length]).astype(np.int64)) for i in ix])
    y_batch = torch.stack([torch.from_numpy((x[i+1 : i + 1 + context_length]).astype(np.int64)) for i in ix])

    # 3. 将数据移动到指定设备
    # 支持 'cuda', 'cuda:0', 'cuda:1' 等所有设备格式
    x_batch = x_batch.to(device, non_blocking=True)
    y_batch = y_batch.to(device, non_blocking=True)

    return x_batch, y_batch