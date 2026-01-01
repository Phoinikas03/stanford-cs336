import regex as re
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional
from multiprocessing import Pool

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def _process_shard(args) -> Dict[str, int]:
    """
    子进程处理一个 shard：
    - texts: 这一批样本的文本列表
    - special_tokens: 需要作为“切分边界”的 special tokens（不参与合并）
    """
    texts, special_tokens = args
    words_count: Dict[str, int] = {}

    pattern = None
    if special_tokens:
        # 和你原来的逻辑一致：用 special token 做 split 分界，不保留 token 本身
        pattern = "|".join(re.escape(tok) for tok in special_tokens)

    for text in texts:
        if pattern:
            parts = re.split(pattern, text)
        else:
            parts = [text]

        for p in parts:
            split_words = re.findall(PAT, p)
            for w in split_words:
                if w in words_count:
                    words_count[w] += 1
                else:
                    words_count[w] = 1

    return words_count

def pretokenization_parallel_hf(dataset,
                                num_processes: int,
                                text_column: str = "text",
                                special_tokens: Optional[List[str]] = None) -> Dict[str, int]:
    """
    并行版本：对 HuggingFace Dataset 做预分词 + 词频统计（多进程）。
    - dataset: HF 的 Dataset，例如 load_dataset("Skylion007/openwebtext", split="train")
    - num_processes: 进程数
    - text_column: 文本所在列名
    - special_tokens: 需要作为切分边界的 special token 列表，一般在hf中已经切分好，不需要再额外传入
    """
    from collections import Counter

    n = len(dataset)
    if n == 0:
        return {}

    shard_size = math.ceil(n / num_processes)

    tasks = []
    for i in range(num_processes):
        start = i * shard_size
        end = min(n, (i + 1) * shard_size)
        if start >= end:
            break
        # 直接把这一段的文本取出来传给子进程
        texts = dataset[start:end][text_column]  # -> List[str]
        tasks.append((texts, special_tokens or []))

    merged = Counter()
    with Pool(processes=len(tasks)) as pool:
        for shard_counts in pool.map(_process_shard, tasks):
            merged.update(shard_counts)

    return dict(merged)

if __name__ == "__main__":
    pretokenization("../dataset/TinyStoriesV2-GPT4-valid.txt", 32, "plain_text", None)