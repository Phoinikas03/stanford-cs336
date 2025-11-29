import os
import cProfile
import regex as re
from typing import BinaryIO, List
from multiprocessing import Pool
from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1): # 只需要找n - 1个分界线
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))




def _process_chunk(args):
    """子进程处理一个 chunk：读取 -> 解码 -> 按 special token 切分"""
    path, start, end, pattern = args
    with open(path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    # 按特殊 token 分割
    parts = re.split(pattern, chunk)
    # 你可以选择去掉空字符串
    # parts = [p for p in parts if p]   # 可选
    # byte_parts = [t.encode("utf-8") for t in parts]
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words_count = {}

    for p in parts:
        split_words = re.findall(PAT, p)
        # words_count.update(split_words)
        for w in split_words: 
            if w in words_count:
                words_count[w] += 1
            else:
                words_count[w] = 1
    
    return words_count

## Usage
def pretokenization_serial(path: str, num_chunks: int):
    """
    串行版本的预分词 + 词频统计：
    - 不用多进程
    - 逻辑跟 _process_chunk 完全一致
    - 方便你打断点调试
    """
    # 1. 先算 chunk 边界（和并行版保持一致）
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_chunks, b"<|endoftext|>")

    special_tokens = ["<|endoftext|>"]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    # 2. 串行遍历每个 chunk，直接调用 _process_chunk
    total_counts: Dict[str, int] = {}

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        # 这里不走多进程，直接在本进程调用
        chunk_counts = _process_chunk((path, start, end, pattern))

        # 3. 合并每个 chunk 的词频
        for k, v in chunk_counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    return total_counts
    

def pretokenization_parallel(path: str, num_processes: int):
    # BPE “不跨词合并”，这是由 pretokenization 的 split 保证的
    # BPE 不会生成跨词（如 "I like"）的 token
    
    with open(path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    special_tokens = ["<|endoftext|>"]
    pattern = "|".join(re.escape(tok) for tok in special_tokens)

    # 给每个进程准备 (path, start, end, pattern) 参数
    tasks = [
        (path, start, end, pattern)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    total_parts: List[str] = []

    # 并行处理所有 chunk
    with Pool(processes=num_processes) as pool:
        for parts in pool.map(_process_chunk, tasks):
            # 把每个进程的结果加到总列表中
            total_parts.append(parts)

    merged = Counter()
    for d in total_parts:
        merged.update(d)
    return dict(merged)


if __name__ == "__main__":
    # pretokenization("../dataset/TinyStoriesV2-GPT4-valid.txt", 32)
    cProfile.run('pretokenization_parallel("../dataset/TinyStoriesV2-GPT4-train.txt", 128)')
    