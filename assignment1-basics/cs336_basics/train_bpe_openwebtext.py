import os
import pickle
import cProfile
from pathlib import Path

from cs336_basics.bpe import train_bpe  # or import your run_train_bpe function

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """
    Wrapper around the train_bpe function from cs336_basics.
    """
    return train_bpe(input_path, vocab_size, special_tokens)


def train_bpe_tinystories():
    """
    Train a byte-level BPE tokenizer on the openwebtext dataset (32k vocab)
    and serialize the vocabulary and merges to disk.
    """

    # === Configuration ===
    input_path = "../dataset/openwebtext/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    # Output directory (recommended structure)
    output_dir = Path("../artifacts")
    output_dir.mkdir(exist_ok=True)

    vocab_path = output_dir / "openwebtext_vocab.pkl"
    merges_path = output_dir / "openwebtext_merges.pkl"

    # === Train BPE ===
    print(f"Training BPE on: {input_path}")
    vocab, merges = run_train_bpe(input_path, vocab_size, special_tokens)

    # === Serialize ===
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    with open(merges_path, "wb") as f:
        pickle.dump(merges, f)

    longest_id, longest_token = max(vocab.items(), key=lambda x: len(x[1]))
    print("Longest token ID:", longest_id)
    print("Bytes length:", len(longest_token))
    print("Token (decoded):", longest_token.decode('utf-8', errors='replace'))
    print("*" * 50)
    # 最长的token:ssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssssss
    # 这是一条典型的二倍递归合并链：
    # (303,    b's', b's')                                →   ss
    # (6415,   b'ss', b'ss')                              →   ssss
    # (6595,   b'ssss', b'ssss')                          →   ssssssss
    # (6645,   b'ssssssss', b'ssssssss')                  →   16 个 s
    # (6685,   b'ssssssssssssssss', b'ssssssssssssssss')  →   32 个 s
    # (6761,   32个s, 32个s)                              →   64 个 s
    # (6913,   64个s, 64个s)                              →   128 个 s
    # (7258,   128个s, 128个s)                            →   256 个 s
    # (8569,   256个s, 256个s)                            →   512 个 s   ← 你看到的长 token 来自这里
    # 为什么会出现这样的 512 个 s 的 token？
    # 因为：
    # openwebtext 数据中包含很长的 ssssssssss 模式（例如蛇的声音、某种重复字符）

    # BPE 在后期 merges（6000 之后）不再有太多高频 pair 可合并

    # 于是它会把剩余的高频字符重复模式不断合并

    # 这会形成 指数增长 的长 token，BPE 的标准行为完全允许出现。


    print(f"Saved vocab to:  {vocab_path}")
    print(f"Saved merges to: {merges_path}")


if __name__ == "__main__":
    train_bpe_tinystories()
    # cProfile.run('train_bpe_tinystories()', 'openwebtext.prof')
