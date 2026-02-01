from __future__ import annotations

import json
import os
from pathlib import Path

import tiktoken

# 这俩和 tests 里的保持一致
from tests.adapters import get_tokenizer
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

VOCAB_PATH = FIXTURES_PATH / "gpt2_vocab.json"
MERGES_PATH = FIXTURES_PATH / "gpt2_merges.txt"


def get_tokenizer_from_vocab_merges_path(
    vocab_path: str | os.PathLike,
    merges_path: str | os.PathLike,
    special_tokens: list[str] | None = None,
):
    """
    直接拷贝 tests/test_tokenizer.py 里的同名函数，
    这样保证和单元测试完全一致。
    """
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with open(vocab_path) as vocab_f:
        gpt2_vocab = json.load(vocab_f)

    gpt2_bpe_merges = []
    with open(merges_path) as f:
        for line in f:
            cleaned_line = line.rstrip()
            if cleaned_line and len(cleaned_line.split(" ")) == 2:
                gpt2_bpe_merges.append(tuple(cleaned_line.split(" ")))

    vocab = {
        gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
        for gpt2_vocab_item, gpt2_vocab_index in gpt2_vocab.items()
    }

    if special_tokens:
        for special_token in special_tokens:
            byte_encoded_special_token = special_token.encode("utf-8")
            if byte_encoded_special_token not in set(vocab.values()):
                vocab[len(vocab)] = byte_encoded_special_token

    merges = [
        (
            bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
            bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
        )
        for merge_token_1, merge_token_2 in gpt2_bpe_merges
    ]
    return get_tokenizer(vocab, merges, special_tokens)


def debug_overlapping_special_tokens():
    tokenizer = get_tokenizer_from_vocab_merges_path(
        vocab_path=VOCAB_PATH,
        merges_path=MERGES_PATH,
        special_tokens=["<|endoftext|>", "<|endoftext|><|endoftext|>"],
    )

    test_string = "Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>"

    print("=== 原始字符串 ===")
    print(repr(test_string))
    print()

    # 1. encode
    ids = tokenizer.encode(test_string)

    print("=== 编码得到的 token ids ===")
    print(ids)
    print(f"共 {len(ids)} 个 id")
    print()

    # 2. 逐 token decode 看看每个 id 对应什么
    print("=== 逐 token 解码结果（id -> 字符串） ===")
    decoded_tokens = [tokenizer.decode([tid]) for tid in ids]
    for i, (tid, s) in enumerate(zip(ids, decoded_tokens)):
        print(f"{i:02d}: id={tid:<6d} repr={s!r}")
    print()

    # 3. 数一数两种 special token 的出现次数
    single = "<|endoftext|>"
    double = "<|endoftext|><|endoftext|>"

    cnt_single = decoded_tokens.count(single)
    cnt_double = decoded_tokens.count(double)

    print("=== special token 统计 ===")
    print(f"{single!r}  出现次数: {cnt_single}")
    print(f"{double!r} 出现次数: {cnt_double}")
    print("（测试期望：single 出现 1 次，double 出现 1 次）")
    print()

    # 4. 整体 roundtrip
    roundtrip = tokenizer.decode(ids)
    print("=== 整体 decode 结果 ===")
    print(repr(roundtrip))
    print("与原始字符串是否相等？", roundtrip == test_string)


if __name__ == "__main__":
    debug_overlapping_special_tokens()
