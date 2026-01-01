#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sample 10 documents from TinyStories and OpenWebText (stored as .txt, separated by <|endoftext|>),
encode them with your previously-trained tokenizers (10K and 32K vocab),
and compute compression ratio (bytes / token) for each tokenizer.
"""

import os
import random
import pickle
from typing import List

# ====== 配置区域：根据你的实际情况修改 ======

# TinyStories 和 OpenWebText 的 txt 路径（数据格式：多篇 text，用 <|endoftext|> 分割）
TINYSTORIES_TXT_PATH = "../dataset/tinystories/TinyStoriesV2-GPT4-valid.txt"
OPENWEBTEXT_TXT_PATH = "../dataset/openwebtext/owt_valid.txt"

# 你训练 TinyStories tokenizer（10K vocab）时保存的 merges 和 vocab（pickle）
TINYSTORIES_MERGES_PKL = "../artifacts/tinystories_merges.pkl"
TINYSTORIES_VOCAB_PKL = "../artifacts/tinystories_vocab.pkl"

# 你训练 OpenWebText tokenizer（32K vocab）时保存的 merges 和 vocab（pickle）
OPENWEBTEXT_MERGES_PKL = "../artifacts/openwebtext_merges.pkl"
OPENWEBTEXT_VOCAB_PKL = "../artifacts/openwebtext_vocab.pkl"

# 从每个语料中采样的文档数
NUM_DOCS = 10

# 随机种子保证可复现
SEED = 42

# 你的自定义 tokenizer 类（示例：你需要用自己的类替换掉这个 import）
# 假设接口类似：MyBPETokenizer(merges, vocab)，并且有 encode(text) -> List[int]
from tokenizer import Tokenizer  # TODO: 用你实际的模块名 / 类名替换


# ====== 工具函数 ======

def load_texts_from_txt(path: str, sep: str = "<|endoftext|>") -> List[str]:
    """
    从单个 txt 文件中读取所有文本，根据特殊分隔符 sep 分割成多篇文档。
    会 strip 每段并去掉空字符串。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    docs = [doc.strip() for doc in raw.split(sep)]
    # 过滤掉完全为空的文档
    docs = [doc for doc in docs if len(doc) > 0]
    return docs


def sample_docs(texts: List[str], num_docs: int, seed: int = 42) -> List[str]:
    """
    从文本列表中随机采样 num_docs 个文档。
    如果数量不够，则使用全部文档。
    """
    random.seed(seed)
    if len(texts) <= num_docs:
        return texts
    return random.sample(texts, num_docs)


def load_merges_and_vocab(merges_pkl: str, vocab_pkl: str):
    """
    从 pickle 文件中加载 merges 和 vocab。
    具体结构取决于你训练时如何保存，这里只负责 load。
    """
    with open(merges_pkl, "rb") as f:
        merges = pickle.load(f)
    with open(vocab_pkl, "rb") as f:
        vocab = pickle.load(f)
    return merges, vocab


def build_tokenizer(merges_pkl: str, vocab_pkl: str) -> Tokenizer:
    """
    从 merges/vocab pickle 创建你的自定义 tokenizer 实例。
    假设 MyBPETokenizer 的构造函数接受 merges 和 vocab 这两个参数。
    如果你的类接口不同，请在这里做相应修改。
    """
    merges, vocab = load_merges_and_vocab(merges_pkl, vocab_pkl)
    tokenizer = Tokenizer(merges=merges, vocab=vocab)
    return tokenizer


def compute_compression_ratio(tokenizer, texts: List[str]) -> float:
    """
    对给定的 texts，用 tokenizer 编码，计算压缩率：
        compression_ratio = total_bytes / total_tokens

    total_bytes: 原始 UTF-8 编码的字节数总和
    total_tokens: tokenizer 输出的 token 数总和

    要求：tokenizer.encode(text) 返回 token id 列表 List[int]
    如果你的 encode 接口不同，比如返回对象里有 .ids 字段，请在这里适配。
    """
    total_bytes = 0
    total_tokens = 0

    for text in texts:
        # 计算原始字节数（UTF-8）
        b = len(text.encode("utf-8"))
        total_bytes += b

        # 使用 tokenizer 编码为 ID
        token_ids = tokenizer.encode(text)  # 这里假设返回的是 List[int]
        num_tokens = len(token_ids)
        total_tokens += num_tokens

    if total_tokens == 0:
        raise ValueError("Total number of tokens is zero. Check tokenizer or texts.")

    compression_ratio = total_bytes / total_tokens
    return compression_ratio

import time

def estimate_throughput(tokenizer, texts, repeat: int = 5) -> float:
    """
    粗略估计 tokenizer 吞吐量（bytes/second）。

    参数：
      tokenizer: 你的 tokenizer 实例，提供 encode(text) 接口。
      texts: 一组代表性样本文档（比如你已经采样好的 10 篇）。
      repeat: 为了拉长时间、减少噪音，把这批文档重复 encode repeat 次。

    返回：
      throughput_bytes_per_sec: 每秒处理的字节数（bytes/s）
    """
    # 先算总字节数（重复后的）
    single_batch_bytes = sum(len(t.encode("utf-8")) for t in texts)
    total_bytes = single_batch_bytes * repeat

    # 预热一次，避免第一次调用 overhead 影响结果（可选）
    for t in texts:
        _ = tokenizer.encode(t)

    start = time.perf_counter()
    for _ in range(repeat):
        for t in texts:
            _ = tokenizer.encode(t)
    end = time.perf_counter()

    elapsed = end - start
    if elapsed <= 0:
        raise RuntimeError("Elapsed time is non-positive, timing failed.")

    throughput_bytes_per_sec = total_bytes / elapsed
    return throughput_bytes_per_sec


# ====== 主逻辑 ======

def main():
    #######################################################################################################################
    # 1. 加载文本数据并切分文档
    print("Loading TinyStories texts...")
    tinystories_all_docs = load_texts_from_txt(TINYSTORIES_TXT_PATH)

    print("Loading OpenWebText texts...")
    openwebtext_all_docs = load_texts_from_txt(OPENWEBTEXT_TXT_PATH)

    # 2. 随机采样文档
    print(f"Sampling {NUM_DOCS} documents from TinyStories...")
    tinystories_docs = sample_docs(tinystories_all_docs, NUM_DOCS, seed=SEED)

    print(f"Sampling {NUM_DOCS} documents from OpenWebText...")
    openwebtext_docs = sample_docs(openwebtext_all_docs, NUM_DOCS, seed=SEED)

    # 3. 构建两个 tokenizer（10K & 32K）
    print("Building TinyStories tokenizer (10K vocab)...")
    tiny_tokenizer = build_tokenizer(TINYSTORIES_MERGES_PKL, TINYSTORIES_VOCAB_PKL)

    print("Building OpenWebText tokenizer (32K vocab)...")
    owt_tokenizer = build_tokenizer(OPENWEBTEXT_MERGES_PKL, OPENWEBTEXT_VOCAB_PKL)

    # 4. 计算压缩率：bytes/token
    print("Computing compression ratio for TinyStories tokenizer on TinyStories samples...")
    tiny_ratio = compute_compression_ratio(tiny_tokenizer, tinystories_docs)

    print("Computing compression ratio for OpenWebText tokenizer on OpenWebText samples...")
    owt_ratio = compute_compression_ratio(owt_tokenizer, openwebtext_docs)

    # 5. 输出结果
    print("\n=== Compression Ratios (bytes/token) ===")
    print(f"TinyStories tokenizer (10K vocab) on TinyStories (10 docs): {tiny_ratio:.4f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) on OpenWebText (10 docs): {owt_ratio:.4f} bytes/token")
    # TinyStories tokenizer (10K vocab) on TinyStories (10 docs): 4.0581 bytes/token
    # OpenWebText tokenizer (32K vocab) on OpenWebText (10 docs): 4.2416 bytes/token
    #######################################################################################################################

    # 用 TinyStories tokenizer 去 tokenize OpenWebText 样本
    print("Computing compression ratio for TinyStories tokenizer on OpenWebText samples...")
    tiny_on_owt_ratio = compute_compression_ratio(tiny_tokenizer, openwebtext_docs)

    # 用 OpenWebText tokenizer 去 tokenize TinyStories 样本
    print("Computing compression ratio for OpenWebText tokenizer on TinyStories samples...")
    owt_on_tiny_ratio = compute_compression_ratio(owt_tokenizer, tinystories_docs)

    print(f"TinyStories tokenizer (10K vocab) on OpenWebText (10 docs): {tiny_on_owt_ratio:.4f} bytes/token")
    print(f"OpenWebText tokenizer (32K vocab) on TinyStories (10 docs): {owt_on_tiny_ratio:.4f} bytes/token")
    # TinyStories tokenizer (10K vocab) on OpenWebText (10 docs): 3.1657 bytes/token
    # OpenWebText tokenizer (32K vocab) on TinyStories (10 docs): 3.9427 bytes/token
    #######################################################################################################################

    print("\nEstimating tokenizer throughput (bytes/second)...")

    tiny_throughput = estimate_throughput(tiny_tokenizer, tinystories_docs, repeat=10)
    owt_throughput  = estimate_throughput(owt_tokenizer, openwebtext_docs, repeat=10)

    print(f"TinyStories tokenizer (10K) throughput: {tiny_throughput:.2f} bytes/s")
    print(f"OpenWebText tokenizer (32K) throughput: {owt_throughput:.2f} bytes/s")

    
if __name__ == "__main__":
    main()
