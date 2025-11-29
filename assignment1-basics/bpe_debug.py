# bpe_debug.py
import json
from pathlib import Path

# 引用你的实现
from tests.adapters import run_train_bpe
from tests.common import FIXTURES_PATH, gpt2_bytes_to_unicode

# ===== 你可以调这个参数 =====
DEBUG_VOCAB_SIZE = 10000
PRINT_TOP_N = 20
# ===========================

def load_reference_merges():
    """Load reference merges from GPT2 merge file."""
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    fname = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    with open(fname, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in t1]),
                bytes([gpt2_byte_decoder[token] for token in t2]),
            )
            for t1, t2 in gpt2_reference_merges
        ]
    return reference_merges


def compare_merges(my_merges, ref_merges, N=20):
    """Print first N merges and report first mismatch."""
    print("\n=== My merges (first {}): ===".format(N))
    for i, m in enumerate(my_merges[:N]):
        print(f"{i}: {m}")

    print("\n=== Reference merges (first {}): ===".format(N))
    for i, m in enumerate(ref_merges[:N]):
        print(f"{i}: {m}")

    print("\n=== Comparing step-by-step ===")
    max_compare = min(len(my_merges), len(ref_merges))

    for i in range(max_compare):
        if my_merges[i] != ref_merges[i]:
            print(f"\n❌ MISMATCH at index {i}:")
            print("  my_merge :", my_merges[i])
            print("  ref_merge:", ref_merges[i])
            return i

    print("\nNo mismatch in first {} merges.".format(max_compare))
    return None


if __name__ == "__main__":
    print(f"\n>>> Running train_bpe with vocab_size={DEBUG_VOCAB_SIZE}\n")

    # 1) 加载 corpus 数据
    input_path = FIXTURES_PATH / "corpus.en"

    # 2) 运行你的 BPE（缩小 vocab_size）
    vocab, my_merges = run_train_bpe(
        input_path=input_path,
        vocab_size=DEBUG_VOCAB_SIZE,
        special_tokens=["<|endoftext|>"],
    )

    # 3) 读取 reference merges
    ref_merges = load_reference_merges()

    # 4) 对比，并找到第一个不一致的地方
    mismatch_idx = compare_merges(my_merges, ref_merges, PRINT_TOP_N)

    if mismatch_idx is not None:
        print("\n=== You should inspect what happens at merge index {} ===".format(mismatch_idx))
        print("""
建议在你的 train_bpe 中加入：

    if step == <mismatch_idx>:
        print("DEBUG pair2freq for conflict:")
        for p, f in sorted(pair2freq.items(), key=lambda x: (-x[1], x[0])):
            print(p, f)
        import pdb; pdb.set_trace()

这样你能看到：
  - 当前所有 pair 的频率排序
  - reference 选择的 pair
  - 你的实现选择的 pair
  - 谁的频率一样、tie-break 不一致
""")
    else:
        print("\n🎉 No mismatch detected in the first {} merges".format(PRINT_TOP_N))
