from .bpe_legacy import PairHeap, init_vocabulary, init_tokens
from .pretokenization_txt import pretokenization_parallel, pretokenization_serial
from tqdm import tqdm

class Node:
    def __init__(self, token, word_id):
        self.token = token        # 这个位置的 token id
        self.prev = None          # 前一个 Node
        self.next = None          # 后一个 Node
        self.word_id = word_id    # 属于哪个词（w_id）


def build_linked_words(words_token: list[list[int]]):
    """
    根据 words_token（list[list[token_id]]）构建链表版的 words。
    
    返回：
        word_heads: list[Node | None]  每个 w_id 对应链表的头结点
        word_tails: list[Node | None]  每个 w_id 对应链表的尾结点（可选，方便需要从尾部操作时用）
    """
    word_heads: list[Node | None] = []
    word_tails: list[Node | None] = []

    for w_id, seq in enumerate(words_token):
        head = None
        prev = None

        for tok in seq:
            node = Node(tok, w_id)
            if head is None:
                head = node
            if prev is not None:
                prev.next = node
                node.prev = prev
            prev = node

        # 这个 word 可能是空串，对应 head = None, prev = None
        word_heads.append(head)
        word_tails.append(prev)

    return word_heads, word_tails


def build_pair_index_linked(word_heads, words_id_count):
    pair2freq = defaultdict(int)
    pair2occ = defaultdict(set)

    for head in word_heads:
        if head is None:
            continue
        w_id = head.word_id
        freq = words_id_count[w_id]

        node = head
        while node is not None and node.next is not None:
            pair = (node.token, node.next.token)
            pair2freq[pair] += freq
            pair2occ[pair].add(node)  # 存左节点 node 本身
            node = node.next

    return pair2freq, pair2occ

from collections import defaultdict

def apply_merge(best_pair, new_tok, word_heads, words_count, pair2occ, pair2freq):
    """
    best_pair: (L, R)
    new_tok: 新 token id
    word_heads: list[Node | None]  # 虽然这里用不到，但保持签名兼容
    words_count: list[int]         # 每个 word_id 的频数
    pair2occ: dict[(int,int), set[Node]]  # 每个 pair 的“左节点集合”
    pair2freq: dict[(int,int), int]       # 每个 pair 的频数
    """
    L, R = best_pair

    # pair2occ[best_pair] 现在存的是所有左节点 node （node.token == L, node.next.token == R）
    nodes = list(pair2occ.get(best_pair, set()))  # 拷贝一份，因为下面要修改 pair2occ

    # 清空 best_pair 的出现
    pair2occ[best_pair].clear()
    pair2freq[best_pair] = 0

    affected_pairs = set()

    for node in nodes:
        # 这个 node 可能已经在之前的 merge 中失效了，先校验一下
        right = node.next
        if right is None:
            continue
        if node.token != L or right.token != R:
            continue  # 这条 pair 已过期

        w_id = node.word_id
        freq = words_count[w_id]

        left = node.prev
        after = right.next  # R 的右邻

        # -------------------
        # 左邻居 (x, L) → (x, new_tok)
        # -------------------
        if left is not None:
            x = left.token
            old_pair = (x, L)
            new_pair = (x, new_tok)

            # old_pair 消失：pair2occ 里存的是左节点 left
            if old_pair in pair2occ:
                pair2occ[old_pair].discard(left)
                pair2freq[old_pair] -= freq
                affected_pairs.add(old_pair)

            # new_pair 增加：左节点仍然是 left
            pair2occ.setdefault(new_pair, set()).add(left)
            pair2freq[new_pair] = pair2freq.get(new_pair, 0) + freq
            affected_pairs.add(new_pair)

        # -------------------
        # 右邻居 (R, y) → (new_tok, y)
        # -------------------
        if after is not None:
            y = after.token
            old_pair = (R, y)
            new_pair = (new_tok, y)

            # old_pair 消失：左节点是 right
            if old_pair in pair2occ:
                pair2occ[old_pair].discard(right)
                pair2freq[old_pair] -= freq
                affected_pairs.add(old_pair)

            # new_pair 增加：左节点是合并后的 node
            pair2occ.setdefault(new_pair, set()).add(node)
            pair2freq[new_pair] = pair2freq.get(new_pair, 0) + freq
            affected_pairs.add(new_pair)

        # -------------------
        # 替换 L, R → new_tok（链表版）
        # -------------------
        # 1. 把当前 node 改成 new_tok
        node.token = new_tok

        # 2. 从链表中删掉 right 这个节点
        node.next = after
        if after is not None:
            after.prev = node
        # 不强制清理 right.prev/right.next，Python 会自动 GC

    return affected_pairs   # ★★★★★ 关键！

def train_bpe(input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    words_count = pretokenization_parallel(input_path, 32, special_tokens)
    vocab = {i: bytes([i]) for i in range(256)}
    vocab = init_vocabulary(vocab, special_tokens)
    words_token, pair2freq, words_id_count = init_tokens(words_count)
    

    # 上述信息是数据集本身就有的，legacy中用数组导致pair2occ中，一旦有2个token被合并，这个word后面所有的pos就全无效了
    # 这里用链表作为数据结构

    word_heads, _ = build_linked_words(words_token)
    _ , pair2occ = build_pair_index_linked(word_heads, words_id_count)
    # 这里的pair2occ存储的是一个pair左节点的指针

    heap = PairHeap(pair2freq, vocab)
    merges = []
    # 初始化进度条
    pbar = tqdm(total=vocab_size, desc="Building BPE Vocab", ncols=100)
    pbar.update(len(vocab))  # 初始 vocab 已经有 256 个 token，提前前进

    
    while(len(vocab) < vocab_size):
        # if(len(vocab) % 100 == 0):
        #     print(f"Current vocab size:{len(vocab)}")
        # max_pair = max(pair2freq, key=pair2freq.get)
        # char_pair = tuple(vocab[i].decode('utf-8') for i in max_pair)
        best_pair, freq = heap.pop_best()
        # rev_vocab = {tok_bytes: tok_id for tok_id, tok_bytes in vocab.items()}

        if best_pair == None:
            break

        A, B = best_pair
        new_tok = len(vocab)
        #
        merged_bytes = vocab[A] + vocab[B]
        merges.append((vocab[A], vocab[B]))
        vocab[new_tok] = merged_bytes

        affected_pairs = apply_merge(
            best_pair,
            new_tok,
            word_heads,        # 原来是 words_token
            words_id_count,
            pair2occ,
            pair2freq
        )

        for p in affected_pairs: 
            # 这里的push_updated实际上是：检查这些pair对应的freq，如果大于0就加入其中。真正的丢弃发生在pop时，如果检查到和现在的freq不一致才会丢弃掉状态
            heap.push_updated(p)
        pbar.update(1)  # 每新增 1 个 token，进度条前进 1
        
    pbar.close()
    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe("../dataset/tinystories/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    # cProfile.run('train_bpe()')
    pass