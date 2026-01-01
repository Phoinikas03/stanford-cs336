import heapq
import cProfile
from .pretokenization_txt import pretokenization_parallel, pretokenization_serial
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict

Pair = Tuple[int, int]  # 例如一个 byte pair

class PairHeap:
    def __init__(self, pair_counts: Dict[Pair, int], vocab: Dict[int, bytes]):
        """
        pair_counts: {pair: count}，外面维护的“最新频次字典”
        堆里存的是 (key, pair)，key 用于排序。
        """
        self.pair_counts = pair_counts
        self.vocab = vocab
        self.heap: List[Tuple[Tuple[int, Tuple[int, int]], Pair]] = []

        for pair, cnt in pair_counts.items():
            if cnt > 0:
                key = self._make_key(pair, cnt)
                self.heap.append((key, pair))
        heapq.heapify(self.heap)

    def _lex_key_desc(self, pair: Pair):
        """
        注意这里的"字典序更大"不是 token_id 更大，而是实际的字节表示更大。
        我们希望：
            pair1 的 (vocab[t1], vocab[t2]) 字节串更大
        ⇒  在最小堆里，它对应的 key 更小（优先弹出）。

        做法：
        1. 把 bytes 转成一个包含哨兵 -1 的整数序列 g(bs) = [b0, ..., bk-1, -1]
           这样 g 的字典序与原始 bytes 完全一致（因为 -1 < 0..255）
        2. 再对整个序列做 255 - x 的映射，得到 f(bs)：
               如果 s1 < s2，则 f(s1) > f(s2)
           也就是说，原本“更大”的字节串，现在对应“更小”的 f(bs)。
        3. 返回的是 tuple[int,...]，用于 heap 的比较（tuple 按字典序比较）
        """
        def inv(bs: bytes) -> tuple[int, ...]:
            # g(bs) = [b0, ..., bk-1, -1]
            g = list(bs) + [-1]
            # f(bs) = [255 - x for x in g(bs)]
            return tuple(255 - x for x in g)

        t1 = self.vocab[pair[0]]
        t2 = self.vocab[pair[1]]
        return (inv(t1), inv(t2))


    def _make_key(self, pair: Pair, cnt: int) -> Tuple[int, Tuple[int, int]]:
        """
        构造排序 key：
        - 先按频率降序：用 -cnt
        - 再按 pair 逆字典序：用 (-a, -b)
          这样 heappop 会把“频率最大 + 字典序最大”的 pair 放在最前面
        """
        return (-cnt, self._lex_key_desc(pair))

    def push_updated(self, pair: Pair):
        """
        当你外面更新了 pair_counts[pair] 之后，
        调这个函数往堆里推一个“新版本”记录。
        旧版本留在堆里，等 pop 时发现过期再丢掉。
        """
        cnt = self.pair_counts.get(pair, 0)
        if cnt <= 0:
            # 频次 <= 0 的不用进堆
            return
        key = self._make_key(pair, cnt)
        heapq.heappush(self.heap, (key, pair))

    def pop_best(self) -> Tuple[Optional[Pair], int]:
        """
        取出当前“真实意义上”最佳的 pair：
        - 频率最高
        - 频率相同时 lexicographically 最大
        会自动跳过堆里所有过期记录（lazy update 的关键）
        """
        while self.heap:
            (key, pair) = heapq.heappop(self.heap)
            stored_cnt = -key[0]
            current_cnt = self.pair_counts.get(pair, 0)

            # 如果和当前最新值一致，就是一个“新鲜”的记录
            if current_cnt == stored_cnt and current_cnt > 0:
                return pair, current_cnt
            # 否则是旧的 / 已被删的，丢弃继续

        return None, 0  # 堆空或没有有效 pair

def init_vocabulary(vocab, special_tokens):
    
    # vocab[256] = "<|endoftext|>".encode("utf-8")
    for i, special_token in enumerate(special_tokens):
        vocab[256 + i] = special_token.encode("utf-8")

    return vocab


def convert_word_to_int_pairs(word: str):
    # 返回这个word的utf-8表示，以及相邻字符的pair
    b = word.encode("utf-8")      # UTF-8 编码 → bytes
    ints = list(b)                # bytes → [int, int, int...]
    
    pairs = []
    for i in range(len(ints) - 1):
        pairs.append((ints[i], ints[i+1]))
    return ints, pairs

def init_tokens(words):
    initial_pairs_count = defaultdict(int)
    word_tokens = []
    words_id_count = []
    for w, c in words.items():
        word_token, pairs = convert_word_to_int_pairs(w)
        word_tokens.append(word_token)
        words_id_count.append(c)

        for p in pairs:
            if p in initial_pairs_count:
                initial_pairs_count[p] += c
            else:
                initial_pairs_count[p] = c

    return word_tokens, initial_pairs_count, words_id_count

def build_pair_index(words: List[List[int]]) -> Dict[Tuple[int, int], Set[Tuple[int, int]]]:
    pair2occ: Dict[Tuple[int, int], Set[Tuple[int, int]]] = defaultdict(set)

    for w_id, w in enumerate(words):
        # w 是一个 int 列表，比如 [104, 101, 108, 108, 111]
        for pos in range(len(w) - 1):
            pair = (w[pos], w[pos + 1])
            pair2occ[pair].add((w_id, pos))

    return pair2occ

def apply_merge(best_pair, new_tok, words_token, words_count, pair2occ, pair2freq):
    L, R = best_pair
    # pair2occ构建了这样一个映射：把一个token pair，映射到[(w_id, pos)]，它表明这个pair在单词w_id的pos位置上出现
    # positions存储了当前pair在哪些词中出现
    positions = list(pair2occ.get(best_pair, []))  # 拷贝一份，因为下面要修改

    # 清空 best_pair 的出现
    pair2occ[best_pair].clear()
    pair2freq[best_pair] = 0

    affected_pairs = set()
    sum = 0
    for (w_id, pos) in positions:
        seq = words_token[w_id]
        freq = words_count[w_id]
        sum += freq

        # merge 前，确认位置仍然有效
        if pos >= len(seq) - 1 or seq[pos] != L or seq[pos+1] != R:
            continue

        # -------------------
        # 左邻居 (x, L) → (x, new_tok)
        # -------------------
        if pos - 1 >= 0:
            x = seq[pos-1] # 这里不能仅提取word的前一个字符，要提取它的token表示，因为这个字符有可能已经和更前面的合并了，因此表示发生了变化
            old_pair = (x, L)
            new_pair = (x, new_tok)

            # old_pair 消失
            if old_pair in pair2occ:
                pair2occ[old_pair].discard((w_id, pos-1))
                pair2freq[old_pair] -= freq # 特别注意这里每次修改的是一个word里的pair，所以会影响freq次当前pair出现次数
                affected_pairs.add(old_pair)

            # new_pair 增加
            pair2occ[new_pair].add((w_id, pos-1))
            pair2freq[new_pair] += freq
            affected_pairs.add(new_pair)

        # -------------------
        # 右邻居 (R, y) → (new_tok, y)
        # -------------------
        if pos + 2 < len(seq):
            y = seq[pos+2]
            old_pair = (R, y)
            new_pair = (new_tok, y)

            # old_pair 消失
            if old_pair in pair2occ:
                pair2occ[old_pair].discard((w_id, pos+1))
                pair2freq[old_pair] -= freq
                affected_pairs.add(old_pair)

            # new_pair 增加
            pair2occ[new_pair].add((w_id, pos))
            pair2freq[new_pair] += freq
            affected_pairs.add(new_pair)

        # -------------------
        # 替换 L, R → new_tok
        # -------------------
        seq[pos] = new_tok
        del seq[pos+1]

    return affected_pairs   # ★★★★★ 关键！


def train_bpe(input_path: str ,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,)-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    words_count = pretokenization_parallel(input_path, 32)
    vocab = {i: bytes([i]) for i in range(256)}
    vocab = init_vocabulary(vocab, special_tokens)
    words_token, pair2freq, words_id_count = init_tokens(words_count)
    pair2occ = build_pair_index(words_token)

    heap = PairHeap(pair2freq)
    merges = []
    while(len(vocab) < vocab_size):
        if(len(vocab) % 100 == 0):
            print(f"Current vocab size:{len(vocab)}")
        max_pair = max(pair2freq, key=pair2freq.get)
        char_pair = tuple(vocab[i].decode('utf-8') for i in max_pair)
        best_pair, freq = heap.pop_best()

        if best_pair == None:
            break

        A, B = best_pair
        new_tok = len(vocab)
        merged_bytes = vocab[A] + vocab[B]
        merges.append((vocab[A], vocab[B]))
        vocab[new_tok] = merged_bytes

        affected_pairs = apply_merge(best_pair, new_tok, words_token, words_id_count, pair2occ, pair2freq)

        for p in affected_pairs: 
            # 这里的push_updated实际上是：检查这些pair对应的freq，如果大于0就加入其中。真正的丢弃发生在pop时，如果检查到和现在的freq不一致才会丢弃掉状态
            heap.push_updated(p)

    return vocab, merges

if __name__ == "__main__":
    vocab, merges = train_bpe("../dataset/tinystories/TinyStoriesV2-GPT4-valid.txt", 10000, ["<|endoftext|>"])
    # cProfile.run('train_bpe()')
    pass