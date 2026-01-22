from collections.abc import Iterable, Iterator
from typing import Dict, List, Tuple, Optional
import pickle
import regex as re


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: Optional[List[str]] = None,
    ) -> None:
        """
        Construct a tokenizer from a given vocabulary and list of merges.

        Parameters
        ----------
        vocab : dict[int, bytes]
            Mapping from token id to token bytes.
        merges : list[tuple[bytes, bytes]]
            BPE merges, in order.
        special_tokens : list[str] | None
            Optional list of special tokens as strings.
        """
        # ---- 基础结构：id -> bytes, merges ----
        self.vocab: Dict[int, bytes] = dict(vocab)
        self.merges: List[Tuple[bytes, bytes]] = list(merges)

        # special_tokens 是用户给的字符串列表
        self.special_tokens: List[str] = list(special_tokens or [])

        # 对于special token需要维护一个特别的体系
        # ---- bytes 体系：bytes <-> id ----
        # 用于普通 BPE token（包括初始 0–255 以及后续 merge 出来的 token）
        self._byte2id: Dict[bytes, int] = {b: i for i, b in self.vocab.items()}

        # ---- special token 字符串体系：str <-> id ----
        # 这两个映射是“额外的”，用来在 encode/decode 时识别/还原特殊 token
        self._special_token_to_id: Dict[str, int] = {}
        self._id_to_special_token: Dict[int, str] = {}

        # 把 special_tokens 补充/对齐到 vocab 里，同时建立 str <-> id 的映射
        for st in self.special_tokens:
            # special token 作为字符串，内部用 utf-8 bytes 表示
            b = st.encode("utf-8")

            # 如果这个 bytes 已经在 vocab 里，直接复用它的 id
            if b in self._byte2id:
                tid = self._byte2id[b]
            else:
                # 否则追加到 vocab 末尾
                tid = len(self.vocab)
                self.vocab[tid] = b
                self._byte2id[b] = tid

            # 在“字符串体系”中记录映射
            self._special_token_to_id[st] = tid
            self._id_to_special_token[tid] = st

        # 预计算 merges 的 rank，方便 encode 时快速查找 best pair
        # NOTE: 为什么要维护这个rank
        # 我们在这里必须“找出 rank 最小的可 merge pair 再合并”，
        # 不能看到哪个 pair 能 merge 就随便先 merge 哪个。
        #
        # 原因（非常重要）：
        # 1. merges 列表是按“训练时的频率/重要性顺序”排好的：
        #       merges[0] 的优先级最高（rank 最小），
        #       merges[1] 次之，依此类推。
        #    encode 时必须严格按这个优先级来做 merge，
        #    才能和训练时的统计规律保持一致。
        #
        # 2. BPE 的 decode 并不会“反向拆分 token”，它只会：
        #       bytes = b"".join(vocab[id] for id in ids)
        #       text  = bytes.decode("utf-8", errors="replace")
        #    也就是说，decode 不会看 merges，更不会告诉你
        #    一个 token 是由哪些小 token 组合来的。
        #    token 的内部结构在 decode 阶段是“不可见”的。
        #
        # 3. 举个例子：
        #       假设训练时有这些合法的 merges：
        #           A, B   -> AB
        #           AB, C  -> ABC
        #       所以“ABC”这个子串理想的 encode 是：
        #           [ID_ABC]
        #       如果 encode 的时候你随便 merge，变成：
        #           A + BC 或者 AB + C
        #       最后 decode 出来的文本虽然还是 "ABC"，
        #       但 token 序列变成：
        #           [ID_A, ID_BC]   或   [ID_AB, ID_C]
        #       而不是训练时学到的：
        #           [ID_ABC]
        #
        #    对模型来说，“文本”只是表象，它真正看到的是 token id 序列。
        #    训练时模型只学过 [ID_ABC] 这一个整体 token 的表示，
        #    并没有学过 [ID_A, ID_BC] 这种拆分方式，
        #    这会导致：
        #       - embedding 查表不同
        #       - positional encoding 不同
        #       - 注意力模式完全不同
        #    最终就是：同样的字符串 "ABC"，模型的输入却完全不一样。
        #
        # 4. 换句话说：
        #    BPE 的“正确性标准”不是“decode 后的字符串一样”，
        #    而是“encode 出来的 token 序列要和训练用的 tokenizer 一致”。
        #    如果我们不按 rank（即 merges 的顺序）来选择要 merge 的 pair，
        #    就会产生“训练时不存在的 token 组合”，
        #    这相当于在运行时发明了新的“非官方 token”，模型并不会理解它们。
        #
        # 因此，这里必须遍历当前序列中所有相邻 pair，
        # 找出 rank 最小的那个 pair（即在 merges 中最靠前的那对），
        # 只 merge 这一对，然后再重复这一过程。

        self._merge_ranks: Dict[Tuple[int, int], int] = {}
        for rank, (b1, b2) in enumerate(self.merges):
            id1 = self._byte2id.get(b1)
            id2 = self._byte2id.get(b2)
            if id1 is not None and id2 is not None:
                self._merge_ranks[(id1, id2)] = rank
    # ---------------------------------------------------------------------
    # classmethod: from_files
    # ---------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: Optional[List[str]] = None,
    ) -> "Tokenizer":
        """
        Load vocab + merges from disk and construct a Tokenizer.

        Parameters
        ----------
        vocab_filepath : str
            Path to serialized vocab (e.g., pickle of dict[int, bytes]).
        merges_filepath : str
            Path to serialized merges (e.g., pickle of list[tuple[bytes, bytes]]).
        special_tokens : list[str] | None
            Optional list of special tokens.

        Returns
        -------
        Tokenizer
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    # ---------------------------------------------------------------------
    # encode / encode_iterable / decode
    # ---------------------------------------------------------------------
    def _encode_plain(self, text: str) -> list[int]:
        # 文本 -> bytes
        b = text.encode("utf-8", errors="replace")

        # 初始序列：每个 byte 对应一个 token id（假设 0–255 都在 vocab 中）
        ids = [self._byte2id[bytes([ch])] for ch in b]

        # 边界情况：空字符串
        if not ids:
            return []

        # 如果没配置 merges，直接返回 byte-level ids
        if not getattr(self, "_merge_ranks", None):
            return ids

        # BPE merge 循环
        while True:
            # 找当前序列中 rank 最小的可 merge pair
            # 如果字符串中含
            best_rank = None
            best_pos = None

            # 这里每次遍历找rank最小的pair，但其实可以用优先队列来优化，每次都只找rank最小的pair，这样时间复杂度可以降到O(nlogn)
            for i in range(len(ids) - 1):
                pair = (ids[i], ids[i + 1])
                rank = self._merge_ranks.get(pair)
                if rank is None:
                    continue
                if (best_rank is None) or (rank < best_rank):
                    best_rank = rank
                    best_pos = i

            # 没有可 merge 的 pair，停止
            if best_pos is None:
                break

            i = best_pos
            pair = (ids[i], ids[i + 1])

            # 获取新 token 的 bytes，并找出它对应的 id
            b1 = self.vocab[pair[0]]
            b2 = self.vocab[pair[1]]
            merged_bytes = b1 + b2
            new_id = self._byte2id[merged_bytes]  # 训练时保证存在

            # 把 ids[i], ids[i+1] 替换成 new_id
            ids = ids[:i] + [new_id] + ids[i+2:]

        return ids

    def encode(self, text: str) -> List[int]:
        """
        Encode an input text into a sequence of token IDs.

        Steps (建议思路):
        1. 处理/识别 special tokens（如果有的话，先在字符串层面切分）
        2. 对普通文本部分做 byte-level BPE：
           - 文本 -> bytes
           - 初始序列为每个单字节对应的 token id
           - 迭代应用 merges，直到没有可 merge 或达到稳定
        3. 拼接 special token id 与普通 token id，返回列表

        Parameters
        ----------
        text : str

        Returns
        -------
        list[int]
        """
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if not self.special_tokens:
            ids: List[int] = []
            # pretokenization: 把整个 text 切成若干片段（作业示例中的 ['the', ' cat', ' ate']）
            pretokens = re.findall(PAT, text)
            for tok in pretokens:
                ids.extend(self._encode_plain(tok))
            return ids

        
        # 1. 构造匹配 special tokens 的正则
        #    用 re.escape 防止里面的 <, |, > 之类影响正则
        #    加上 () 捕获组，让 split 后保留分隔符

        #    关键：按长度从长到短排序，避免重叠时短的先吃掉，如果有2个special_token：A和B，且B是A的前缀，那么优先匹配A
        special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(t) for t in special_tokens_sorted) + ")"

        pieces = re.split(special_pattern, text)
        # 举例：
        # text = "hello <system_prompt> world"
        # pieces = ["hello ", "<system_prompt>", " world"]

        ids: list[int] = []

        for piece in pieces:
            if not piece:
                continue

            # 2. 如果整个 piece 就是一个 special token，直接映射
            if piece in self._special_token_to_id:
                ids.append(self._special_token_to_id[piece])
                continue

            pretokens = re.findall(PAT, piece)
            for tok in pretokens:
                ids.extend(self._encode_plain(tok))

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings into a stream of token IDs.

        用于节省内存的场景，比如对大文件逐行 tokenization。

        Parameters
        ----------
        iterable : Iterable[str]

        Yields
        ------
        int
            token id
        """
        for line in iterable:
            ids = self.encode(line)
            for tid in ids:
                yield tid

    def decode(self, ids: List[int]) -> str:
        """
        Decode a sequence of token IDs into text.

        Steps (建议思路):
        1. 把每个 id 转成 bytes（注意处理 unknown / replacement）
        2. 对 special token id，映射回对应的 str
        3. 普通 token bytes 串联后用 UTF-8 decode（errors='replace'）
           参见作业中提到的 Unicode replacement character。

        Parameters
        ----------
        ids : list[int]

        Returns
        -------
        str
        """
        parts: List[str] = []
        bytes_buffer = bytearray()

        for tid in ids:
            # ---- 如果是 special token ----
            if tid in self._id_to_special_token:
                # 先 flush 掉之前累计的普通 bytes
                if bytes_buffer:
                    parts.append(bytes_buffer.decode("utf-8", errors="replace")) 
                    bytes_buffer.clear()

                # 直接插入 special token 的字符串
                parts.append(self._id_to_special_token[tid]) # 特殊字符就直接返回字符串，不再经过decode("utf-8")
                continue

            # ---- 普通 BPE token：加入 bytes buffer ----
            token_bytes = self.vocab[tid]
            bytes_buffer.extend(token_bytes) # 一定是先合并bytes再decode("utf-8")

        # ---- flush 尾部普通 bytes ----
        if bytes_buffer:
            parts.append(bytes_buffer.decode("utf-8", errors="replace"))

        return "".join(parts)


if __name__ == '__main__':
    mytokenizer = Tokenizer.from_files("../artifacts/tinystories_vocab.pkl", "../artifacts/tinystories_merges.pkl")
    test_str = "I like apple."
    encoded_ids = mytokenizer.encode(test_str)
    result = mytokenizer.decode(encoded_ids)
    print(result)