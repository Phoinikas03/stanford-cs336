from collections.abc import Iterable, Iterator
from typing import Dict, List, Tuple, Optional
import pickle
import regex as re
import json
import time
import os

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

import numpy as np
import subprocess
from tqdm import tqdm
from time import sleep
from multiprocessing import Pool, cpu_count

def _encode_batch(args):
    """
    用于multiprocessing的辅助函数，批量处理多行
    带超时保护和错误处理
    
    Parameters
    ----------
    args : tuple
        (lines_list, vocab_filepath, merges_filepath, special_tokens)
    
    Returns
    -------
    list[list[int]]
        每行编码后的token ids列表
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Encoding timeout")
    
    lines_list, vocab_filepath, merges_filepath, special_tokens = args
    # 每个进程只加载一次tokenizer
    tokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=special_tokens)
    
    # 批量处理所有lines
    results = []
    timeout_count = 0
    error_count = 0
    
    for i, line in enumerate(lines_list):
        try:
            # 跳过异常长的行（可能有问题）
            if len(line) > 100000:
                results.append([])
                continue
            
            # 设置10秒超时
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(10)
            
            ids = tokenizer.encode(line)
            results.append(ids)
            
            signal.alarm(0)  # 取消超时
            
        except TimeoutError:
            # 超时时跳过该行
            results.append([])
            timeout_count += 1
            signal.alarm(0)
            if timeout_count == 1:  # 只打印第一次
                print(f"Warning: Line timeout (length={len(line)}), skipping...")
                
        except Exception as e:
            # 其他错误也跳过
            results.append([])
            error_count += 1
            if error_count == 1:  # 只打印第一次
                print(f"Warning: Encoding error: {e}")
    
    if timeout_count > 0 or error_count > 0:
        print(f"Batch summary: {timeout_count} timeouts, {error_count} errors out of {len(lines_list)} lines")
    
    return results

if __name__ == '__main__':
    # vocab_filepath = "../artifacts/tinystories_vocab.pkl"
    # merges_filepath = "../artifacts/tinystories_merges.pkl"
    vocab_filepath = "../artifacts/openwebtext_vocab.pkl"
    merges_filepath = "../artifacts/openwebtext_merges.pkl"
    special_tokens = ["<|endoftext|>"]
    
    mytokenizer = Tokenizer.from_files(vocab_filepath, merges_filepath, special_tokens=special_tokens)
    print("tokenizer info:", f"vocab size: {len(mytokenizer.vocab)}, merges size: {len(mytokenizer.merges)}")
    
    # file_path = "../dataset/tinystories/TinyStoriesV2-GPT4-train.txt"
    file_path = "../dataset/openwebtext/owt_train.txt"
    buffer = []
    total_tokens = 0
    buffer_size = 2000000
    
    # 使用 uint16 可以节省一半空间（如果你 vocab size < 65535）
    # 如果 vocab > 65535，则必须用 np.uint32 或 np.int32
    dtype = np.uint16
    input_path = file_path
    # output_path = "../artifacts/tinystories_train.bin"
    output_path = "../artifacts/openwebtext_train.bin"
    checkpoint_path = output_path + ".checkpoint.json"
    
    # 设置并行处理的参数
    num_workers = 32  
    lines_per_task = 200000  # 每个任务处理的行数，减少通信开销
    task_queue_size = num_workers * 2  # 同时提交的任务数，保持worker忙碌
    checkpoint_interval = 1000000  # 每处理100万行保存一次checkpoint
    
    print(f"Processing {input_path}...")
    print(f"Using {num_workers} workers, {lines_per_task} lines per task...")
    
    # ========== 检查并加载checkpoint ==========
    start_line = 0
    resume_mode = False
    
    if os.path.exists(checkpoint_path):
        print(f"\n✓ 发现checkpoint文件: {checkpoint_path}")
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            start_line = checkpoint_data.get('lines_processed', 0)
            total_tokens = checkpoint_data.get('total_tokens', 0)
            
            print(f"  已处理行数: {start_line:,}")
            print(f"  已生成token数: {total_tokens:,}")
            print(f"  上次保存时间: {checkpoint_data.get('timestamp', 'N/A')}")
            
            response = input("\n是否从checkpoint恢复? (yes/no): ").strip().lower()
            if response == 'yes':
                resume_mode = True
                print(f"✓ 将从第 {start_line + 1} 行继续处理")
            else:
                print("✗ 从头开始处理")
                start_line = 0
                total_tokens = 0
                resume_mode = False
        except Exception as e:
            print(f"⚠️  读取checkpoint失败: {e}")
            print("将从头开始处理")
            start_line = 0
            resume_mode = False
    else:
        print(f"ℹ️  未找到checkpoint文件，从头开始处理")
    
    result = subprocess.run(
            ['wc', '-l', input_path],
            capture_output=True,
            text=True
        )
    total_lines = int(result.stdout.split()[0])
    
    print(f"\n文件总行数: {total_lines:,}")
    if resume_mode:
        print(f"剩余行数: {total_lines - start_line:,}")
    print("")

    # 以追加模式打开输出文件（如果resume），否则覆盖
    file_mode = 'ab' if resume_mode else 'wb'
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, file_mode) as f_out, \
         Pool(processes=num_workers) as pool:
        
        # 创建进度条
        pbar = tqdm(total=total_lines, desc="Tokenizing", initial=start_line)
        
        # 收集一批lines，准备分配给workers
        lines_batch = []
        tasks = []
        current_line = 0
        lines_processed = start_line  # 已完成处理并写入的行数
        last_checkpoint_line = start_line  # 上次保存checkpoint的行数
        
        def save_checkpoint_info():
            """保存checkpoint信息"""
            checkpoint_info = {
                'lines_processed': lines_processed,
                'total_tokens': total_tokens,
                'total_lines': total_lines,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'input_path': input_path,
                'output_path': output_path,
                'progress_percent': round(lines_processed / total_lines * 100, 2)
            }
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_info, f, indent=2)
        
        for line in f_in:
            current_line += 1
            
            # 跳过已处理的行
            if current_line <= start_line:
                if current_line % 1000000 == 0:
                    print(f"跳过已处理行: {current_line:,} / {start_line:,}")
                continue
            
            lines_batch.append(line)
            
            # 当累积了足够的行，创建一个任务
            if len(lines_batch) >= lines_per_task:
                # 每个任务包含整批lines
                task = pool.apply_async(_encode_batch, 
                                       ((lines_batch, vocab_filepath, merges_filepath, special_tokens),))
                tasks.append((task, len(lines_batch)))
                lines_batch = []
                
                # 如果任务队列满了，开始处理完成的任务
                if len(tasks) >= task_queue_size:
                    # 取出最早提交的任务，等待其完成
                    completed_task, task_line_count = tasks.pop(0)
                    batch_results = completed_task.get()  # 返回 list[list[int]]
                    
                    # 将所有结果加入buffer
                    for ids in batch_results:
                        buffer.extend(ids)
                        
                        # 如果缓冲区满了，写入磁盘
                        if len(buffer) >= buffer_size:
                            arr = np.array(buffer, dtype=dtype)
                            f_out.write(arr.tobytes())
                            f_out.flush()  # 强制写入磁盘
                            total_tokens += len(buffer)
                            buffer = []
                    
                    # 更新已处理的行数
                    lines_processed += task_line_count
                    
                    pbar.update(task_line_count)
                    pbar.set_postfix({"tokens": f"{total_tokens:,}"})
                    
                    # 定期保存checkpoint
                    if lines_processed - last_checkpoint_line >= checkpoint_interval:
                        save_checkpoint_info()
                        last_checkpoint_line = lines_processed
                        print(f"\n💾 Checkpoint saved at line {lines_processed:,} ({lines_processed/total_lines*100:.1f}%)")
        
        # 提交最后不足一批的lines
        if lines_batch:
            task = pool.apply_async(_encode_batch, 
                                   ((lines_batch, vocab_filepath, merges_filepath, special_tokens),))
            tasks.append((task, len(lines_batch)))
        
        # 处理所有剩余的任务
        for completed_task, task_line_count in tasks:
            batch_results = completed_task.get()
            for ids in batch_results:
                buffer.extend(ids)
                
                # 如果缓冲区满了，写入磁盘
                if len(buffer) >= buffer_size:
                    arr = np.array(buffer, dtype=dtype)
                    f_out.write(arr.tobytes())
                    f_out.flush()
                    total_tokens += len(buffer)
                    buffer = []
            
            # 更新已处理的行数
            lines_processed += task_line_count
            pbar.update(task_line_count)
            pbar.set_postfix({"tokens": f"{total_tokens:,}"})
        
        pbar.close()
        
        # 写入剩余的数据
        if buffer:
            arr = np.array(buffer, dtype=dtype)
            f_out.write(arr.tobytes())
            f_out.flush()
            total_tokens += len(buffer)
        
        # 保存最终checkpoint
        save_checkpoint_info()
    
    print(f"\n✓ 处理完成!")
    print(f"  总行数: {total_lines:,}")
    print(f"  已处理行数: {lines_processed:,}")
    print(f"  总token数: {total_tokens:,}")
    print(f"  输出文件: {output_path}")
    print(f"  Checkpoint文件: {checkpoint_path}")
    
    # 如果全部处理完成，可以选择删除checkpoint文件
    # if lines_processed >= total_lines:
    #     response = input("\n处理已完成，是否删除checkpoint文件? (yes/no): ").strip().lower()
    #     if response == 'yes':
    #         os.remove(checkpoint_path)
    #         print(f"✓ 已删除checkpoint文件: {checkpoint_path}")
    #     else:
    #         print(f"ℹ️  保留checkpoint文件: {checkpoint_path}")