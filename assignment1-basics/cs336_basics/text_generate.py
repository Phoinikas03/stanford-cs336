#!/usr/bin/env python3
"""
使用KV Cache进行文本生成的脚本

支持的采样方法:
- Greedy decoding
- Temperature sampling
- Top-k sampling
- Top-p (nucleus) sampling

使用方法:
    python text_generate.py --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
                           --prompt "Once upon a time" \
                           --max_tokens 100 \
                           --temperature 0.8 \
                           --top_k 50
"""

import torch
import torch.nn.functional as F
import argparse
import yaml
import os
from pathlib import Path
from typing import Optional, Tuple, List
import time

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer


class KVCache:
    """KV Cache用于加速自回归生成"""
    
    def __init__(self, num_layers: int, batch_size: int, num_heads: int, max_seq_len: int, d_k: int, device: torch.device):
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.d_k = d_k
        self.device = device
        
        # 为每一层创建K和V的缓存
        # 形状: (batch_size, num_heads, seq_len, d_k)
        self.k_cache = [torch.zeros(batch_size, num_heads, 0, d_k, device=device) 
                        for _ in range(num_layers)]
        self.v_cache = [torch.zeros(batch_size, num_heads, 0, d_k, device=device) 
                        for _ in range(num_layers)]
        self.current_length = 0
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """
        更新指定层的KV缓存
        
        Args:
            layer_idx: 层索引
            k: 新的key张量 (batch, num_heads, seq_len, d_k)
            v: 新的value张量 (batch, num_heads, seq_len, d_k)
        """
        # 拼接新的k和v到缓存中
        self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
        self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
    
    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取指定层的KV缓存
        
        Returns:
            (k_cache, v_cache) 元组
        """
        return self.k_cache[layer_idx], self.v_cache[layer_idx]
    
    def get_seq_length(self) -> int:
        """返回当前缓存的序列长度"""
        return self.k_cache[0].shape[2] if len(self.k_cache) > 0 else 0


def forward_with_kv_cache(
    model: TransformerLM,
    input_ids: torch.Tensor,
    kv_cache: Optional[KVCache] = None,
    use_cache: bool = True
) -> Tuple[torch.Tensor, Optional[KVCache]]:
    """
    带KV Cache的前向传播
    
    Args:
        model: TransformerLM模型
        input_ids: 输入token IDs (batch_size, seq_len)
        kv_cache: KV缓存对象
        use_cache: 是否使用和更新缓存
    
    Returns:
        logits: 输出的logits (batch_size, seq_len, vocab_size)
        kv_cache: 更新后的KV缓存
    """
    from einops import rearrange, repeat
    
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # 生成token positions
    if kv_cache is None or kv_cache.get_seq_length() == 0:
        # 首次生成，使用完整序列
        token_positions = torch.arange(seq_len, device=device)
        token_positions = repeat(token_positions, "seq -> b seq", b=batch_size)
    else:
        # 增量生成，position从缓存长度开始
        cache_len = kv_cache.get_seq_length()
        token_positions = torch.arange(cache_len, cache_len + seq_len, device=device)
        token_positions = repeat(token_positions, "seq -> b seq", b=batch_size)
    
    # Embeddings
    x = model.token_embeddings(input_ids)
    
    # 初始化KV缓存（如果需要）
    if use_cache and kv_cache is None:
        d_k = model.layers[0].attn.d_k
        num_heads = model.layers[0].attn.num_heads
        kv_cache = KVCache(
            num_layers=len(model.layers),
            batch_size=batch_size,
            num_heads=num_heads,
            max_seq_len=1024,  # 可以设置更大
            d_k=d_k,
            device=device
        )
    
    # Transformer Blocks with KV Cache
    for layer_idx, layer in enumerate(model.layers):
        x = forward_layer_with_cache(layer, x, token_positions, kv_cache, layer_idx, use_cache)
    
    # Final layer norm
    x = model.ln_final(x)
    
    # LM head
    logits = model.lm_head(x)
    
    return logits, kv_cache


def forward_layer_with_cache(
    layer,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    kv_cache: Optional[KVCache],
    layer_idx: int,
    use_cache: bool
) -> torch.Tensor:
    """
    单层Transformer Block的带缓存前向传播
    """
    from einops import rearrange
    
    # Pre-norm
    normed_x = layer.ln1(x)
    
    # Multi-head attention with KV cache
    attn_output = forward_attention_with_cache(
        layer.attn, normed_x, token_positions, kv_cache, layer_idx, use_cache
    )
    
    # Residual connection
    x = x + attn_output
    
    # Feed-forward with pre-norm
    normed_x = layer.ln2(x)
    ff_output = layer.ffn(normed_x)
    
    # Residual connection
    x = x + ff_output
    
    return x


def forward_attention_with_cache(
    attn,
    x: torch.Tensor,
    token_positions: torch.Tensor,
    kv_cache: Optional[KVCache],
    layer_idx: int,
    use_cache: bool
) -> torch.Tensor:
    """
    多头注意力的带缓存前向传播
    """
    from einops import rearrange
    from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
    
    # 投影得到Q, K, V
    Q = attn.w_q(x)
    K = attn.w_k(x)
    V = attn.w_v(x)
    
    # 分头
    Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=attn.num_heads)
    K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=attn.num_heads)
    V = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=attn.num_heads)
    
    # 应用RoPE
    Q = attn.rope(Q, token_positions)
    K = attn.rope(K, token_positions)
    
    # 使用KV Cache
    if use_cache and kv_cache is not None:
        # 更新缓存
        kv_cache.update(layer_idx, K, V)
        # 获取完整的K和V（包括历史）
        K, V = kv_cache.get(layer_idx)
    
    # 计算注意力
    # 创建因果掩码
    q_len = Q.shape[-2]
    k_len = K.shape[-2]
    mask = torch.tril(torch.ones(q_len, k_len, device=Q.device, dtype=torch.bool))
    
    # 如果使用缓存，掩码应该允许查询关注所有缓存的key
    if use_cache and kv_cache is not None and kv_cache.get_seq_length() > q_len:
        # 查询只有最后几个token，可以关注所有之前的token
        mask = torch.ones(q_len, k_len, device=Q.device, dtype=torch.bool)
    
    mask = rearrange(mask, "query key -> 1 query key")
    
    attention_output = ScaledDotProductAttention()(Q, K, V, mask)
    attention_output = rearrange(attention_output, "... heads seq d_v -> ... seq (heads d_v)")
    
    # 输出投影
    output = attn.w_o(attention_output)
    
    return output


def sample_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Top-k采样"""
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    return logits


def sample_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Top-p (nucleus) 采样"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过top_p的token
    sorted_indices_to_remove = cumulative_probs > top_p
    # 保留至少一个token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    use_cache: bool = True,
    device: torch.device = torch.device('cpu'),
    verbose: bool = True
) -> str:
    """
    使用KV Cache生成文本
    
    Args:
        model: TransformerLM模型
        tokenizer: Tokenizer
        prompt: 输入提示文本
        max_tokens: 最大生成token数
        temperature: 温度参数（越高越随机，越低越确定）
        top_k: Top-k采样参数
        top_p: Top-p采样参数
        use_cache: 是否使用KV cache
        device: 设备
        verbose: 是否打印生成过程
    
    Returns:
        生成的完整文本
    """
    model.eval()
    
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    if verbose:
        print(f"Prompt: {prompt}")
        print(f"Prompt tokens: {input_ids.shape[1]}")
        print(f"Using KV Cache: {use_cache}")
        if use_cache:
            print("⚡ Prefill phase will process the prompt once, then decode phase will generate incrementally")
        print("-" * 60)
        print("Generating", end="", flush=True)
    
    generated_ids = input_ids.clone()
    kv_cache = None
    
    start_time = time.time()
    
    for i in range(max_tokens):
        # ========== Prefill vs Decode 阶段 ==========
        # 第1次循环（i=0）：Prefill阶段
        #   - kv_cache = None
        #   - 输入完整prompt：input_for_model = generated_ids
        #   - forward_with_kv_cache会处理所有token并创建KV cache
        #   - 这是最耗时的阶段，需要O(n²)的注意力计算
        # 
        # 第2次及以后（i>0）：Decode阶段
        #   - kv_cache != None（已在prefill中创建）
        #   - 只输入最后一个token：input_for_model = generated_ids[:, -1:]
        #   - 之前的K、V已经缓存，只需计算新token的K、V
        #   - 每步只需O(n)的计算，大幅加速
        # ========================================
        
        if use_cache and kv_cache is not None:
            # Decode阶段：只输入最后一个token
            input_for_model = generated_ids[:, -1:]
        else:
            # Prefill阶段（或不使用cache）：输入完整序列
            input_for_model = generated_ids
        
        # 前向传播
        logits, kv_cache = forward_with_kv_cache(
            model, input_for_model, kv_cache, use_cache=use_cache
        )
        
        # 获取最后一个位置的logits
        next_token_logits = logits[:, -1, :]
        
        # 应用temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # 应用top-k采样
        if top_k is not None:
            next_token_logits = sample_top_k(next_token_logits, top_k)
        
        # 应用top-p采样
        if top_p is not None:
            next_token_logits = sample_top_p(next_token_logits, top_p)
        
        # 采样下一个token
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # 添加到生成序列
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # 解码并打印（如果verbose）
        if verbose:
            next_token_text = tokenizer.decode(next_token[0].tolist())
            print(next_token_text, end="", flush=True)
        
        # 检查是否生成了结束符（假设有的话）
        # if next_token.item() == tokenizer.eos_token_id:
        #     break
    
    end_time = time.time()
    
    # 解码完整序列
    generated_text = tokenizer.decode(generated_ids[0].tolist())
    
    if verbose:
        print("\n" + "-" * 60)
        print(f"Generated {generated_ids.shape[1] - input_ids.shape[1]} tokens in {end_time - start_time:.2f}s")
        print(f"Tokens/second: {(generated_ids.shape[1] - input_ids.shape[1]) / (end_time - start_time):.2f}")
        print(f"Using KV Cache: {use_cache}")
    
    return generated_text


def load_model_from_checkpoint(checkpoint_path: str, device: torch.device = torch.device('cpu')):
    """
    从checkpoint加载模型
    
    Args:
        checkpoint_path: checkpoint文件路径
        device: 设备
    
    Returns:
        model: 加载的模型
        config: 训练配置
    """
    checkpoint_dir = Path(checkpoint_path).parent
    config_path = checkpoint_dir / "config.yaml"
    
    # 加载配置
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 初始化模型
    model = TransformerLM(
        vocab_size=config['model']['vocab_size'],
        context_length=config['model']['context_length'],
        d_model=config['model']['d_model'],
        num_layers=config['model']['num_layers'],
        num_heads=config['model']['num_heads'],
        d_ff=config['model']['d_ff'],
        rope_theta=config['model']['rope_theta'],
    )
    
    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✓ Model loaded from: {checkpoint_path}")
    print(f"✓ Model parameters: {model.get_num_params():,}")
    print(f"✓ Training step: {checkpoint['iteration']}")
    
    return model, config


def main():
    parser = argparse.ArgumentParser(description="使用KV Cache生成文本")
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Checkpoint文件路径")
    parser.add_argument("--vocab", type=str, default="../artifacts/tinystories_vocab.pkl",
                        help="Vocab文件路径")
    parser.add_argument("--merges", type=str, default="../artifacts/tinystories_merges.pkl",
                        help="Merges文件路径")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="输入提示文本")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="最大生成token数")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="温度参数（越高越随机）")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k采样参数")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p采样参数")
    parser.add_argument("--no_cache", action="store_true",
                        help="禁用KV cache（用于比较速度）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备 (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    
    args = parser.parse_args()
    
    # 设置随机种子
    if args.seed is not None:
        torch.manual_seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型
    print("\n" + "=" * 60)
    print("Loading model...")
    print("=" * 60)
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载tokenizer
    print("\n" + "=" * 60)
    print("Loading tokenizer...")
    print("=" * 60)
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges
    )
    print(f"✓ Tokenizer loaded: vocab_size={len(tokenizer.vocab)}")
    
    # 生成文本
    print("\n" + "=" * 60)
    print("Generating text...")
    print("=" * 60)
    generated_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        use_cache=not args.no_cache,
        device=device,
        verbose=True
    )
    
    print("\n" + "=" * 60)
    print("Complete generated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == "__main__":
    main()
