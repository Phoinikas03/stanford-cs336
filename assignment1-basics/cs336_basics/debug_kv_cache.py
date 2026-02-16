#!/usr/bin/env python3
"""
KV Cache调试工具 - 详细显示Prefill和Decode过程

使用方法:
    python debug_kv_cache.py --checkpoint YOUR_CHECKPOINT
"""

import argparse
import torch
from cs336_basics.text_generate import load_model_from_checkpoint, forward_with_kv_cache, KVCache
from cs336_basics.tokenizer import Tokenizer
import torch.nn.functional as F


def debug_generate(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 10,
    device: torch.device = torch.device('cpu')
):
    """
    详细展示KV Cache的工作过程
    """
    model.eval()
    
    # Tokenize
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    
    print("=" * 80)
    print("KV Cache 调试模式")
    print("=" * 80)
    print(f"\nPrompt: {prompt}")
    print(f"Prompt tokens: {tokenizer.decode(input_ids[0].tolist())}")
    print(f"Token IDs: {input_ids[0].tolist()}")
    print(f"Sequence length: {input_ids.shape[1]}")
    print("\n" + "=" * 80)
    
    generated_ids = input_ids.clone()
    kv_cache = None
    
    for step in range(max_tokens):
        print(f"\n{'='*80}")
        print(f"Step {step + 1}/{max_tokens}")
        print(f"{'='*80}")
        
        # 判断是Prefill还是Decode
        if kv_cache is None:
            phase = "PREFILL"
            input_for_model = generated_ids
            print(f"📋 Phase: {phase}")
            print(f"📥 Input shape: {input_for_model.shape}")
            print(f"📥 Input tokens: {tokenizer.decode(input_for_model[0].tolist())}")
            print(f"💾 KV Cache: Not initialized")
        else:
            phase = "DECODE"
            input_for_model = generated_ids[:, -1:]
            cache_len = kv_cache.get_seq_length()
            print(f"📋 Phase: {phase}")
            print(f"📥 Input shape: {input_for_model.shape} (只有最后1个token)")
            print(f"📥 Input token: {tokenizer.decode(input_for_model[0].tolist())}")
            print(f"💾 KV Cache length: {cache_len} tokens")
            print(f"💾 Cache contains: {tokenizer.decode(generated_ids[0, :cache_len].tolist())}")
        
        # 前向传播
        print(f"\n⚙️  Forward pass...")
        with torch.no_grad():
            logits, kv_cache = forward_with_kv_cache(
                model, input_for_model, kv_cache, use_cache=True
            )
        
        print(f"✓ Output logits shape: {logits.shape}")
        
        # 获取下一个token
        next_token_logits = logits[:, -1, :]
        probs = F.softmax(next_token_logits, dim=-1)
        
        # 显示top-5预测
        top5_probs, top5_indices = torch.topk(probs, 5)
        print(f"\n🎯 Top-5 predictions:")
        for i, (prob, idx) in enumerate(zip(top5_probs[0], top5_indices[0])):
            token_text = tokenizer.decode([idx.item()])
            print(f"   {i+1}. '{token_text}' (prob: {prob.item():.4f})")
        
        # 采样
        next_token = torch.multinomial(probs, num_samples=1)
        next_token_text = tokenizer.decode(next_token[0].tolist())
        
        print(f"\n✨ Selected token: '{next_token_text}' (ID: {next_token[0].item()})")
        
        # 更新生成序列
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        
        # 显示当前生成的完整序列
        current_text = tokenizer.decode(generated_ids[0].tolist())
        print(f"\n📝 Current sequence ({generated_ids.shape[1]} tokens):")
        print(f"   {current_text}")
        
        # 显示cache状态
        if kv_cache is not None:
            new_cache_len = kv_cache.get_seq_length()
            print(f"\n💾 KV Cache updated: {cache_len if phase == 'DECODE' else 0} → {new_cache_len} tokens")
            
            # 显示每层cache的形状
            if step == 0:  # 只在第一步显示详细信息
                print(f"\n   Cache structure (per layer):")
                k, v = kv_cache.get(0)  # 获取第一层的cache
                print(f"   - K shape: {k.shape}")
                print(f"   - V shape: {v.shape}")
                print(f"   Format: (batch_size, num_heads, seq_len, d_k)")
        
        print(f"\n{'='*80}")
        
        if step < max_tokens - 1:
            input("\n👉 按Enter继续下一步...")
    
    print("\n" + "=" * 80)
    print("生成完成！")
    print("=" * 80)
    
    final_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"\n📜 Final generated text:")
    print(f"{final_text}")
    print(f"\n📊 Statistics:")
    print(f"   - Total tokens generated: {generated_ids.shape[1] - input_ids.shape[1]}")
    print(f"   - Final sequence length: {generated_ids.shape[1]}")
    print(f"   - KV Cache final length: {kv_cache.get_seq_length()}")
    
    return final_text


def main():
    parser = argparse.ArgumentParser(description="KV Cache调试工具")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint文件路径")
    parser.add_argument("--vocab", type=str, default="../artifacts/tinystories_vocab.pkl",
                        help="Vocab文件路径")
    parser.add_argument("--merges", type=str, default="../artifacts/tinystories_merges.pkl",
                        help="Merges文件路径")
    parser.add_argument("--prompt", type=str, default="Once upon a time",
                        help="输入提示文本")
    parser.add_argument("--max_tokens", type=int, default=10,
                        help="生成token数（建议<=10以便观察）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    
    # 加载模型
    print("Loading model...")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges
    )
    print(f"✓ Ready!\n")
    
    # 运行调试生成
    debug_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        device=device
    )


if __name__ == "__main__":
    main()
