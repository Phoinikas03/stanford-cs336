#!/usr/bin/env python3
"""
KV Cache性能基准测试

对比有/无KV Cache的生成速度

使用方法:
    python benchmark_generation.py --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt
"""

import argparse
import torch
import time
from cs336_basics.text_generate import load_model_from_checkpoint, generate
from cs336_basics.tokenizer import Tokenizer


def benchmark(
    model,
    tokenizer,
    prompts,
    max_tokens,
    device,
    use_cache=True
):
    """运行benchmark测试"""
    total_time = 0
    total_tokens = 0
    
    for prompt in prompts:
        start_time = time.time()
        
        generated = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.8,
            use_cache=use_cache,
            device=device,
            verbose=False
        )
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 计算生成的token数（总长度 - prompt长度）
        prompt_tokens = len(tokenizer.encode(prompt))
        generated_tokens = len(tokenizer.encode(generated)) - prompt_tokens
        
        total_time += elapsed
        total_tokens += generated_tokens
    
    avg_time = total_time / len(prompts)
    tokens_per_second = total_tokens / total_time
    
    return {
        'total_time': total_time,
        'avg_time': avg_time,
        'total_tokens': total_tokens,
        'tokens_per_second': tokens_per_second
    }


def main():
    parser = argparse.ArgumentParser(description="KV Cache性能基准测试")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint文件路径")
    parser.add_argument("--vocab", type=str, default="../artifacts/tinystories_vocab.pkl",
                        help="Vocab文件路径")
    parser.add_argument("--merges", type=str, default="../artifacts/tinystories_merges.pkl",
                        help="Merges文件路径")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="每次生成的token数")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="测试样本数")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    args = parser.parse_args()
    
    # 测试prompts
    test_prompts = [
        "Once upon a time",
        "In a faraway land",
        "The brave knight",
        "A magical wizard",
        "The princess lived",
        "Long ago there was",
        "In the deep forest",
        "Under the starry sky",
        "The young hero",
        "In the ancient castle"
    ][:args.num_samples]
    
    device = torch.device(args.device)
    
    print("=" * 80)
    print("  KV Cache Performance Benchmark")
    print("=" * 80)
    
    # 加载模型
    print("\nLoading model...")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges
    )
    
    print(f"\nBenchmark settings:")
    print(f"  Device: {device}")
    print(f"  Number of samples: {args.num_samples}")
    print(f"  Tokens per sample: {args.max_tokens}")
    print(f"  Model parameters: {model.get_num_params():,}")
    
    # Warm up
    print("\nWarming up...")
    for _ in range(2):
        generate(
            model=model,
            tokenizer=tokenizer,
            prompt="warm up",
            max_tokens=10,
            use_cache=True,
            device=device,
            verbose=False
        )
    
    print("\n" + "=" * 80)
    print("Running benchmark...")
    print("=" * 80)
    
    # 测试有KV Cache
    print("\n[1/2] Testing WITH KV Cache...")
    results_with_cache = benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_tokens=args.max_tokens,
        device=device,
        use_cache=True
    )
    
    # 测试无KV Cache
    print("[2/2] Testing WITHOUT KV Cache...")
    results_without_cache = benchmark(
        model=model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_tokens=args.max_tokens,
        device=device,
        use_cache=False
    )
    
    # 显示结果
    print("\n" + "=" * 80)
    print("  Results")
    print("=" * 80)
    
    print(f"\n{'Metric':<30} {'With Cache':<20} {'Without Cache':<20} {'Speedup':<15}")
    print("-" * 85)
    
    print(f"{'Total time (s)':<30} {results_with_cache['total_time']:<20.2f} "
          f"{results_without_cache['total_time']:<20.2f} "
          f"{results_without_cache['total_time'] / results_with_cache['total_time']:<15.1f}x")
    
    print(f"{'Avg time per sample (s)':<30} {results_with_cache['avg_time']:<20.2f} "
          f"{results_without_cache['avg_time']:<20.2f} "
          f"{results_without_cache['avg_time'] / results_with_cache['avg_time']:<15.1f}x")
    
    print(f"{'Total tokens generated':<30} {results_with_cache['total_tokens']:<20} "
          f"{results_without_cache['total_tokens']:<20}")
    
    print(f"{'Tokens per second':<30} {results_with_cache['tokens_per_second']:<20.1f} "
          f"{results_without_cache['tokens_per_second']:<20.1f} "
          f"{results_with_cache['tokens_per_second'] / results_without_cache['tokens_per_second']:<15.1f}x")
    
    print("\n" + "=" * 80)
    print("Summary:")
    print("=" * 80)
    speedup = results_without_cache['total_time'] / results_with_cache['total_time']
    print(f"✨ KV Cache provides {speedup:.1f}x speedup!")
    print(f"✨ Generation speed: {results_with_cache['tokens_per_second']:.1f} tokens/s (with cache)")
    print(f"✨ Generation speed: {results_without_cache['tokens_per_second']:.1f} tokens/s (without cache)")
    
    if speedup < 2:
        print("\n⚠️  Warning: Speedup is lower than expected. This might be because:")
        print("   - Sequence length is too short (cache overhead dominates)")
        print("   - Device is CPU (limited by compute, not memory bandwidth)")
        print("   - Model is too small")
    else:
        print(f"\n✅ KV Cache is working efficiently!")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
