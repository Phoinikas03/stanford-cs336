#!/usr/bin/env python3
"""
交互式文本生成

使用方法:
    python interactive_generate.py --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt
"""

import argparse
import torch
from cs336_basics.text_generate import load_model_from_checkpoint, generate
from cs336_basics.tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="交互式文本生成")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Checkpoint文件路径")
    parser.add_argument("--vocab", type=str, default="../artifacts/tinystories_vocab.pkl",
                        help="Vocab文件路径")
    parser.add_argument("--merges", type=str, default="../artifacts/tinystories_merges.pkl",
                        help="Merges文件路径")
    parser.add_argument("--max_tokens", type=int, default=100,
                        help="每次生成的最大token数")
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="温度参数")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k采样")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Top-p采样")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="设备")
    
    args = parser.parse_args()
    
    # 加载模型
    device = torch.device(args.device)
    print(f"Loading model on {device}...")
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    
    # 加载tokenizer
    print("Loading tokenizer...")
    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab,
        merges_filepath=args.merges
    )
    
    print("\n" + "=" * 70)
    print("  Interactive Text Generation")
    print("=" * 70)
    print(f"Model: {args.checkpoint}")
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Device: {device}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temperature: {args.temperature}")
    if args.top_k:
        print(f"Top-k: {args.top_k}")
    if args.top_p:
        print(f"Top-p: {args.top_p}")
    print("=" * 70)
    print("\nCommands:")
    print("  - Enter your prompt and press Enter to generate")
    print("  - Type 'quit', 'exit', or 'q' to exit")
    print("  - Type 'help' for more commands")
    print("=" * 70)
    
    # 交互循环
    while True:
        try:
            print("\n" + "-" * 70)
            prompt = input("📝 Prompt: ").strip()
            
            if not prompt:
                continue
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋")
                break
            
            if prompt.lower() == 'help':
                print("\nAvailable commands:")
                print("  quit, exit, q  - Exit the program")
                print("  help           - Show this help message")
                print("  clear          - Clear the screen")
                continue
            
            if prompt.lower() == 'clear':
                import os
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            
            print("\n🤖 Generating...\n")
            
            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                use_cache=True,
                device=device,
                verbose=False
            )
            
            print("\n" + "-" * 70)
            print("✨ Generated text:")
            print("-" * 70)
            print(generated_text)
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with a new prompt.")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
