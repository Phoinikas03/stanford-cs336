#!/usr/bin/env python3
"""
诊断tokenizer性能问题

测试tokenizer的正则表达式是否会导致hang
"""

import regex as re
import time

# 这是tokenizer中使用的正则表达式
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

# 测试不同的输入
test_cases = [
    ("简单文本", "Hello world"),
    ("带标点", "Hello, world!"),
    ("带数字", "The year is 2024"),
    ("单词", "Once upon a time"),
    # 可能有问题的输入
    ("大量空格", " " * 100),
    ("混合空格和文本", "   hello   world   "),
    ("空字符串", ""),
]

print("测试tokenizer正则表达式性能...")
print("=" * 60)

for name, text in test_cases:
    print(f"\n测试: {name}")
    print(f"输入: {repr(text[:50])}{'...' if len(text) > 50 else ''}")
    
    try:
        start = time.time()
        # 设置超时
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("正则匹配超时!")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5秒超时
        
        matches = re.findall(PAT, text)
        
        signal.alarm(0)  # 取消超时
        
        elapsed = time.time() - start
        print(f"✓ 完成: {len(matches)} 个匹配，耗时 {elapsed:.4f}秒")
        if elapsed > 1.0:
            print(f"⚠️  警告: 耗时过长!")
        
    except TimeoutError as e:
        print(f"❌ 超时: {e}")
    except Exception as e:
        print(f"❌ 错误: {e}")

print("\n" + "=" * 60)
print("诊断完成")

# 建议
print("\n建议:")
print("如果看到超时或耗时过长，说明正则表达式存在性能问题。")
print("可能的解决方案:")
print("1. 简化正则表达式")
print("2. 使用更高效的tokenization方法")
print("3. 对特殊输入进行预处理")
