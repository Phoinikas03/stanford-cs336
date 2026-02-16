# 文本生成指南 - 使用KV Cache

本文档说明如何使用训练好的模型进行文本生成。

## 什么是KV Cache？

KV Cache是一种优化技术，用于加速自回归文本生成：

### 传统方法（无缓存）
- 每生成一个token，需要重新计算整个序列的Key和Value
- 时间复杂度：O(n²)，n为序列长度
- 生成100个token需要计算：1+2+3+...+100 = 5050次注意力操作

### 使用KV Cache
- 缓存之前计算过的Key和Value
- 每次只需计算新token的Key和Value
- 时间复杂度：O(n)
- 生成100个token只需要：100次注意力操作
- **速度提升：约50倍（对于长序列）**

## 功能特性

✅ **KV Cache加速** - 显著提升生成速度
✅ **多种采样方法** - Greedy、Temperature、Top-k、Top-p
✅ **自动加载配置** - 从checkpoint目录自动读取模型配置
✅ **实时生成显示** - 边生成边显示文本
✅ **性能统计** - 显示生成速度（tokens/second）

## 使用方法

### 基本用法

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

### 完整参数说明

```bash
python text_generate.py \
    --checkpoint PATH              # Checkpoint文件路径（必需）
    --vocab PATH                   # Vocab文件路径（默认：../artifacts/tinystories_vocab.pkl）
    --merges PATH                  # Merges文件路径（默认：../artifacts/tinystories_merges.pkl）
    --prompt "TEXT"                # 输入提示文本
    --max_tokens 100               # 最大生成token数
    --temperature 1.0              # 温度参数（0.1-2.0）
    --top_k 50                     # Top-k采样
    --top_p 0.9                    # Top-p采样
    --no_cache                     # 禁用KV cache（用于性能对比）
    --device cuda                  # 设备（cuda/cpu）
    --seed 42                      # 随机种子（可复现）
```

## 采样方法

### 1. Greedy Decoding（贪婪解码）

每次选择概率最高的token：

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The princess" \
    --temperature 0.0001  # 接近0，近似greedy
```

**特点：**
- ✅ 确定性输出（相同输入总是产生相同输出）
- ✅ 连贯性较好
- ❌ 可能重复、无聊
- ❌ 缺乏创造性

### 2. Temperature Sampling（温度采样）

控制生成的随机性：

```bash
# 低温度 - 更保守、更可预测
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The princess" \
    --temperature 0.5

# 中等温度 - 平衡
python text_generate.py \
    --temperature 1.0

# 高温度 - 更随机、更有创造性
python text_generate.py \
    --temperature 1.5
```

**温度参数指南：**
- `0.1-0.5`: 保守、可预测、适合事实性内容
- `0.7-0.9`: 平衡、自然的对话
- `1.0`: 与训练分布一致
- `1.2-2.0`: 随机、有创造性、可能不连贯

### 3. Top-k Sampling

只从概率最高的k个token中采样：

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The princess" \
    --temperature 0.8 \
    --top_k 50
```

**参数建议：**
- `k=10`: 非常保守
- `k=40-50`: 推荐值（平衡质量和多样性）
- `k=100+`: 更有创造性

### 4. Top-p (Nucleus) Sampling

动态选择概率和达到p的最小token集合：

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The princess" \
    --temperature 0.8 \
    --top_p 0.9
```

**参数建议：**
- `p=0.9`: 推荐值（常用于ChatGPT等）
- `p=0.95`: 更多样化
- `p=0.85`: 更保守

### 5. 组合使用

可以同时使用多种采样方法：

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The princess" \
    --temperature 0.8 \
    --top_k 50 \
    --top_p 0.9
```

## 性能对比

### 有KV Cache vs 无KV Cache

```bash
# 使用KV Cache（快速）
time python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "Once upon a time" \
    --max_tokens 100

# 不使用KV Cache（慢）
time python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "Once upon a time" \
    --max_tokens 100 \
    --no_cache
```

**预期结果：**
- 使用KV Cache: ~5秒，~20 tokens/s
- 不使用KV Cache: ~100秒，~1 tokens/s
- **加速比：20倍+**

## 示例场景

### 故事生成

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "Once upon a time, in a faraway kingdom," \
    --max_tokens 200 \
    --temperature 0.9 \
    --top_p 0.95
```

### 对话续写

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "Hello, how are you?" \
    --max_tokens 50 \
    --temperature 0.7 \
    --top_k 40
```

### 确定性生成（可复现）

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_xxx/checkpoint_step_1000.pt \
    --prompt "The quick brown fox" \
    --max_tokens 50 \
    --temperature 0.8 \
    --seed 42  # 固定随机种子
```

## 批量生成

创建脚本批量生成多个样本：

```bash
#!/bin/bash
# generate_samples.sh

CHECKPOINT="../checkpoints/run_xxx/checkpoint_step_1000.pt"
PROMPTS=(
    "Once upon a time"
    "In a magical forest"
    "The brave knight"
    "A mysterious wizard"
)

for i in "${!PROMPTS[@]}"; do
    echo "Generating sample $((i+1))..."
    python text_generate.py \
        --checkpoint $CHECKPOINT \
        --prompt "${PROMPTS[$i]}" \
        --max_tokens 100 \
        --temperature 0.9 \
        --top_p 0.95 \
        > "sample_$((i+1)).txt"
done
```

## 交互式生成

创建交互式文本生成脚本：

```python
# interactive_generate.py
import argparse
from text_generate import load_model_from_checkpoint, generate
from cs336_basics.tokenizer import Tokenizer
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", default="../artifacts/tinystories.model")
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_model_from_checkpoint(args.checkpoint, device)
    tokenizer = Tokenizer(args.tokenizer)
    
    print("Interactive Text Generation (type 'quit' to exit)")
    print("=" * 60)
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit', 'q']:
            break
        
        generated = generate(
            model, tokenizer, prompt,
            max_tokens=100,
            temperature=0.9,
            top_p=0.95,
            device=device
        )
        print("\nGenerated:")
        print(generated)
        print("-" * 60)

if __name__ == "__main__":
    main()
```

## 常见问题

### Q: 生成的文本不够好怎么办？

A: 尝试以下方法：
1. 使用训练更久的checkpoint（更高的step）
2. 调整temperature（通常0.7-0.9效果较好）
3. 使用top-p=0.9或top-k=50
4. 改进prompt（更具体、更长的提示通常更好）

### Q: 生成速度慢怎么办？

A: 
1. 确保使用GPU（`--device cuda`）
2. 确保KV Cache已启用（不要用`--no_cache`）
3. 减小max_tokens
4. 使用更小的模型

### Q: 如何让生成更有创造性？

A:
- 提高temperature（1.2-1.5）
- 使用top-p=0.95
- 增加top-k（100+）

### Q: 如何让生成更保守/可靠？

A:
- 降低temperature（0.5-0.7）
- 使用top-p=0.85
- 减小top-k（20-30）

### Q: 生成的文本重复怎么办？

A:
- 提高temperature
- 使用top-k或top-p采样
- 检查模型是否训练充分

## KV Cache实现细节

### 工作原理

1. **初始阶段**（处理prompt）：
   - 正常前向传播计算所有token的K、V
   - 将K、V缓存起来

2. **生成阶段**（每步）：
   - 只输入新生成的1个token
   - 计算该token的Q、K、V
   - 将新的K、V追加到缓存
   - 使用缓存的完整K、V进行注意力计算

3. **内存使用**：
   - 每层缓存：`2 × batch_size × num_heads × seq_len × d_k`
   - 例如：6层，16头，序列长度100，d_k=32
   - 内存：`2 × 1 × 16 × 100 × 32 × 4 bytes ≈ 400KB`

### 性能分析

**生成n个token的计算量：**

- **无缓存**：
  - 第1步：计算1个token的注意力
  - 第2步：计算2个token的注意力
  - ...
  - 第n步：计算n个token的注意力
  - 总计：O(n²) 复杂度

- **有缓存**：
  - 每步只计算1个新token
  - 总计：O(n) 复杂度
  - **理论加速：n倍**（实际约为n/2倍，因为prompt处理开销）

## 相关文件

- `text_generate.py` - 文本生成主脚本
- `transformer_lm.py` - Transformer模型定义
- `multi_head_attention.py` - 注意力机制
- `tokenizer.py` - Tokenizer
- `checkpoint.py` - Checkpoint管理

## 下一步

1. 尝试不同的采样参数，找到最适合你任务的配置
2. 使用训练更久的checkpoint
3. 实现beam search（束搜索）以获得更好的结果
4. 添加重复惩罚（repetition penalty）
5. 实现批量生成以提高吞吐量

Happy generating! 🎉
