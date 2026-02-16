# 文本生成快速开始 🚀

5分钟快速上手使用KV Cache生成文本！

## 第一步：找到你的checkpoint

```bash
# 列出所有可用的checkpoint
ls -lt ../checkpoints/*/checkpoint_step_*.pt | head -5
```

选择一个checkpoint，例如：
```
../checkpoints/run_20260216_180753/checkpoint_step_1000.pt
```

## 第二步：快速测试

### 方法1：使用测试脚本（推荐）

```bash
./quick_test_generation.sh ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt
```

这会运行4个测试：
1. ✅ 基本生成
2. ✅ Top-k采样
3. ✅ Top-p采样
4. ✅ KV Cache性能对比

### 方法2：手动运行

```bash
python text_generate.py \
    --checkpoint ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt \
    --prompt "Once upon a time" \
    --max_tokens 100
```

## 第三步：交互式生成

```bash
python interactive_generate.py \
    --checkpoint ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt
```

然后输入你的prompt：
```
📝 Prompt: The brave knight
🤖 Generating...

✨ Generated text:
The brave knight rode through the forest on his white horse...
```

## 常用命令

### 1. 创意故事生成
```bash
python text_generate.py \
    --checkpoint YOUR_CHECKPOINT \
    --prompt "In a magical kingdom" \
    --max_tokens 200 \
    --temperature 0.9 \
    --top_p 0.95
```

### 2. 保守生成（更可靠）
```bash
python text_generate.py \
    --checkpoint YOUR_CHECKPOINT \
    --prompt "The quick brown fox" \
    --max_tokens 100 \
    --temperature 0.5 \
    --top_k 40
```

### 3. 性能测试
```bash
python benchmark_generation.py \
    --checkpoint YOUR_CHECKPOINT \
    --max_tokens 50 \
    --num_samples 5
```

## 参数速查表

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| `--temperature` | 0.8-1.0 | 控制随机性（越高越随机） |
| `--top_k` | 40-50 | 从概率最高的k个token中采样 |
| `--top_p` | 0.9-0.95 | Nucleus采样（动态截断） |
| `--max_tokens` | 100-200 | 最大生成token数 |

## 采样策略选择

| 场景 | 推荐配置 |
|------|----------|
| 🎨 创意写作 | `temperature=1.0, top_p=0.95` |
| 📝 正式写作 | `temperature=0.7, top_k=40` |
| 💬 对话 | `temperature=0.8, top_p=0.9` |
| 🎯 确定性输出 | `temperature=0.1` (接近greedy) |

## 故障排除

### 问题：生成速度很慢

**解决方案：**
```bash
# 确保使用GPU
python text_generate.py --checkpoint YOUR_CHECKPOINT --device cuda ...

# 检查KV Cache是否启用（默认启用）
# 不要使用 --no_cache 标志
```

### 问题：生成的文本质量不好

**解决方案：**
1. 使用训练更久的checkpoint（更高的step）
2. 调整temperature（尝试0.7-0.9）
3. 使用top-p=0.9或top-k=50
4. 改进prompt（更具体的提示）

### 问题：生成重复内容

**解决方案：**
```bash
# 提高temperature
python text_generate.py --temperature 1.0 ...

# 使用top-p采样
python text_generate.py --temperature 0.8 --top_p 0.95 ...
```

## KV Cache性能

**预期性能（GPU）：**
- 使用KV Cache: ~20-50 tokens/s
- 不使用KV Cache: ~1-5 tokens/s
- **加速比：10-20x**

运行benchmark查看你的性能：
```bash
python benchmark_generation.py --checkpoint YOUR_CHECKPOINT
```

## 下一步

1. 📖 阅读完整文档：`cat README_GENERATION.md`
2. 🔧 调整参数找到最佳配置
3. 🎮 尝试不同的prompt
4. 📊 比较不同checkpoint的效果

## Tokenizer说明

脚本会自动使用以下默认tokenizer文件：
- Vocab: `../artifacts/tinystories_vocab.pkl`
- Merges: `../artifacts/tinystories_merges.pkl`

如果需要使用其他tokenizer，可以指定：
```bash
python text_generate.py \
    --checkpoint YOUR_CHECKPOINT \
    --vocab ../artifacts/openwebtext_vocab.pkl \
    --merges ../artifacts/openwebtext_merges.pkl \
    --prompt "Your prompt"
```

## 示例输出

```bash
$ python text_generate.py \
    --checkpoint ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt \
    --prompt "Once upon a time" \
    --max_tokens 100 \
    --temperature 0.9

Using device: cuda
==========================================
Loading model...
==========================================
✓ Model loaded from: ../checkpoints/run_20260216_180753/checkpoint_step_1000.pt
✓ Model parameters: 51,404,800
✓ Training step: 1000

==========================================
Generating text...
==========================================
Prompt: Once upon a time
Prompt tokens: 4
------------------------------------------------------------
Generating, there was a little girl named Lily. She loved to play 
outside in the sunshine. One day, she saw a big tree with red 
apples. "I want an apple!" said Lily...

------------------------------------------------------------
Generated 96 tokens in 4.82s
Tokens/second: 19.92
Using KV Cache: True
```

## 需要帮助？

- 📖 完整文档：`README_GENERATION.md`
- 🐛 遇到问题：检查checkpoint是否存在，tokenizer是否正确
- 💡 获取更多示例：查看README中的示例部分

开始生成吧！✨
