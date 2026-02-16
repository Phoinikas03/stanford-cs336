# KV Cache 详解：Prefill vs Decode

## 问题：为什么使用KV Cache后，只输入最后一个token？

这是因为文本生成分为两个阶段：**Prefill** 和 **Decode**。

## 两个阶段详解

### 阶段1：Prefill（预填充）

**发生时机**：生成的第一步

**输入**：完整的prompt（例如："Once upon a time"）

**过程**：
```python
# 第1次循环（i=0）
kv_cache = None  # 缓存还未创建
input_for_model = generated_ids  # 完整prompt：["Once", "upon", "a", "time"]

# 前向传播
logits, kv_cache = forward_with_kv_cache(model, input_for_model, kv_cache, use_cache=True)

# 这一步会：
# 1. 计算所有token的Q、K、V
# 2. 计算attention(Q, K, V)
# 3. 将K和V缓存起来
# 4. 返回所有位置的logits
```

**计算量**：
- 需要计算所有token之间的注意力
- 复杂度：O(n²)，其中n是prompt长度
- 这是最耗时的步骤

**结果**：
- 生成第一个新token："there"
- KV Cache已建立，包含prompt的所有K、V

### 阶段2：Decode（解码）

**发生时机**：生成的第2步及以后

**输入**：只有最后一个token

**过程**：
```python
# 第2次循环（i=1）
kv_cache != None  # 缓存已存在
input_for_model = generated_ids[:, -1:]  # 只有最后一个token：["there"]

# 前向传播
logits, kv_cache = forward_with_kv_cache(model, input_for_model, kv_cache, use_cache=True)

# 这一步会：
# 1. 只计算新token的Q、K、V
# 2. 从cache中获取之前的K、V
# 3. 将新的K、V追加到cache
# 4. 计算attention(Q_new, K_all, V_all)
# 5. 只返回新token位置的logits
```

**计算量**：
- 只需计算新token的K、V
- 复杂度：O(n)，其中n是当前序列长度
- 大幅加速！

**结果**：
- 生成下一个token："was"
- 更新KV Cache，追加新token的K、V

## 完整示例

假设我们要生成："Once upon a time there was a princess"

```
初始：prompt = ["Once", "upon", "a", "time"]

=== Prefill阶段（第1次循环）===
输入：["Once", "upon", "a", "time"]  # 4个token
计算：所有4个token的K、V，并缓存
      计算所有token之间的注意力（4×4=16次）
输出：生成 "there"
KV Cache状态：[K₁, K₂, K₃, K₄] 和 [V₁, V₂, V₃, V₄]

=== Decode阶段（第2次循环）===
输入：["there"]  # 1个token
计算：只计算"there"的K、V
      从cache获取之前的K、V
      只需计算5个注意力分数（不是5×5=25次）
输出：生成 "was"
KV Cache状态：[K₁, K₂, K₃, K₄, K₅] 和 [V₁, V₂, V₃, V₄, V₅]

=== Decode阶段（第3次循环）===
输入：["was"]  # 1个token
计算：只计算"was"的K、V
      从cache获取之前的K、V
      只需计算6个注意力分数
输出：生成 "a"
KV Cache状态：[K₁, K₂, K₃, K₄, K₅, K₆] 和 [V₁, V₂, V₃, V₄, V₅, V₆]

... 以此类推
```

## 性能对比

### 生成100个token的计算量

**不使用KV Cache：**
```
第1步：1×1 = 1 次注意力计算
第2步：2×2 = 4 次注意力计算
第3步：3×3 = 9 次注意力计算
...
第100步：100×100 = 10,000 次注意力计算

总计：1 + 4 + 9 + ... + 10,000 ≈ 338,350 次
```

**使用KV Cache：**
```
Prefill：1×1 = 1 次
第2步：1 次（只计算新token）
第3步：1 次
...
第100步：1 次

总计：1 + 1 + 1 + ... + 1 = 100 次
```

**加速比：338,350 / 100 ≈ 3,383倍（理论值）**

实际加速比约为10-50倍，因为：
- Prefill仍需要完整计算
- 内存访问开销
- 其他计算（FFN、归一化等）无法缓存

## 代码实现要点

### 1. KV Cache数据结构

```python
class KVCache:
    def __init__(self, num_layers, batch_size, num_heads, max_seq_len, d_k, device):
        # 为每一层存储K和V
        self.k_cache = [torch.zeros(batch_size, num_heads, 0, d_k, device=device) 
                        for _ in range(num_layers)]
        self.v_cache = [torch.zeros(batch_size, num_heads, 0, d_k, device=device) 
                        for _ in range(num_layers)]
    
    def update(self, layer_idx, k, v):
        """追加新的K和V"""
        self.k_cache[layer_idx] = torch.cat([self.k_cache[layer_idx], k], dim=2)
        self.v_cache[layer_idx] = torch.cat([self.v_cache[layer_idx], v], dim=2)
```

### 2. 注意力计算

```python
def forward_attention_with_cache(attn, x, token_positions, kv_cache, layer_idx, use_cache):
    # 计算当前token的Q、K、V
    Q = attn.w_q(x)
    K = attn.w_k(x)
    V = attn.w_v(x)
    
    # 应用RoPE
    Q = attn.rope(Q, token_positions)
    K = attn.rope(K, token_positions)
    
    if use_cache and kv_cache is not None:
        # 更新缓存
        kv_cache.update(layer_idx, K, V)
        # 获取完整的K和V（包括历史）
        K, V = kv_cache.get(layer_idx)
    
    # 计算注意力
    # Q: (batch, heads, 1, d_k) - 只有新token
    # K: (batch, heads, seq_len, d_k) - 所有历史token
    # V: (batch, heads, seq_len, d_k) - 所有历史token
    output = attention(Q, K, V)
    
    return output
```

### 3. Token位置编码

```python
# Prefill阶段：positions = [0, 1, 2, 3]
token_positions = torch.arange(seq_len, device=device)

# Decode阶段：position = [4]（接着之前的）
cache_len = kv_cache.get_seq_length()  # 4
token_positions = torch.arange(cache_len, cache_len + 1, device=device)
```

## 内存使用

每一层的KV Cache占用：
```
内存 = 2 × batch_size × num_heads × seq_len × d_k × 4 bytes
```

示例（生成100个token）：
```
配置：
- 6层
- 16个头
- d_k = 32
- batch_size = 1
- seq_len = 100

总内存 = 2 × 1 × 16 × 100 × 32 × 4 bytes × 6 layers
       = 6,144,000 bytes
       ≈ 6 MB
```

这相对于模型权重（通常几百MB到几GB）来说很小。

## 常见问题

### Q1: 为什么第一次循环要输入完整prompt？

A: 因为KV Cache还未建立，需要计算所有token的K和V。这就是Prefill阶段。

### Q2: 为什么之后只输入一个token？

A: 因为之前的K和V已经缓存了，只需要计算新token的K和V，然后与缓存的K、V一起做注意力计算。

### Q3: 每次生成都要重新Prefill吗？

A: 不需要！只有第一次需要。之后的每一步都是Decode，直到生成完成或达到最大长度。

### Q4: 如果想继续生成，可以复用cache吗？

A: 可以！只要保留KV Cache，就可以继续从任何位置生成。这就是对话系统能够保持上下文的原因。

### Q5: KV Cache会一直增长吗？

A: 是的。每生成一个token，cache就会增长。当达到max_seq_len时，通常需要：
- 截断旧的token
- 或停止生成
- 或使用滑动窗口注意力

## 可视化对比

```
不使用KV Cache：
Step 1: [Prompt] → Model → Token1
Step 2: [Prompt + Token1] → Model → Token2
Step 3: [Prompt + Token1 + Token2] → Model → Token3
...
每步都要重新计算所有之前的K和V

使用KV Cache：
Step 1 (Prefill): [Prompt] → Model → Token1 + Cache[Prompt]
Step 2 (Decode):  [Token1] → Model + Cache[Prompt] → Token2 + Cache[Prompt + Token1]
Step 3 (Decode):  [Token2] → Model + Cache[Prompt + Token1] → Token3 + Cache[...]
...
每步只计算新token，复用缓存
```

## 总结

| 特性 | Prefill阶段 | Decode阶段 |
|------|-------------|------------|
| 发生时机 | 第1次循环 | 第2次及以后 |
| 输入 | 完整prompt | 单个token |
| KV Cache状态 | 创建并初始化 | 增量更新 |
| 计算复杂度 | O(n²) | O(n) |
| 相对耗时 | 较长 | 很短 |
| 输出 | 第一个生成token | 每次一个token |

KV Cache是现代LLM推理优化的核心技术之一，理解它的工作原理对于高效部署模型至关重要！

## 相关资源

- `text_generate.py` - 完整实现
- `README_GENERATION.md` - 使用指南
- `benchmark_generation.py` - 性能测试
