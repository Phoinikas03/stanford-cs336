def param_accounting(
    vocab_size: int,
    context_length: int,  # 实际上 RoPE 不消耗参数，这个入参只用于逻辑检查
    d_model: int,
    num_layers: int,
    d_ff: int):
    """
    计算 LLaMA 架构 (SwiGLU, RoPE, No Bias) 的参数量
    """
    
    # 1. Token Embeddings
    # 形状: [vocab, d_model]
    embedding_params = vocab_size * d_model
    
    # 2. Position Embeddings
    # LLaMA 使用 RoPE，是数学旋转，没有训练参数。
    # 如果是 GPT-2，则是 context_length * d_model
    pos_embedding_params = 0 

    # 3. Attention Weights
    # 包含 W_q, W_k, W_v 和 W_o (输出投影)
    # 这里的 4 代表: Q, K, V, O
    # 忽略了 bias (LLaMA 通常没有 bias)
    attention_params = 4 * d_model * d_model * num_layers

    # 4. FFN (SwiGLU) Weights
    # 包含 Gate_proj, Up_proj, Down_proj
    # 形状分别是 [d, d_ff], [d, d_ff], [d_ff, d]
    # 所以是 3 个矩阵
    ffn_params = 3 * d_model * d_ff * num_layers

    # 5. Layer Norms (RMSNorm)
    # 每个 Block 有 2 个 RMSNorm (Attention前 + FFN前)
    # 还有一个 Final RMSNorm 在最后
    # 每个 RMSNorm 只有一个缩放参数 gamma (维度 d_model)
    norm_params = (2 * num_layers + 1) * d_model

    # 6. Logits Head
    # 形状: [d_model, vocab]
    # LLaMA 通常不共享输入输出权重 (Untied Embeddings)
    head_params = d_model * vocab_size

    total_params = (
        embedding_params +
        pos_embedding_params +
        attention_params +
        ffn_params +
        norm_params +
        head_params
    )
    
    return total_params

def FLOPS_accounting(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
):
    """
    计算 Transformer 模型在前向传播 (Forward Pass) 中的浮点运算次数 (FLOPs)。
    基于 LLaMA 架构假设：
    1. 使用 SwiGLU FFN (3 个矩阵)。
    2. 计算基于矩阵乘法 (MatMul)，忽略 LayerNorm/RoPE/Elem-wise 等非主导计算。
    3. 这是一个 Token 序列 (Batch=1, Length=L) 的完整计算量。
    """
    
    # -------------------------------------------------------
    # 1. Attention 模块 (Per Layer)
    # -------------------------------------------------------
    
    # A. 线性投影 (Q, K, V, O)
    # 包含 4 个矩阵乘法：W_q, W_k, W_v, W_o
    # 形状变换: [L, d] x [d, d] -> [L, d]
    # 计算量: 4 x (2 * L * d^2)
    flops_attn_proj = 8 * context_length * (d_model ** 2)
    
    # B. Attention 核心计算 (QK^T 和 Score*V)
    # 步骤 1: QK^T -> [L, d] x [d, L] -> [L, L] (计算 Attention Scores)
    # 步骤 2: Score*V -> [L, L] x [L, d] -> [L, d] (计算 Context)
    # 计算量: 2 x (2 * L^2 * d)
    # 注意：这是导致长文本推理变慢的 O(L^2) 项
    flops_attn_core = 4 * (context_length ** 2) * d_model
    
    # Attention 层总和
    layer_attn_flops = flops_attn_proj + flops_attn_core

    # -------------------------------------------------------
    # 2. FFN 模块 (Per Layer, SwiGLU)
    # -------------------------------------------------------
    
    # 现代 LLM (SwiGLU) 包含 3 个矩阵: Gate_proj, Up_proj, Down_proj
    # Gate/Up: [L, d] x [d, d_ff]
    # Down:    [L, d_ff] x [d_ff, d]
    # 计算量: 3 x (2 * L * d * d_ff)
    # 若是老式 GPT-2 (GELU)，系数应为 2 而不是 3 (仅 Up/Down)
    layer_ffn_flops = 6 * context_length * d_model * d_ff

    # -------------------------------------------------------
    # 3. 汇总与输出层
    # -------------------------------------------------------
    
    # 单个 Transformer Block 的总 FLOPs
    flops_per_layer = layer_attn_flops + layer_ffn_flops
    
    # 所有层的总 FLOPs
    total_body_flops = flops_per_layer * num_layers
    
    # Logits Head (Un-embedding)
    # 将隐状态映射回词表概率
    # 形状变换: [L, d] x [d, V] -> [L, V]
    # 计算量: 2 * L * d * V
    flops_logits = 2 * context_length * d_model * vocab_size
    
    return {
        "total_flops": total_body_flops + flops_logits,
        "attn_flops": layer_attn_flops * num_layers,
        "ffn_flops": layer_ffn_flops * num_layers,
        "logits_flops": flops_logits,
    }

if __name__ == "__main__":
    param_num = param_accounting(
        vocab_size=50257,
        context_length=1024,
        d_model=1600,
        num_layers=48,
        d_ff=6400,
    )

    print(f"Number of parameters: {param_num / 1e6:.2f}M")
    print(f"Assume each param is 4 bytes, the total memory is {param_num * 4 / 1e9:.2f}GB")

    flops_num = FLOPS_accounting(
        vocab_size=50257,
        context_length=10240,
        d_model=1600,
        num_layers=48,
        d_ff=6400,
    )
    print("GPT2-XL:")
    print(f"Number of FLOPS: {flops_num['total_flops'] / 1e12:.2f}TFLOPs")
    print(f"Attention FLOPS: {flops_num['attn_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num['attn_flops'] / flops_num['total_flops'] * 100:.2f}% of the total FLOPS")
    print(f"FFN FLOPS: {flops_num['ffn_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num['ffn_flops'] / flops_num['total_flops'] * 100:.2f}% of the total FLOPS")
    print(f"Logits FLOPS: {flops_num['logits_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num['logits_flops'] / flops_num['total_flops'] * 100:.2f}% of the total FLOPS")

    flops_num_gpt2_small = FLOPS_accounting(
        vocab_size=50257,
        context_length=1024,
        d_model=768,
        num_layers=12,
        d_ff=3072,
    )

    print("GPT2-Small:")
    print(f"Number of FLOPS: {flops_num_gpt2_small['total_flops'] / 1e12:.2f}TFLOPs")
    print(f"Attention FLOPS: {flops_num_gpt2_small['attn_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num_gpt2_small['attn_flops'] / flops_num_gpt2_small['total_flops'] * 100:.2f}% of the total FLOPS")
    print(f"FFN FLOPS: {flops_num_gpt2_small['ffn_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num_gpt2_small['ffn_flops'] / flops_num_gpt2_small['total_flops'] * 100:.2f}% of the total FLOPS")
    print(f"Logits FLOPS: {flops_num_gpt2_small['logits_flops'] / 1e12:.2f}TFLOPs, taking up {flops_num_gpt2_small['logits_flops'] / flops_num_gpt2_small['total_flops'] * 100:.2f}% of the total FLOPS")
