def estimate_training_memory_corrected(
    batch_size,
    context_length,
    vocab_size,
    n_layers,
    d_model,
    n_heads,
    d_ff_ratio=4,
    bytes_per_param=4,
    print_breakdown=True
):
    # ==========================================
    # 1. 参数数量 (Parameters)
    # ==========================================
    # Token Embeddings (V * D)
    # RoPE 不需要可学习的位置编码矩阵 (L * D)，所以这里只有 Token Emb
    param_embeddings = vocab_size * d_model
    
    # Transformer Block parameters per layer:
    # 1. Attention: 4 * D^2 (W_q, W_k, W_v, W_o) + 2*D (RMSNorm scale)
    #    注：RoPE 不增加参数
    param_attn_layer = 4 * (d_model ** 2) + d_model
    
    # 2. FFN (SwiGLU): 3 * (D * d_ff) + D (RMSNorm scale)
    #    SwiGLU 有 W_gate, W_val, W_out 三个矩阵
    d_ff = d_ff_ratio * d_model
    param_ffn_layer = 3 * (d_model * d_ff) + d_model
    
    # Total per layer
    param_per_layer = param_attn_layer + param_ffn_layer
    
    # Output Head (Unembedding): D * V
    # 假设不共享权重 (Untied embeddings)，如果是 Tied，这部分可以去掉或减半
    param_output_head = d_model * vocab_size + d_model # + final norm
    
    # 总参数量
    total_params = param_embeddings + (n_layers * param_per_layer) + param_output_head

    # ==========================================
    # 2. 静态显存 (Static Memory)
    # ==========================================
    # Parameters: float32
    mem_params = total_params * bytes_per_param
    
    # Gradients: 与参数一一对应
    mem_grads = total_params * bytes_per_param
    
    # Optimizer States (AdamW): 存储 m 和 v 两个状态
    mem_optimizer = total_params * 2 * bytes_per_param
    
    # 总静态显存
    total_static_mem = mem_params + mem_grads + mem_optimizer

    # ==========================================
    # 3.激活显存 (Activations)
    # ==========================================
    
    # 3.1 Norms Activations
    # 保存两个 Norm 层的输入 (Input to Pre-Attn Norm, Input to Pre-FFN Norm)
    act_norms = 2 * d_model 

    # 3.2 Attention Activations (Dense part)
    # - Input to QKV Linear (1*L*D)
    # - Input to Output Linear (Context) (1*L*D)
    act_attn_dense = 2 * d_model
    
    # 3.3 Attention Matrix (Scores + Probs)
    # [H, L, L] * 2
    act_attn_matrix = 2 * n_heads * (context_length ** 2)

    # 3.4 FFN (SwiGLU) Activations
    # - Input to FFN Linear (1*L*D)
    # - SwiGLU Internals (4 * d_ff * L) -> (16*L*D)
    act_ffn = (1 * d_model) + (4 * d_ff) 
    
    # 单层总计
    act_per_layer = (act_norms + act_attn_dense + act_ffn) * context_length + act_attn_matrix
    
    # 非层激活 (Logits + Final Norm input)
    # Logits (L*V) + Final Norm input (L*D)
    act_output = (context_length * vocab_size) + (context_length * d_model)
    
    # 总激活
    total_act_mem = batch_size * ((n_layers * act_per_layer) + act_output) * bytes_per_param
    
    # ==========================================
    # 4. 总显存 (Total Memory)
    # ==========================================
    total_mem_bytes = total_static_mem + total_act_mem
    total_mem_gb = total_mem_bytes / (1024**3)

    if print_breakdown:
        print(f"--- Corrected Memory Estimation ---")
        print(f"Batch: {batch_size}, Layers: {n_layers}, Dim: {d_model}")
        print(f"Static Memory:  {total_static_mem / 1024**3:.2f} GB")
        print(f"Activation Mem: {total_act_mem / 1024**3:.2f} GB")
        print(f"Total Memory:   {total_mem_gb:.2f} GB")
        
    return total_mem_gb

# 重新运行估算
estimate_training_memory_corrected(
    batch_size=2,
    context_length=1024,
    vocab_size=50257,
    n_layers=48,
    d_model=1600,
    n_heads=25
)

import sys
import os
from pathlib import Path

# Add current directory to sys.path to allow sibling imports
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from p3_6_transformer_accounting import FLOPS_accounting, param_accounting

forward_flops_num_gpt2_xl = FLOPS_accounting(
    vocab_size=50257,
    context_length=1024,
    d_model=1600,
    num_layers=48,
    d_ff=6400,
)['total_flops']

param_num_gpt2_xl = param_accounting(
    vocab_size=50257,
    context_length=1024,
    d_model=1600,
    num_layers=48,
    d_ff=6400,
)
backward_flops_num_gpt2_xl = forward_flops_num_gpt2_xl * 2
optimizer_flops_num_gpt2_xl = param_num_gpt2_xl * 12
print(f'backward_flops_num_gpt2_xl: {backward_flops_num_gpt2_xl / 1e12:.2f} TFLOPS')
print(f'optimizer_flops_num_gpt2_xl: {optimizer_flops_num_gpt2_xl / 1e12:.2f} TFLOPS')