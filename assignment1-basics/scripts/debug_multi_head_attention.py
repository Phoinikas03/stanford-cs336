"""
调试脚本：用于分析 MultiHeadSelfAttentionWithRope 的中间计算过程
"""
import torch
import numpy as np
from einops import rearrange
from cs336_basics.multi_head_attention import MultiHeadSelfAttentionWithRope
from cs336_basics.rope import RotaryPositionalEmbeddings


def load_test_data():
    """加载测试数据"""
    # 使用固定的随机种子以保证可重现性
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 从测试配置中获取参数
    d_model = 64
    num_heads = 4
    d_k = d_model // num_heads  # 16
    seq_len = 12
    batch_size = 1
    max_seq_len = 16
    theta = 10000.0
    
    # 生成测试数据（模拟测试中的数据）
    in_embeddings = torch.randn(batch_size, seq_len, d_model)
    token_positions = torch.arange(seq_len).unsqueeze(0)  # (1, 12)
    
    # 生成随机权重
    q_proj_weight = torch.randn(d_model, d_model) * 0.1
    k_proj_weight = torch.randn(d_model, d_model) * 0.1
    v_proj_weight = torch.randn(d_model, d_model) * 0.1
    o_proj_weight = torch.randn(d_model, d_model) * 0.1
    
    return {
        'd_model': d_model,
        'num_heads': num_heads,
        'd_k': d_k,
        'seq_len': seq_len,
        'batch_size': batch_size,
        'max_seq_len': max_seq_len,
        'theta': theta,
        'in_embeddings': in_embeddings,
        'token_positions': token_positions,
        'q_proj_weight': q_proj_weight,
        'k_proj_weight': k_proj_weight,
        'v_proj_weight': v_proj_weight,
        'o_proj_weight': o_proj_weight,
    }


def debug_rope_forward(rope_module, x, token_positions):
    """调试 RoPE 的 forward 过程"""
    print("\n" + "="*80)
    print("调试 RoPE Forward")
    print("="*80)
    
    print(f"\n输入 x 的形状: {x.shape}")
    print(f"token_positions 的形状: {token_positions.shape}")
    print(f"token_positions 的值: {token_positions}")
    
    # 获取 cos 和 sin
    cos = rope_module.cos_sin[0][token_positions]
    sin = rope_module.cos_sin[1][token_positions]
    print(f"\ncos 的形状: {cos.shape}")
    print(f"sin 的形状: {sin.shape}")
    print(f"cos 的前几个值: {cos[0, 0, :5]}")
    print(f"sin 的前几个值: {sin[0, 0, :5]}")
    
    # 拆分 x 为 x1 和 x2
    x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)
    print(f"\nx1 的形状: {x1.shape}")
    print(f"x2 的形状: {x2.shape}")
    print(f"x1 的前几个值: {x1[0, 0, 0, :5]}")
    print(f"x2 的前几个值: {x2[0, 0, 0, :5]}")
    
    # 检查维度匹配
    print(f"\n检查广播前的维度:")
    print(f"  cos.ndim = {cos.ndim}, x1.ndim = {x1.ndim}")
    
    # 添加维度
    cos_expanded = cos
    sin_expanded = sin
    while cos_expanded.ndim < x1.ndim:
        cos_expanded = cos_expanded.unsqueeze(-3)
        sin_expanded = sin_expanded.unsqueeze(-3)
    
    print(f"\n广播后的形状:")
    print(f"  cos_expanded: {cos_expanded.shape}")
    print(f"  sin_expanded: {sin_expanded.shape}")
    
    # 计算旋转
    x1_rot = cos_expanded * x1 - sin_expanded * x2
    x2_rot = sin_expanded * x1 + cos_expanded * x2
    
    print(f"\nx1_rot 的形状: {x1_rot.shape}")
    print(f"x2_rot 的形状: {x2_rot.shape}")
    print(f"x1_rot 的前几个值: {x1_rot[0, 0, 0, :5]}")
    print(f"x2_rot 的前几个值: {x2_rot[0, 0, 0, :5]}")
    
    # 重新组合
    result = rearrange([x1_rot, x2_rot], "xy ... x_half -> ... (x_half xy)")
    print(f"\n最终 result 的形状: {result.shape}")
    print(f"result 的前几个值: {result[0, 0, 0, :10]}")
    
    return result


def debug_multihead_attention_with_rope():
    """逐步调试 MultiHeadSelfAttentionWithRope"""
    print("\n" + "="*80)
    print("开始调试 MultiHeadSelfAttentionWithRope")
    print("="*80)
    
    # 加载测试数据
    data = load_test_data()
    
    print(f"\n测试参数:")
    print(f"  d_model: {data['d_model']}")
    print(f"  num_heads: {data['num_heads']}")
    print(f"  d_k (per head): {data['d_k']}")
    print(f"  seq_len: {data['seq_len']}")
    print(f"  batch_size: {data['batch_size']}")
    print(f"  max_seq_len: {data['max_seq_len']}")
    print(f"  theta: {data['theta']}")
    
    # 创建模型
    model = MultiHeadSelfAttentionWithRope(
        d_model=data['d_model'],
        d_in=data['d_model'],
        d_out=data['d_model'],
        num_heads=data['num_heads'],
        theta=data['theta'],
        max_seq_len=data['max_seq_len']
    )
    
    # 设置权重
    model.w_q.weight.data = data['q_proj_weight']
    model.w_k.weight.data = data['k_proj_weight']
    model.w_v.weight.data = data['v_proj_weight']
    model.w_o.weight.data = data['o_proj_weight']
    
    x = data['in_embeddings']
    token_positions = data['token_positions']
    
    print(f"\n输入形状:")
    print(f"  x: {x.shape}")
    print(f"  token_positions: {token_positions.shape}")
    print(f"  x 的前几个值: {x[0, 0, :5]}")
    
    # 步骤 1: 线性投影
    print("\n" + "-"*80)
    print("步骤 1: 线性投影 Q, K, V")
    print("-"*80)
    Q = model.w_q(x)
    K = model.w_k(x)
    V = model.w_v(x)
    print(f"Q 的形状: {Q.shape}")
    print(f"K 的形状: {K.shape}")
    print(f"V 的形状: {V.shape}")
    print(f"Q 的前几个值: {Q[0, 0, :5]}")
    
    # 步骤 2: 分头
    print("\n" + "-"*80)
    print("步骤 2: 分头 (Reshape)")
    print("-"*80)
    Q = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=model.num_heads)
    K = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=model.num_heads)
    V = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=model.num_heads)
    print(f"Q 的形状: {Q.shape}")
    print(f"K 的形状: {K.shape}")
    print(f"V 的形状: {V.shape}")
    print(f"Q[0, 0, 0, :] (第1个头, 第1个token): {Q[0, 0, 0, :]}")
    
    # 步骤 3: 应用 RoPE
    print("\n" + "-"*80)
    print("步骤 3: 应用 RoPE 到 Q 和 K")
    print("-"*80)
    print("应用 RoPE 到 Q:")
    Q_rope = debug_rope_forward(model.rope, Q, token_positions)
    
    print("\n应用 RoPE 到 K:")
    K_rope = debug_rope_forward(model.rope, K, token_positions)
    
    print(f"\n应用 RoPE 后:")
    print(f"  Q_rope 的形状: {Q_rope.shape}")
    print(f"  K_rope 的形状: {K_rope.shape}")
    print(f"  Q_rope[0, 0, 0, :] (第1个头, 第1个token): {Q_rope[0, 0, 0, :]}")
    
    # 步骤 4: 计算注意力
    print("\n" + "-"*80)
    print("步骤 4: 计算 Scaled Dot-Product Attention")
    print("-"*80)
    from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
    attention_output = ScaledDotProductAttention()(Q_rope, K_rope, V)
    print(f"attention_output 的形状: {attention_output.shape}")
    print(f"attention_output[0, 0, 0, :] (第1个头, 第1个token): {attention_output[0, 0, 0, :]}")
    
    # 步骤 5: 合并多头
    print("\n" + "-"*80)
    print("步骤 5: 合并多头")
    print("-"*80)
    attention_output = rearrange(attention_output, "... heads seq d_v -> ... seq (heads d_v)")
    print(f"合并后的形状: {attention_output.shape}")
    print(f"attention_output[0, 0, :10]: {attention_output[0, 0, :10]}")
    
    # 步骤 6: 输出投影
    print("\n" + "-"*80)
    print("步骤 6: 输出投影")
    print("-"*80)
    output = model.w_o(attention_output)
    print(f"最终输出的形状: {output.shape}")
    print(f"output[0, 0, :10]: {output[0, 0, :10]}")
    
    # 与直接调用比较
    print("\n" + "="*80)
    print("验证：与直接调用 model(x, token_positions) 的结果比较")
    print("="*80)
    with torch.no_grad():
        direct_output = model(x, token_positions)
    print(f"直接调用的输出形状: {direct_output.shape}")
    print(f"直接调用的输出[0, 0, :10]: {direct_output[0, 0, :10]}")
    
    # 检查是否一致
    if torch.allclose(output, direct_output, atol=1e-6):
        print("\n✓ 逐步计算与直接调用的结果一致！")
    else:
        print("\n✗ 逐步计算与直接调用的结果不一致！")
        print(f"最大差异: {(output - direct_output).abs().max()}")
    
    return output


if __name__ == "__main__":
    output = debug_multihead_attention_with_rope()
    print("\n" + "="*80)
    print("调试完成")
    print("="*80)
