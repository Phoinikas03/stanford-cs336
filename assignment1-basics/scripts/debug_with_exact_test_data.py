"""
使用完全一致的测试数据进行调试
完全复刻 pytest 测试中的数据生成方式
"""
import torch
import numpy as np
from einops import rearrange
import sys
import os
from pathlib import Path
import json

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cs336_basics.multi_head_attention import MultiHeadSelfAttentionWithRope
from cs336_basics.rope import RotaryPositionalEmbeddings


def load_exact_test_data():
    """
    完全复刻测试 fixtures 的数据生成方式
    """
    # 从 conftest.py 中的 fixtures 复制的参数
    batch_size = 4
    n_queries = 12
    n_heads = 4
    d_head = 16
    d_model = n_heads * d_head  # 64
    n_keys = 16
    theta = 10000.0
    
    # 生成输入数据 (使用相同的随机种子)
    torch.manual_seed(4)
    in_embeddings = torch.randn(batch_size, n_queries, d_model)
    
    # 生成位置 ID
    pos_ids = torch.arange(0, n_queries)
    
    # 加载模型权重
    fixtures_path = project_root / "tests" / "fixtures" / "ts_tests"
    state_dict = torch.load(fixtures_path / "model.pt", map_location="cpu")
    config = json.load(open(fixtures_path / "model_config.json"))
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    # 提取权重
    q_proj_weight = state_dict["layers.0.attn.q_proj.weight"]
    k_proj_weight = state_dict["layers.0.attn.k_proj.weight"]
    v_proj_weight = state_dict["layers.0.attn.v_proj.weight"]
    o_proj_weight = state_dict["layers.0.attn.output_proj.weight"]
    
    # 转换 pos_ids 为与测试相同的格式
    pos_ids = rearrange(pos_ids, "seq -> 1 seq")
    
    return {
        'batch_size': batch_size,
        'n_queries': n_queries,
        'n_heads': n_heads,
        'd_head': d_head,
        'd_model': d_model,
        'n_keys': n_keys,
        'theta': theta,
        'in_embeddings': in_embeddings,
        'pos_ids': pos_ids,
        'q_proj_weight': q_proj_weight,
        'k_proj_weight': k_proj_weight,
        'v_proj_weight': v_proj_weight,
        'o_proj_weight': o_proj_weight,
        'state_dict': state_dict,
        'config': config,
    }


def debug_step_by_step():
    """逐步调试，使用真实测试数据"""
    print("="*80)
    print("使用真实测试数据逐步调试 MultiHeadSelfAttentionWithRope")
    print("="*80)
    
    # 加载真实测试数据
    data = load_exact_test_data()
    
    print(f"\n测试参数 (与 pytest 完全一致):")
    print(f"  batch_size: {data['batch_size']}")
    print(f"  n_queries (seq_len): {data['n_queries']}")
    print(f"  n_heads: {data['n_heads']}")
    print(f"  d_head: {data['d_head']}")
    print(f"  d_model: {data['d_model']}")
    print(f"  max_seq_len (n_keys): {data['n_keys']}")
    print(f"  theta: {data['theta']}")
    
    print(f"\n输入数据:")
    print(f"  in_embeddings.shape: {data['in_embeddings'].shape}")
    print(f"  pos_ids.shape: {data['pos_ids'].shape}")
    print(f"  pos_ids: {data['pos_ids']}")
    print(f"  in_embeddings[0, 0, :5]: {data['in_embeddings'][0, 0, :5]}")
    
    print(f"\n权重矩阵:")
    print(f"  q_proj_weight.shape: {data['q_proj_weight'].shape}")
    print(f"  k_proj_weight.shape: {data['k_proj_weight'].shape}")
    print(f"  v_proj_weight.shape: {data['v_proj_weight'].shape}")
    print(f"  o_proj_weight.shape: {data['o_proj_weight'].shape}")
    
    # 创建模型
    model = MultiHeadSelfAttentionWithRope(
        d_model=data['d_model'],
        d_in=data['d_model'],
        d_out=data['d_model'],
        num_heads=data['n_heads'],
        theta=data['theta'],
        max_seq_len=data['n_keys']
    )
    
    # 设置权重
    model.w_q.weight.data = data['q_proj_weight']
    model.w_k.weight.data = data['k_proj_weight']
    model.w_v.weight.data = data['v_proj_weight']
    model.w_o.weight.data = data['o_proj_weight']
    
    x = data['in_embeddings']
    token_positions = data['pos_ids']
    
    # 开始逐步执行
    print("\n" + "="*80)
    print("步骤 1: 线性投影 Q, K, V")
    print("="*80)
    Q = model.w_q(x)
    K = model.w_k(x)
    V = model.w_v(x)
    print(f"Q.shape: {Q.shape}")
    print(f"K.shape: {K.shape}")
    print(f"V.shape: {V.shape}")
    print(f"Q[0, 0, :5]: {Q[0, 0, :5]}")
    print(f"K[0, 0, :5]: {K[0, 0, :5]}")
    
    print("\n" + "="*80)
    print("步骤 2: 分头 Reshape")
    print("="*80)
    Q_reshaped = rearrange(Q, "... seq (heads d_k) -> ... heads seq d_k", heads=model.num_heads)
    K_reshaped = rearrange(K, "... seq (heads d_k) -> ... heads seq d_k", heads=model.num_heads)
    V_reshaped = rearrange(V, "... seq (heads d_v) -> ... heads seq d_v", heads=model.num_heads)
    print(f"Q_reshaped.shape: {Q_reshaped.shape}")
    print(f"K_reshaped.shape: {K_reshaped.shape}")
    print(f"V_reshaped.shape: {V_reshaped.shape}")
    print(f"每个头的维度 d_k: {Q_reshaped.shape[-1]}")
    print(f"Q_reshaped[0, 0, 0, :]: {Q_reshaped[0, 0, 0, :]}")  # batch=0, head=0, seq=0
    
    print("\n" + "="*80)
    print("步骤 3: 应用 RoPE")
    print("="*80)
    print(f"RoPE 模块配置:")
    print(f"  theta: {model.rope.theta}")
    print(f"  d_k: {model.rope.d_k}")
    print(f"  max_seq_len: {model.rope.max_seq_len}")
    print(f"  cos_sin buffer 形状: {model.rope.cos_sin.shape}")
    
    # 手动执行 RoPE 以便调试
    print(f"\n应用 RoPE 到 Q:")
    print(f"  输入 Q_reshaped.shape: {Q_reshaped.shape}")
    print(f"  token_positions.shape: {token_positions.shape}")
    
    # 获取 cos 和 sin
    cos = model.rope.cos_sin[0][token_positions]
    sin = model.rope.cos_sin[1][token_positions]
    print(f"  cos.shape: {cos.shape}")
    print(f"  sin.shape: {sin.shape}")
    print(f"  cos[0, 0, :3]: {cos[0, 0, :3]}")
    print(f"  sin[0, 0, :3]: {sin[0, 0, :3]}")
    
    # 拆分 Q 为 x1 和 x2
    q1, q2 = rearrange(Q_reshaped, "... (half_d xy) -> xy ... half_d", xy=2)
    print(f"  q1.shape: {q1.shape}")
    print(f"  q2.shape: {q2.shape}")
    print(f"  Q_reshaped[0, 0, 0, :8]: {Q_reshaped[0, 0, 0, :8]}")
    print(f"  q1[0, 0, 0, :4] (偶数索引): {q1[0, 0, 0, :4]}")
    print(f"  q2[0, 0, 0, :4] (奇数索引): {q2[0, 0, 0, :4]}")
    
    # 扩展 cos/sin 维度
    cos_expanded = cos
    sin_expanded = sin
    print(f"\n  维度匹配:")
    print(f"    cos.ndim = {cos.ndim}, q1.ndim = {q1.ndim}")
    while cos_expanded.ndim < q1.ndim:
        cos_expanded = cos_expanded.unsqueeze(-3)
        sin_expanded = sin_expanded.unsqueeze(-3)
        print(f"    扩展后 cos_expanded.shape: {cos_expanded.shape}")
    
    # 应用旋转
    q1_rot = cos_expanded * q1 - sin_expanded * q2
    q2_rot = sin_expanded * q1 + cos_expanded * q2
    print(f"\n  旋转后:")
    print(f"    q1_rot.shape: {q1_rot.shape}")
    print(f"    q2_rot.shape: {q2_rot.shape}")
    print(f"    q1_rot[0, 0, 0, :4]: {q1_rot[0, 0, 0, :4]}")
    print(f"    q2_rot[0, 0, 0, :4]: {q2_rot[0, 0, 0, :4]}")
    
    # 重组
    Q_rope_manual = rearrange([q1_rot, q2_rot], "xy ... x_half -> ... (x_half xy)")
    print(f"\n  重组后:")
    print(f"    Q_rope_manual.shape: {Q_rope_manual.shape}")
    print(f"    Q_rope_manual[0, 0, 0, :8]: {Q_rope_manual[0, 0, 0, :8]}")
    
    # 使用模型的 RoPE
    Q_rope = model.rope(Q_reshaped, token_positions)
    K_rope = model.rope(K_reshaped, token_positions)
    print(f"\n  使用模型 RoPE:")
    print(f"    Q_rope.shape: {Q_rope.shape}")
    print(f"    K_rope.shape: {K_rope.shape}")
    print(f"    Q_rope[0, 0, 0, :8]: {Q_rope[0, 0, 0, :8]}")
    
    # 比较手动和模型 RoPE
    if torch.allclose(Q_rope_manual, Q_rope, atol=1e-6):
        print(f"\n  ✓ 手动 RoPE 与模型 RoPE 一致")
    else:
        print(f"\n  ✗ 手动 RoPE 与模型 RoPE 不一致!")
        print(f"    最大差异: {(Q_rope_manual - Q_rope).abs().max()}")
    
    print("\n" + "="*80)
    print("步骤 4: 计算 Scaled Dot-Product Attention")
    print("="*80)
    from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention
    attention_output = ScaledDotProductAttention()(Q_rope, K_rope, V_reshaped)
    print(f"attention_output.shape: {attention_output.shape}")
    print(f"attention_output[0, 0, 0, :5]: {attention_output[0, 0, 0, :5]}")
    
    print("\n" + "="*80)
    print("步骤 5: 合并多头")
    print("="*80)
    attention_output_merged = rearrange(attention_output, "... heads seq d_v -> ... seq (heads d_v)")
    print(f"attention_output_merged.shape: {attention_output_merged.shape}")
    print(f"attention_output_merged[0, 0, :5]: {attention_output_merged[0, 0, :5]}")
    
    print("\n" + "="*80)
    print("步骤 6: 输出投影")
    print("="*80)
    output = model.w_o(attention_output_merged)
    print(f"output.shape: {output.shape}")
    print(f"output[0, 0, :5]: {output[0, 0, :5]}")
    
    print("\n" + "="*80)
    print("验证：与模型直接调用比较")
    print("="*80)
    with torch.no_grad():
        direct_output = model(x, token_positions)
    print(f"direct_output.shape: {direct_output.shape}")
    print(f"direct_output[0, 0, :5]: {direct_output[0, 0, :5]}")
    
    if torch.allclose(output, direct_output, atol=1e-6):
        print("\n✓ 手动计算与模型直接调用一致")
    else:
        print("\n✗ 手动计算与模型直接调用不一致!")
        print(f"  最大差异: {(output - direct_output).abs().max()}")
    
    # 加载期望的输出（从 snapshot）
    print("\n" + "="*80)
    print("与测试 snapshot 比较")
    print("="*80)
    snapshot_path = project_root / "tests" / "_snapshots" / "test_multihead_self_attention_with_rope.npz"
    if snapshot_path.exists():
        expected = np.load(snapshot_path)
        expected_output = torch.from_numpy(expected['array'])
        print(f"expected_output.shape: {expected_output.shape}")
        print(f"expected_output[0, 0, :5]: {expected_output[0, 0, :5]}")
        
        print(f"\nactual_output[0, 0, :5]:   {direct_output[0, 0, :5]}")
        print(f"expected_output[0, 0, :5]: {expected_output[0, 0, :5]}")
        print(f"差异[0, 0, :5]:            {(direct_output - expected_output)[0, 0, :5]}")
        
        diff = (direct_output - expected_output).abs()
        print(f"\n差异统计:")
        print(f"  最大绝对差异: {diff.max()}")
        print(f"  平均绝对差异: {diff.mean()}")
        print(f"  差异 > 0.01 的元素数: {(diff > 0.01).sum()}/{diff.numel()}")
        print(f"  差异 > 0.1 的元素数: {(diff > 0.1).sum()}/{diff.numel()}")
        
        # 找出最大差异的位置
        max_diff_idx = diff.argmax()
        max_diff_pos = np.unravel_index(max_diff_idx.item(), diff.shape)
        print(f"\n最大差异位置: {max_diff_pos}")
        print(f"  actual: {direct_output[max_diff_pos]}")
        print(f"  expected: {expected_output[max_diff_pos]}")
        print(f"  diff: {diff[max_diff_pos]}")
    else:
        print(f"未找到 snapshot 文件: {snapshot_path}")
    
    return output, direct_output


if __name__ == "__main__":
    try:
        output, direct_output = debug_step_by_step()
        print("\n" + "="*80)
        print("调试完成")
        print("="*80)
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()
