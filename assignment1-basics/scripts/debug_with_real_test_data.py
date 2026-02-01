"""
使用真实测试数据进行调试
从 pytest 测试中提取实际数据来重现问题
"""
import torch
import numpy as np
from einops import rearrange
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.multi_head_attention import MultiHeadSelfAttentionWithRope
from cs336_basics.rope import RotaryPositionalEmbeddings
from tests.adapters import run_multihead_self_attention_with_rope


def load_real_test_data():
    """从测试 fixtures 加载真实数据"""
    import pytest
    from tests.conftest import in_embeddings, d_model, n_heads, ts_state_dict, n_keys, theta, pos_ids
    
    # 创建一个临时的测试环境
    class DummyRequest:
        pass
    
    request = DummyRequest()
    
    # 获取测试数据
    embeddings = in_embeddings(request)
    model_dim = d_model(request)
    heads = n_heads(request)
    state_dict = ts_state_dict(request)
    keys = n_keys(request)
    theta_val = theta(request)
    positions = pos_ids(request)
    
    return embeddings, model_dim, heads, state_dict, keys, theta_val, positions


def manual_rope_application(rope_module, x, token_positions, step_name=""):
    """手动执行 RoPE 操作，打印每一步"""
    print(f"\n{'='*60}")
    print(f"手动执行 RoPE: {step_name}")
    print(f"{'='*60}")
    
    print(f"输入 x 形状: {x.shape}")
    print(f"token_positions 形状: {token_positions.shape}")
    print(f"token_positions 值:\n{token_positions}")
    
    # 检查 cos_sin buffer
    print(f"\ncos_sin buffer 形状: {rope_module.cos_sin.shape}")
    print(f"cos_sin[0] 形状 (cos cache): {rope_module.cos_sin[0].shape}")
    print(f"cos_sin[1] 形状 (sin cache): {rope_module.cos_sin[1].shape}")
    
    # 索引 cos 和 sin
    print(f"\n索引操作: rope_module.cos_sin[0][token_positions]")
    cos = rope_module.cos_sin[0][token_positions]
    sin = rope_module.cos_sin[1][token_positions]
    print(f"cos 形状: {cos.shape}")
    print(f"sin 形状: {sin.shape}")
    
    # 显示 cos/sin 的一些值
    print(f"\ncos 的统计信息:")
    print(f"  min: {cos.min():.6f}, max: {cos.max():.6f}, mean: {cos.mean():.6f}")
    print(f"  第一个位置 (pos=0) 的 cos 值: {cos[0, 0, :5]}")
    print(f"\nsin 的统计信息:")
    print(f"  min: {sin.min():.6f}, max: {sin.max():.6f}, mean: {sin.mean():.6f}")
    print(f"  第一个位置 (pos=0) 的 sin 值: {sin[0, 0, :5]}")
    
    # 拆分 x
    print(f"\n拆分 x 为 x1 和 x2 (偶数索引和奇数索引)")
    x1, x2 = rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)
    print(f"x1 形状: {x1.shape}")
    print(f"x2 形状: {x2.shape}")
    
    print(f"\nx 的原始值 (第一个 batch, 第一个 head, 第一个 token, 前10维):")
    print(f"  {x[0, 0, 0, :10]}")
    print(f"x1 的值 (偶数索引):")
    print(f"  {x1[0, 0, 0, :5]}")
    print(f"x2 的值 (奇数索引):")
    print(f"  {x2[0, 0, 0, :5]}")
    
    # 检查维度匹配并扩展
    print(f"\n维度匹配检查:")
    print(f"  cos.ndim = {cos.ndim}, x1.ndim = {x1.ndim}")
    
    cos_expanded = cos
    sin_expanded = sin
    expand_count = 0
    while cos_expanded.ndim < x1.ndim:
        print(f"  扩展维度 {expand_count + 1}: 在位置 -3 插入新维度")
        cos_expanded = cos_expanded.unsqueeze(-3)
        sin_expanded = sin_expanded.unsqueeze(-3)
        expand_count += 1
    
    print(f"  扩展后 cos 形状: {cos_expanded.shape}")
    print(f"  扩展后 sin 形状: {sin_expanded.shape}")
    
    # 应用旋转公式
    print(f"\n应用旋转公式:")
    print(f"  x1_rot = cos * x1 - sin * x2")
    print(f"  x2_rot = sin * x1 + cos * x2")
    
    x1_rot = cos_expanded * x1 - sin_expanded * x2
    x2_rot = sin_expanded * x1 + cos_expanded * x2
    
    print(f"\nx1_rot 形状: {x1_rot.shape}")
    print(f"x2_rot 形状: {x2_rot.shape}")
    print(f"x1_rot 的前5个值: {x1_rot[0, 0, 0, :5]}")
    print(f"x2_rot 的前5个值: {x2_rot[0, 0, 0, :5]}")
    
    # 重新组合
    print(f"\n重新组合 (交织拼接):")
    result = rearrange([x1_rot, x2_rot], "xy ... x_half -> ... (x_half xy)")
    print(f"result 形状: {result.shape}")
    print(f"result 的前10个值: {result[0, 0, 0, :10]}")
    
    # 与模块的输出比较
    print(f"\n与 rope_module.forward() 的输出比较:")
    with torch.no_grad():
        module_output = rope_module(x, token_positions)
    print(f"module_output 形状: {module_output.shape}")
    print(f"module_output 的前10个值: {module_output[0, 0, 0, :10]}")
    
    if torch.allclose(result, module_output, atol=1e-6):
        print("\n✓ 手动计算与模块输出一致")
    else:
        print("\n✗ 手动计算与模块输出不一致！")
        print(f"  最大差异: {(result - module_output).abs().max():.8f}")
    
    return result


def compare_with_without_rope():
    """比较有无 RoPE 的多头注意力"""
    print("\n" + "="*80)
    print("比较有无 RoPE 的多头注意力")
    print("="*80)
    
    # 这里需要实际的测试数据
    # 由于直接加载 fixture 比较复杂，我们可以运行测试并捕获数据
    print("\n请直接运行 pytest 来获取实际数据")
    print("或者查看测试失败时的输出中的 ACTUAL 和 DESIRED 值")


def simple_debug():
    """简化的调试：使用小规模数据"""
    print("\n" + "="*80)
    print("简化调试：使用小规模数据测试 RoPE")
    print("="*80)
    
    # 小规模测试
    d_k = 4  # 每个头的维度
    seq_len = 3
    batch_size = 1
    num_heads = 2
    
    # 创建简单的输入
    x = torch.randn(batch_size, num_heads, seq_len, d_k)
    token_positions = torch.arange(seq_len).unsqueeze(0)  # (1, 3)
    
    print(f"\n测试配置:")
    print(f"  d_k: {d_k}")
    print(f"  seq_len: {seq_len}")
    print(f"  batch_size: {batch_size}")
    print(f"  num_heads: {num_heads}")
    print(f"  x 形状: {x.shape}")
    print(f"  token_positions: {token_positions}")
    
    # 创建 RoPE 模块
    rope = RotaryPositionalEmbeddings(theta=10000.0, d_k=d_k, max_seq_len=10)
    
    # 手动执行
    manual_rope_application(rope, x, token_positions, "小规模测试")


if __name__ == "__main__":
    print("RoPE 调试工具")
    print("="*80)
    
    # 运行简化调试
    simple_debug()
    
    print("\n\n提示：")
    print("1. 运行此脚本查看 RoPE 的详细执行过程")
    print("2. 运行 pytest -k test_rope -v 来测试单独的 RoPE 模块")
    print("3. 如果 test_rope 通过但 test_multihead_self_attention_with_rope 失败，")
    print("   问题可能在多头注意力的集成部分")
