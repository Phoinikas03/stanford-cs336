#!/usr/bin/env python3
"""
DDP训练错误诊断脚本

用于排查多GPU训练时的错误
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

# 添加父目录到sys.path以支持导入cs336_basics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def diagnose_ddp_error():
    """诊断DDP训练环境和模型"""
    
    print("=" * 60)
    print("DDP训练环境诊断")
    print("=" * 60)
    
    # 1. 检查CUDA可用性
    print("\n【1】CUDA环境:")
    print(f"  CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    当前已用: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
    
    # 2. 检查分布式环境变量
    print("\n【2】分布式环境变量:")
    env_vars = ["RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"]
    for var in env_vars:
        value = os.environ.get(var, "未设置")
        print(f"  {var}: {value}")
    
    # 3. 尝试导入和测试模型
    print("\n【3】模型测试:")
    try:
        from cs336_basics.transformer_lm import TransformerLM
        from cs336_basics.get_batch import get_batch
        
        # 创建测试模型（使用较小的配置）
        test_config = {
            "vocab_size": 10000,
            "context_length": 256,
            "d_model": 128,  # 较小的模型用于测试
            "num_layers": 2,
            "num_heads": 8,
            "d_ff": 512,
            "rope_theta": 10000,
        }
        
        print(f"  创建测试模型...")
        model = TransformerLM(**test_config)
        print(f"  ✓ 模型创建成功")
        print(f"  ✓ 参数数量: {model.get_num_params():,}")
        
        # 测试forward pass（CPU）
        print(f"\n  测试CPU forward pass...")
        batch_size = 4
        seq_len = 32
        test_input = torch.randint(0, test_config["vocab_size"], (batch_size, seq_len))
        
        try:
            with torch.no_grad():
                output = model(test_input)
            print(f"  ✓ CPU forward pass成功")
            print(f"    输入形状: {test_input.shape}")
            print(f"    输出形状: {output.shape}")
        except Exception as e:
            print(f"  ✗ CPU forward pass失败:")
            print(f"    错误: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # 测试CUDA forward pass
        if torch.cuda.is_available():
            print(f"\n  测试CUDA forward pass...")
            device = torch.device("cuda:0")
            model = model.to(device)
            test_input = test_input.to(device)
            
            try:
                with torch.no_grad():
                    output = model(test_input)
                print(f"  ✓ CUDA forward pass成功")
                print(f"    输入形状: {test_input.shape}")
                print(f"    输出形状: {output.shape}")
                print(f"    GPU显存使用: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
            except Exception as e:
                print(f"  ✗ CUDA forward pass失败:")
                print(f"    错误: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
            
            # 测试更大的batch size
            print(f"\n  测试更大batch size (batch_size=64, seq_len=256)...")
            batch_size = 64
            seq_len = 256
            test_input_large = torch.randint(0, test_config["vocab_size"], 
                                            (batch_size, seq_len), device=device)
            
            try:
                torch.cuda.empty_cache()
                mem_before = torch.cuda.memory_allocated(0) / 1024**2
                
                with torch.no_grad():
                    output = model(test_input_large)
                
                mem_after = torch.cuda.memory_allocated(0) / 1024**2
                print(f"  ✓ 大batch size测试成功")
                print(f"    显存增量: {mem_after - mem_before:.2f} MB")
                print(f"    总显存使用: {mem_after:.2f} MB")
            except torch.cuda.OutOfMemoryError:
                print(f"  ⚠️  GPU显存不足（OOM）")
                print(f"    当前batch_size={batch_size}, seq_len={seq_len}对于此GPU可能过大")
                print(f"    建议：减小batch_size_per_gpu或使用更大显存的GPU")
                return False
            except Exception as e:
                print(f"  ✗ 大batch size测试失败:")
                print(f"    错误: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        # 4. 测试RoPE
        print(f"\n【4】RoPE测试:")
        from cs336_basics.rope import RotaryPositionalEmbeddings
        
        d_k = test_config["d_model"] // test_config["num_heads"]
        rope = RotaryPositionalEmbeddings(
            theta=test_config["rope_theta"],
            d_k=d_k,
            max_seq_len=test_config["context_length"],
            device=torch.device("cuda:0") if torch.cuda.is_available() else None
        )
        
        print(f"  RoPE配置:")
        print(f"    theta: {test_config['rope_theta']}")
        print(f"    d_k: {d_k}")
        print(f"    max_seq_len: {test_config['context_length']}")
        
        # 测试不同形状的输入
        test_cases = [
            ("without head dim", (4, 32, d_k)),  # [batch, seq, d_k]
            ("with head dim", (4, 8, 32, d_k)),  # [batch, head, seq, d_k]
        ]
        
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        for name, shape in test_cases:
            print(f"\n  测试 {name}: {shape}")
            x = torch.randn(shape, device=device)
            
            # token_positions形状应该是 [batch, seq]
            batch_size = shape[0]
            seq_len = shape[-2] if len(shape) == 4 else shape[1]
            token_positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            
            print(f"    输入形状: {x.shape}")
            print(f"    token_positions形状: {token_positions.shape}")
            
            # 检查token_positions是否超出范围
            max_pos = token_positions.max().item()
            if max_pos >= test_config["context_length"]:
                print(f"    ⚠️  token_positions最大值 ({max_pos}) >= max_seq_len ({test_config['context_length']})")
                print(f"    这会导致IndexError!")
                return False
            
            try:
                output = rope(x, token_positions)
                print(f"    ✓ 输出形状: {output.shape}")
                assert output.shape == x.shape, f"输出形状不匹配！{output.shape} != {x.shape}"
            except Exception as e:
                print(f"    ✗ RoPE测试失败:")
                print(f"      错误: {type(e).__name__}: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        print(f"\n【5】结论:")
        print(f"  ✓ 所有基础测试通过")
        print(f"  ✓ 模型在单GPU上运行正常")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"  ✗ 测试失败: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ddp_initialization():
    """测试DDP初始化"""
    print("\n" + "=" * 60)
    print("测试DDP初始化")
    print("=" * 60)
    
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == -1:
        print("\n⚠️  未检测到DDP环境变量")
        print("  这不是错误，只是表示当前不是在DDP模式下运行")
        print("\n  要测试DDP，请使用:")
        print("    torchrun --nproc_per_node=4 debug_ddp_error.py")
        return False
    
    print(f"\n  检测到DDP环境:")
    print(f"    local_rank: {local_rank}")
    print(f"    world_size: {world_size}")
    
    try:
        # 初始化进程组
        print(f"\n  初始化DDP进程组...")
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        
        print(f"  ✓ 进程组初始化成功")
        print(f"    rank: {dist.get_rank()}")
        print(f"    world_size: {dist.get_world_size()}")
        
        # 测试创建DDP模型
        print(f"\n  测试创建DDP模型...")
        from cs336_basics.transformer_lm import TransformerLM
        
        model = TransformerLM(
            vocab_size=10000,
            context_length=256,
            d_model=128,
            num_layers=2,
            num_heads=8,
            d_ff=512,
            rope_theta=10000,
        )
        
        device = torch.device(f"cuda:{local_rank}")
        model = model.to(device)
        model = DDP(model, device_ids=[local_rank])
        
        print(f"  ✓ DDP模型创建成功")
        
        # 测试forward pass
        print(f"\n  测试DDP forward pass...")
        test_input = torch.randint(0, 10000, (4, 32), device=device)
        
        with torch.no_grad():
            output = model(test_input)
        
        print(f"  ✓ DDP forward pass成功")
        print(f"    输入形状: {test_input.shape}")
        print(f"    输出形状: {output.shape}")
        
        # 清理
        dist.destroy_process_group()
        print(f"\n  ✓ DDP测试完成")
        return True
        
    except Exception as e:
        print(f"\n  ✗ DDP测试失败:")
        print(f"    错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "🔍 DDP训练错误诊断工具" + "\n")
    
    # 运行基础诊断
    success = diagnose_ddp_error()
    
    # 如果在DDP环境中，测试DDP初始化
    if "LOCAL_RANK" in os.environ:
        success = success and test_ddp_initialization()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有诊断测试通过")
        print("\n如果训练仍然失败，请:")
        print("  1. 查看完整的错误堆栈跟踪")
        print("  2. 检查训练数据的形状和设备")
        print("  3. 尝试减小batch_size_per_gpu")
    else:
        print("✗ 发现问题，请根据上述输出修复")
    print("=" * 60 + "\n")
    
    sys.exit(0 if success else 1)
