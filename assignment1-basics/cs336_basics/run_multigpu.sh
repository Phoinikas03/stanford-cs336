#!/bin/bash

# 多GPU训练启动脚本

# 显式指定使用0-3号GPU
# export CUDA_VISIBLE_DEVICES=0,1,2,3

# GPU数量设置为4
NUM_GPUS=8

echo "使用GPU: ${CUDA_VISIBLE_DEVICES}"
echo "GPU数量: ${NUM_GPUS}"
echo "开始多GPU分布式训练..."

# 使用torchrun启动分布式训练
torchrun --nproc_per_node=${NUM_GPUS} train_loop_multigpu.py

# ============ 其他配置选项 ============
# 如果想使用所有可用GPU（自动检测）：
# NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
# torchrun --nproc_per_node=${NUM_GPUS} train_loop_multigpu.py

# 如果想使用其他GPU组合，修改 CUDA_VISIBLE_DEVICES：
# export CUDA_VISIBLE_DEVICES=4,5,6,7  # 使用4-7号GPU
# export CUDA_VISIBLE_DEVICES=0,2,4,6  # 使用0,2,4,6号GPU
