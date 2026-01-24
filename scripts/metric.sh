#!/bin/bash

# TravelUAV 指标计算脚本
# 用于计算评估结果的各项指标：SR, OSR, NE, SPL

# ========== 配置参数 ==========
# 评估结果根目录（修改为你的路径）
ROOT_DIR='/sim/data/TravelUAV_data/eval_closeloop'

# 需要分析的评估目录列表（对应你的输出目录名）
# 例如：eval_qwen, eval_llama, baseline 等
ANALYSIS_LIST="eval_qwen"

# 路径类型分类
# - full:  所有路径
# - easy:  简单路径（路径长度 <= 250）
# - hard:  困难路径（路径长度 > 250）
PATH_TYPE_LIST="full easy hard"

# ========== 执行指标计算 ==========
echo "========================================="
echo "开始计算评估指标..."
echo "根目录: $ROOT_DIR"
echo "分析目录: $ANALYSIS_LIST"
echo "路径类型: $PATH_TYPE_LIST"
echo "========================================="

CUDA_VISIBLE_DEVICES=0 python3 /home/yyx/TravelUAV/utils/metric.py \
    --root_dir "$ROOT_DIR" \
    --analysis_list $ANALYSIS_LIST \
    --path_type_list $PATH_TYPE_LIST

echo "========================================="
echo "指标计算完成！"
echo "========================================="

