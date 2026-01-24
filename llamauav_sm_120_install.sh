#!/bin/bash
# ============================================================================
# llamauav_sm_120 环境安装脚本
# ============================================================================
# 适用于 Blackwell GPU (RTX PRO 6000, sm_120)
# PyTorch: 2.7.1+cu128
# CUDA: 12.8
# ============================================================================

set -e  # 遇到错误立即退出

echo "========================================"
echo "llamauav_sm_120 环境安装脚本"
echo "========================================"

# 1. 激活 conda 环境
echo ""
echo "步骤 1: 激活 conda..."
source /home/yyx/miniconda3/etc/profile.d/conda.sh

# 2. 创建或激活环境
echo ""
echo "步骤 2: 激活 llamauav_sm_120 环境..."
conda activate llamauav_sm_120

# 3. 卸载旧版 PyTorch（如果存在）
echo ""
echo "步骤 3: 卸载旧版 PyTorch..."
pip uninstall -y torch torchvision torchaudio

# 4. 安装 PyTorch 2.7.1+cu128（支持 Blackwell sm_120）
echo ""
echo "步骤 4: 安装 PyTorch 2.7.1+cu128..."
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 torchaudio==2.7.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 5. 安装 PyTorch 相关依赖
echo ""
echo "步骤 5: 安装 PyTorch 相关依赖..."
pip install numpy==1.26.4

# 6. 安装 transformers 和相关库
echo ""
echo "步骤 6: 安装 transformers 和相关库..."
pip install transformers==4.57.3
pip install bitsandbytes==0.49.1

# 7. 配置 HuggingFace 镜像（加速下载，避免网络超时）
echo ""
echo "步骤 7: 配置 HuggingFace 镜像..."
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
echo "HuggingFace 镜像已配置: https://hf-mirror.com"

# 8. 安装其他依赖
echo ""
echo "步骤 8: 安装其他依赖..."
pip install Pillow  # PIL 图像处理
pip install einops   # 张量操作
pip install msgpackrpc  # AirSim 通信
pip install backports.ssl_match_hostname  # AirSim 依赖
pip install opencv-python  # OpenCV 图像处理
pip install yacs  # 配置管理
pip install airsim  # AirSim SDK
pip install numba  # JIT 编译
pip install accelerate  # HuggingFace 加速
pip install datasets  # 数据集处理
pip install scikit-learn  # 机器学习工具

# 9. 验证安装
echo ""
echo "步骤 9: 验证安装..."
python3 -c "
import torch
print('PyTorch 版本:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA 版本:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('计算能力:', torch.cuda.get_device_capability(0))
"

# 10. 下载 bert-base-uncased 到本地缓存（避免网络超时）
echo ""
echo "步骤 10: 下载 bert-base-uncased 到本地缓存..."
python3 -c "
import os
from pathlib import Path

bert_cache = Path('/home/yyx/.cache/huggingface/hub/models--bert-base-uncased')
if bert_cache.exists():
    print('✓ bert-base-uncased 本地缓存已存在')
    snapshots = list((bert_cache / 'snapshots').glob('*'))
    if snapshots:
        print(f'  快照数量: {len(snapshots)}')
        for snap in snapshots:
            files = list(snap.glob('*'))
            print(f'  文件数量: {len(files)}')
            for f in files[:10]:  # 只显示前10个文件
                print(f'    - {f.name}')
else:
    print('⚠ bert-base-uncased 本地缓存不存在，开始下载...')
    print('  这可能需要几分钟时间...')
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModel.from_pretrained('bert-base-uncased')
    print('✓ bert-base-uncased 下载完成')
"

# 11. 验证 GroundingDINO C++ 扩展状态
echo ""
echo "步骤 11: 验证 GroundingDINO C++ 扩展状态..."
python3 -c "
try:
    from groundingdino import _C
    print('✓ GroundingDINO C++ 扩展已加载')
except:
    print('⚠ GroundingDINO C++ 扩展未加载，使用纯 PyTorch 实现')
    print('  注意: 这不会影响功能，但可能有轻微性能影响')
"

echo ""
echo "========================================"
echo "✅ 安装完成！"
echo "========================================"
echo ""
echo "重要配置:"
echo "  - PyTorch: 2.7.1+cu128 (支持 Blackwell sm_120)"
echo "  - CUDA: 12.8"
echo "  - HuggingFace 镜像: https://hf-mirror.com"
echo "  - bert-base-uncased: 已下载到本地缓存"
echo ""
echo "已安装的核心依赖:"
echo "  - transformers==4.57.3"
echo "  - bitsandbytes==0.49.1 (4-bit 量化)"
echo "  - msgpackrpc, airsim (AirSim 仿真)"
echo "  - opencv-python (图像处理)"
echo "  - accelerate (HuggingFace 加速)"
echo ""
echo "验证命令:"
echo "  conda activate llamauav_sm_120"
echo "  python3 -c \"import torch; print(f'CUDA 可用: {torch.cuda.is_available()}')\""
echo ""
echo "运行集成测试:"
echo "  python3 run_all_tests.py"
echo "  # 或使用 bash 脚本:"
echo "  bash scripts/run_tests.sh"
echo ""
echo "运行评估:"
echo "  # 1. 启动仿真服务器:"
echo "  cd /home/yyx/TravelUAV/airsim_plugin"
echo "  python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/extracted"
echo ""
echo "  # 2. 运行评估:"
echo "  cd /home/yyx/TravelUAV"
echo "  bash scripts/eval_qwen.sh"
echo ""
echo "运行时警告说明:"
echo "  - C++ ops 警告: 已修复，代码会自动使用纯 PyTorch 实现"
echo "  - torch.meshgrid 警告: 已修复，所有调用已添加 indexing='ij' 参数"
echo "  - HuggingFace 连接超时: 已配置镜像 https://hf-mirror.com"
echo ""
echo "已修复的问题:"
echo "  ✓ GroundingDINO C++ 扩展加载失败 - 自动回退到纯 PyTorch 实现"
echo "  ✓ torch.meshgrid 索引参数缺失 - 添加 indexing='ij' 参数"
echo "  ✓ HuggingFace 网络超时 - 配置镜像加速下载"
echo "  ✓ bert-base-uncased 本地缓存验证 - 避免重复下载"
echo ""
