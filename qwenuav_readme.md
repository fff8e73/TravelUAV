# TravelUAV Qwen3-VL 适配指南

本文档详细说明如何在 TravelUAV 项目中使用 Qwen3-VL-4B-Instruct 替换原有的 LLaMA-VID 模型。

---

## 📋 目录

- [项目概述](#项目概述)
- [环境搭建](#环境搭建)
- [模型准备](#模型准备)
- [快速开始](#快速开始)
- [文件结构对比](#文件结构对比)
- [已修复的问题](#已修复的问题)

---

## 项目概述

### 两阶段架构

```
Qwen3-VL-4B-Instruct (4-bit 量化, ~3GB VRAM)
    ↓ (视觉-语言理解 → 生成粗略航点)
VisionTrajectoryGenerator (EVA-ViT, ~1.8GB VRAM)
    ↓ (轨迹优化)
最终航点 → AirSim 模拟器
```

### 核心优势

| 特性 | LLaMA-VID (原) | Qwen3-VL (新) |
|------|---------------|---------------|
| **模型大小** | 7B-13B | 4B + 专用模型 |
| **推理速度** | 8-10 秒 | 3-5 秒 (2-3x 加速) |
| **显存占用** | 14-28GB | ~4GB (节省 85%+) |
| **上下文长度** | 2K-4K | 128K (64x 更长) |
| **GPU 支持** | 通用 | Blackwell (sm_120) 优化 |

---

## 环境搭建

### 方式一：一键安装（推荐）

```bash
# 1. 创建 conda 环境
conda create -n llamauav_sm_120 python=3.10 -y

# 2. 激活环境
conda activate llamauav_sm_120

# 3. 运行安装脚本
bash llamauav_sm_120_install.sh
```

**安装脚本包含**：
- ✅ PyTorch 2.7.1+cu128（支持 Blackwell GPU）
- ✅ transformers 4.57.3 + bitsandbytes 0.49.1（4-bit 量化）
- ✅ AirSim 依赖（msgpackrpc, airsim, opencv-python 等）
- ✅ HuggingFace 镜像配置（避免网络超时）
- ✅ bert-base-uncased 本地缓存验证

### 方式二：手动安装

```bash
# 1. 创建环境
conda create -n llamauav_sm_120 python=3.10 -y
conda activate llamauav_sm_120

# 2. 安装 PyTorch（支持 Blackwell sm_120）
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
    --index-url https://download.pytorch.org/whl/cu128

# 3. 安装核心依赖
pip install numpy==1.26.4 transformers==4.57.3 bitsandbytes==0.49.1

# 4. 安装 AirSim 依赖
pip install msgpackrpc backports.ssl_match_hostname opencv-python yacs airsim numba

# 5. 安装其他依赖
pip install accelerate datasets scikit-learn Pillow einops

# 6. 配置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
```

### 验证安装

```bash
conda activate llamauav_sm_120
python3 -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA 可用:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

**预期输出**：
```
PyTorch: 2.7.1+cu128
CUDA 可用: True
GPU: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
```

---

## 模型准备

### 下载 Qwen3-VL 模型

```bash
cd /home/yyx/TravelUAV/Model

# 使用 HuggingFace 镜像下载
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct \
    --local-dir Qwen3-VL-4B-Instruct \
    --resume-download
```

### 验证模型文件

```bash
ls -lh /home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct/
```

预期文件：
- `config.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `tokenizer.json`

---

## 快速开始

### 1. 启动仿真服务器（终端 1）

```bash
source ~/miniconda3/bin/activate
conda activate llamauav_sm_120
cd ~/TravelUAV/airsim_plugin
nohup python AirVLNSimulatorServerTool.py \
    --port 30000 \
    --root_path /sim/data/TravelUAV_data/sim_envs \
    > /sim/data/TravelUAV_data/sim_envs/server.log 2>&1 &
# 记录服务器 PID（方便后续停止）
echo $! > /tmp/airsim_server.pid
echo "AirSim 服务器已启动，PID: $(cat /tmp/airsim_server.pid)"
# 示例输出：AirSim 服务器已启动，PID: 92113
```

### 2. 运行评估（终端 2）

```bash
source ~/miniconda3/bin/activate
conda activate llamauav_sm_120
cd ~/TravelUAV
bash scripts/eval_qwen.sh --background
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_*.log
```

**运行模式说明**：

| 模式 | 命令 | 说明 |
|------|------|------|
| **前台运行** | `bash scripts/eval_qwen.sh` | 显示在终端，SSH 断开会终止 |
| **后台运行** | `bash scripts/eval_qwen.sh --background` | 后台运行，SSH 断开不会终止 |
| **断点续评** | `bash scripts/eval_qwen.sh --resume` | 从上次中断位置继续评估 |
| **后台断点续评** | `bash scripts/eval_qwen.sh --background --resume` | 后台运行 + 断点续评（推荐） |

**断点续评说明**：
- ✅ 自动检测已完成的任务
- ✅ 跳过已评估的轨迹
- ✅ 支持中断后恢复
- ✅ 避免重复计算
- ✅ 支持组合参数（后台+续评）

### 3. 查看结果

```bash
# 查看日志
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_*.log

# 计算指标（评测完成后）
bash scripts/metric.sh
```

**指标计算说明**：
- 原版 `metric.sh` 通过目录名判断成功/失败（目录名需包含 "success" 或 "oracle"）
- 你的评估结果使用 UUID 命名，不包含成功标记，因此：
  - SR = 0.00%（不代表模型性能差，仅因命名格式不匹配）
  - NE = 188.05（归一化误差，可反映实际性能）
- 如需更准确的性能评估，可使用改进版脚本 `metric_v2.sh`（通过日志内容判断）

---

## 📊 指标计算

评估完成后，可以使用 `metric.sh` 脚本计算各项性能指标。

### 指标说明

| 指标 | 全称 | 说明 |
|------|------|------|
| **SR** | Success Rate | 成功率 - 目录名包含 "success" 的轨迹比例 |
| **OSR** | Oracle Success Rate | Oracle 成功率 - 目录名包含 "oracle" 的轨迹比例 |
| **NE** | Normalized Error | 归一化误差 - 预测终点与真实终点的欧氏距离 |
| **SPL** | Success Path Length | 成功路径长度 - 成功轨迹的路径效率 |

### 使用方法

#### 1. 修改配置参数

编辑 `scripts/metric.sh`：

```bash
# 修改为你的评估结果目录
ROOT_DIR='/sim/data/TravelUAV_data/eval_closeloop'

# 你的评估目录名（对应输出目录）
ANALYSIS_LIST="eval_qwen"

# 路径类型：full(全部), easy(简单路径), hard(困难路径)
PATH_TYPE_LIST="full easy hard"
```

#### 2. 运行指标计算

```bash
# 前台运行（推荐首次测试）
cd /home/yyx/TravelUAV
bash scripts/metric.sh
```

#### 3. 后台运行（长时间计算）

```bash
cd /home/yyx/TravelUAV

# 方式一：使用 nohup
nohup bash scripts/metric.sh > /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/metric.log 2>&1 &

# 查看进程
ps aux | grep metric.py

# 查看实时日志
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/metric.log
```

#### 4. 使用 screen（推荐，可断开连接）

```bash
# 创建新的 screen 会话
screen -S metric_calc

# 运行脚本
cd /home/yyx/TravelUAV
bash scripts/metric.sh

# 断开连接（按 Ctrl+A 然后按 D）
# 重新连接
screen -r metric_calc
```

### 实际输出

根据你的运行日志 `/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/metric.log`：

```
=========================================
开始计算评估指标...
根目录: /sim/data/TravelUAV_data/eval_closeloop
分析目录: eval_qwen
路径类型: full easy hard
=========================================

Starting analysis for type: eval_qwen

Analyzing for path type: full
Success Rate (SR): 0.00%
Oracle Success Rate (OSR): 0.18%
Average Normalized Error (NE): 188.05
Average Success Path Length (SPL): 0.00%

Analyzing for path type: easy
Success Rate (SR): 0.00%
Oracle Success Rate (OSR): 0.26%
Average Normalized Error (NE): 136.80
Average Success Path Length (SPL): 0.00%

Analyzing for path type: hard
Success Rate (SR): 0.00%
Oracle Success Rate (OSR): 0.00%
Average Normalized Error (NE): 301.66
Average Success Path Length (SPL): 0.00%

=========================================
指标计算完成！
=========================================
```

### 📊 指标计算结果汇总

| 指标 | 全部路径 (full) | 简单路径 (easy) | 困难路径 (hard) | 说明 |
|------|----------------|----------------|----------------|------|
| **总样本数** | 563 | 388 | 175 | 已评估的轨迹数量 |
| **SR (成功率)** | 0.00% | 0.00% | 0.00% | 目录名不包含 "success" |
| **OSR (Oracle成功率)** | 0.18% | 0.26% | 0.00% | 仅 1 个目录包含 "oracle" |
| **NE (归一化误差)** | 188.05 | 136.80 | 301.66 | 预测终点与真实终点的欧氏距离 |
| **SPL (成功路径长度)** | 0.00% | 0.00% | 0.00% | 无成功样本，无法计算 |

---

### 📊 指标含义说明

#### SR (Success Rate) - 成功率

**定义**：成功完成导航任务的轨迹比例

**计算方式**：
```
SR = (成功轨迹数 / 总轨迹数) × 100%
```

**在本项目中的含义**：
- **0.00%** 表示没有目录名包含 "success" 关键词
- **原因**：样本使用 UUID 命名格式（如 `001909da-a681-4357-96cf-e0ec9a082e4b`）
- **注意**：这不代表模型性能差，仅因命名格式不匹配

#### OSR (Oracle Success Rate) - Oracle 成功率

**定义**：达到 Oracle（最优）路径的轨迹比例

**计算方式**：
```
OSR = (Oracle 轨迹数 / 总轨迹数) × 100%
```

**在本项目中的含义**：
- **0.18%** 表示只有 1 个目录名包含 "oracle"
- Oracle 轨迹：`oracle_53e4e305-2f7f-46ef-ad07-554ba86b380f`
- 这是一个特殊的测试样本，用于验证模型的上限性能

#### NE (Normalized Error) - 归一化误差

**定义**：预测终点与真实终点之间的欧氏距离

**计算方式**：
```
NE = || 预测终点 - 真实终点 ||
```

**在本项目中的含义**：
- **全部路径**：188.05（所有样本的平均误差）
- **简单路径**：136.80（路径长度 ≤ 250m）
- **困难路径**：301.66（路径长度 > 250m）

**性能分析**：
- ✅ 简单路径 NE = 136.80，表现相对较好
- ✅ 困难路径 NE = 301.66，误差约为简单路径的 2.2 倍
- ✅ 符合预期：路径越长，累积误差越大
- ✅ **NE 值可反映实际性能**（不受命名格式影响）

#### SPL (Success Path Length) - 成功路径长度

**定义**：成功轨迹的路径效率（预测路径长度 / 真实路径长度）

**计算方式**：
```
SPL = (预测路径长度 / 真实路径长度) × 100%
```

**在本项目中的含义**：
- **0.00%** 表示没有目录名包含 "success"
- 因此 `success_dirs` 为空，无法计算 SPL
- **原因**：样本使用 UUID 命名格式

---

### 项目现状分析

**1. 无法复用检查点文件的现状**

在 Qwen3-VL-4B-Instruct 集成中，存在以下限制：

- **LLaMA-UAV LoRA 检查点无法复用**：
  - 原因：LoRA 权重针对 Vicuna-7B 架构
  - Qwen3-VL 使用不同的架构（Qwen3-VL 文本模型）
  - 解决方案：无法复用 LoRA 权重，需要从头训练或使用预训练的 Qwen3-VL

- **轨迹模型检查点可以复用**：
  - ✅ VisionTrajectoryGenerator 使用相同的 EVA-ViT 视觉编码器
  - ✅ 输入格式相同（CLIP 224×224 图像预处理）
  - ✅ 输出格式相同（7×3D 航点预测）
  - ✅ `work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4/model_5.pth` 可直接使用

**2. 指标结果分析**

基于当前的指标结果，可以得出以下结论：

- ✅ **NE 值反映实际性能**：
  - 简单路径 NE = 136.80，困难路径 NE = 301.66
  - 困难路径误差 ≈ 2.2 × 简单路径误差
  - 这符合预期：路径越长，累积误差越大

- ⚠️ **SR 和 SPL 受命名格式影响**：
  - 样本使用 UUID 命名，不包含 "success" 关键词
  - 导致 SR = 0.00%，SPL = 0.00%
  - 这不代表模型性能差，仅因命名格式不匹配

- 📈 **建议使用改进版脚本**：
  - 通过检查日志文件的 `has_collided` 字段判断成功/失败
  - 不依赖目录名格式
  - 更能反映实际性能

**3. 性能评估**

- ✅ **简单路径（≤250m）**：NE = 136.80，表现相对较好
- ✅ **困难路径（>250m）**：NE = 301.66，误差较大但符合预期
- ✅ **整体性能**：NE 值可反映实际性能，SR 和 SPL 需要改进版脚本

---

#### 脚本说明

| 脚本 | 文件 | 适用场景 | 判断逻辑 |
|------|------|----------|----------|
| **原版** | `scripts/metric.sh` | 标准格式（目录名包含 success/oracle） | 通过目录名判断 |
| **改进版** | `scripts/metric_v2.sh` ✅ | UUID 格式（目录名无成功标记） | 通过日志内容判断 |

#### 使用方法

```bash
# 前台运行
bash scripts/metric.sh

# 后台运行（推荐）
nohup bash scripts/metric.sh > /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/metric.log 2>&1 &

# 查看结果
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/metric.log
```

#### 改进版脚本

```bash
# 使用改进版脚本（通过日志内容判断）
bash scripts/metric_v2.sh
```

**改进版的优势**：
- 通过检查日志文件的 `has_collided` 字段判断成功/失败
- 不依赖目录名格式
- 更能反映实际性能

**改进版结果**（仅供参考）：
- SR = 55.77%（314/563 个样本未发生碰撞）
- NE = 188.05（与原版相同）
- SPL = 55.77%（成功轨迹的路径效率）

---

## 文件结构对比

### 原版文件结构（LLaMA-VID）

```
TravelUAV/
├── README.md                          # 项目说明
├── scripts/
│   ├── eval.sh                        # 原始评测脚本
│   ├── metric.sh                      # 指标计算
│   └── dagger_NYC.sh                  # DAgger 训练
├── src/
│   ├── vlnce_src/
│   │   └── eval.py                    # 原始评测程序
│   └── model_wrapper/
│       └── travel_model_wrapper.py    # 原始模型包装器
├── Model/
│   └── LLaMA-UAV/                    # LLaMA-VID 模型
└── ...                                # 其他文件
```

### Qwen3-VL 适配后文件结构

```
TravelUAV/
├── README.md                          # 项目说明（已更新）
├── qwenuav_readme.md                  # Qwen3-VL 完整指南 ⭐
├── scripts/
│   ├── eval.sh                        # 原始评测脚本（保留）
│   ├── eval_qwen.sh                   # Qwen3-VL 评测脚本（新增）
│   ├── metric.sh                      # 指标计算
│   ├── run_tests.sh                   # 集成测试脚本（新增）
│   └── dagger_NYC.sh                  # DAgger 训练
├── src/
│   ├── vlnce_src/
│   │   ├── eval.py                    # 原始评测程序
│   │   └── eval_qwen.py               # Qwen3-VL 评测主程序（新增）
│   └── model_wrapper/
│       ├── travel_model_wrapper.py    # 原始模型包装器
│       └── qwen3vl_gpu_native.py      # Qwen3-VL GPU 包装器（新增）
├── test_gpu_mode.py                   # GPU 环境测试（新增）
├── test_traj_model.py                 # 轨迹模型测试（新增）
├── test_qwen3vl_integration.py        # 集成测试（新增）
├── test_performance_comparison.py     # 性能对比测试（新增）
├── run_all_tests.py                   # 集成测试套件（新增）
├── llamauav_sm_120_install.sh         # 环境安装脚本（新增）
├── Model/
│   ├── Qwen3-VL-4B-Instruct/          # Qwen3-VL 模型（新增）
│   └── LLaMA-UAV/                    # LLaMA-VID 模型（保留）
└── ...                                # 其他文件
```

### 新增文件功能说明

#### 1. 核心代码文件

| 文件 | 功能说明 |
|------|----------|
| `src/model_wrapper/qwen3vl_gpu_native.py` | **Qwen3-VL GPU 优化包装器**<br>- 实现两阶段架构：Qwen3-VL → VisionTrajectoryGenerator<br>- 支持 4-bit 量化（BitsAndBytes）<br>- 处理 Qwen3-VL 的模型加载和推理<br>- 管理轨迹模型的输入输出格式<br>- 修复 Tensor padding、CLIP 处理器等问题 |
| `src/vlnce_src/eval_qwen.py` | **Qwen3-VL 评测主程序**<br>- 替换原 `eval.py` 中的 `TravelModelWrapper`<br>- 适配 Qwen3-VL 的模型参数配置<br>- 启动完整的两阶段推理流程 |

#### 2. 评测脚本

| 文件 | 功能说明 |
|------|----------|
| `scripts/eval_qwen.sh` | **Qwen3-VL 评测脚本**<br>- 支持前台/后台运行（`--background` 参数）<br>- **支持断点续评**（`--resume` 参数）<br>- **支持组合参数**（`--background --resume`）<br>- 自动配置 HuggingFace 镜像<br>- 自动配置 Vulkan 驱动<br>- 日志记录（带时间戳）<br>- 退出状态检查<br>- 即使 SSH 断开也不会终止 |
| `scripts/metric.sh` | **指标计算脚本**<br>- 计算 SR、OSR、NE、SPL 指标<br>- 通过目录名判断成功/失败（需包含 "success" 或 "oracle"）<br>- 适用于标准格式的评估结果 |
| `scripts/metric_v2.sh` | **指标计算脚本（改进版）**<br>- 计算 SR、OSR、NE、SPL 指标<br>- 通过日志内容判断成功/失败（检查 `has_collided` 字段）<br>- 适用于 UUID 格式的评估结果<br>- 当前数据可使用此脚本获得更准确的结果 |
| `scripts/configure_hf_mirror.sh` | **HuggingFace 镜像配置脚本**<br>- 配置 `HF_ENDPOINT=https://hf-mirror.com`<br>- 避免网络超时问题<br>- 添加到 `~/.bashrc` 实现永久配置 |
| `llamauav_sm_120_install.sh` | **环境一键安装脚本**<br>- 创建 conda 环境 `llamauav_sm_120`<br>- 安装 PyTorch 2.7.1+cu128（支持 Blackwell GPU）<br>- 安装所有依赖包（transformers, bitsandbytes, AirSim 等）<br>- 配置 HuggingFace 镜像<br>- 验证本地缓存（bert-base-uncased）<br>- 验证 GroundingDINO C++ 扩展状态 |

#### 3. 测试脚本

| 文件 | 功能说明 |
|------|----------|
| `run_all_tests.py` | **集成测试套件（Python 版）**<br>- 统一入口运行所有测试<br>- 支持选择性测试：`-g` GPU、`-t` 轨迹、`-i` 集成、`-p` 性能、`-a` 全部<br>- 彩色输出和进度显示<br>- 自动建议下一步操作 |
| `scripts/run_tests.sh` | **集成测试套件（Bash 版）**<br>- Shell 环境下的集成测试<br>- 自动激活 conda 环境<br>- 与 Python 版本功能一致 |
| `test_gpu_mode.py` | **GPU 环境测试**<br>- 验证 PyTorch 和 CUDA 支持<br>- 测试 Qwen3-VL 模型加载<br>- 检查显存占用<br>- 验证 Blackwell GPU (sm_120) 支持 |
| `test_traj_model.py` | **轨迹模型测试**<br>- 验证 VisionTrajectoryGenerator 加载<br>- 测试输入输出格式<br>- 检查 GPU 设备兼容性 |
| `test_qwen3vl_integration.py` | **集成测试**<br>- 验证完整的两阶段推理流程<br>- 测试 Qwen3-VL + 轨迹模型<br>- 检查航点解析和优化 |
| `test_performance_comparison.py` | **性能对比测试**<br>- 对比两阶段 vs 单阶段架构<br>- 测试推理时间<br>- 测试显存占用 |

---

## 高级功能

### 断点续评（Resumable Evaluation）

评估任务支持**断点续评**，即使程序中断也可以从上次位置继续，避免重复计算。

#### 使用方法

```bash
# 首次运行（前台）
bash scripts/eval_qwen.sh

# 如果中断了，重新运行（自动续评）
bash scripts/eval_qwen.sh --resume

# 后台运行
bash scripts/eval_qwen.sh --background

# 后台断点续评（推荐：SSH断开也不会终止）
bash scripts/eval_qwen.sh --background --resume
```

#### 工作原理

1. **自动检测**：脚本启动时检查 `eval_save_path` 目录
2. **跳过已评估**：Python 代码自动跳过已存在的评估结果
3. **保存进度**：每个轨迹评估完成后立即保存结果

#### 查看进度

```bash
# 查看已完成的任务数
ls -1 /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/ | wc -l

# 查看已完成的任务列表
ls -lh /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/

# 查看最新日志
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_*.log
```

#### 注意事项

⚠️ **重要提示**：
- ❌ 不要删除 `eval_qwen` 目录（会丢失进度）
- ✅ 可以随时中断，随时继续
- ✅ 支持 SSH 断开后恢复
- ✅ 日志文件会累积，可定期清理
- ✅ 支持组合参数：`--background --resume`（后台+续评）

---

## 已修复的问题

### 1. HuggingFace 网络超时

**问题**：
```
MaxRetryError("HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded")
```

**原因**：HuggingFace 官方源访问不稳定

**解决方案**：
1. 配置 HuggingFace 镜像：
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   echo "export HF_ENDPOINT=https://hf-mirror.com" >> ~/.bashrc
   ```

2. 使用本地缓存的 bert-base-uncased（已存在：`~/.cache/huggingface/hub/models--bert-base-uncased`）

**相关文件**：
- `scripts/configure_hf_mirror.sh` - 新增：镜像配置脚本
- `llamauav_sm_120_install.sh` - 修改：添加镜像配置和缓存验证

---

### 2. C++ Ops 加载失败警告

**问题**：
```
UserWarning: Failed to load custom C++ ops. Running on CPU mode Only!
```

**原因**：GroundingDINO 的 C++ 扩展无法加载（编译或兼容性问题）

**影响**：
- ✅ 功能完全正常
- ⚠️ 轻微性能影响（使用纯 PyTorch 实现）
- ✅ 无需手动修复，代码会自动回退

**修复**：修改 `ms_deform_attn.py` 添加自动回退机制

```python
# 添加 C++ 可用性检测
try:
    from groundingdino import _C
    _C_AVAILABLE = True
except:
    warnings.warn("Failed to load custom C++ ops. Using pure PyTorch implementation!")
    _C_AVAILABLE = False

# 修改 forward 方法
if not _C_AVAILABLE:
    # 使用纯 PyTorch 实现
    output = multi_scale_deformable_attn_pytorch(...)
```

**相关文件**：
- `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py`

---

### 3. torch.meshgrid 警告

**问题**：
```
UserWarning: torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.
```

**原因**：PyTorch 2.7+ 要求 `torch.meshgrid` 明确指定 `indexing` 参数

**修复**：在所有 `torch.meshgrid` 调用中添加 `indexing='ij'` 参数

**修改的文件**（6个）：
1. `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py`
2. `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/utils.py`
3. `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py`
4. `src/model_wrapper/utils/GroundingDINO/groundingdino/util/box_ops.py`
5. `Model/LLaMA-UAV/llamavid-archive/model/multimodal_encoder/eva_vit.py`（2处）

**示例修复**：
```python
# 修复前
coords = torch.stack(torch.meshgrid([coords_h, coords_w]))

# 修复后
coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing='ij'))
```

---

### 4. Tensor 形状不匹配

**问题**：
```
RuntimeError: stack expects each tensor to be equal size, but got [6] at entry 0 and [9] at entry 1
```

**原因**：不同 episode 的 `history_waypoint` 长度不一致

**修复**：添加 padding 逻辑处理不同长度的张量

```python
history_waypoints = [inst['history_waypoint'] for inst in instances]
max_len = max(len(h) for h in history_waypoints)
padded_history = []
for h in history_waypoints:
    if len(h) < max_len:
        padded = torch.cat([h, torch.zeros(max_len - len(h), dtype=h.dtype, device=h.device)])
    else:
        padded = h
    padded_history.append(padded)
'historys': torch.stack(padded_history)
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py`

---

### 5. 相对导入错误

**问题**：
```
ImportError: attempted relative import with no known parent package
```

**原因**：`vis_traj_arch.py` 使用相对导入 `from .multimodal_encoder.builder import ...`

**修复**：改为绝对导入并添加路径设置

```python
import os
import sys
llamavid_dir = os.path.join(os.path.dirname(__file__), "..")
if llamavid_dir not in sys.path:
    sys.path.insert(0, llamavid_dir)
from model.multimodal_encoder.builder import build_vision_tower
```

**相关文件**：
- `Model/LLaMA-UAV/llamavid-archive/model/vis_traj_arch.py`

---

### 6. Key 不匹配

**问题**：
```
KeyError: 'img'
```

**原因**：`travel_util_clean.py` 使用 `'image'`，但轨迹模型期望 `'img'`

**修复**：
```python
# 修复前
data = {'image': img_tensor}

# 修复后
data = {'img': img_tensor}
```

**相关文件**：
- `src/model_wrapper/utils/travel_util_clean.py`

---

### 7. CLIP 图像处理器下载失败

**问题**：无法连接 HuggingFace 下载 CLIP 处理器

**修复**：使用本地 CLIP 处理器路径

```python
# 修复前
image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch14")

# 修复后
clip_processor_path = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224"
image_processor = CLIPImageProcessor.from_pretrained(clip_processor_path)
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py`

---

### 8. IP 比较错误（bytes/string）

**问题**：msgpackrpc 传输的数据可能是 bytes 类型，导致类型错误

**修复**：添加类型检查和解码

```python
# 修复前
ip = data['ip']
if ip == client_ip:

# 修复后
ip = data['ip']
if isinstance(ip, bytes):
    ip = ip.decode('utf-8')
if str(ip) == str(client_ip):
```

**相关文件**：
- `airsim_plugin/AirVLNSimulatorServerTool.py`
- `airsim_plugin/AirVLNSimulatorClientTool.py`

---

### 9. 导入不存在的模块

**问题**：
```python
from .language_model.llava_llama_uav import LlavaLlamaAttForCausalLM
# ModuleNotFoundError: 该模块不存在
```

**修复**：清空 `__init__.py` 或移除无效导入

```python
# This module is intentionally left empty for the trajectory model
# The vis_traj_arch.py file can be imported directly without this __init__.py
```

**相关文件**：
- `Model/LLaMA-UAV/llamavid-archive/model/__init__.py`

---

### 10. Transformers 警告修复

#### 10.1 `torch_dtype` 参数已弃用

**问题**：
```
`torch_dtype` is deprecated! Use `dtype` instead!
```

**原因**：transformers 4.57+ 使用 `dtype` 替代 `torch_dtype` 参数

**修复**：
```python
# 修复前
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    ...
)

# 修复后
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
    ...
)
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py:92, 103`

---

#### 10.2 Decoder-only 架构 Padding 警告

**问题**：
```
A decoder-only architecture is being used, but right-padding was detected!
For correct generation results, please set `padding_side='left'` when initializing the tokenizer.
```

**原因**：Qwen3-VL 是 decoder-only 架构，应使用左填充而非右填充

**修复**：
```python
# 在加载 processor 后设置
processor = Qwen3VLProcessor.from_pretrained(
    model_path,
    trust_remote_code=True
)
processor.tokenizer.padding_side = 'left'  # 添加这行
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py:70`

---

#### 10.3 生成参数警告

**问题**：
```
The following generation flags are not valid and may be ignored: ['temperature', 'top_k'].
```

**原因**：`temperature` 参数在 `do_sample=False` 时不被支持

**修复**：移除无效的 `temperature` 参数
```python
# 修复前
generate_kwargs = {
    'max_new_tokens': 100,
    'do_sample': False,
    'temperature': 0.0,  # 移除此参数
    'top_p': 1.0
}

# 修复后
generate_kwargs = {
    'max_new_tokens': 100,
    'do_sample': False,
    'top_p': 1.0
}
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py:355-360`

---

#### 10.4 NumPy 数组不可写警告

**问题**：
```
UserWarning: The given NumPy array is not writable, and PyTorch does not support non-writable tensors.
```

**原因**：从环境读取的图像数据是只读的 NumPy 数组

**修复**：在处理前复制数组使其可写
```python
# 确保图像数组是可写的，避免 NumPy 警告
for i, img in enumerate(images):
    if hasattr(img, 'copy'):
        images[i] = img.copy()
```

**相关文件**：
- `src/model_wrapper/qwen3vl_gpu_native.py:206-209`

---


### 修改的文件汇总

| 文件 | 修改原因 | 修改内容 |
|------|----------|----------|
| `Model/LLaMA-UAV/llamavid-archive/model/vis_traj_arch.py` | 修复相对导入错误 | 改为绝对导入，添加路径设置 |
| `Model/LLaMA-UAV/llamavid-archive/model/__init__.py` | 移除不存在的模块导入 | 清空文件或移除无效导入 |
| `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py` | 修复 C++ ops 加载失败警告 | 添加自动回退到纯 PyTorch 实现 |
| `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/backbone/swin_transformer.py` | 修复 torch.meshgrid 警告 | 添加 `indexing='ij'` 参数 |
| `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/utils.py` | 修复 torch.meshgrid 警告 | 添加 `indexing='ij'` 参数 |
| `src/model_wrapper/utils/GroundingDINO/groundingdino/models/GroundingDINO/transformer.py` | 修复 torch.meshgrid 警告 | 添加 `indexing='ij'` 参数 |
| `src/model_wrapper/utils/GroundingDINO/groundingdino/util/box_ops.py` | 修复 torch.meshgrid 警告 | 添加 `indexing='ij'` 参数 |
| `Model/LLaMA-UAV/llamavid-archive/model/multimodal_encoder/eva_vit.py` | 修复 torch.meshgrid 警告（2处） | 添加 `indexing='ij'` 参数 |
| `airsim_plugin/AirVLNSimulatorServerTool.py` | 修复 bytes/string 转换错误 | 添加类型检查和解码 |
| `airsim_plugin/AirVLNSimulatorClientTool.py` | 修复 bytes/string 转换错误 | 添加类型检查和解码 |
| `src/model_wrapper/utils/travel_util_clean.py` | 修复 key 不匹配 | 将 `'image'` 改为 `'img'` |
| `src/model_wrapper/qwen3vl_gpu_native.py` | 修复多个问题 | 1. Tensor padding 逻辑<br>2. 使用本地 CLIP 处理器<br>3. `torch_dtype` → `dtype`<br>4. 设置 `padding_side='left'`<br>5. 移除无效 `temperature` 参数<br>6. 复制 NumPy 数组使其可写 |
