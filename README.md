# TravelUAV Benchmark - 外部模型接口

## Contents

- [简介](#简介)
- [前置准备](#前置准备)
- [快速开始](#快速开始)
- [详细接口标准](#详细接口标准)
- [完整工作流程](#完整工作流程)
- [评估结果](#评估结果)

---

## 简介

**TravelUAV** 是一个基于 AirSim 的无人机视觉语言导航（VLN）仿真平台。本项目（TravelUAV Benchmark - 外部模型接口）在原版基础上提供了基于 HTTP 的 Client-Server 接口，允许外部研究者使用自己的模型在 TravelUAV Benchmark 上进行评测，而无需修改 Benchmark 代码或了解内部实现细节。

**核心特点**:
- ✅ 完全解耦：Benchmark 和模型独立部署
- ✅ 语言无关：Server 可用任何语言实现
- ✅ 简单易用：只需实现 2 个 HTTP 接口
- ✅ 参考设计：基于 Isaac-Drone-Navigation-Benchmark

**相关链接**:
- 论文: [arXiv:2410.07087](https://arxiv.org/abs/2410.07087)
- 项目主页: [OpenUAV](https://prince687028.github.io/Travel/)
- 数据集: [HuggingFace](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV)
- 仿真环境: [HuggingFace](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env)

---

## 前置准备
### 安装依赖

```bash
conda create -n openuav python=3.10 -y
conda activate openuav

# 安装 PyTorch（用于基础数据处理）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装接口必需的依赖（不包含模型相关）
pip install -r requirements_interface.txt
```

### 数据集下载
```bash
cd /sim/data
mkdir -p TravelUAV_data
export HF_ENDPOINT=https://hf-mirror.com
# 下载数据集分割信息
nohup huggingface-cli download wangxiangyu0814/TravelUAV_data_json --repo-type dataset --local-dir /sim/data/TravelUAV_data/TravelUAV_data_json --force-download --resume-download
# 下载数据集
nohup huggingface-cli download wangxiangyu0814/TravelUAV --repo-type dataset --local-dir /sim/data/TravelUAV_data --force-download --resume-download
```
- 据集分割信息路径：/sim/data/TravelUAV_data/TravelUAV_data_json
- 数据集路径：/sim/data/TravelUAV_data/extracted

### 仿真环境下载
```bash
cd /sim/data/TravelUAV_data/
mkdir -p sim_envs
cd sim_envs
# 后台运行并实时查看日志
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  wangxiangyu0814/TravelUAV_env \
  --repo-type dataset \
  --local-dir /sim/data/TravelUAV_data/sim_envs \
  --force-download \
  --resume-download \
  --include "*" > sim_envs.log 2>&1 &
# 实时查看日志（监控下载进度和错误）
tail -f sim_envs.log

# 解压下载的仿真环境包
7z x carla_town_envs.zip -o./ -aoa
7z x closeloop_envs.zip -o./ -aoa
mkdir -p extra_envs
7z x BattlefieldKitDesert.zip -o./extra_envs/ -aoa
7z x BrushifyCountryRoads.zip -o./extra_envs/ -aoa
7z x BrushifyForestPack.zip -o./extra_envs/ -aoa
7z x BrushifyUrban.zip -o./extra_envs/ -aoa
7z x Japanese_Street.zip -o./extra_envs/ -aoa
7z x London_Street.zip -o./extra_envs/ -aoa
7z x NordicHarbour.zip -o./extra_envs/ -aoa
7z x WesterTown.zip -o./extra_envs/ -aoa
```
---

## 快速开始

### 步骤 1: 启动 AirSim 仿真环境

**⚠️ 重要**: 在运行评测之前，必须先启动 AirSim 仿真服务器！

```bash
conda activate openuav
cd ~/TravelUAV/airsim_plugin
nohup python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs > sim.log 2>&1 &
```

**说明**:
- `--port 30000`: AirSim 服务器端口（默认 30000）
- `--root_path`: 仿真环境数据路径

### 步骤 2: 启动外部模型服务器

在**新的终端**中启动你的模型服务器：

```bash
conda activate openuav
cd ~/TravelUAV
nohup python server/travel_model_server.py --host 0.0.0.0 --port 9010 > server.log 2>&1 &
```

**预期输出**:
```
⏳ Loading External Model...
✅ Model Loaded Successfully!
🚀 TravelUAV Model Server running on 0.0.0.0:9010
📡 Endpoints:
   - POST /reset       : Reset episode state
   - POST /act         : Single inference
   - POST /act_batch   : Batch inference
   - GET  /health      : Health check
```

**测试服务器连通性**:
```bash
curl http://127.0.0.1:9010/health
```

### 步骤 3: 运行 Benchmark 评测

在**第三个终端**中运行评测脚本：

```bash
conda activate openuav
cd ~/TravelUAV
nohup bash scripts/eval_http.sh > eval_http_$(date +%Y%m%d_%H%M%S).log 2>&1 &
bash scripts/metrics.sh
```
---

## 详细接口标准

详细接口标准查看[API_INTERFACE.md](docs/API_INTERFACE.md)

---
## 完整工作流程

```
┌─────────────────────────────────────────────────────────────┐
│  终端1: AirSim仿真环境                                        │
│  $ cd airsim_plugin                                          │
│  $ python AirVLNSimulatorServerTool.py --port 30000 \        │
│      --root_path /sim/data/TravelUAV_data/sim_envs          │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 提供仿真环境
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  终端3: TravelUAV Benchmark (Client)                         │
│  $ python src/vlnce_src/eval_http.py \                      │
│      --server_url http://127.0.0.1:9010                     │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP请求
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  终端2: 外部模型服务器 (Server)                               │
│  $ python server/travel_model_server.py --port 9010         │
└─────────────────────────────────────────────────────────────┘
```

---


## 评估结果

评测完成后，结果会保存在 `args.eval_save_path` 指定的目录中，包括：

- **指标文件**: Success Rate (SR), SPL, Distance to Goal 等
- **轨迹文件**: 每个 episode 的完整轨迹
- **日志文件**: 详细的评测日志

