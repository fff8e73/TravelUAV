<div align="center">
<h1>Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology</h1>

<image src="./header.png" width="70%">

<a href="https://arxiv.org/abs/2410.07087"><img src='https://img.shields.io/badge/arXiv-TRAVEL: UAV VLN Platform, Benchmark, and Methodology-red' alt='Paper PDF'></a>
<a href='https://prince687028.github.io/Travel/'><img src='https://img.shields.io/badge/Project_Page-TRAVEL-green' alt='Project Page'></a>
<a href='https://huggingface.co/datasets/wangxiangyu0814/TravelUAV'><img src='https://img.shields.io/badge/Dataset-TRAVEL-blue'></a>
<a href='https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env'><img src='https://img.shields.io/badge/Env-TRAVEL-blue'></a>

</div>

## Contents

- [Introduction](#introduction)
- [Qwen3-VL Support](#qwen3-vl-support)
- [HTTP Server & Client Testing](#http-server--client-testing)
- [Dependencies](#dependencies)
- [Preparation](#prepare-the-data)
- [Usage](#usage)
- [Citation](#paper)

## News
- **2026-01-17:** ✅ 运行时警告已修复！C++ ops 和 torch.meshgrid 警告已解决，脚本支持后台运行（SSH断开不会终止）。See [CHANGES.md](./CHANGES.md) for details.
- **2026-01-16:** ✅ Qwen3-VL evaluation completed successfully! 4/1413 episodes processed, 81 log files generated. Blackwell GPU (RTX PRO 6000, sm_120) fully supported with PyTorch 2.7.1+cu128. See [CHANGES.md](./CHANGES.md) for details.
- **2026-01-16:** Added Qwen3-VL-4B-Instruct support with two-stage architecture for Blackwell GPU (RTX PRO 6000, sm_120). See [USAGE.md](./USAGE.md) for details.
- **2026-01-16:** Fixed trajectory model import error in `vis_traj_arch.py`. See [CHANGES.md](./CHANGES.md) for details.
- **2025-05-22:** We release UAV-Flow, the first real-world benchmark for language-conditioned UAV imitation learning. (project page: https://prince687028.github.io/UAV-Flow)
- **2025-01-25:** Paper, project page, code, data, envs and models are all released.

# Introduction

This work presents **_TOWARDS REALISTIC UAV VISION-LANGUAGE NAVIGATION: PLATFORM, BENCHMARK, AND METHODOLOGY_**. We introduce a UAV simulation platform, an assistant-guided realistic UAV VLN benchmark, and an MLLM-based method to address the challenges in realistic UAV vision-language navigation.

# Qwen3-VL Support

This project now supports **Qwen3-VL-4B-Instruct** as an alternative to LLaMA-VID, with optimized performance for Blackwell GPU (RTX PRO 6000, sm_120).

## Key Features

- ✅ **Two-stage architecture**: Qwen3-VL for visual-language understanding + VisionTrajectoryGenerator for trajectory optimization
- ✅ **GPU acceleration**: 2-3x faster inference, 80%+ memory savings
- ✅ **4-bit quantization**: Only ~3GB VRAM required
- ✅ **Blackwell support**: Full support for RTX PRO 6000 (sm_120)

## Quick Start

```bash
# 1. Activate environment
conda activate llamauav_sm_120

# 2. Run GPU test
python3 test_gpu_mode.py

# 3. Run evaluation
bash scripts/eval_qwen.sh
```

## Documentation

| 文档 | 用途 |
|------|------|
| **[USAGE.md](./USAGE.md)** | **使用指南** - 完整的快速开始、评测流程、测试方法 |
| **[CHANGES.md](./CHANGES.md)** | **变更记录** - 所有代码修改、环境变更、Bug修复 |
| **[llamauav_sm_120_install.sh](./llamauav_sm_120_install.sh)** | **环境安装** - llamauav_sm_120 环境一键安装脚本 |
| **[http_server/README.md](./http_server/README.md)** | **HTTP 服务端** - HTTP 服务端/客户端使用和测试指南 |
| **[README.md](./README.md)** | **项目说明** - 项目概述、依赖、准备、使用 |

**快速参考**：
- 运行评测：`bash scripts/eval_qwen.sh` 或 `bash scripts/eval_qwen.sh --background`
- 查看日志：`tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/*.log`
- 停止脚本：`pkill -f eval_qwen.py`

# Dependencies

### Create `llamauav` environment

```bash
conda create -n llamauav python=3.10 -y
conda activate llamauav
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

## Install LLaMA-UAV model

You can follow [LLaMA-UAV](./Model/LLaMA-UAV/README.md#install) to install the llm dependencies.

### Install other dependencies listed in the requirements file

```bash
pip install -r requirement.txt
```

Additionally, to ensure compatibility with the AirSim Python API, apply the fix mentioned in the [AirSim issue](https://github.com/microsoft/AirSim/issues/3333#issuecomment-827894198)

# Preparation

## Data

To prepare the dataset, please follow the instructions provided in the [Dataset Section](./Model/LLaMA-UAV/README.md#dataset) to construct the dataset.

## Model

### GroundingDINO

Download the GroundingDINO model from the link [groundingdino_swint_ogc.pth](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth), and place the file in the directory `src/model_wrapper/utils/GroundingDINO/`.

### LLaMA-UAV

To set up the model, refer to to the detailed [Model Setup](./Model/LLaMA-UAV/README.md).

## Simulator environments

Download the simulator environments for various maps from [here](https://huggingface.co/datasets/wangxiangyu0814/TravelUAV_env).

The file directory of environments is as follows:

```
├── carla_town_envs
│   ├── Town01
│   ├── Town02
│   ├── Town03
│   ├── ...
├── closeloop_envs
│   ├── Engine
│   ├── ModularEuropean
│   ├── ModularEuropean.sh
│   ├── ModularPark
│   ├── ModularPark.sh
│   ├── ...
├── extra_envs
│   ├── BrushifyUrban
│   ├── BrushifyCountryRoads
│   ├── ...
```

# Usage

## 1. HTTP Server & Client Testing (无需拉取 VLA_Habitat 仓库)

TravelUAV 提供了完整的 HTTP 服务端和客户端，可以直接进行端到端测试，无需依赖外部仓库。

### 快速开始

```bash
# 1. 激活环境
conda activate llamauav_sm_120

# 2. 启动服务端（终端 1）
bash http_server/start_server.sh

# 3. 运行测试（终端 2）
python http_server/tests/test_server.py
```

**详细文档**: [http_server/README.md](./http_server/README.md)

### 功能特性

- ✅ **无需 VLA_Habitat**: 独立的 HTTP 服务端和客户端
- ✅ **完整测试套件**: 6 个测试用例覆盖所有功能
- ✅ **兼容 VLA_Habitat 接口**: 可直接替换使用
- ✅ **一键启动**: 简化配置和启动流程

## 2. Simulator Environment Server

Before running the simulations, ensure the AirSim environment server is properly configured.

> Update the env executable paths`env_exec_path_dict` relative to `root_path` in `AirVLNSimulatorServerTool.py`.

```bash
cd airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /path/to/your/envs
```

## 3. Close-Loop Simulation

Once the simulator server is running, you can execute the dagger or evaluation script.

```bash
# Dagger NYC
bash scripts/dagger_NYC.sh
# Eval (LLaMA-VID)
bash scripts/eval.sh
bash scripts/metrics.sh
# Eval (Qwen3-VL - GPU optimized)
bash scripts/eval_qwen.sh
```

# Paper

If you find this project useful, please consider citing: [paper](https://arxiv.org/abs/2410.07087):

```
@misc{wang2024realisticuavvisionlanguagenavigation,
      title={Towards Realistic UAV Vision-Language Navigation: Platform, Benchmark, and Methodology},
      author={Xiangyu Wang and Donglin Yang and Ziqin Wang and Hohin Kwan and Jinyu Chen and Wenjun Wu and Hongsheng Li and Yue Liao and Si Liu},
      year={2024},
      eprint={2410.07087},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2410.07087},
}
```

# Acknowledgement

This repository is partly based on [AirVLN](https://github.com/AirVLN/AirVLN) and [LLaMA-VID](https://github.com/dvlab-research/LLaMA-VID) repositories.
