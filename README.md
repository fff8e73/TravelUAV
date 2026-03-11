# TravelUAV Benchmark - 外部模型接口

## Contents

- [简介](#简介)
- [前置要求](#安装依赖)
- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [评估结果](#评估结果)
- [常见问题排查](#常见问题排查)

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

## 安装依赖

```bash
conda create -n openuav python=3.10 -y
conda activate openuav

# 安装 PyTorch（用于基础数据处理）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装接口必需的依赖（不包含模型相关）
pip install -r requirements_interface.txt
```

---

## 快速开始

### 步骤 1: 启动 AirSim 仿真环境

**⚠️ 重要**: 在运行评测之前，必须先启动 AirSim 仿真服务器！

```bash
conda activate openuav
cd ~/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
```

**说明**:
- `--port 30000`: AirSim 服务器端口（默认 30000）
- `--root_path`: 仿真环境数据路径

### 步骤 2: 启动外部模型服务器

在**新的终端**中启动你的模型服务器：

```bash
conda activate openuav
cd ~/TravelUAV
python server/travel_model_server.py --host 0.0.0.0 --port 9010
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
bash scripts/eval_http.sh > eval_http.log
bash scripts/metrics.sh
```

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

---

## 常见问题排查

### 问题 1: AirSim 连接失败

**错误信息**:
```
AssertionError: error port
```

**解决方案**:
1. 确认 AirSim 服务器已启动（步骤 1）
2. 检查端口 30000 是否被占用：`lsof -i :30000`
3. 确认防火墙未阻止端口

### 问题 2: 模型服务器连接失败

**错误信息**:
```
[HttpClient] Query failed for batch 0: Connection refused
```

**解决方案**:
1. 确认模型服务器已启动（步骤 2）
2. 测试连通性：`curl http://127.0.0.1:9010/health`
3. 检查端口 9010 是否被占用：`lsof -i :9010`

### 问题 3: 请求超时

**错误信息**:
```
[HttpClient] Query failed: Timeout
```

**解决方案**:
1. 增加超时时间：`--timeout 600`
2. 优化模型推理速度
3. 检查### 问题 4 GPU 是否可用

: 图像数据传输慢

**解决方案**:
1. 确保 Client 和 Server 在同一台机器上（使用 localhost）
2. 考虑降低图像分辨率
3. 使用批量推理接口（`/act_batch`）


