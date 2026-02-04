# TravelUAV Benchmark 外部模型接口 - 快速开始

> 🚁 为外部模型提供标准化的无人机视觉导航评测接口

---

## 📖 简介

TravelUAV Benchmark 提供了一个基于HTTP的Client-Server接口，允许外部研究者使用自己的模型在TravelUAV上进行评测，而无需修改Benchmark代码或了解内部实现细节。

**核心特点**:
- ✅ 完全解耦：Benchmark和模型独立部署
- ✅ 语言无关：Server可用任何语言实现
- ✅ 简单易用：只需实现2个HTTP接口
- ✅ 参考设计：基于Isaac-Drone-Navigation-Benchmark

---

## 🚀 快速开始

### 前置要求

- Python 3.8+
- AirSim仿真环境
- 必需的Python包：`fastapi`, `json_numpy`, `requests`, `numpy`

### 步骤1: 安装依赖

```bash
conda create -n openuav python=3.10 -y
conda activate openuav

# 安装PyTorch（用于基础数据处理）
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128

# 安装接口必需的依赖（不包含模型相关）
pip install -r requirements_interface.txt
```

**依赖说明**:
- `fastapi`: 现代化的HTTP服务器框架（Server端）
- `uvicorn`: ASGI服务器
- `json_numpy`: numpy数组的JSON序列化
- `requests`: HTTP客户端（Client端）
- `numpy`, `opencv-python`, `scipy`, `pandas`: 数据处理
- `airsim`: AirSim Python API
- `tqdm`, `pyyaml`, `pillow`: 工具库

### 步骤2: 启动AirSim仿真环境

**⚠️ 重要**: 在运行评测之前，必须先启动AirSim仿真服务器！

```bash
cd ~/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
```

**说明**:
- `--port 30000`: AirSim服务器端口（默认30000）
- `--root_path`: 仿真环境数据路径

**预期输出**:
```
[AirSim] Starting simulator server on port 30000...
[AirSim] Loading environments from /sim/data/TravelUAV_data/sim_envs
[AirSim] Server ready!
```

### 步骤3: 启动外部模型服务器

在**新的终端**中启动你的模型服务器：

```bash
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

### 步骤4: 运行Benchmark评测

在**第三个终端**中运行评测脚本：

**使用便捷脚本**

```bash
cd ~/TravelUAV
bash scripts/eval_http.sh
```

**预期输出**:
```
🔗 [HttpClient] Connecting to server: http://127.0.0.1:9010
🔧 Initializing evaluation environment...
🌐 Initializing HTTP Client: http://127.0.0.1:9010
🤖 Assist setting: always_help=False, use_gt=False
🚀 Starting HTTP Client Evaluation (Total: 100 episodes)
[HttpClient] Reset env_id=0 successfully
Step: 0 	 Completed: 0 / 100
...
✅ All episodes completed!
🎉 Evaluation completed successfully!
```

---

## 📂 完整工作流程

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

## 📊 评估结果

评测完成后，结果会保存在 `args.eval_save_path` 指定的目录中，包括：

- **指标文件**: Success Rate (SR), SPL, Distance to Goal等
- **轨迹文件**: 每个episode的完整轨迹
- **日志文件**: 详细的评测日志

---

## 🐛 常见问题排查

### 问题1: AirSim连接失败

**错误信息**:
```
AssertionError: error port
```

**解决方案**:
1. 确认AirSim服务器已启动（步骤2）
2. 检查端口30000是否被占用：`lsof -i :30000`
3. 确认防火墙未阻止端口

### 问题2: 模型服务器连接失败

**错误信息**:
```
[HttpClient] Query failed for batch 0: Connection refused
```

**解决方案**:
1. 确认模型服务器已启动（步骤3）
2. 测试连通性：`curl http://127.0.0.1:9010/health`
3. 检查端口9010是否被占用：`lsof -i :9010`

### 问题3: 请求超时

**错误信息**:
```
[HttpClient] Query failed: Timeout
```

**解决方案**:
1. 增加超时时间：`--timeout 600`
2. 优化模型推理速度
3. 检查GPU是否可用

### 问题4: 图像数据传输慢

**解决方案**:
1. 确保Client和Server在同一台机器上（使用localhost）
2. 考虑降低图像分辨率
3. 使用批量推理接口（`/act_batch`）

---
## 🔄 接口更新计划（2026-02-04）

### 更新内容

#### 1. **动作格式变更**
参考 Isaac-Drone-Navigation-Benchmark，接口返回格式改为相对位移：

**新格式**:
```python
{
    "action": [[dx1, dy1, dz1, dyaw1],
               [dx2, dy2, dz2, dyaw2],
               ...
               [dxN, dyN, dzN, dyawN]]  # shape: [N, 4]
}
```

**坐标系定义**:
- `dx`: 前后位移 (+前, -后)
- `dy`: 左右位移 (+左, -右)
- `dz`: 垂直位移 (+上, -下)
- `dyaw`: 偏航角增量 (弧度，逆时针为正)

**说明**:
- 返回 `[N, 4]` 的 numpy.array，N 是步数
- 每一步的 action 是**相对于上一步**的相对位置
- Client 负责将相对位移转换为世界坐标航点

#### 2. **框架迁移**
- 从 Flask 迁移到 FastAPI
- 提升性能和异步支持
- 更好的类型检查和文档生成

