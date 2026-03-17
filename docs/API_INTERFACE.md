# TravelUAV Benchmark 外部模型接口文档

**版本**: 2.1
**日期**: 2026-03-10
**参考项目**: Isaac-Drone-Navigation-Benchmark (对齐 A 标准)

---

## 📋 概述

本文档描述了TravelUAV Benchmark与外部模型服务器的HTTP接口规范。该接口允许外部研究者使用自己的模型在TravelUAV Benchmark上进行评测，而无需修改Benchmark代码。

### 架构设计

```
┌─────────────────────────────────────┐
│   TravelUAV Benchmark (Client)      │
│   - 环境管理 (AirSim)                │
│   - 数据加载                         │
│   - 评估指标计算                     │
│   - HttpClient                      │
└──────────────┬──────────────────────┘
               │ HTTP + json_numpy
               │
┌──────────────▼──────────────────────┐
│   External Model Server             │
│   - 模型推理                         │
│   - 数据预处理                       │
│   - 坐标变换                         │
│   - 状态管理                         │
└─────────────────────────────────────┘
```

---

## 🔌 接口规范

### 1. Reset 接口

**用途**: 每个新Episode开始时调用，清空模型的历史状态

**Endpoint**: `POST /reset`

**Request**:
```json
{
    "type": "reset",
    "env_id": 0,
    "episode_id": "batch_0_0"
}
```

**Response**:
```json
{
    "status": "ok"
}
```

**说明**:
- `env_id`: 环境ID（0到batch_size-1）
- `episode_id`: Episode标识符
- Server端应清空该环境的Hidden State、History Buffer等

---

### 2. Act 接口（核心推理）

**用途**: 发送当前观测数据，接收下一个航点和停止信号

**Endpoint**: `POST /act`

**Request** (A 标准 7+1 字段):
```json
{
    "rgb": [
        [[[R, G, B], ...], ...],  // frontcamera, shape: (256, 256, 3)
        [[[R, G, B], ...], ...],  // leftcamera
        [[[R, G, B], ...], ...],  // rightcamera
        [[[R, G, B], ...], ...],  // rearcamera
        [[[R, G, B], ...], ...]   // downcamera
    ],
    "depth": [
        [[depth], ...],  // frontcamera, shape: (256, 256), uint8
        [[depth], ...],  // leftcamera
        [[depth], ...],  // rightcamera
        [[depth], ...],  // rearcamera
        [[depth], ...]   // downcamera
    ],
    "instruction": "Fly to the red ball",
    "step": 10,
    "compass": [0.5],
    "gps": [1.2, 0.8],
    "collision": false,
    "episode_id": "batch_0_0"
}
```

**Request 字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `rgb` | List[np.ndarray] | 是 | 5个相机的RGB图像列表，每个shape: (256, 256, 3)，uint8<br>索引: [0]前视 [1]左视 [2]右视 [3]后视 [4]下视 |
| `depth` | List[np.ndarray] | 否 | 5个相机的深度图列表，每个shape: (256, 256)，uint8<br>索引: [0]前视 [1]左视 [2]右视 [3]后视 [4]下视 |
| `instruction` | str | 是 | 自然语言指令 |
| `step` | int | 是 | 当前步数 |
| `compass` | numpy.ndarray | 否 | 相对朝向，shape: [1]，弧度，从 IMU rotation 提取 yaw |
| `gps` | numpy.ndarray | 否 | 相对起点位置，shape: [2]，投影到起点局部坐标系的 [x, y] |
| `collision` | bool | 否 | 是否碰撞 |
| `episode_id` | str | 否 | Episode标识符，格式: `batch_{index_data}_{env_id}` |

**Response** (A 标准 [N,4] 相对动作):
```json
{
    "action": [
        [0.1, 0.0, 0.0, 0.0],
        [0.1, 0.0, 0.0, 0.0],
        ...
    ]
}
```

**Response 字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `action` | numpy.ndarray | 形状 [N, 4]，每行 [dx, dy, dz, dyaw]，N 由模型决定 |
| `dx` | float | 局部坐标系下 +dx 表示前进方向位移（米） |
| `dy` | float | 局部坐标系下 +dy 表示向左方向位移（米） |
| `dz` | float | 局部坐标系下 +dz 表示向上方向位移（米） |
| `dyaw` | float | 局部坐标系下 +dyaw 表示逆时针旋转增量（弧度） |

**注意**: Server 返回的动作数量 N 是动态的，由模型自行决定。Client 端会根据 Server 返回的实际 N 值从 Buffer 中取出相应数量的动作进行处理。

**停止信号**: 隐式判定，当 `sqrt(dx² + dy² + dz²) < threshold` 时视为停止（默认 threshold = 1e-5）

Client 端会遍历 Server 返回的每一个动作，依次进行坐标转换并生成递进航点序列。遇到第一个满足停止条件的动作时停止遍历，剩余动作保留在 Buffer 中供后续使用。

---

### 3. Health Check 接口（可选）

**用途**: 检查服务器状态

**Endpoint**: `GET /health`

**Response**:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "active_episodes": 3
}
```

---

## 📦 数据格式

### 坐标系说明

**NED 坐标系** (TravelUAV 采用):
- +X: 北 (North)
- +Y: 东 (East)
- +Z: 下 (Down)

**A 标准局部坐标系** (模型输出):
- +dx: 前进方向
- +dy: 左侧方向
- +dz: 上方方向
- +dyaw: 逆时针旋转

**Client 端坐标转换**:
Client 端 (`http_client.py`) 会将 A 标准的局部动作转换为世界坐标航点：
```python
# A 标准 → NED 世界坐标
dx_airsim = dx
dy_airsim = -dy  # A标准+左 → AirSim -右
dz_airsim = -dz  # A标准+上 → AirSim -下

delta_local = [dx_airsim, dy_airsim, dz_airsim]
delta_world = current_rot @ delta_local
waypoint_world = current_pos + delta_world
```

### 坐标变换示例

```python
# 世界坐标 → 局部坐标
initial_pos = np.array(obs["initial_position"])
initial_rot = np.array(obs["initial_rotation"])
current_pos = np.array(obs["current_position"])

relative_pos = initial_rot.T @ (current_pos - initial_pos)

# 局部坐标 → 世界坐标
current_rot = np.array(obs["current_rotation"])
waypoint_local = np.array([dx, dy, dz])

waypoint_world = current_rot @ waypoint_local + current_pos
```

### 图像格式

- **RGB**: numpy.ndarray, shape: [H, W, 3], dtype: uint8, 范围: [0, 255]
- **Depth**: numpy.ndarray, shape: [H, W], dtype: float32, 单位: 米

---

## 🚀 使用方法

### 1. 启动Server（外部模型侧）

```bash
cd /home/yyx/TravelUAV
python server/travel_model_server.py --host 0.0.0.0 --port 9009
```

**输出**:
```
⏳ Loading External Model...
✅ Model Loaded Successfully!
🚀 TravelUAV Model Server (FastAPI) running on 0.0.0.0:9009
📡 Endpoints:
   - POST /reset       : Reset episode state
   - POST /act         : Single inference (A标准 [N,4] 动作)
   - GET  /health      : Health check
```

### 2. 运行Client（TravelUAV Benchmark侧）

```bash
cd /home/yyx/TravelUAV
python src/vlnce_src/eval_http.py \
    --server_url http://127.0.0.1:9009 \
    --timeout 300
```

**参数说明**:
- `--server_url`: 外部模型服务器地址
- `--timeout`: HTTP请求超时时间（秒）

---


## 📊 评估指标

TravelUAV Benchmark会自动计算以下指标：

- **Success Rate (SR)**: 成功到达目标的比例
- **Success weighted by Path Length (SPL)**: 考虑路径长度的成功率
- **Distance to Goal (DTG)**: 最终距离目标的距离
- **Collision Rate**: 碰撞率
- **Navigation Error (NE)**: 导航误差

---

## 🐛 调试技巧

### 1. 测试Server连通性

```bash
curl http://127.0.0.1:9009/health
```

### 2. 手动发送测试请求

```python
import requests
import json_numpy
import numpy as np

json_numpy.patch()

# 测试reset
resp = requests.post("http://127.0.0.1:9009/reset",
    data=json_numpy.dumps({"env_id": 0}),
    headers={"Content-Type": "application/json"})
print(resp.json())

# 测试act (A 标准格式)
obs = {
    "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "depth": np.random.rand(480, 640).astype(np.float32),
    "instruction": "Fly forward",
    "step": 0,
    "compass": np.array([0.0], dtype=np.float32),
    "gps": np.array([0.0, 0.0], dtype=np.float32),
    "collision": False
}

resp = requests.post("http://127.0.0.1:9009/act",
    data=json_numpy.dumps(obs),
    headers={"Content-Type": "application/json"})
print(json_numpy.loads(resp.text))
# 输出: {"action": [[dx, dy, dz, dyaw], ...]}
```

### 3. 查看Client日志

```bash
# Client端会输出详细日志
[HttpClient] Connecting to server: http://127.0.0.1:9009
[HttpClient] Reset env_id=0 successfully
[HttpClient] Query failed for batch 0: Connection refused
```

---

## ⚠️ 常见问题

### Q1: 图像数据太大，传输慢怎么办？

**A**: 可以考虑：
1. 降低图像分辨率
2. 使用JPEG压缩（需修改Client和Server）
3. 使用WebSocket长连接
4. 部署在同一台机器上（localhost）

### Q2: 模型需要 target_position 怎么办？

**A**: A 标准 obs 的 7+1 字段中不包含 `target_position`。Client 端可以在构造 obs 时将 `target_position` 作为额外可选字段附加，Server 端透传给模型。

### Q3: 如何调整 Buffer K 值？

**A**: 在初始化 `HttpClient` 时设置 `k_exec_steps` 参数。初始 K=1（每步都请求），验证正确性后可逐步增大 K 以减少 HTTP 请求次数。

### Q4: Server 返回 N=0 动作怎么办？

**A**: Client 端会抛出 `ValueError: Server returned empty action list (N=0), this is an error`。Server 必须返回至少一个动作（N >= 1）。

---

## 📚 参考资料

- **json_numpy文档**: https://pypi.org/project/json-numpy/
- **FastAPI文档**: https://fastapi.tiangolo.com/
- **uvicorn文档**: https://www.uvicorn.org/
- **AirSim文档**: https://microsoft.github.io/AirSim/

---

## 📝 更新日志

### v2.1 (2026-03-10)
- 移除 `assist_notice` 字段，碰撞检测完全依赖 AirSim 仿真器
- 添加 `episode_id` 字段用于 Episode 标识
- 支持动态动作数量 N（由 Server 返回的 action 数组长度决定）
- 停止判定改为遍历检查每一个动作，遇到停止动作后停止遍历
- Buffer 根据 Server 返回的 N 值动态取出相应数量的动作

### v2.0 (2026-03-04)
- 对齐 Isaac-Drone-Navigation-Benchmark 接口标准
- Act 接口 Request 改为 A 标准 7+1 字段
- Act 接口 Response 改为 `{"action": ndarray[N,4]}` 格式
- 停止信号改为隐式判定（位移 < 阈值）
- 框架从 Flask 迁移到 FastAPI + uvicorn
- 坐标系明确为 NED

### v1.0 (2026-02-03)
- 初始版本
- 实现基于HTTP的Client-Server接口
- 支持单个推理和批量推理
