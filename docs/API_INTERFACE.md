# TravelUAV Benchmark 外部模型接口文档

**版本**: 1.0
**日期**: 2026-02-03
**参考项目**: Isaac-Drone-Navigation-Benchmark

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

**Request**:
```json
{
    "observation": {
        "rgb": [H, W, 3],
        "depth": [H, W],
        "instruction": "Fly to the red ball",
        "current_position": [x, y, z],
        "current_rotation": [[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]],
        "current_orientation": [qw, qx, qy, qz],
        "target_position": [x, y, z],
        "history_positions": [[x1, y1, z1], [x2, y2, z2], ...],
        "initial_rotation": [[...], [...], [...]],
        "initial_position": [x, y, z],
        "assist_notice": "cruise",
        "timestep": 10
    }
}
```

**Request 字段说明**:

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `rgb` | numpy.ndarray | 是 | RGB图像，shape: [H, W, 3]，前视相机 |
| `depth` | numpy.ndarray | 否 | 深度图，shape: [H, W] |
| `instruction` | str | 是 | 自然语言指令 |
| `current_position` | list[float] | 是 | 当前位置 [x, y, z]，世界坐标系 |
| `current_rotation` | list[list[float]] | 是 | 当前旋转矩阵 [3, 3] |
| `current_orientation` | list[float] | 否 | 当前朝向四元数 [qw, qx, qy, qz] |
| `target_position` | list[float] | 是 | 目标位置 [x, y, z]，世界坐标系 |
| `history_positions` | list[list[float]] | 是 | 历史位置序列 |
| `initial_rotation` | list[list[float]] | 是 | 起点旋转矩阵 [3, 3] |
| `initial_position` | list[float] | 是 | 起点位置 [x, y, z] |
| `assist_notice` | str | 否 | 助手提示，如 "cruise", "take off" |
| `timestep` | int | 是 | 当前时间步 |

**Response**:
```json
{
    "waypoints": [[x, y, z]],
    "stop": false
}
```

**Response 字段说明**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `waypoints` | list[list[float]] | 下一个航点列表，通常只包含一个元素 [x, y, z]，世界坐标系 |
| `stop` | bool | 是否到达目标（True表示停止导航） |

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

1. **世界坐标系**: AirSim的全局坐标系，单位：米
2. **局部坐标系**: 相对于起点的坐标系，需要Server端自行转换

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

### 1. 安装依赖

```bash
pip install flask json_numpy requests numpy
```

### 2. 启动Server（外部模型侧）

```bash
cd /home/yyx/TravelUAV
python server/travel_model_server.py --host 0.0.0.0 --port 9009
```

**输出**:
```
⏳ Loading External Model...
✅ Model Loaded Successfully!
🚀 TravelUAV Model Server running on 0.0.0.0:9009
📡 Endpoints:
   - POST /reset       : Reset episode state
   - POST /act         : Single inference
   - POST /act_batch   : Batch inference
   - GET  /health      : Health check
```

### 3. 运行Client（TravelUAV Benchmark侧）

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

## 📂 文件结构

```
TravelUAV/
├── src/
│   ├── model_wrapper/
│   │   └── http_client.py          # HTTP客户端实现
│   └── vlnce_src/
│       ├── eval.py                 # 原始评测脚本（本地模型）
│       └── eval_http.py            # HTTP评测脚本（外部模型）
│
├── server/
│   └── travel_model_server.py      # 外部模型服务器模板
│
└── docs/
    └── API_INTERFACE.md             # 本文档
```

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
    data=json_numpy.dumps({"type": "reset", "env_id": 0}),
    headers={"Content-Type": "application/json"})
print(resp.json())

# 测试act
obs = {
    "rgb": np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
    "instruction": "Test",
    "current_position": [0, 0, 1.5],
    "current_rotation": np.eye(3).tolist(),
    "target_position": [5, 5, 1.5],
    "history_positions": [[0, 0, 1.5]],
    "initial_position": [0, 0, 1.5],
    "initial_rotation": np.eye(3).tolist(),
    "timestep": 0
}

resp = requests.post("http://127.0.0.1:9009/act",
    data=json_numpy.dumps({"observation": obs}),
    headers={"Content-Type": "application/json"})
print(json_numpy.loads(resp.text))
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

### Q2: Server端需要计算rot_to_targets吗？

**A**: 是的，Server端需要自行计算旋转矩阵。参考 `travel_util.py` 中的 `rotation_matrix_from_vector` 函数。

### Q3: 如何处理多视角图像？

**A**: 当前Client只发送前视相机图像。如果需要多视角，可以修改 `http_client.py` 中的 `_extract_observation` 方法。

### Q4: 如何支持批量推理？

**A**: 实现 `/act_batch` 接口，一次处理多个观测。需要修改Client端的 `query_batch` 方法。

---

## 📚 参考资料

- **json_numpy文档**: https://pypi.org/project/json-numpy/
- **Flask文档**: https://flask.palletsprojects.com/
- **AirSim文档**: https://microsoft.github.io/AirSim/

---

## 📝 更新日志

### v1.0 (2026-02-03)
- 初始版本
- 实现基于HTTP的Client-Server接口
- 支持单个推理和批量推理
