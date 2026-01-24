# TravelUAV HTTP 服务端

TravelUAV HTTP 服务端提供 HTTP 接口，供客户端调用生成轨迹动作。**无需拉取 VLA_Habitat 仓库**即可进行完整的测试。

---

## 📁 目录结构

```
http_server/
├── server/                 # 服务端代码
│   └── server.py          # HTTP 服务端主程序
├── client/                 # 客户端代码
│   └── client.py          # HTTP 客户端（兼容 VLA_Habitat 接口）
├── tests/                  # 测试脚本
│   ├── test_server.py     # 完整的服务端测试（6 个测试用例）
│   └── test_client.py     # 客户端快速测试
├── docs/                   # 文档
├── start_server.sh         # 服务端启动脚本
└── README.md              # 本文件
```

---

## 🚀 快速开始

### 前提条件

1. **Conda 环境**: 已激活 `llamauav_sm_120`
   ```bash
   conda activate llamauav_sm_120
   ```

2. **模型文件**: 确保模型文件存在
   - Qwen3-VL 模型: `/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct/`
   - 轨迹模型: `/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4/`

3. **依赖安装**: 确保已安装必要依赖
   ```bash
   pip install fastapi uvicorn pillow requests numpy
   ```

---

## 🏗️ 启动服务端

### 方式 1: 使用启动脚本（推荐）

```bash
cd /home/yyx/TravelUAV
bash http_server/start_server.sh
```

**可选参数**:
```bash
# 使用自定义端口
bash http_server/start_server.sh --port 8080

# 监听所有接口
bash http_server/start_server.sh --host 0.0.0.0 --port 80

# 禁用 4-bit 量化
bash http_server/start_server.sh --no-4bit

# 查看帮助
bash http_server/start_server.sh --help
```

### 方式 2: 直接运行 Python

```bash
cd /home/yyx/TravelUAV
python http_server/server/server.py --port 9000
```

### 启动成功标志

服务端启动后会显示:
```
============================================================
加载 TravelUAV 模型...
============================================================
模型路径: /home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct
轨迹模型路径: /home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4
4-bit 量化: True
✓ 模型加载成功
  设备: cuda:0
  显存占用: 3.45 GB
✓ TravelUAV HTTP 服务端初始化完成
  监听地址: 127.0.0.1:9000
============================================================
启动 TravelUAV HTTP 服务器
============================================================
监听地址: 127.0.0.1:9000
端点: POST http://127.0.0.1:9000/act
============================================================
```

---

## 🧪 测试服务端

### 方式 1: 完整测试（推荐）

在**新终端**中运行:

```bash
cd /home/yyx/TravelUAV
python http_server/tests/test_server.py
```

这个脚本会运行 **6 个测试用例**:

| 测试 | 说明 | 端点 |
|------|------|------|
| 1. 健康检查 | 测试 `/health` 端点 | `GET /health` |
| 2. 根端点 | 测试 `/` 端点 | `GET /` |
| 3. act 端点（简单） | 测试简单数据请求 | `POST /act` |
| 4. act 端点（完整） | 测试完整数据请求 | `POST /act` |
| 5. 客户端库 | 测试 `HTTPTrajectoryClient` | - |
| 6. 多次请求 | 测试并发/重复请求 | `POST /act` |

**测试输出示例**:
```
============================================================
TravelUAV HTTP 服务端独立测试
============================================================

前提条件:
  1. conda 环境已激活: conda activate llamauav_sm_120
  2. 服务端已启动: python http_server/server/server.py
  3. 服务端监听: http://127.0.0.1:9000

按 Enter 键开始测试...

============================================================
测试 1: 健康检查
============================================================
✓ 健康检查成功
  状态码: 200
  响应: {'status': 'healthy', 'model_loaded': True, 'device': 'cuda:0'}

============================================================
测试总结
============================================================
✓ 通过: 健康检查
✓ 通过: 根端点
✓ 通过: act 端点（简单）
✓ 通过: act 端点（完整）
✓ 通过: 客户端库
✓ 通过: 多次请求

总计: 6/6 测试通过
============================================================

🎉 所有测试通过！HTTP 服务端工作正常。
```

### 方式 2: 客户端快速测试

```bash
cd /home/yyx/TravelUAV
python http_server/tests/test_client.py
```

### 方式 3: 使用 curl 手动测试

```bash
# 测试健康检查
curl http://127.0.0.1:9000/health

# 测试根端点
curl http://127.0.0.1:9000/

# 测试 act 端点
curl -X POST http://127.0.0.1:9000/act \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "rgb": [[0, 0, 0]],
      "gps": [0.0, 0.0],
      "yaw": 0.0,
      "camera_height": 1.0,
      "instruction": "向前飞行 10 米",
      "step_id": 0
    }
  }'
```

---

## 📡 API 接口说明

### 健康检查

**请求**: `GET /health`

**响应**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda:0"
}
```

### 生成动作

**请求**: `POST /act`

**请求体**:
```json
{
  "observation": {
    "rgb": [[...]],      // (H, W, 3) numpy array 或 list
    "depth": [[...]],    // (H, W) depth image (可选)
    "gps": [x, y],       // GPS 坐标
    "yaw": 0.0,          // 朝向（弧度）
    "camera_height": 1.0,
    "instruction": "文本指令",
    "step_id": 0
  }
}
```

**响应**:
```json
{
  "actions": [1]         // Habitat 动作 ID 列表
}
```

### Habitat 动作 ID 含义

| ID | 动作 | 说明 |
|----|------|------|
| 0 | STOP | 停止 |
| 1 | MOVE_FORWARD | 向前移动 |
| 2 | TURN_LEFT | 左转 |
| 3 | TURN_RIGHT | 右转 |
| 5 | LOOK_DOWN | 向下看 |

---

## 💻 客户端使用

### 基本用法

```python
from http_server.client.client import HTTPTrajectoryClient
import numpy as np

# 创建客户端
client = HTTPTrajectoryClient(
    server_url="http://127.0.0.1:9000",
    timeout=5.0
)

# 准备观察数据
observation = {
    "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    "gps": np.array([0.0, 0.0]),
    "yaw": 0.0,
    "camera_height": 1.0,
    "instruction": "向前飞行 10 米",
    "step_id": 0
}

# 查询动作
actions = client.query(observation)
print(f"动作: {actions}")  # 例如: [1]

# 关闭客户端
client.close()
```

### 兼容 VLA_Habitat 接口

`HTTPTrajectoryClient` 兼容 VLA_Habitat 的 `BaseTrajectoryClient` 接口:

```python
from http_server.client.client import HTTPTrajectoryClient

# 可以直接替换 VLA_Habitat 的 TrajectoryClient
client = HTTPTrajectoryClient(server_url="http://127.0.0.1:9000")

# reset 方法（兼容接口）
client.reset(instruction="向前飞行 10 米")

# query 方法（返回 Habitat 动作 ID）
actions = client.query(observation)
```

---

## 🔍 故障排除

### 问题 1: 无法连接到服务器

**症状**:
```
✗ 无法连接到服务器: http://127.0.0.1:9000
```

**解决方案**:
1. 检查服务端是否正在运行
2. 检查端口是否正确（默认 9000）
3. 检查防火墙设置

### 问题 2: 模型加载失败

**症状**:
```
✗ 模型加载失败: 模型目录不存在
```

**解决方案**:
1. 检查模型路径是否正确
2. 确保模型文件已下载
3. 检查磁盘空间

### 问题 3: 端口被占用

**症状**:
```
错误: 端口 9000 已被占用
```

**解决方案**:
```bash
# 查找占用端口的进程
netstat -tuln | grep 9000

# 或使用其他端口
bash http_server/start_server.sh --port 8080
```

### 问题 4: CUDA 内存不足

**症状**:
```
CUDA out of memory
```

**解决方案**:
1. 禁用 4-bit 量化（需要更多显存）
   ```bash
   bash http_server/start_server.sh --no-4bit
   ```
2. 降低图像分辨率
3. 重启服务端释放显存

---

## 📊 性能测试

### 单次请求延迟

```bash
# 使用 time 命令测试
time curl -X POST http://127.0.0.1:9000/act \
  -H "Content-Type: application/json" \
  -d '{"observation": {...}}'
```

### 并发测试

```bash
# 使用 ab (Apache Bench) 测试并发
ab -n 100 -c 10 -p test_data.json -T application/json \
   http://127.0.0.1:9000/act
```

---

## 🔧 高级配置

### 自定义模型路径

```bash
bash http_server/start_server.sh \
  --model_path /path/to/your/Qwen3-VL \
  --traj_model_path /path/to/your/traj_model
```

### 监听所有接口

```bash
bash http_server/start_server.sh --host 0.0.0.0 --port 80
```

**注意**: 监听所有接口时，请确保防火墙允许外部访问。

### 禁用 4-bit 量化

```bash
bash http_server/start_server.sh --no-4bit
```

**说明**: 禁用 4-bit 量化会增加显存占用，但可能提高推理精度。

---

## 📝 示例脚本

### 示例 1: 简单测试

```python
# test_simple.py
from http_server.client.client import HTTPTrajectoryClient
import numpy as np

client = HTTPTrajectoryClient()

observation = {
    "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    "gps": np.array([0.0, 0.0]),
    "yaw": 0.0,
    "camera_height": 1.0,
    "instruction": "向前飞行 10 米",
    "step_id": 0
}

actions = client.query(observation)
print(f"动作: {actions}")

client.close()
```

### 示例 2: 模拟导航循环

```python
# test_navigation.py
from http_server.client.client import HTTPTrajectoryClient
import numpy as np
import time

client = HTTPTrajectoryClient()

# 模拟导航循环
for step in range(10):
    observation = {
        "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "gps": np.array([float(step * 2), 0.0]),
        "yaw": 0.0,
        "camera_height": 1.0,
        "instruction": "向前飞行 20 米",
        "step_id": step
    }

    actions = client.query(observation)
    print(f"步骤 {step}: 动作 {actions}")

    time.sleep(0.5)

client.close()
```

---

## 📚 相关文档

- [项目 README](../README.md) - 项目整体说明
- [QUICK_START_HTTP.md](../QUICK_START_HTTP.md) - HTTP 快速指南
- [TRAVELUAV_HTTP_SERVER.md](../TRAVELUAV_HTTP_SERVER.md) - HTTP 服务端完整文档

---

## 🎯 总结

✅ **无需拉取 VLA_Habitat 仓库**
✅ **完整的测试套件**
✅ **兼容 VLA_Habitat 接口**
✅ **一键启动服务端**
✅ **详细的测试文档**

通过本 HTTP 服务端，您可以直接在 TravelUAV 项目中进行完整的端到端测试，无需依赖外部仓库。
