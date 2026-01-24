# HTTP 服务端快速参考

## 🚀 3 步开始测试

```bash
# 1. 激活环境
conda activate llamauav_sm_120

# 2. 启动服务端（终端 1）
bash http_server/start_server.sh

# 3. 运行测试（终端 2）
python http_server/tests/test_server.py
```

---

## 📋 常用命令

### 启动服务端

| 命令 | 说明 |
|------|------|
| `bash http_server/start_server.sh` | 默认配置启动 (127.0.0.1:9000) |
| `bash http_server/start_server.sh --port 8080` | 使用端口 8080 |
| `bash http_server/start_server.sh --host 0.0.0.0` | 监听所有接口 |
| `bash http_server/start_server.sh --no-4bit` | 禁用 4-bit 量化 |

### 运行测试

| 命令 | 说明 |
|------|------|
| `python http_server/tests/test_server.py` | 完整测试（6 个用例） |
| `python http_server/tests/test_client.py` | 客户端快速测试 |
| `curl http://127.0.0.1:9000/health` | 健康检查 |

### 客户端使用

```python
from http_server.client.client import HTTPTrajectoryClient
import numpy as np

client = HTTPTrajectoryClient(server_url="http://127.0.0.1:9000")

actions = client.query({
    "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
    "gps": np.array([0.0, 0.0]),
    "yaw": 0.0,
    "camera_height": 1.0,
    "instruction": "向前飞行 10 米",
    "step_id": 0
})

print(f"动作: {actions}")  # [1]
client.close()
```

---

## 🔌 API 端点

### GET /health
健康检查
```bash
curl http://127.0.0.1:9000/health
```

### POST /act
生成轨迹动作
```bash
curl -X POST http://127.0.0.1:9000/act \
  -H "Content-Type: application/json" \
  -d '{"observation": {...}}'
```

**请求字段**:
- `rgb`: RGB 图像 (H, W, 3)
- `gps`: GPS 坐标 [x, y]
- `yaw`: 朝向（弧度）
- `camera_height`: 相机高度
- `instruction`: 导航指令
- `step_id`: 当前步数（可选）

**响应**:
```json
{"actions": [1]}  // Habitat 动作 ID
```

---

## 🎯 Habitat 动作 ID

| ID | 动作 | 说明 |
|----|------|------|
| 0 | STOP | 停止 |
| 1 | MOVE_FORWARD | 向前移动 |
| 2 | TURN_LEFT | 左转 |
| 3 | TURN_RIGHT | 右转 |
| 5 | LOOK_DOWN | 向下看 |

---

## 📁 文件结构

```
http_server/
├── server/server.py          # 服务端主程序
├── client/client.py          # 客户端（兼容 VLA_Habitat）
├── tests/test_server.py      # 完整测试
├── tests/test_client.py      # 客户端测试
├── start_server.sh           # 启动脚本
├── README.md                 # 详细文档
└── QUICK_REFERENCE.md        # 本文件（快速参考）
```

---

## 🔧 故障排除

### 端口被占用
```bash
# 查找占用端口的进程
netstat -tuln | grep 9000

# 使用其他端口
bash http_server/start_server.sh --port 8080
```

### 无法连接服务器
1. 检查服务端是否正在运行
2. 检查端口是否正确
3. 检查防火墙设置

### CUDA 内存不足
```bash
# 禁用 4-bit 量化（需要更多显存）
bash http_server/start_server.sh --no-4bit
```

### 模型加载失败
检查模型路径是否正确：
```bash
ls -lh /home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct/
ls -lh /home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/
```

---

## 📚 更多信息

- **详细文档**: [README.md](./README.md)
- **项目主页**: [../README.md](../README.md)
- **使用指南**: [../USAGE.md](../USAGE.md)
