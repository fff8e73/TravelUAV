#!/bin/bash
# 独立测试 TravelUAV HTTP 服务端（使用 curl）

set -e

echo "============================================================
TravelUAV HTTP 服务端独立测试
============================================================
"
echo "前提条件："
echo "  1. conda 环境已激活: conda activate llamauav_sm_120"
echo "  2. 服务端已启动: bash scripts/start_traveluav_server.sh"
echo "  3. 服务端监听: http://127.0.0.1:9000"
echo ""

# 检查服务端是否运行
echo "检查服务端状态..."
if ! curl -s http://127.0.0.1:9000/health > /dev/null 2>&1; then
    echo "✗ 服务端未运行或无法访问"
    echo ""
    echo "请先启动服务端："
    echo "  conda activate llamauav_sm_120"
    echo "  bash scripts/start_traveluav_server.sh"
    exit 1
fi

echo "✓ 服务端正在运行"
echo ""

# 测试 1: 健康检查
echo "============================================================"
echo "测试 1: 健康检查"
echo "============================================================"
curl -s http://127.0.0.1:9000/health | python3 -m json.tool
echo ""

# 测试 2: 根端点
echo "============================================================"
echo "测试 2: 根端点"
echo "============================================================"
curl -s http://127.0.0.1:9000/ | python3 -m json.tool
echo ""

# 测试 3: act 端点（简单数据）
echo "============================================================"
echo "测试 3: act 端点（简单数据）"
echo "============================================================"
curl -s -X POST http://127.0.0.1:9000/act \
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
  }' | python3 -m json.tool
echo ""

# 测试 4: act 端点（完整数据）
echo "============================================================"
echo "测试 4: act 端点（完整数据）"
echo "============================================================"
curl -s -X POST http://127.0.0.1:9000/act \
  -H "Content-Type: application/json" \
  -d '{
    "observation": {
      "rgb": [[0, 0, 0], [1, 1, 1], [2, 2, 2]],
      "depth": [[0.5, 0.6], [0.7, 0.8]],
      "gps": [10.5, 20.3],
      "yaw": 1.57,
      "camera_height": 1.5,
      "instruction": "飞到建筑物前的空地",
      "step_id": 5
    }
  }' | python3 -m json.tool
echo ""

# 测试 5: 使用 Python 客户端
echo "============================================================"
echo "测试 5: 使用 Python 客户端"
echo "============================================================"
python3 /home/yyx/TravelUAV/scripts/eval/HTTPTrajectoryClient_TravelUAV.py
echo ""

echo "============================================================
所有测试完成！
============================================================
"
echo "✓ HTTP 服务端工作正常"
echo ""
echo "现在你可以："
echo "  1. 在 VLA_Habitat 项目中使用此服务端"
echo "  2. 复制客户端代码: cp scripts/eval/HTTPTrajectoryClient_TravelUAV.py /path/to/VLA_Habitat/internnav/evaluator/HTTPTrajectoryClient.py"
echo "  3. 在 VLA_Habitat 中调用: client = HTTPTrajectoryClient(server_url='http://127.0.0.1:9000')"
