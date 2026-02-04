"""
TravelUAV Model Server
外部模型服务器端实现 - 基于Flask
"""
from flask import Flask, request, jsonify
import json_numpy
import numpy as np
import torch
from typing import Dict, Any

# 开启 numpy 序列化支持
json_numpy.patch()

app = Flask(__name__)

# ============================================================
# 全局变量：存储每个环境的状态
# ============================================================
# 用于存储每个 env_id 的历史状态（如果模型需要维护状态）
episode_states = {}

# ============================================================
# [MODEL INIT] 在这里加载你的模型
# ============================================================
print("⏳ Loading External Model...")

# 示例：加载你的自定义模型
# from your_model import YourNavigationModel
# model = YourNavigationModel.from_pretrained("path/to/checkpoint")
# model.eval()
# model.to("cuda")
# tokenizer = ...
# image_processor = ...

print("✅ Model Loaded Successfully!")

# ============================================================
# [RESET] 重置接口
# ============================================================
@app.route("/reset", methods=["POST"])
def reset():
    """
    重置接口：当新的 Episode 开始时调用
    作用：清空该环境的历史状态（Hidden State、History Buffer等）
    """
    req = json_numpy.loads(request.data)
    env_id = req.get("env_id", 0)
    episode_id = req.get("episode_id", "unknown")

    # 清空该环境的状态
    if env_id in episode_states:
        del episode_states[env_id]

    # 初始化新的状态
    episode_states[env_id] = {
        "episode_id": episode_id,
        "history": [],
        "step_count": 0
    }

    print(f"[Server] Reset env_id={env_id}, episode_id={episode_id}")

    return jsonify({"status": "ok"})

# ============================================================
# [ACT] 核心推理接口
# ============================================================
@app.route("/act", methods=["POST"])
def act():
    """
    核心推理接口

    Input (JSON):
        {
            "observation": {
                "rgb": [H, W, 3],                    # RGB图像
                "depth": [H, W],                     # 深度图（可选）
                "instruction": "...",                # 自然语言指令
                "current_position": [x, y, z],       # 当前位置（世界坐标系）
                "current_rotation": [[...], ...],    # 当前旋转矩阵 [3, 3]
                "current_orientation": [qw,qx,qy,qz],# 当前朝向（四元数，可选）
                "target_position": [x, y, z],        # 目标位置
                "history_positions": [[x,y,z], ...], # 历史位置序列
                "initial_rotation": [[...], ...],    # 起点旋转矩阵
                "initial_position": [x, y, z],       # 起点位置
                "assist_notice": "cruise",           # 助手提示
                "timestep": 10                       # 当前时间步
            }
        }

    Output (JSON):
        {
            "waypoints": [[x, y, z]],  # 下一个航点（世界坐标系）
            "stop": false              # 是否到达目标
        }
    """
    # 解析请求
    req = json_numpy.loads(request.data)
    obs = req["observation"]

    # ========================================================
    # [EXTRACT] 提取输入数据
    # ========================================================
    rgb_img = obs.get("rgb")                          # [H, W, 3]
    depth = obs.get("depth")                          # [H, W]
    instruction = obs.get("instruction", "")          # str
    current_pos = np.array(obs.get("current_position", [0, 0, 0]))  # [x, y, z]
    current_rot = np.array(obs.get("current_rotation", np.eye(3)))  # [3, 3]
    target_pos = np.array(obs.get("target_position", [0, 0, 0]))    # [x, y, z]
    history_positions = obs.get("history_positions", [])
    assist_notice = obs.get("assist_notice", "cruise")
    timestep = obs.get("timestep", 0)

    # ========================================================
    # [PREPROCESS] 数据预处理
    # ========================================================
    # 示例：计算相对坐标（局部坐标系）
    initial_pos = np.array(obs.get("initial_position", current_pos))
    initial_rot = np.array(obs.get("initial_rotation", np.eye(3)))

    # 转换到相对起点的局部坐标系
    relative_pos = current_rot.T @ (current_pos - initial_pos)
    relative_target = current_rot.T @ (target_pos - initial_pos)

    # 示例：图像预处理
    # if rgb_img is not None:
    #     tensor_img = torch.from_numpy(rgb_img).permute(2, 0, 1).float() / 255.0
    #     tensor_img = tensor_img.unsqueeze(0).to("cuda")

    # 示例：文本预处理
    # inputs = tokenizer(instruction, return_tensors="pt").to("cuda")

    # ========================================================
    # [INFERENCE] 模型推理
    # ========================================================
    # 示例：调用你的模型
    # with torch.no_grad():
    #     pred_waypoint, pred_stop = model(
    #         image=tensor_img,
    #         instruction=inputs,
    #         current_pos=relative_pos,
    #         target_pos=relative_target,
    #         history=history_positions
    #     )
    #     waypoint_local = pred_waypoint.cpu().numpy()[0]  # [x, y, z] 局部坐标
    #     stop = pred_stop.item() > 0.5

    # ========================================================
    # [MOCK IMPLEMENTATION] 测试用的Mock实现
    # ========================================================
    # 简单的几何导航：朝目标点移动
    direction = target_pos - current_pos
    dist = np.linalg.norm(direction)

    if dist > 0.5:
        # 归一化方向，乘以步长（0.5米）
        waypoint_world = current_pos + (direction / dist) * 0.5
        stop = False
        print(f"[Server] Step {timestep} | Dist: {dist:.2f}m | Moving towards target")
    else:
        # 到达目标
        waypoint_world = target_pos
        stop = True
        print(f"[Server] Step {timestep} | Target Reached! (Dist: {dist:.2f}m)")

    # ========================================================
    # [POSTPROCESS] 后处理
    # ========================================================
    # 如果模型输出的是局部坐标，需要转换回世界坐标
    # waypoint_world = current_rot @ waypoint_local + current_pos

    # 限制航点范围（防止飞出地图）
    # waypoint_world = np.clip(waypoint_world, -100, 100)

    # ========================================================
    # [RETURN] 返回结果
    # ========================================================
    response = {
        "waypoints": [waypoint_world.tolist()],  # 返回列表格式
        "stop": stop
    }

    return json_numpy.dumps(response)

# ============================================================
# [OPTIONAL] 批量推理接口（可选，用于提升性能）
# ============================================================
@app.route("/act_batch", methods=["POST"])
def act_batch():
    """
    批量推理接口：一次处理多个观测

    Input:
        {
            "observations": [obs1, obs2, ...]
        }

    Output:
        {
            "waypoints": [[x,y,z], [x,y,z], ...],
            "stops": [false, false, true, ...]
        }
    """
    req = json_numpy.loads(request.data)
    observations = req["observations"]

    batch_waypoints = []
    batch_stops = []

    for obs in observations:
        # 复用 act() 的逻辑
        # 这里简化处理，实际应该批量推理
        result = act_single(obs)
        batch_waypoints.append(result["waypoints"][0])
        batch_stops.append(result["stop"])

    return json_numpy.dumps({
        "waypoints": batch_waypoints,
        "stops": batch_stops
    })

def act_single(obs: Dict[str, Any]) -> Dict[str, Any]:
    """单个观测的推理逻辑（供批量接口调用）"""
    # 这里可以复用 act() 的逻辑
    # 为了简化，这里只是示例
    return {"waypoints": [[0, 0, 0]], "stop": False}

# ============================================================
# [HEALTH CHECK] 健康检查接口
# ============================================================
@app.route("/health", methods=["GET"])
def health():
    """健康检查接口"""
    return jsonify({
        "status": "healthy",
        "model_loaded": True,
        "active_episodes": len(episode_states)
    })

# ============================================================
# [MAIN] 启动服务器
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TravelUAV Model Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9009, help="Server port")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print(f"🚀 TravelUAV Model Server running on {args.host}:{args.port}")
    print(f"📡 Endpoints:")
    print(f"   - POST /reset       : Reset episode state")
    print(f"   - POST /act         : Single inference")
    print(f"   - POST /act_batch   : Batch inference")
    print(f"   - GET  /health      : Health check")

    app.run(host=args.host, port=args.port, debug=args.debug)
