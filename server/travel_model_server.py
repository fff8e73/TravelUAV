"""
TravelUAV Model Server
外部模型服务器端实现 - 基于FastAPI（薄透传层）
对齐 Isaac-Drone-Navigation-Benchmark 接口标准
"""
from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn
import json_numpy
import numpy as np
from typing import Dict, Any

# 开启 numpy 序列化支持
json_numpy.patch()

app = FastAPI()

# ============================================================
# 全局变量：存储每个环境的状态
# ============================================================
episode_states = {}

# ============================================================
# [MODEL INIT] 在这里加载你的模型
# ============================================================
# 注意：模型本身已直接接受 A 标准 obs、直接输出 [N,4]
# Server 仅做薄透传，不做格式转换
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
@app.post("/reset")
async def reset(request: Request):
    """
    重置接口：当新的 Episode 开始时调用
    作用：清空该环境的历史状态（Hidden State、History Buffer等）
    """
    req = json_numpy.loads(await request.body())
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

    return {"status": "ok"}


# ============================================================
# [ACT] 核心推理接口 - 薄透传层
# ============================================================
@app.post("/act")
async def act(request: Request):
    """
    核心推理接口 - 薄透传层

    Input (A 标准 JSON):
        {
            "rgb": [H, W, 3],              # RGB图像
            "depth": [H, W],               # 深度图（可选）
            "instruction": "...",          # 自然语言指令
            "step": int,                   # 当前 episode 已执行的步数
            "compass": [yaw_rad],          # 相对朝向（可选）
            "gps": [x, y],                 # 相对位置（可选）
            "collision": bool,             # 是否碰撞（可选）
            "assist_notice": "..."         # 助手提示（可选）
        }

    Output (A 标准 JSON):
        {
            "action": [[dx, dy, dz, dyaw], ...]  # [N, 4] 相对动作
        }

    注意：Server 不做任何格式转换，直接透传 obs 给模型，模型输出直接返回
    """
    # 解析请求（直接透传）
    req = json_numpy.loads(await request.body())
    obs = req  # A 标准 obs 直接作为模型输入

    # ========================================================
    # [INFERENCE] 模型推理
    # ========================================================
    # 注意：这里应该调用实际的模型
    # 示例：
    # with torch.no_grad():
    #     pred_actions = model.predict(obs)  # → ndarray[N,4]
    # response = {"action": pred_actions}

    # ========================================================
    # [MOCK IMPLEMENTATION] 测试用的 Mock 实现
    # ========================================================
    # 模拟模型输出 [N, 4] 相对动作
    # N = 8 (前7步运动 + 最后1步stop)
    N = 8
    pred_actions = np.zeros((N, 4), dtype=np.float32)

    # 获取当前步数，用于随机噪声
    step = obs.get("step", 0)

    # 随机噪声（模拟真实模型的输出变化）
    np.random.seed(step)
    random_noise = np.random.randn(N, 4) * 0.01

    # 在 8 步内部进行方向循环和高度变化
    for i in range(N):
        # 最后一步（第8步）返回 stop 信号
        if i == N - 1:
            pred_actions[i] = [
                1e-6,   # dx (几乎为0)
                1e-6,   # dy (几乎为0)
                1e-6,   # dz (几乎为0)
                0.0     # dyaw
            ]
        else:
            # 前7步：方向循环 + 高度交替
            # 方向循环：每 4 步一循环（前→左→后→右）
            direction_idx = i % 4
            if direction_idx == 0:
                dx, dy, dz= 0.5, 0.0, 0.0  # 向前
            elif direction_idx == 1:
                dx, dy, dz= 0.0, 0.5, 0.0  # 向左
            elif direction_idx == 2:
                dx, dy, dz= 0.0, 0.0, 0.5  # 向上
            else:
                dx, dy, dz= 0.0, -0.5, 0.0  # 向右

            # 微小偏航角变化
            dyaw = 0.05 * ((step % 10) - 5) / 5

            # 加入随机噪声
            pred_actions[i] = [
                dx + random_noise[i, 0],
                dy + random_noise[i, 1],
                dz + random_noise[i, 2],
                dyaw + random_noise[i, 3]
            ]

    # 构建响应（A 标准格式）
    response = {
        "action": pred_actions
    }

    # 返回（使用 Response 确保正确设置 media_type）
    return Response(
        content=json_numpy.dumps(response),
        media_type="application/json"
    )


# ============================================================
# [HEALTH CHECK] 健康检查接口
# ============================================================
@app.get("/health")
async def health():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": True,
        "active_episodes": len(episode_states)
    }


# ============================================================
# [MAIN] 启动服务器
# ============================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TravelUAV Model Server (FastAPI)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=9009, help="Server port")
    args = parser.parse_args()

    print(f"🚀 TravelUAV Model Server (FastAPI) running on {args.host}:{args.port}")
    print(f"📡 Endpoints:")
    print(f"   - POST /reset       : Reset episode state")
    print(f"   - POST /act         : Single inference (A标准 [N,4] 动作)")
    print(f"   - GET  /health      : Health check")

    uvicorn.run(app, host=args.host, port=args.port)
