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
    # N = 16 (假设模型一次输出 16 步动作)
    N = 16
    pred_actions = np.zeros((N, 4), dtype=np.float32)

    # 复杂的测试逻辑：根据 step 生成多样化的动作序列
    step = obs.get("step", 0)

    # 模式1：每 20 步改变一次方向（前→左→右→后→前...循环）
    direction_cycle = (step // 20) % 4  # 0,1,2,3 循环

    # 模式2：每 50 步执行一次"停止"（小位移）来测试 stop 判定
    is_stop_phase = (step % 50) >= 45  # step 45-49 为停止阶段

    # 模式3：偶尔加入高度变化（每 30 步上升/下降）
    height_change = (step % 60) < 30  # 前30步上升，后30步下降

    # 模式4：加入微小的随机扰动（模拟真实模型的输出变化）
    np.random.seed(step)  # 用 step 作为种子，保证可复现
    random_noise = np.random.randn(N, 4) * 0.01  # 微小噪声

    for i in range(N):
        if is_stop_phase:
            # 停止阶段：返回极小的位移（测试 stop 判定）
            # 位移 < 1e-5 会被识别为 stop
            pred_actions[i] = [
                1e-6,   # dx (几乎为0)
                1e-6,   # dy (几乎为0)
                1e-6,   # dz (几乎为0)
                0.0     # dyaw
            ]
        else:
            # 正常运动阶段
            if direction_cycle == 0:
                # 向前
                dx, dy = 0.1, 0.0
            elif direction_cycle == 1:
                # 向左
                dx, dy = 0.0, 0.1
            elif direction_cycle == 2:
                # 向后
                dx, dy = -0.1, 0.0
            else:
                # 向右
                dx, dy = 0.0, -0.1

            # 高度变化
            dz = 0.02 if height_change else -0.02

            # 微小偏航角变化
            dyaw = 0.01 * ((step % 10) - 5) / 5  # -0.01 到 0.01 之间变化

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
