"""
TravelUAV HTTP 服务端

提供 HTTP 接口供客户端调用，接收观察数据并返回轨迹动作。

接口格式:
- GET / - 基本信息
- GET /health - 健康检查
- POST /act - 生成轨迹动作

请求格式:
{
    "observation": {
        "rgb": <numpy.ndarray>,      # (H, W, 3) RGB image
        "depth": <numpy.ndarray>,    # (H, W) depth image (optional)
        "gps": <numpy.ndarray>,      # (2,) GPS coordinates [x, y]
        "yaw": <float>,              # Compass heading in radians
        "camera_height": <float>,    # Camera height relative to initial
        "instruction": <str>,        # Navigation instruction text
        "step_id": <int>             # Current step number
    }
}

响应格式:
{
    "actions": <list[int]>           # Habitat action IDs [0, 1, 2, 3, 5]
}
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
    from PIL import Image
except ImportError as e:
    print(f"缺少依赖: {e}")
    print("请安装: pip install fastapi uvicorn pillow")
    sys.exit(1)


@dataclass
class TravelUAVModelArgs:
    """TravelUAV 模型参数"""
    model_path: str = "/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct"
    traj_model_path: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4"
    vision_tower: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth"
    image_processor: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224"
    use_4bit: bool = True


@dataclass
class TravelUAVDataArgs:
    """TravelUAV 数据参数"""
    input_prompt: str = None
    refine_prompt: bool = True


class TravelUAVHTTPServer:
    """
    TravelUAV HTTP 服务端

    接收来自客户端的请求，使用 TravelUAV 模型生成动作
    """

    def __init__(self, model_args: TravelUAVModelArgs, host: str = "127.0.0.1", port: int = 9000):
        self.host = host
        self.port = port
        self.model_args = model_args
        self.data_args = TravelUAVDataArgs()

        # 初始化 FastAPI
        self.app = FastAPI(title="TravelUAV HTTP Server")

        # 加载模型
        self.model_wrapper = None
        self._load_model()

        # 设置路由
        self._setup_routes()

        print(f"✓ TravelUAV HTTP 服务端初始化完成")
        print(f"  监听地址: {host}:{port}")
        print(f"  模型设备: {self.model_wrapper.model.device if self.model_wrapper else 'N/A'}")

    def _load_model(self):
        """加载 TravelUAV 模型"""
        print("=" * 60)
        print("加载 TravelUAV 模型...")
        print("=" * 60)

        try:
            from src.model_wrapper.qwen3vl_gpu_native import Qwen3VLGPUNativeWrapper

            print(f"模型路径: {self.model_args.model_path}")
            print(f"轨迹模型路径: {self.model_args.traj_model_path}")
            print(f"4-bit 量化: {self.model_args.use_4bit}")

            # 检查模型文件是否存在
            if not Path(self.model_args.model_path).exists():
                raise FileNotFoundError(f"模型目录不存在: {self.model_args.model_path}")

            if not Path(self.model_args.traj_model_path).exists():
                raise FileNotFoundError(f"轨迹模型目录不存在: {self.model_args.traj_model_path}")

            # 加载模型
            self.model_wrapper = Qwen3VLGPUNativeWrapper(
                model_args=self.model_args,
                data_args=self.data_args,
                use_traj_model=True
            )

            # 设置为评估模式
            self.model_wrapper.eval()

            print(f"✓ 模型加载成功")
            print(f"  设备: {self.model_wrapper.model.device}")
            print(f"  显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        except Exception as e:
            print(f"✗ 模型加载失败: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _setup_routes(self):
        """设置 HTTP 路由"""

        @self.app.get("/")
        async def root():
            """健康检查"""
            return {
                "status": "ok",
                "service": "TravelUAV",
                "model": "Qwen3-VL-4B-Instruct",
                "device": str(self.model_wrapper.model.device) if self.model_wrapper else "N/A"
            }

        @self.app.post("/act")
        async def act(request: Dict):
            """
            接收客户端的请求，生成动作
            """
            try:
                # 验证请求
                if "observation" not in request:
                    raise HTTPException(status_code=400, detail="Missing 'observation' in request")

                obs = request["observation"]

                # 验证必需字段
                required_fields = ["rgb", "gps", "yaw", "camera_height", "instruction"]
                for field in required_fields:
                    if field not in obs:
                        raise HTTPException(status_code=400, detail=f"Missing required field: {field}")

                # 处理观察数据
                actions = self._process_observation(obs)

                return {"actions": actions}

            except HTTPException:
                raise
            except Exception as e:
                print(f"处理请求时出错: {e}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.get("/health")
        async def health():
            """健康检查端点"""
            return {
                "status": "healthy",
                "model_loaded": self.model_wrapper is not None,
                "device": str(self.model_wrapper.model.device) if self.model_wrapper else None
            }

    def _process_observation(self, obs: Dict) -> List[int]:
        """
        处理观察数据并生成动作

        Args:
            obs: 观察数据字典

        Returns:
            Habitat 动作 ID 列表
        """
        # 提取观察数据
        rgb_array = obs["rgb"]  # numpy array
        gps = obs["gps"]  # [x, y]
        yaw = obs["yaw"]  # radians
        camera_height = obs["camera_height"]
        instruction = obs["instruction"]
        step_id = obs.get("step_id", 0)

        # 转换 RGB 为 PIL Image
        if isinstance(rgb_array, np.ndarray):
            # 确保是 uint8 类型
            if rgb_array.dtype != np.uint8:
                rgb_array = (rgb_array * 255).astype(np.uint8)
            image = Image.fromarray(rgb_array).convert("RGB")
        elif isinstance(rgb_array, list):
            # 如果是 list，先转换为 numpy array
            rgb_array = np.array(rgb_array, dtype=np.uint8)
            image = Image.fromarray(rgb_array).convert("RGB")
        else:
            # 其他类型（不应该出现）
            raise ValueError(f"Unsupported RGB type: {type(rgb_array)}")

        # 构建模拟的 episode 数据
        episode = [{
            'instruction': instruction,
            'rgb': [image],
            'sensors': {
                'imu': {
                    'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                },
                'state': {
                    'position': [float(gps[0]), float(gps[1]), float(camera_height)]
                }
            }
        }]

        # 目标位置（从 GPS 和 yaw 推断）
        target_distance = 10.0
        target_x = float(gps[0]) + target_distance * np.cos(yaw)
        target_y = float(gps[1]) + target_distance * np.sin(yaw)
        target_z = float(camera_height)
        target_position = [target_x, target_y, target_z]

        # 准备输入
        try:
            inputs, rot_to_targets = self.model_wrapper.prepare_inputs(
                episodes=[episode],
                target_positions=[target_position]
            )

            # 运行模型生成航点
            refined_waypoints = self.model_wrapper.run(
                inputs=inputs,
                episodes=[episode],
                rot_to_targets=rot_to_targets
            )

            # 将航点转换为 Habitat 动作
            actions = self._waypoints_to_actions(refined_waypoints[0], episode[0])

            return actions

        except Exception as e:
            print(f"模型推理出错: {e}")
            import traceback
            traceback.print_exc()
            # 返回默认动作（停止）
            return [0]

    def _waypoints_to_actions(self, waypoint: np.ndarray, episode: Dict) -> List[int]:
        """
        将航点转换为 Habitat 动作

        Args:
            waypoint: 航点 [x, y, z]
            episode: 当前 episode 数据

        Returns:
            Habitat 动作 ID 列表
        """
        # Habitat 动作 ID:
        # 0: STOP
        # 1: MOVE_FORWARD
        # 2: TURN_LEFT
        # 3: TURN_RIGHT
        # 5: LOOK_DOWN

        actions = []

        # 提取当前位置
        current_pos = np.array(episode['sensors']['state']['position'])

        # 计算到航点的方向向量
        target_vec = waypoint[:3] - current_pos
        distance = np.linalg.norm(target_vec)

        # 如果距离很近，停止
        if distance < 0.5:
            return [0]

        # 简化的动作决策
        if distance > 2.0:
            # 距离较远，向前移动
            actions.append(1)  # MOVE_FORWARD
        else:
            # 距离较近，停止
            actions.append(0)  # STOP

        return actions

    def run(self):
        """启动 HTTP 服务器"""
        print("=" * 60)
        print(f"启动 TravelUAV HTTP 服务器")
        print("=" * 60)
        print(f"监听地址: {self.host}:{self.port}")
        print(f"端点: POST http://{self.host}:{self.port}/act")
        print("=" * 60)

        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


def main():
    parser = argparse.ArgumentParser(description="TravelUAV HTTP 服务端")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务器主机地址")
    parser.add_argument("--port", type=int, default=9000, help="服务器端口")
    parser.add_argument("--model_path", type=str, default="/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct", help="Qwen3-VL 模型路径")
    parser.add_argument("--traj_model_path", type=str, default="/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4", help="轨迹模型路径")
    parser.add_argument("--no_4bit", action="store_true", help="禁用 4-bit 量化")

    args = parser.parse_args()

    # 检查 CUDA
    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，模型将在 CPU 上运行（可能很慢）")

    # 创建模型参数
    model_args = TravelUAVModelArgs(
        model_path=args.model_path,
        traj_model_path=args.traj_model_path,
        use_4bit=not args.no_4bit
    )

    # 创建并启动服务器
    server = TravelUAVHTTPServer(
        model_args=model_args,
        host=args.host,
        port=args.port
    )

    server.run()


if __name__ == "__main__":
    main()
