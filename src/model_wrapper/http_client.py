"""
TravelUAV HTTP Client
用于连接外部模型服务器的客户端
"""
import requests
import json_numpy
import numpy as np
from typing import List, Dict, Any

# 开启 numpy 序列化支持
json_numpy.patch()


class HttpClient:
    """
    HTTP客户端 - 用于TravelUAV Benchmark连接外部模型Server
    不依赖BaseModelWrapper，直接提供eval.py所需的接口
    """

    def __init__(self, server_url: str = "http://127.0.0.1:9009", timeout: int = 300):
        """
        初始化HTTP客户端

        Args:
            server_url: 外部模型服务器地址
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        print(f"🔗 [HttpClient] Connecting to server: {self.server_url}")

    def reset(self, env_id: int = 0, **kwargs):
        """
        通知Server重置特定环境的状态（如Hidden State、History Buffer）

        Args:
            env_id: 环境ID
            **kwargs: 其他参数（如episode_id, scene_id等）
        """
        payload = {"type": "reset", "env_id": env_id, **kwargs}
        try:
            resp = requests.post(
                f"{self.server_url}/reset",
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=2.0
            )
            resp.raise_for_status()
            print(f"[HttpClient] Reset env_id={env_id} successfully")
        except Exception as e:
            print(f"[HttpClient] Reset warning: {e}")

    def _extract_observation(self, episode: List[Dict], target_position: np.ndarray,
                            assist_notice: str = None) -> Dict[str, Any]:
        """
        从episode中提取观测数据，构造发送给Server的observation

        Args:
            episode: 单个episode的历史观测序列
            target_position: 目标位置 [x, y, z]
            assist_notice: 助手提示（可选）

        Returns:
            observation字典
        """
        # 获取最新的观测帧
        current_frame = episode[-1]
        first_frame = episode[0]

        obs = {}

        # 1. RGB图像（取第一个视角，即前视相机）
        if 'rgb' in current_frame and len(current_frame['rgb']) > 0:
            obs['rgb'] = current_frame['rgb'][0]  # shape: [H, W, 3]

        # 2. 深度图（可选）
        if 'depth' in current_frame and len(current_frame['depth']) > 0:
            obs['depth'] = current_frame['depth'][0]

        # 3. 自然语言指令
        if 'instruction' in current_frame:
            obs['instruction'] = current_frame['instruction']
        elif 'instruction' in first_frame:
            obs['instruction'] = first_frame['instruction']
        else:
            obs['instruction'] = "Navigate to the target"

        # 4. 当前位置
        if 'sensors' in current_frame and 'state' in current_frame['sensors']:
            obs['current_position'] = current_frame['sensors']['state']['position']

        # 5. 当前旋转矩阵
        if 'sensors' in current_frame and 'imu' in current_frame['sensors']:
            obs['current_rotation'] = current_frame['sensors']['imu']['rotation']

        # 6. 当前朝向（四元数，可选）
        if 'sensors' in current_frame and 'state' in current_frame['sensors']:
            if 'orientation' in current_frame['sensors']['state']:
                obs['current_orientation'] = current_frame['sensors']['state']['orientation']

        # 7. 目标位置
        obs['target_position'] = target_position.tolist() if isinstance(target_position, np.ndarray) else target_position

        # 8. 历史位置序列
        history_positions = []
        for frame in episode:
            if 'sensors' in frame and 'state' in frame['sensors']:
                history_positions.append(frame['sensors']['state']['position'])
        obs['history_positions'] = history_positions

        # 9. 历史旋转序列（可选，用于Server端计算相对坐标）
        if 'sensors' in first_frame and 'imu' in first_frame['sensors']:
            obs['initial_rotation'] = first_frame['sensors']['imu']['rotation']
        if 'sensors' in first_frame and 'state' in first_frame['sensors']:
            obs['initial_position'] = first_frame['sensors']['state']['position']

        # 10. 助手提示
        if assist_notice is not None:
            obs['assist_notice'] = assist_notice
        else:
            # 根据历史长度推断阶段
            obs['assist_notice'] = 'cruise' if len(episode) > 20 else 'take off'

        # 11. 时间步信息
        obs['timestep'] = len(episode) - 1

        return obs

    def query_batch(self, episodes: List[List[Dict]], target_positions: List[np.ndarray],
                   assist_notices: List[str] = None) -> tuple:
        """
        批量查询：发送观测数据，接收航点和停止信号

        Args:
            episodes: episode列表，每个episode是观测序列
            target_positions: 目标位置列表
            assist_notices: 助手提示列表（可选）

        Returns:
            (refined_waypoints, predict_dones)
            - refined_waypoints: List[np.ndarray]，每个元素是 [x, y, z]
            - predict_dones: List[bool]
        """
        batch_size = len(episodes)
        refined_waypoints = []
        predict_dones = []

        # 逐个发送请求（也可以改为批量接口）
        for i in range(batch_size):
            assist_notice = assist_notices[i] if assist_notices is not None else None

            # 提取观测数据
            obs = self._extract_observation(episodes[i], target_positions[i], assist_notice)

            # 构造请求
            payload = {"observation": obs}

            try:
                # 发送请求
                resp = requests.post(
                    f"{self.server_url}/act",
                    data=json_numpy.dumps(payload),
                    headers={"Content-Type": "application/json"},
                    timeout=self.timeout
                )
                resp.raise_for_status()

                # 解析响应
                result = json_numpy.loads(resp.text)

                # 提取航点（取第一个，因为Server返回的是列表）
                waypoints = result.get("waypoints", [[0, 0, 0]])
                if len(waypoints) > 0:
                    refined_waypoints.append(np.array(waypoints[0]))
                else:
                    refined_waypoints.append(np.array([0, 0, 0]))

                # 提取停止信号
                predict_dones.append(result.get("stop", False))

            except Exception as e:
                print(f"[HttpClient] Query failed for batch {i}: {e}")
                # 返回零向量和False作为fallback
                refined_waypoints.append(np.array([0, 0, 0]))
                predict_dones.append(False)

        return refined_waypoints, predict_dones

    def eval(self):
        """
        兼容接口：设置为评估模式（对于HTTP客户端无实际作用）
        """
        pass
