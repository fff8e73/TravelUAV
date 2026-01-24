"""
TravelUAV HTTP 客户端适配器

兼容 VLA_Habitat 的 HTTPTrajectoryClient 接口
用于 VLA_Habitat 项目调用 TravelUAV 服务端
"""

import json
import numpy as np
import requests
from typing import Dict, List, Optional, Any


class BaseTrajectoryClient:
    """VLA_Habitat 的基类接口"""

    def reset(self, instruction: str, **kwargs):
        pass

    def query(self, obs: dict) -> list[int]:
        """
        返回 Habitat action id list
        """
        raise NotImplementedError


class HTTPTrajectoryClient(BaseTrajectoryClient):
    """
    HTTP 轨迹客户端 - 适配 VLA_Habitat 接口

    通过 HTTP 调用 TravelUAV 服务端生成轨迹动作
    """

    def __init__(self, server_url: str = "http://127.0.0.1:9000", timeout: float = 5.0):
        """
        初始化 HTTP 客户端

        Args:
            server_url: TravelUAV 服务器 URL
            timeout: 请求超时时间（秒）
        """
        self.server_url = server_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

        # 测试连接
        self._test_connection()

    def _test_connection(self):
        """测试与服务器的连接"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=self.timeout)
            if response.status_code == 200:
                print(f"✓ 成功连接到 TravelUAV 服务器: {self.server_url}")
                print(f"  响应: {response.json()}")
            else:
                print(f"⚠ 服务器返回状态码: {response.status_code}")
        except requests.exceptions.ConnectionError as e:
            print(f"✗ 无法连接到服务器: {self.server_url}")
            print(f"  请确保服务器正在运行")
            raise
        except Exception as e:
            print(f"⚠ 连接测试出错: {e}")

    def reset(self, instruction: str, **kwargs):
        """
        重置客户端状态

        注意: 对于 HTTP 客户端，这通常不需要做任何操作
        因为服务器是无状态的

        Args:
            instruction: 导航指令（兼容基类接口）
            **kwargs: 其他可选参数
        """
        print(f"重置 HTTP 客户端状态，指令: {instruction}")

    def query(self, obs: dict) -> list[int]:
        """
        查询动作 - 兼容 VLA_Habitat 接口

        Args:
            obs: 观察数据字典，包含:
                - rgb: RGB 图像 (H, W, 3) numpy array
                - depth: 深度图像 (H, W) numpy array (可选)
                - gps: GPS 坐标 [x, y]
                - yaw: 朝向角度（弧度）
                - camera_height: 相机高度
                - instruction: 导航指令文本
                - step_id: 当前步数（可选）

        Returns:
            Habitat 动作 ID 列表
        """
        # 构建请求数据
        request_data = {
            "observation": self._prepare_observation(obs)
        }

        try:
            # 发送 POST 请求
            response = self.session.post(
                f"{self.server_url}/act",
                json=request_data,
                timeout=self.timeout
            )

            # 检查响应
            response.raise_for_status()

            # 解析响应
            result = response.json()

            if "actions" not in result:
                raise ValueError(f"响应缺少 'actions' 字段: {result}")

            actions = result["actions"]

            # 验证动作格式
            if not isinstance(actions, list):
                raise ValueError(f"actions 应该是列表，得到: {type(actions)}")

            return actions

        except requests.exceptions.Timeout:
            print(f"请求超时 (timeout={self.timeout}s)")
            return [0]  # 返回 STOP 动作

        except requests.exceptions.ConnectionError as e:
            print(f"连接错误: {e}")
            return [0]  # 返回 STOP 动作

        except Exception as e:
            print(f"查询出错: {e}")
            import traceback
            traceback.print_exc()
            return [0]  # 返回 STOP 动作

    def _prepare_observation(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """
        准备观察数据用于序列化

        Args:
            observation: 原始观察数据

        Returns:
            可序列化的观察数据
        """
        prepared = {}

        for key, value in observation.items():
            if key == "rgb" or key == "depth":
                # 确保 numpy 数组是可序列化的
                if isinstance(value, np.ndarray):
                    # 转换为列表（json_numpy 会自动处理）
                    prepared[key] = value.tolist() if value.size < 10000 else value
                else:
                    prepared[key] = value
            elif key == "gps":
                # 确保 GPS 是列表
                if isinstance(value, np.ndarray):
                    prepared[key] = value.tolist()
                elif isinstance(value, (list, tuple)):
                    prepared[key] = list(value)
                else:
                    prepared[key] = value
            else:
                prepared[key] = value

        return prepared

    def close(self):
        """关闭客户端会话"""
        self.session.close()
        print("HTTP 客户端已关闭")


# 兼容 VLA_Habitat 的接口别名
TrajectoryClient = HTTPTrajectoryClient


def test_client():
    """测试客户端"""
    print("=" * 60)
    print("测试 TravelUAV HTTP 客户端")
    print("=" * 60)

    # 创建客户端
    client = HTTPTrajectoryClient(
        server_url="http://127.0.0.1:9000",
        timeout=5.0
    )

    # 创建测试观察数据
    test_observation = {
        "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "depth": np.random.rand(256, 256).astype(np.float32),
        "gps": np.array([0.0, 0.0]),
        "yaw": 0.0,
        "camera_height": 1.0,
        "instruction": "向前飞行 10 米",
        "step_id": 0
    }

    print(f"测试观察数据:")
    print(f"  RGB 形状: {test_observation['rgb'].shape}")
    print(f"  GPS: {test_observation['gps']}")
    print(f"  指令: {test_observation['instruction']}")

    # 查询动作
    print("\n查询动作...")
    actions = client.query(test_observation)

    print(f"\n✓ 查询成功!")
    print(f"  动作: {actions}")

    # 关闭客户端
    client.close()

    return True


if __name__ == "__main__":
    import sys
    try:
        test_client()
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
