"""
快速测试 HTTP 客户端

使用示例:
    python http_server/tests/test_client.py
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
from http_server.client.client import HTTPTrajectoryClient


def main():
    print("=" * 60)
    print("TravelUAV HTTP 客户端快速测试")
    print("=" * 60)
    print()
    print("前提条件:")
    print("  1. conda 环境已激活: conda activate llamauav_sm_120")
    print("  2. 服务端已启动: python http_server/server/server.py")
    print("  3. 服务端监听: http://127.0.0.1:9000")
    print()

    try:
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

        print("\n🎉 客户端测试完成！")
        return 0

    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
