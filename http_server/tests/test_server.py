"""
TravelUAV HTTP 服务端独立测试

无需 VLA_Habitat 项目，直接测试服务端和客户端
"""

import sys
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import requests
import time


def test_health_check():
    """测试健康检查端点"""
    print("=" * 60)
    print("测试 1: 健康检查")
    print("=" * 60)

    try:
        response = requests.get("http://127.0.0.1:9000/health", timeout=5)
        print(f"✓ 健康检查成功")
        print(f"  状态码: {response.status_code}")
        print(f"  响应: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ 健康检查失败: {e}")
        return False


def test_root_endpoint():
    """测试根端点"""
    print("\n" + "=" * 60)
    print("测试 2: 根端点")
    print("=" * 60)

    try:
        response = requests.get("http://127.0.0.1:9000/", timeout=5)
        print(f"✓ 根端点访问成功")
        print(f"  状态码: {response.status_code}")
        print(f"  响应: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ 根端点访问失败: {e}")
        return False


def test_act_endpoint_simple():
    """测试 act 端点（简单数据）"""
    print("\n" + "=" * 60)
    print("测试 3: act 端点（简单数据）")
    print("=" * 60)

    # 创建简单的测试数据
    test_data = {
        "observation": {
            "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tolist(),  # 正确尺寸的 RGB 数据
            "gps": [0.0, 0.0],
            "yaw": 0.0,
            "camera_height": 1.0,
            "instruction": "向前飞行 10 米",
            "step_id": 0
        }
    }

    try:
        response = requests.post(
            "http://127.0.0.1:9000/act",
            json=test_data,
            timeout=10
        )

        print(f"✓ act 端点请求成功")
        print(f"  状态码: {response.status_code}")
        print(f"  响应: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ act 端点请求失败: {e}")
        return False


def test_act_endpoint_full():
    """测试 act 端点（完整数据）"""
    print("\n" + "=" * 60)
    print("测试 4: act 端点（完整数据）")
    print("=" * 60)

    # 创建完整的测试数据（模拟真实场景）
    test_data = {
        "observation": {
            "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tolist(),
            "depth": np.random.rand(256, 256).astype(np.float32).tolist(),
            "gps": [10.5, 20.3],
            "yaw": 1.57,  # 90 度
            "camera_height": 1.5,
            "instruction": "飞到建筑物前的空地",
            "step_id": 5
        }
    }

    try:
        response = requests.post(
            "http://127.0.0.1:9000/act",
            json=test_data,
            timeout=30
        )

        print(f"✓ act 端点请求成功")
        print(f"  状态码: {response.status_code}")
        result = response.json()
        print(f"  响应: {result}")

        # 验证响应格式
        if "actions" in result:
            print(f"  动作: {result['actions']}")
            return True
        else:
            print(f"✗ 响应缺少 'actions' 字段")
            return False

    except Exception as e:
        print(f"✗ act 端点请求失败: {e}")
        return False


def test_client_library():
    """测试使用客户端库"""
    print("\n" + "=" * 60)
    print("测试 5: 使用客户端库")
    print("=" * 60)

    try:
        # 导入客户端
        from http_server.client.client import HTTPTrajectoryClient

        # 创建客户端
        client = HTTPTrajectoryClient(
            server_url="http://127.0.0.1:9000",
            timeout=5.0
        )

        # 创建测试观察数据
        test_observation = {
            "rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
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

    except Exception as e:
        print(f"✗ 客户端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_requests():
    """测试多次请求"""
    print("\n" + "=" * 60)
    print("测试 6: 多次请求测试")
    print("=" * 60)

    success_count = 0
    total_requests = 5

    for i in range(total_requests):
        print(f"\n请求 {i+1}/{total_requests}...")

        test_data = {
            "observation": {
                "rgb": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8).tolist(),
                "gps": [float(i * 10), float(i * 5)],
                "yaw": float(i),
                "camera_height": 1.0,
                "instruction": f"测试指令 {i+1}",
                "step_id": i
            }
        }

        try:
            response = requests.post(
                "http://127.0.0.1:9000/act",
                json=test_data,
                timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                if "actions" in result:
                    print(f"  ✓ 成功，动作: {result['actions']}")
                    success_count += 1
                else:
                    print(f"  ✗ 响应格式错误")
            else:
                print(f"  ✗ 状态码错误: {response.status_code}")

        except Exception as e:
            print(f"  ✗ 请求失败: {e}")

        time.sleep(0.5)  # 短暂延迟

    print(f"\n{'='*60}")
    print(f"多次请求测试完成: {success_count}/{total_requests} 成功")
    print(f"{'='*60}")

    return success_count == total_requests


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("TravelUAV HTTP 服务端独立测试")
    print("=" * 60)
    print()
    print("前提条件:")
    print("  1. conda 环境已激活: conda activate llamauav_sm_120")
    print("  2. 服务端已启动: python http_server/server/server.py")
    print("  3. 服务端监听: http://127.0.0.1:9000")
    print()

    # 等待用户确认
    input("按 Enter 键开始测试...")

    results = []

    # 运行测试
    results.append(("健康检查", test_health_check()))
    results.append(("根端点", test_root_endpoint()))
    results.append(("act 端点（简单）", test_act_endpoint_simple()))
    results.append(("act 端点（完整）", test_act_endpoint_full()))
    results.append(("客户端库", test_client_library()))
    results.append(("多次请求", test_multiple_requests()))

    # 输出总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    for name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {name}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    print()
    print(f"总计: {passed}/{total} 测试通过")
    print("=" * 60)

    if passed == total:
        print("\n🎉 所有测试通过！HTTP 服务端工作正常。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败，请检查服务端状态。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
