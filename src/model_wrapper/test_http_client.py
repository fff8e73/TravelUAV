import numpy as np
import sys
sys.path.insert(0, '/home/yyx/TravelUAV')
from src.model_wrapper.http_client import HttpClient

# 初始化客户端（不需要真实连接 Server）
client = HttpClient(server_url="http://127.0.0.1:9009")

def test_forward():
    """测试：向前 1 米"""
    action = np.array([1.0, 0.0, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, -10.0])
    current_rot = np.eye(3)  # 朝北

    waypoint = client._relative_action_to_waypoint(action, current_pos, current_rot)
    expected = np.array([1.0, 0.0, -10.0])

    print(f"Forward: waypoint={waypoint}, expected={expected}")
    assert np.allclose(waypoint, expected), f"Forward failed: {waypoint} != {expected}"

def test_left():
    """测试：向左 1 米"""
    action = np.array([0.0, 1.0, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, -10.0])
    current_rot = np.eye(3)  # 朝北

    waypoint = client._relative_action_to_waypoint(action, current_pos, current_rot)
    expected = np.array([0.0, -1.0, -10.0])

    print(f"Left: waypoint={waypoint}, expected={expected}")
    assert np.allclose(waypoint, expected), f"Left failed: {waypoint} != {expected}"

def test_up():
    """测试：向上 1 米"""
    action = np.array([0.0, 0.0, 1.0, 0.0])
    current_pos = np.array([0.0, 0.0, -10.0])
    current_rot = np.eye(3)  # 朝北

    waypoint = client._relative_action_to_waypoint(action, current_pos, current_rot)
    expected = np.array([0.0, 0.0, -11.0])

    print(f"Up: waypoint={waypoint}, expected={expected}")
    assert np.allclose(waypoint, expected), f"Up failed: {waypoint} != {expected}"

def test_yaw_90():
    """测试：朝东时向前 1 米"""
    action = np.array([1.0, 0.0, 0.0, 0.0])
    current_pos = np.array([0.0, 0.0, -10.0])
    # 朝东的旋转矩阵
    current_rot = np.array([
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0]
    ])

    waypoint = client._relative_action_to_waypoint(action, current_pos, current_rot)
    expected = np.array([0.0, -1.0, -10.0])

    print(f"Yaw90: waypoint={waypoint}, expected={expected}")
    assert np.allclose(waypoint, expected), f"Yaw90 failed: {waypoint} != {expected}"

if __name__ == "__main__":
    test_forward()
    test_left()
    test_up()
    test_yaw_90()
    print("✅ All coordinate mapping tests passed!")