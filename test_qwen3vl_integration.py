"""
Qwen3-VL + 轨迹模型集成测试脚本（两阶段架构）

验证完整的两阶段推理流程：
1. Qwen3-VL 生成粗略航点
2. 专用轨迹模型优化航点
"""

import torch
from src.model_wrapper.qwen3vl_gpu_native import Qwen3VLGPUNativeWrapper
from dataclasses import dataclass
from PIL import Image
import numpy as np

@dataclass
class ModelArgs:
    model_path: str = "/home/yyx/TravelUAV/Model/Qwen3-VL-4B-Instruct"
    traj_model_path: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor"
    vision_tower: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth"
    image_processor: str = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224"
    use_4bit: bool = True

@dataclass
class DataArgs:
    input_prompt: str = None
    refine_prompt: bool = True

print("=" * 70)
print("Qwen3-VL + 轨迹模型集成测试（两阶段架构）")
print("=" * 70)

# 1. 初始化模型
print("\n1. 加载模型...")
model_args = ModelArgs()
data_args = DataArgs()

try:
    wrapper = Qwen3VLGPUNativeWrapper(model_args, data_args, use_traj_model=True)
    print("  ✓ 模型加载成功")
except Exception as e:
    print(f"  ✗ 模型加载失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 2. 创建测试数据
print("\n2. 创建测试数据...")
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

test_episode = [{
    'instruction': '向前飞行 10 米',
    'rgb': [test_image],
    'sensors': {
        'imu': {'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        'state': {'position': [0, 0, 0]}
    }
}]
test_target = [10, 0, 5]

print(f"  - 测试指令: {test_episode[0]['instruction']}")
print(f"  - 目标位置: {test_target}")
print(f"  - 图像尺寸: {test_image.size}")

# 3. 准备输入
print("\n3. 准备输入...")
try:
    inputs, rot_to_targets = wrapper.prepare_inputs([test_episode], [test_target])
    print("  ✓ 输入准备成功")
    print(f"    - input_ids 形状: {inputs['input_ids'].shape}")
    print(f"    - pixel_values 形状: {inputs['pixel_values'].shape if 'pixel_values' in inputs else 'N/A'}")
    print(f"    - image_grid_thw 形状: {inputs['image_grid_thw'].shape if 'image_grid_thw' in inputs else 'N/A'}")
except Exception as e:
    print(f"  ✗ 输入准备失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 4. 运行推理（两阶段）
print("\n4. 运行推理（两阶段）...")

# 阶段 1: Qwen3-VL 生成粗略航点
print("  阶段 1: Qwen3-VL 生成粗略航点...")
try:
    waypoints_llm = wrapper.run_llm_model(inputs)
    print(f"    ✓ 粗略航点生成成功")
    print(f"      - 形状: {waypoints_llm.shape}")
    print(f"      - 值: {waypoints_llm}")
except Exception as e:
    print(f"    ✗ 阶段 1 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 阶段 2: 专用轨迹模型优化
print("  阶段 2: 专用轨迹模型优化...")
try:
    waypoints = wrapper.run_traj_model(test_episode, waypoints_llm, rot_to_targets)
    print(f"    ✓ 轨迹优化成功")
    print(f"      - 形状: {waypoints.shape}")
    print(f"      - 值: {waypoints}")
except Exception as e:
    print(f"    ✗ 阶段 2 失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 5. 结果验证
print("\n5. 结果验证...")
print(f"  ✓ 推理成功!")
print(f"    - 输出航点形状: {waypoints.shape}")
print(f"    - 输出航点值:\n{waypoints}")
print(f"    - 显存占用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 6. 测试端到端运行
print("\n6. 端到端运行测试...")
try:
    print("  运行完整推理流程...")
    waypoints_full = wrapper.run(inputs, test_episode, rot_to_targets)
    print(f"  ✓ 端到端运行成功")
    print(f"    - 输出形状: {waypoints_full.shape}")
    print(f"    - 输出值:\n{waypoints_full}")
except Exception as e:
    print(f"  ✗ 端到端运行失败: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# 7. 清理
print("\n7. 清理资源...")
del wrapper, inputs, waypoints_llm, waypoints, waypoints_full
torch.cuda.empty_cache()
print(f"  ✓ 资源已清理")
print(f"    - 清理后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

print("\n" + "=" * 70)
print("✅ 集成测试完成！")
print("=" * 70)
