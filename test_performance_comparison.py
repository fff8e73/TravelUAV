"""
性能对比测试脚本（两阶段架构）

对比测试：
1. 两阶段: Qwen3-VL + 专用轨迹模型
2. 单阶段: Qwen3-VL (端到端)
"""

import time
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
print("性能对比测试（两阶段架构）")
print("=" * 70)

# 测试配置
test_configs = [
    {"use_traj_model": True, "name": "两阶段: Qwen3-VL + 专用轨迹模型"},
    {"use_traj_model": False, "name": "单阶段: Qwen3-VL (端到端)"},
]

# 创建测试数据
print("\n创建测试数据...")
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))

test_episode = [{
    'instruction': '向前飞行',
    'rgb': [test_image],
    'sensors': {
        'imu': {'rotation': [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
        'state': {'position': [0, 0, 0]}
    }
}]
test_target = [10, 0, 5]

results = []

for config in test_configs:
    print(f"\n{'='*70}")
    print(f"测试配置: {config['name']}")
    print(f"{'='*70}")

    # 初始化
    model_args = ModelArgs()
    data_args = DataArgs()

    print("\n1. 加载模型...")
    wrapper = Qwen3VLGPUNativeWrapper(
        model_args,
        data_args,
        use_traj_model=config['use_traj_model']
    )
    print("  ✓ 模型加载成功")

    # 预热
    print("\n2. 预热运行...")
    for i in range(2):
        inputs, rot_to_targets = wrapper.prepare_inputs([test_episode], [test_target])
        _ = wrapper.run(inputs, test_episode, rot_to_targets)
        print(f"  预热 {i+1}/2 完成")
    print("  ✓ 预热完成")

    # 性能测试
    print("\n3. 性能测试 (10 次推理)...")
    times = []
    memory_usage = []

    for i in range(10):
        inputs, rot_to_targets = wrapper.prepare_inputs([test_episode], [test_target])

        start_time = time.time()
        waypoints = wrapper.run(inputs, test_episode, rot_to_targets)
        end_time = time.time()

        elapsed = end_time - start_time
        times.append(elapsed)
        memory = torch.cuda.memory_allocated() / 1024**3
        memory_usage.append(memory)

        print(f"  运行 {i+1}/10: {elapsed:.2f} 秒, 显存: {memory:.2f} GB")

    # 统计结果
    avg_time = sum(times) / len(times)
    avg_memory = sum(memory_usage) / len(memory_usage)
    min_time = min(times)
    max_time = max(times)
    fps = 1 / avg_time

    print(f"\n4. 性能统计:")
    print(f"  - 平均推理时间: {avg_time:.2f} 秒")
    print(f"  - 最快/最慢: {min_time:.2f} / {max_time:.2f} 秒")
    print(f"  - 平均显存占用: {avg_memory:.2f} GB")
    print(f"  - FPS: {fps:.2f}")
    print(f"  - 输出航点: {waypoints}")

    results.append({
        "name": config["name"],
        "avg_time": avg_time,
        "avg_memory": avg_memory,
        "fps": fps
    })

    # 清理
    print("\n5. 清理资源...")
    del wrapper, inputs, waypoints
    torch.cuda.empty_cache()
    print(f"  ✓ 资源已清理")
    print(f"    - 清理后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# 对比总结
print(f"\n{'='*70}")
print("性能对比总结")
print(f"{'='*70}")

for result in results:
    print(f"\n{result['name']}:")
    print(f"  - 平均推理时间: {result['avg_time']:.2f} 秒")
    print(f"  - 平均显存占用: {result['avg_memory']:.2f} GB")
    print(f"  - FPS: {result['fps']:.2f}")

# 计算改进
if len(results) >= 2:
    time_improvement = (results[1]["avg_time"] - results[0]["avg_time"]) / results[1]["avg_time"] * 100
    memory_improvement = (results[1]["avg_memory"] - results[0]["avg_memory"]) / results[1]["avg_memory"] * 100

    print(f"\n{'='*70}")
    print("改进分析")
    print(f"{'='*70}")
    print(f"\n两阶段 vs 单阶段:")
    print(f"  - 推理时间: {time_improvement:+.1f}% ({results[0]['avg_time']:.2f} vs {results[1]['avg_time']:.2f} 秒)")
    print(f"  - 显存占用: {memory_improvement:+.1f}% ({results[0]['avg_memory']:.2f} vs {results[1]['avg_memory']:.2f} GB)")

    if time_improvement > 0:
        print(f"\n  ✅ 两阶段架构更快 (快 {time_improvement:.1f}%)")
    else:
        print(f"\n  ⚠️  单阶段架构更快 (快 {abs(time_improvement):.1f}%)")

    if memory_improvement > 0:
        print(f"  ✅ 两阶段架构显存更低 (低 {memory_improvement:.1f}%)")
    else:
        print(f"  ⚠️  单阶段架构显存更低 (低 {abs(memory_improvement):.1f}%)")

print(f"\n{'='*70}")
print("✅ 性能测试完成！")
print(f"{'='*70}")
