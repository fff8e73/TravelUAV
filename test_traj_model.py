"""
专用轨迹预测模型验证脚本

验证 VisionTrajectoryGenerator (阶段 2) 的加载和基本功能
"""

import torch
import sys
from pathlib import Path

print("=" * 70)
print("专用轨迹预测模型验证")
print("=" * 70)

# 1. 检查模型文件是否存在
print("\n1. 检查模型文件:")
model_path = "/home/yyx/TravelUAV/Model/LLaMA-UAV/work_dirs/traj_predictor"
weight_path = f"{model_path}/model_5.pth"

if Path(weight_path).exists():
    print(f"  ✓ 轨迹权重文件存在: {weight_path}")
    file_size = Path(weight_path).stat().st_size / 1024**2
    print(f"    - 文件大小: {file_size:.2f} MB")
else:
    print(f"  ✗ 轨迹权重文件不存在: {weight_path}")
    sys.exit(1)

# 2. 检查视觉编码器文件
print("\n2. 检查视觉编码器文件:")
vision_tower_path = "/home/yyx/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth"
if Path(vision_tower_path).exists():
    print(f"  ✓ EVA-ViT 权重文件存在: {vision_tower_path}")
    file_size = Path(vision_tower_path).stat().st_size / 1024**2
    print(f"    - 文件大小: {file_size:.2f} MB")
else:
    print(f"  ⚠ EVA-ViT 权重文件不存在: {vision_tower_path}")
    print(f"    轨迹模型可能需要从配置文件加载视觉编码器")

# 3. 加载轨迹预测模型
print("\n3. 加载轨迹预测模型:")
try:
    sys.path.insert(0, "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/model")
    from vis_traj_arch import VisionTrajectoryGenerator
    from src.model_wrapper.utils.travel_util_clean import generate_vision_tower_config
    import transformers

    print("  正在加载模型配置...")

    # 生成视觉配置
    vision_config = generate_vision_tower_config(
        "/home/yyx/TravelUAV/Model/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth",
        "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid/processor/clip-patch14-224"
    )

    # 加载配置
    config = transformers.AutoConfig.from_pretrained(vision_config, trust_remote_code=True)
    traj_model = VisionTrajectoryGenerator(config)

    print("  ✓ 模型配置加载成功")

    # 4. 加载权重
    print("\n4. 加载模型权重:")
    print(f"  加载权重: {weight_path}")

    traj_weights = torch.load(weight_path, map_location='cpu')
    print(f"  - 权重文件中的键: {len(traj_weights)} 个")

    # 转换数据类型以节省内存
    traj_weights = {k: v.to(torch.bfloat16) for k, v in traj_weights.items()}

    # 加载权重
    traj_model.load_state_dict(traj_weights, strict=False)
    print("  ✓ 权重加载成功")

    # 5. 模型信息统计
    print("\n5. 模型信息统计:")
    total_params = sum(p.numel() for p in traj_model.parameters())
    trainable_params = sum(p.numel() for p in traj_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    print(f"  - 总参数量: {total_params:,}")
    print(f"  - 可训练参数: {trainable_params:,}")
    print(f"  - 冻结参数: {frozen_params:,}")
    print(f"  - 可训练比例: {trainable_params / total_params * 100:.2f}%")

    # 6. GPU 转换测试
    print("\n6. GPU 转换测试:")
    if torch.cuda.is_available():
        print(f"  GPU 可用: {torch.cuda.get_device_name(0)}")
        print(f"  当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # 转移到 GPU
        traj_model.to(dtype=torch.bfloat16, device='cuda')
        print(f"  ✓ 模型已转移到 GPU")
        print(f"  转移后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # 前向传播测试
        print("\n7. 前向传播测试:")
        batch_size = 2
        print(f"  创建测试输入 (batch_size={batch_size})...")

        # 模拟输入
        test_inputs = {
            'img': torch.randn(batch_size, 5, 3, 224, 224).to('cuda'),
            'coarse_waypoints': torch.randn(batch_size, 4).to('cuda')
        }

        print("  运行前向传播...")
        with torch.no_grad():
            output = traj_model(test_inputs, None)

        print(f"  ✓ 前向传播成功")
        print(f"    - 输出形状: {output.shape}")
        print(f"    - 输出范围: [{output.min():.3f}, {output.max():.3f}]")
        print(f"    - 当前显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

        # 清理
        del traj_model, traj_weights, test_inputs, output
        torch.cuda.empty_cache()
        print(f"    - 清理后显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("  ⚠ GPU 不可用，跳过 GPU 测试")

    print("\n" + "=" * 70)
    print("✅ 轨迹模型验证完成！")
    print("=" * 70)

except Exception as e:
    print(f"\n✗ 轨迹模型验证失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
