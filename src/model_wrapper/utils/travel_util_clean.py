"""
TravelUAV 工具函数 - 清理版本
移除了 llamavid 和 llava 依赖，适配 Qwen3-VL 集成
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Sequence
import torch
import numpy as np
from scipy.spatial.transform import Rotation as R

# Qwen3-VL 集成不需要这些函数
# 保留占位符以兼容旧代码

LLAMAVID_AVAILABLE = False


def generate_vision_tower_config(vision_tower, image_processor):
    """
    生成视觉塔配置（用于轨迹预测模型）
    """
    default_vision_config = {
        "model_type": "clip",
        "hidden_act": "silu",
        "hidden_size": 4096,
        "image_aspect_ratio": "square",
        "image_grid_pinpoints": None,
        "image_processor": "./llamavid/processor/clip-patch14-224",
        "initializer_range": 0.02,
        "intermediate_size": 11008,
        "max_position_embeddings": 4096,
        "max_token": 2048,
        "mm_hidden_size": 1408,
        "mm_projector_type": "mlp2x_gelu",
        "mm_use_im_patch_token": False,
        "mm_use_im_start_end": False,
        "mm_vision_select_feature": "patch",
        "mm_vision_select_layer": -2,
        "mm_vision_tower": "./model_zoo/LAVIS/eva_vit_g.pth",
        "torch_dtype": "float16"
    }
    default_vision_config['image_processor'] = image_processor
    default_vision_config['mm_vision_tower'] = vision_tower

    # 创建配置文件
    cf_path = os.path.join(os.path.split(vision_tower)[0], 'config.json')
    os.makedirs(os.path.dirname(cf_path), exist_ok=True)
    with open(cf_path, 'w') as f:
        json.dump(default_vision_config, f, indent=2)

    return cf_path


def prepare_data_to_traj_model(episodes, waypoints, image_processor, rot_to_targets=None):
    """
    准备轨迹预测模型的输入数据
    """
    image_list = []
    target_list = []

    for i in range(len(episodes)):
        info = episodes[i]
        rot_to_target = None

        if rot_to_targets is not None:
            if rot_to_targets[i] is not None:
                rot_to_target = rot_to_targets[i]

        target = waypoints[i][0:3]
        rot_0 = info[0]['sensors']['imu']["rotation"]
        rot = info[-1]['sensors']['imu']["rotation"]

        if rot_to_target is not None:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(rot_to_target) @ np.array(target)
        else:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(target)

        image_list.append(info[-1]['rgb'][0])
        target_list.append(target)

    images = np.stack(image_list, axis=0)
    image = image_processor.preprocess(images, return_tensors='pt')['pixel_values']

    # Ensure image is 4D: [batch, channels, height, width]
    # CLIPImageProcessor returns [batch, channels, height, width] when return_tensors='pt'
    # But we need to ensure it's properly shaped for the trajectory model
    if image.dim() == 3:
        # If somehow 3D, add batch dimension
        image = image.unsqueeze(0)

    # 构建输入字典
    inputs = {
        'img': image,
        'target': torch.tensor(np.stack(target_list, axis=0), dtype=torch.float32)
    }

    return inputs


def transform_to_world(waypoints, episodes):
    """
    将相对坐标转换为世界坐标
    waypoints can be either:
    - 1D array of shape (3,) for a single waypoint
    - 2D array of shape (n, 3) for multiple waypoints
    - 1D array of shape (3*n,) for multiple waypoints (flattened)
    """
    waypoints_world = []

    for i, waypoint in enumerate(waypoints):
        ep = episodes[i]
        pos = ep[-1]["sensors"]["state"]["position"]
        rot = ep[-1]["sensors"]["imu"]["rotation"]

        # Handle different waypoint shapes
        waypoint = np.array(waypoint)
        if waypoint.ndim == 1:
            # 1D array - could be (3,) or (3*n,)
            if len(waypoint) % 3 == 0:
                # Multiple waypoints, reshape to (n, 3)
                waypoint = waypoint.reshape(-1, 3)
            else:
                # Single waypoint
                waypoint = waypoint.reshape(1, 3)

        # 转换到世界坐标系 for each waypoint
        waypoint_world = []
        for wp in waypoint:
            wp_world = np.array(rot) @ np.array(wp) + np.asarray(pos)
            waypoint_world.append(wp_world)

        waypoints_world.append(np.array(waypoint_world))

    return np.array(waypoints_world)


# 兼容性占位符
def load_model(args):
    """已废弃"""
    raise RuntimeError(
        "load_model() 已废弃。\n"
        "请使用 src.model_wrapper.qwen3vl_hf.Qwen3VLHFWrapper"
    )


def load_traj_model(model_args):
    """已废弃 - 轨迹模型在 Qwen3VLHFWrapper 中加载"""
    raise RuntimeError(
        "load_traj_model() 已废弃。\n"
        "请使用 src.model_wrapper.qwen3vl_hf.Qwen3VLHFWrapper"
    )


# 导出的类（用于轨迹预测）
from types import SimpleNamespace

VisionTrajectoryGenerator = None  # 在 Qwen3VLHFWrapper 中直接使用原实现

# 兼容性常量
IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN = '<image>'
DEFAULT_IM_START_TOKEN = '<im_start>'
DEFAULT_IM_END_TOKEN = '<im_end>'
WAYPOINT_INPUT_TOKEN = 2
WAYPOINT_LABEL_TOKEN = 3
DEFAULT_HISTORY_TOKEN = '<history>'
DEFAULT_WP_TOKEN = '<wp>'