# TravelUAV Model Wrapper 接口详解
## 1. prepare_inputs() 方法
### 方法签名:


def prepare_inputs(self, episodes, target_positions, assist_notices=None)
### 输入参数:

episodes - List[List[Dict]]
- 类型: 列表的列表，每个元素是一个episode的历史观测序列
- 结构: episodes[i] 表示第i个batch的episode历史
- 每个观测帧包含:
```python
{
    'rgb': [numpy.ndarray],  # 多视角RGB图像列表，shape: [N_views, H, W, 3]
    'depth': [...],          # 深度图（可选）
    'instruction': str,      # 自然语言指令，如 "Fly to the red ball"
    'sensors': {
        'imu': {
            'rotation': numpy.ndarray  # 旋转矩阵 [3, 3]
        },
        'state': {
            'position': [x, y, z],     # 世界坐标系位置
            'orientation': [qw, qx, qy, qz]  # 四元数朝向
        }
    }
}
```
target_positions - List[numpy.ndarray]
- 类型: 目标位置列表
- 格式: [numpy.array([x, y, z]), ...]
- 坐标系: 世界坐标系

assist_notices - List[str] (可选)
- 类型: 助手提示列表
- 示例: ["cruise", "take off", "偏左了"]
- 输出返回:

返回一个元组 (inputs_device, rot_to_targets)

inputs_device - Dict[str, torch.Tensor]
这是一个字典，包含模型推理所需的所有输入：
```python

{
    # 文本输入
    'input_ids': torch.Tensor,        # shape: [batch_size, seq_len]
    'labels': torch.Tensor,           # shape: [batch_size, seq_len]
    'attention_mask': torch.Tensor,   # shape: [batch_size, seq_len]
    
    # 视觉输入
    'images': List[torch.Tensor],     # 每个元素 shape: [N_views, 3, H, W]
    
    # 历史轨迹
    'historys': List[torch.Tensor],   # 每个元素 shape: [N_history_points * 3]
                                      # 展平的历史航点序列
    
    # 朝向信息
    'orientations': torch.Tensor,     # shape: [batch_size, 3]
                                      # 相对于起点的欧拉角 (pitch, roll, yaw)
    
    # 提示文本
    'prompts': List[str],             # 原始指令文本列表
    
    # 控制标志
    'return_waypoints': True,         # 是否返回航点
    'use_cache': False                # 是否使用KV缓存
}
```
关键处理逻辑:

1. 坐标变换: 将世界坐标系转换为相对于起点的局部坐标系
2. 旋转对齐: 计算旋转矩阵 rotation_to_target，使x轴指向目标
3. 历史轨迹: 提取并变换历史航点到局部坐标系
4. 文本构造: 构造包含阶段信息、位移、位置的提示文本
```python
Stage: cruise
Previous displacement: 0.5,0.2,-0.1
Current position: 10.2,5.3,1.5
Current image: <image>
Instruction: Fly to the red ball
```
rot_to_targets - List[numpy.ndarray]
- 类型: 旋转矩阵列表
- 格式: 每个元素是 [3, 3] 的旋转矩阵
- 作用: 将局部坐标系对齐到目标方向（x轴指向目标）
---
## 2. run() 方法
### 方法签名:


def run(self, inputs, episodes, rot_to_targets)
### 输入参数:

inputs: 来自 prepare_inputs() 的 inputs_device
episodes: 同上
rot_to_targets: 同上
### 输出返回:

refined_waypoints - List[numpy.ndarray]
- 类型: 精炼后的航点列表
- 格式: [numpy.array([x, y, z]), numpy.array([x, y, z]), ...]
- 长度: batch_size
- 坐标系: 世界坐标系（已转换回去）
- 含义: 每个元素是下一步要飞往的目标航点
### 内部处理流程:

```python
# 第一阶段：LLM模型预测粗略航点
waypoints_llm = self.model(**inputs)  # shape: [batch_size, 4]
# 输出格式: [direction_x, direction_y, direction_z, distance]

# 归一化方向并乘以距离
waypoint_new = waypoint[:3] / norm(waypoint[:3]) * waypoint[3]
# 结果: [x, y, z] 在局部坐标系

# 第二阶段：Trajectory模型精炼航点
refined_waypoints = self.traj_model(...)  # shape: [batch_size, 3]
# 输出格式: [dx, dy, dz] 在无人机局部坐标系

# 第三阶段：转换到世界坐标系
waypoint_world = rot @ waypoint + current_position
# 最终输出: [x, y, z] 在世界坐标系
```
### 示例输出:


[
    array([12.5, 8.3, 1.8]),  # 第1个无人机的下一个航点
    array([5.2, -3.1, 2.0]),  # 第2个无人机的下一个航点
    ...
]
---
## 3. predict_done() 方法
### 方法签名:


def predict_done(self, episodes, object_infos)
### 输入参数:

episodes - List[List[Dict]]
- 同 prepare_inputs() 的 episodes

object_infos - List[Dict]
- 类型: 目标物体信息列表
- 结构:
```python
{
    'object_name': str,        # 目标物体名称，如 "red ball"
    'object_category': str,    # 物体类别
    'target_position': [x, y, z]  # 目标位置
}
```
### 输出返回:

prediction_dones - List[bool]
- 类型: 布尔值列表
- 长度: batch_size
- 含义: 每个元素表示对应的无人机是否已到达目标
- 判断依据: 使用 GroundingDINO 模型检测当前视野中是否出现目标物体
### 内部处理逻辑:

```python
# 使用 DinoMonitor 单例
dino_monitor = DinoMonitor.get_instance()

for i in range(len(episodes)):
    # 提取当前帧的RGB图像
    current_rgb = episodes[i][-1]['rgb']
    
    # 使用 GroundingDINO 检测目标物体
    detection_result = dino_monitor.get_dino_results(
        episode=episodes[i],
        object_info=object_infos[i]
    )
    
    # detection_result 是布尔值:
    # True: 检测到目标物体，认为已到达
    # False: 未检测到，继续导航
    
    prediction_dones.append(detection_result)
```
示例输出:

[False, False, True, False]  # 第3个无人机已到达目标
---
## 完整数据流示意图
```bash
┌─────────────────────────────────────────────────────────────┐
│                    eval.py 主循环                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  1. prepare_inputs()                                         │
│     输入:                                                     │
│       - episodes: [                                          │
│           [{rgb, depth, instruction, sensors}, ...],  # ep1 │
│           [{rgb, depth, instruction, sensors}, ...]   # ep2 │
│         ]                                                    │
│       - target_positions: [array([x,y,z]), ...]             │
│       - assist_notices: ["cruise", ...]                     │
│                                                              │
│     输出:                                                     │
│       - inputs_device: {                                     │
│           input_ids, images, historys, orientations, ...    │
│         }                                                    │
│       - rot_to_targets: [rotation_matrix_3x3, ...]          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  2. run()                                                    │
│     输入:                                                     │
│       - inputs_device (from step 1)                         │
│       - episodes (same as above)                            │
│       - rot_to_targets (from step 1)                        │
│                                                              │
│     内部流程:                                                 │
│       LLM Model → [dir_x, dir_y, dir_z, dist]              │
│       Traj Model → [dx, dy, dz] (局部坐标)                  │
│       Transform → [x, y, z] (世界坐标)                      │
│                                                              │
│     输出:                                                     │
│       - refined_waypoints: [                                │
│           array([12.5, 8.3, 1.8]),  # 世界坐标系            │
│           array([5.2, -3.1, 2.0]),                          │
│           ...                                               │
│         ]                                                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  eval_env.makeActions(refined_waypoints)                    │
│  → 无人机飞向航点                                            │
│  → 获取新观测 outputs = eval_env.get_obs()                  │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  3. predict_done()                                           │
│     输入:                                                     │
│       - episodes: (更新后的episode，包含新观测)              │
│       - object_infos: [                                      │
│           {object_name: "red ball", ...},                   │
│           ...                                               │
│         ]                                                    │
│                                                              │
│     内部流程:                                                 │
│       GroundingDINO 检测当前视野中的目标物体                 │
│                                                              │
│     输出:                                                     │
│       - prediction_dones: [False, False, True, False]       │
└─────────────────────────────────────────────────────────────┘
```
---

## 关键要点总结
### 坐标系变换
1. 输入: 世界坐标系
2. prepare_inputs: 转换到相对起点的局部坐标系，并旋转对齐目标
3. run (LLM): 输出局部坐标系的方向+距离
4. run (Traj): 输出无人机局部坐标系的增量
5. run (最终): 转换回世界坐标系
### 数据维度
- Batch维度: 所有方法都支持批处理，batch_size 通常为 8
- 历史维度: episodes[i] 包含从起点到当前的所有观测帧
- 视角维度: RGB图像包含多个视角（前、左、右、后、下）
### 模型架构
- LLM模型: 基于LLaMA的视觉-语言-动作模型，输出粗略航点
- Traj模型: 基于视觉的轨迹精炼器，输出精确的局部增量
- DINO模型: GroundingDINO用于目标检测，判断是否到达