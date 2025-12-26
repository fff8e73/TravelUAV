#!/bin/bash
# change the dataset_path to your own path

root_dir=/home/yyx/TravelUAV # TravelUAV directory
data_dir=/home/yyx/TravelUAV/data/TravelUAV_data_json/data
model_dir=/home/yyx/TravelUAV/Model/LLaMA-UAV

# python -u：-u参数禁用输出缓冲，让终端实时显示脚本运行日志（方便查看评估进度/报错）
# run_type 运行类型：指定为"eval"（评估模式），可选值通常还有"train"（训练）、"test"（测试）
# name 评估任务名称
# gpu_id 指定使用的GPU编号
# simulator_tool_port 仿真服务器端口：需与AirVLNSimulatorServerTool.py启动时的--port保持一致（否则无法连接仿真环境）
# DDP_MASTER_PORT DDP分布式训练的主端口：若仅单卡评估，该参数可忽略；多卡时需指定未被占用的端口（避免端口冲突）
# batchSize 批次大小：每次同时评估2个导航任务（根据GPU显存调整，显存充足可改为4/8，显存不足则改为1）
# always_help 是否始终启用"专家辅助"：True表示无人机导航时遇到困难会触发专家提示（如避障建议），False则完全自主
# use_gt 是否使用真值信息（Ground Truth）：True表示评估时加载场景真值（如地图、障碍物位置），用于辅助判断导航是否正确
# maxWaypoints 最大导航航点数：无人机一次导航任务的最大规划航点数量（200足够覆盖绝大多数场景，避免因航点不足导致任务失败）
# dataset_path 评估数据集路径：存放导航场景的回放数据、图像数据
# eval_save_path 评估结果保存路径：评估日志、轨迹数据、失败案例会存到这里（后续给metrics.sh计算指标用）
# model_path 待评估的模型路径：Dagger训练后生成的最优模型（核心参数，必须指向正确的.pth/.ckpt模型文件）
# model_base 基础语言模型路径：Vicuna-7B-v1.5预训练权重（LLaMA-UAV的基础模型）
# vision_tower 视觉塔模型路径：EVA-ViT-G视觉编码器（用于处理无人机相机图像）
# image_processor 图像处理器路径：CLIP的图像预处理工具（用于将无人机图像转为模型可识别的格式）
# traj_model_path 轨迹预测模型路径：负责规划无人机导航轨迹的模型
# eval_json_path 评估集配置文件：存放评估场景的起点、终点、任务描述等信息（"seen_valset"表示"已见过的场景"，对应训练时的场景分布）
# map_spawn_area_json_path 地图生成区域配置：存放各个场景的无人机 spawn 点（起飞点）信息（确保场景与spawn点匹配）
# object_name_json_path 物体描述配置：存放场景中各类物体（如墙、树、建筑）的名称和描述（用于模型识别障碍物）
# groundingdino_config GroundingDINO配置文件：目标检测模型的配置（用于识别场景中的物体）
# groundingdino_model_path GroundingDINO模型路径：目标检测预训练权重（辅助无人机识别障碍物）
CUDA_VISIBLE_DEVICES=0 python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name TravelLLM \
    --gpu_id 0 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 2 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /sim/data/TravelUAV_data/extracted/ \
    --eval_save_path /sim/data/TravelUAV_data/eval_closeloop/eval_test \
    --model_path $model_dir/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32 \
    --model_base $model_dir/model_zoo/vicuna-7b-v1.5 \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --traj_model_path $model_dir/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
    --eval_json_path $data_dir/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path $data_dir/meta/map_spawnarea_info.json \
    --object_name_json_path $data_dir/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth