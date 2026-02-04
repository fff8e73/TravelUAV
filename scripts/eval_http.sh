#!/bin/bash
# HTTP Client Evaluation Script for TravelUAV
# 使用外部模型服务器进行评测，无需加载本地模型

# ============================================================
# 重要提示：运行此脚本前，请先启动 AirSim 仿真服务器！
# ============================================================
# 在另一个终端中运行：
# cd ~/TravelUAV/airsim_plugin
# python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
# ============================================================

root_dir=/home/yyx/TravelUAV # TravelUAV directory
data_dir=/sim/data/TravelUAV_data/TravelUAV_data_json/data

# HTTP Client 配置
server_url="http://127.0.0.1:9010"  # 外部模型服务器地址
timeout=300  # HTTP请求超时时间（秒）

# 评测配置参数说明：
# run_type 运行类型：指定为"eval"（评估模式）
# name 评估任务名称
# gpu_id 指定使用的GPU编号（用于环境渲染，不用于模型推理）
# simulator_tool_port 仿真服务器端口：需与AirVLNSimulatorServerTool.py启动时的--port保持一致
# DDP_MASTER_PORT DDP分布式训练的主端口：单卡评估时可忽略
# batchSize 批次大小：每次同时评估的导航任务数量（根据仿真环境性能调整）
# always_help 是否始终启用"专家辅助"：True表示遇到困难会触发专家提示，False则完全自主
# use_gt 是否使用真值信息（Ground Truth）：True表示加载场景真值辅助判断
# maxWaypoints 最大导航航点数：无人机一次导航任务的最大规划航点数量
# dataset_path 评估数据集路径：存放导航场景的回放数据、图像数据
# eval_save_path 评估结果保存路径：评估日志、轨迹数据、失败案例会存到这里
# eval_json_path 评估集配置文件：存放评估场景的起点、终点、任务描述等信息
# map_spawn_area_json_path 地图生成区域配置：存放各个场景的无人机spawn点信息
# object_name_json_path 物体描述配置：存放场景中各类物体的名称和描述

python -u $root_dir/src/vlnce_src/eval_http.py \
    --server_url $server_url \
    --timeout $timeout \
    --run_type eval \
    --name TravelLLM_HTTP \
    --gpu_id 0 \
    --simulator_tool_port 30000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 2 \
    --always_help False \
    --use_gt False \
    --maxWaypoints 200 \
    --activate_maps NewYorkCity \
    --dataset_path /sim/data/TravelUAV_data/extracted/ \
    --eval_save_path /sim/data/TravelUAV_data/eval_closeloop/eval_http_test \
    --eval_json_path $data_dir/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path $data_dir/meta/map_spawnarea_info.json \
    --object_name_json_path $data_dir/meta/object_description.json
