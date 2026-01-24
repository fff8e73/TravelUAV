#!/bin/bash
# 使用 Qwen3-VL 运行评测脚本
# 注意：
# 1. 请确保已激活 conda 环境: conda activate llamauav_sm_120
# 2. 请确保 AirSim 仿真服务器已启动:
#    cd /home/yyx/TravelUAV/airsim_plugin
#    python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/extracted
# 3. 配置 HuggingFace 镜像（避免网络超时）
#    export HF_ENDPOINT=https://hf-mirror.com
# 4. 配置 Vulkan 驱动（解决 GPU 厂商 ID 识别问题）
#    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
#
# 使用方法：
#   bash scripts/eval_qwen.sh              # 前台运行（显示在终端）
#   bash scripts/eval_qwen.sh --background # 后台运行（SSH断开不会终止）
#   bash scripts/eval_qwen.sh --resume     # 断点续评（从上次中断位置继续）

# 配置 HuggingFace 镜像（避免 bert-base-uncased 下载超时）
export HF_ENDPOINT=https://hf-mirror.com
echo "HuggingFace 镜像: $HF_ENDPOINT"

# 配置 Vulkan 驱动（解决 GPU 厂商 ID 识别问题）
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
echo "Vulkan ICD: $VK_ICD_FILENAMES"

root_dir=/home/yyx/TravelUAV
data_dir=/sim/data/TravelUAV_data/TravelUAV_data_json/data
model_dir=/home/yyx/TravelUAV/Model

# 创建日志目录
log_dir=/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs
mkdir -p $log_dir

# 评估结果保存路径
eval_save_path=/sim/data/TravelUAV_data/eval_closeloop/eval_qwen

# 检查是否存在之前的评估结果
check_resume() {
    if [ -d "$eval_save_path" ] && [ "$(ls -A $eval_save_path 2>/dev/null)" ]; then
        echo "✅ 检测到之前的评估结果"
        echo "   目录: $eval_save_path"
        echo "   已完成的任务数: $(ls -1 $eval_save_path | wc -l)"
        echo ""
        return 0
    else
        echo "❌ 未检测到之前的评估结果，将从头开始"
        echo ""
        return 1
    fi
}

# 生成带时间戳的日志文件名
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file=$log_dir/eval_qwen_${timestamp}.log

# 检查运行模式
resume_mode=false
background_mode=false

# 解析命令行参数（支持多个参数）
for arg in "$@"; do
    case "$arg" in
        --background)
            background_mode=true
            ;;
        --resume)
            resume_mode=true
            ;;
    esac
done

# 显示运行模式
echo "========================================"
if [ "$background_mode" = true ] && [ "$resume_mode" = true ]; then
    echo "开始后台断点续评模式"
elif [ "$background_mode" = true ]; then
    echo "开始后台运行评估脚本"
elif [ "$resume_mode" = true ]; then
    echo "开始断点续评模式"
else
    echo "开始运行评估脚本"
fi
echo "日志文件: $log_file"
echo "========================================"
echo ""

# 检查断点续评
if [ "$resume_mode" = true ]; then
    check_resume
    echo "将从上次中断的位置继续评估..."
    echo ""
fi

# 显示后台运行提示
if [ "$background_mode" = true ]; then
    echo "脚本将在后台运行，即使 SSH 断开也不会终止"
    echo "使用以下命令查看日志："
    echo "  tail -f $log_file"
    echo ""
    echo "使用以下命令查看进程："
    echo "  ps aux | grep eval_qwen"
    echo ""
    echo "使用以下命令停止脚本："
    echo "  pkill -f eval_qwen.py"
    echo ""
fi

# 运行评测命令（输出到日志文件和终端）
# 参数说明：
# - run_type: eval (评估模式)
# - name: 任务名称
# - gpu_id: GPU 编号
# - simulator_tool_port: 仿真服务器端口（需与服务器端口一致）
# - batchSize: 批次大小（根据显存调整）
# - always_help: 是否启用专家辅助
# - use_gt: 是否使用真值信息
# - maxWaypoints: 最大导航航点数
# - model_path: 模型路径（Qwen3-VL 模型）
# - model_base: 基础语言模型（Qwen3-VL 不需要）
# - vision_tower: 视觉塔模型路径
# - image_processor: 图像处理器路径
# - traj_model_path: 轨迹预测模型路径
# - eval_json_path: 评估集配置文件
# - map_spawn_area_json_path: 地图生成区域配置
# - object_name_json_path: 物体描述配置
# - groundingdino_config: GroundingDINO 配置文件
# - groundingdino_model_path: GroundingDINO 模型路径

# 根据模式运行评测
if [ "$background_mode" = true ]; then
    # 后台运行模式（使用 nohup 和 disown）
    nohup bash -c "
        export HF_ENDPOINT=https://hf-mirror.com
        export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
        CUDA_VISIBLE_DEVICES=0 python3 -u $root_dir/src/vlnce_src/eval_qwen.py \
            --run_type eval \
            --name TravelLLM_Qwen \
            --gpu_id 0 \
            --simulator_tool_port 30000 \
            --DDP_MASTER_PORT 80005 \
            --batchSize 2 \
            --always_help True \
            --use_gt True \
            --maxWaypoints 200 \
            --dataset_path /sim/data/TravelUAV_data/extracted/ \
            --eval_save_path /sim/data/TravelUAV_data/eval_closeloop/eval_qwen \
            --model_path $model_dir/Qwen3-VL-4B-Instruct \
            --model_base \\\\\"\\\\\" \
            --vision_tower $model_dir/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
            --image_processor $model_dir/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224 \
            --traj_model_path $model_dir/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
            --eval_json_path $data_dir/uav_dataset/seen_valset.json \
            --map_spawn_area_json_path $data_dir/meta/map_spawnarea_info.json \
            --object_name_json_path $data_dir/meta/object_description.json \
            --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
            --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth \
            >> $log_file 2>&1
    " &

    # 获取后台进程的 PID
    BG_PID=$!

    # 将后台进程放入后台，不受 SIGHUP 影响
    disown $BG_PID

    echo "✅ 评估脚本已在后台运行"
    echo "   进程 PID: $BG_PID"
    echo "   日志文件: $log_file"
    echo ""
    echo "即使关闭 SSH 终端，脚本也会继续运行"
    echo ""
else
    # 前台运行模式（使用 tee 同时输出到终端和日志文件）
    export HF_ENDPOINT=https://hf-mirror.com
    export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

    CUDA_VISIBLE_DEVICES=0 python3 -u $root_dir/src/vlnce_src/eval_qwen.py \
        --run_type eval \
        --name TravelLLM_Qwen \
        --gpu_id 0 \
        --simulator_tool_port 30000 \
        --DDP_MASTER_PORT 80005 \
        --batchSize 2 \
        --always_help True \
        --use_gt True \
        --maxWaypoints 200 \
        --dataset_path /sim/data/TravelUAV_data/extracted/ \
        --eval_save_path /sim/data/TravelUAV_data/eval_closeloop/eval_qwen \
        --model_path $model_dir/Qwen3-VL-4B-Instruct \
        --model_base \"\" \
        --vision_tower $model_dir/LLaMA-UAV/model_zoo/LAVIS/eva_vit_g.pth \
        --image_processor $model_dir/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224 \
        --traj_model_path $model_dir/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4 \
        --eval_json_path $data_dir/uav_dataset/seen_valset.json \
        --map_spawn_area_json_path $data_dir/meta/map_spawnarea_info.json \
        --object_name_json_path $data_dir/meta/object_description.json \
        --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
        --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth 2>&1 | tee $log_file

    # 检查退出状态
    exit_code=${PIPESTATUS[0]}

    echo ""
    echo "========================================"
    if [ $exit_code -eq 0 ]; then
        echo "✅ 评估完成！"
    else
        echo "❌ 评估失败（退出码: $exit_code）"
        echo ""
        echo "如需断点续评，请运行:"
        echo "  bash scripts/eval_qwen.sh --resume"
    fi
    echo "日志文件: $log_file"
    echo "========================================"
fi

# 清理（如果启动了服务器）
# kill $SERVER_PID
