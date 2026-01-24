#!/bin/bash

# TravelUAV HTTP 服务端启动脚本
# 用于将 TravelUAV 项目作为服务端暴露给 VLA_Habitat 客户端调用

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}TravelUAV HTTP 服务端启动脚本${NC}"
echo -e "${GREEN}========================================${NC}"

# 检查 conda 环境
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}警告: 未检测到 conda 环境${NC}"
    echo -e "${YELLOW}请先激活环境: conda activate llamauav_sm_120${NC}"
    exit 1
fi

echo -e "当前环境: ${GREEN}${CONDA_DEFAULT_ENV}${NC}"

# 检查项目目录
PROJECT_DIR="/home/yyx/TravelUAV"
if [ ! -d "$PROJECT_DIR" ]; then
    echo -e "${RED}错误: 项目目录不存在: $PROJECT_DIR${NC}"
    exit 1
fi

cd "$PROJECT_DIR"
echo -e "项目目录: ${GREEN}${PROJECT_DIR}${NC}"

# 检查模型文件
MODEL_PATH="$PROJECT_DIR/Model/Qwen3-VL-4B-Instruct"
TRAJ_MODEL_PATH="$PROJECT_DIR/Model/LLaMA-UAV/work_dirs/traj_predictor_bs_128_drop_0.1_lr_5e-4"

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: Qwen3-VL 模型不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请下载模型: huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir $MODEL_PATH${NC}"
    exit 1
fi

if [ ! -d "$TRAJ_MODEL_PATH" ]; then
    echo -e "${RED}错误: 轨迹模型不存在: $TRAJ_MODEL_PATH${NC}"
    exit 1
fi

echo -e "模型路径: ${GREEN}${MODEL_PATH}${NC}"
echo -e "轨迹模型路径: ${GREEN}${TRAJ_MODEL_PATH}${NC}"

# 解析命令行参数
HOST="127.0.0.1"
PORT="9000"
USE_4BIT=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --no-4bit)
            USE_4BIT=false
            shift
            ;;
        --help)
            echo ""
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --host HOST      服务器主机地址 (默认: 127.0.0.1)"
            echo "  --port PORT      服务器端口 (默认: 9000)"
            echo "  --no-4bit        禁用 4-bit 量化"
            echo "  --help           显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                          # 使用默认配置启动"
            echo "  $0 --port 8080              # 使用端口 8080"
            echo "  $0 --host 0.0.0.0 --port 80 # 监听所有接口，端口 80"
            echo "  $0 --no-4bit                # 禁用 4-bit 量化"
            exit 0
            ;;
        *)
            echo -e "${RED}未知参数: $1${NC}"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${GREEN}启动配置:${NC}"
echo "  主机: $HOST"
echo "  端口: $PORT"
echo "  4-bit 量化: $USE_4BIT"

# 构建启动命令
CMD="python scripts/eval/server_traveluav.py --host $HOST --port $PORT --model_path $MODEL_PATH --traj_model_path $TRAJ_MODEL_PATH"

if [ "$USE_4BIT" = "false" ]; then
    CMD="$CMD --no_4bit"
fi

echo ""
echo -e "${GREEN}启动命令:${NC}"
echo "  $CMD"
echo ""

# 检查端口是否被占用
echo -e "${YELLOW}检查端口 $PORT 是否被占用...${NC}"
if netstat -tuln 2>/dev/null | grep -q ":$PORT "; then
    echo -e "${RED}错误: 端口 $PORT 已被占用${NC}"
    echo -e "${YELLOW}请使用其他端口或关闭占用该端口的进程${NC}"
    exit 1
fi

echo -e "${GREEN}端口 $PORT 可用${NC}"

# 启动服务器
echo ""
echo -e "${GREEN}正在启动 TravelUAV HTTP 服务器...${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

exec $CMD
