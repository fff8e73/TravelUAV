#!/bin/bash
# ============================================================================
# TravelUAV Qwen3-VL 测试脚本 - 集成版
# ============================================================================
# 功能：运行所有 Qwen3-VL 相关测试
# 作者：Claude Code
# 创建时间：2026-01-16
# ============================================================================

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 项目根目录
ROOT_DIR="/home/yyx/TravelUAV"

# 默认测试选项
TEST_GPU=false
TEST_TRAJ=false
TEST_INTEGRATION=false
TEST_PERFORMANCE=false
TEST_ALL=false

# 显示帮助信息
show_help() {
    echo "=========================================================================="
    echo "TravelUAV Qwen3-VL 测试脚本"
    echo "=========================================================================="
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -g, --gpu          运行 GPU 环境测试"
    echo "  -t, --traj         运行轨迹模型测试"
    echo "  -i, --integration  运行集成测试（两阶段架构）"
    echo "  -p, --performance  运行性能对比测试"
    echo "  -a, --all          运行所有测试"
    echo "  -h, --help         显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 -a               # 运行所有测试"
    echo "  $0 -g -t            # 运行 GPU 和轨迹模型测试"
    echo "  $0 --integration    # 仅运行集成测试"
    echo ""
    echo "=========================================================================="
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gpu)
            TEST_GPU=true
            shift
            ;;
        -t|--traj)
            TEST_TRAJ=true
            shift
            ;;
        -i|--integration)
            TEST_INTEGRATION=true
            shift
            ;;
        -p|--performance)
            TEST_PERFORMANCE=true
            shift
            ;;
        -a|--all)
            TEST_ALL=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 如果没有指定任何测试，默认运行所有测试
if [[ "$TEST_GPU" == false && "$TEST_TRAJ" == false && "$TEST_INTEGRATION" == false && "$TEST_PERFORMANCE" == false && "$TEST_ALL" == false ]]; then
    TEST_ALL=true
fi

# 如果指定了 --all，设置所有测试为 true
if [[ "$TEST_ALL" == true ]]; then
    TEST_GPU=true
    TEST_TRAJ=true
    TEST_INTEGRATION=true
    TEST_PERFORMANCE=true
fi

# 激活 conda 环境
echo -e "${BLUE}激活 conda 环境...${NC}"
conda activate llamauav_sm_120
echo -e "${GREEN}✓ 环境激活成功${NC}"

# 1. GPU 环境测试
run_gpu_test() {
    echo ""
    echo "=========================================================================="
    echo -e "${BLUE}测试 1: GPU 环境测试${NC}"
    echo "=========================================================================="
    echo "验证 Blackwell GPU (RTX PRO 6000) 支持和 Qwen3-VL 模型"
    echo ""

    python3 $ROOT_DIR/test_gpu_mode.py
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ GPU 环境测试通过${NC}"
        return 0
    else
        echo -e "${RED}✗ GPU 环境测试失败 (退出码: $exit_code)${NC}"
        return 1
    fi
}

# 2. 轨迹模型测试
run_traj_test() {
    echo ""
    echo "=========================================================================="
    echo -e "${BLUE}测试 2: 轨迹模型测试${NC}"
    echo "=========================================================================="
    echo "验证 VisionTrajectoryGenerator (阶段 2) 的加载和基本功能"
    echo ""

    python3 $ROOT_DIR/test_traj_model.py
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ 轨迹模型测试通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 轨迹模型测试失败 (退出码: $exit_code)${NC}"
        return 1
    fi
}

# 3. 集成测试
run_integration_test() {
    echo ""
    echo "=========================================================================="
    echo -e "${BLUE}测试 3: 集成测试（两阶段架构）${NC}"
    echo "=========================================================================="
    echo "验证完整的两阶段推理流程：Qwen3-VL + 专用轨迹模型"
    echo ""

    python3 $ROOT_DIR/test_qwen3vl_integration.py
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ 集成测试通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 集成测试失败 (退出码: $exit_code)${NC}"
        return 1
    fi
}

# 4. 性能对比测试
run_performance_test() {
    echo ""
    echo "=========================================================================="
    echo -e "${BLUE}测试 4: 性能对比测试${NC}"
    echo "=========================================================================="
    echo "对比测试：两阶段 vs 单阶段架构"
    echo ""

    python3 $ROOT_DIR/test_performance_comparison.py
    local exit_code=$?

    if [[ $exit_code -eq 0 ]]; then
        echo -e "${GREEN}✓ 性能测试通过${NC}"
        return 0
    else
        echo -e "${RED}✗ 性能测试失败 (退出码: $exit_code)${NC}"
        return 1
    fi
}

# 主执行流程
main() {
    echo "=========================================================================="
    echo -e "${GREEN}TravelUAV Qwen3-VL 测试套件${NC}"
    echo "=========================================================================="
    echo ""
    echo "测试配置:"
    echo "  - GPU 环境测试:      $([[ "$TEST_GPU" == true ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo "  - 轨迹模型测试:      $([[ "$TEST_TRAJ" == true ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo "  - 集成测试:          $([[ "$TEST_INTEGRATION" == true ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo "  - 性能对比测试:      $([[ "$TEST_PERFORMANCE" == true ]] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}")"
    echo ""
    echo "=========================================================================="
    echo ""

    # 记录测试结果
    declare -A results
    local failed_tests=()

    # 运行选中的测试
    if [[ "$TEST_GPU" == true ]]; then
        if run_gpu_test; then
            results["GPU"]=0
        else
            results["GPU"]=1
            failed_tests+=("GPU")
        fi
    fi

    if [[ "$TEST_TRAJ" == true ]]; then
        if run_traj_test; then
            results["TRAJ"]=0
        else
            results["TRAJ"]=1
            failed_tests+=("轨迹模型")
        fi
    fi

    if [[ "$TEST_INTEGRATION" == true ]]; then
        if run_integration_test; then
            results["INTEGRATION"]=0
        else
            results["INTEGRATION"]=1
            failed_tests+=("集成测试")
        fi
    fi

    if [[ "$TEST_PERFORMANCE" == true ]]; then
        if run_performance_test; then
            results["PERFORMANCE"]=0
        else
            results["PERFORMANCE"]=1
            failed_tests+=("性能测试")
        fi
    fi

    # 输出总结
    echo ""
    echo "=========================================================================="
    echo -e "${BLUE}测试总结${NC}"
    echo "=========================================================================="
    echo ""

    local all_passed=true
    for test_name in "${!results[@]}"; do
        if [[ ${results[$test_name]} -eq 0 ]]; then
            echo -e "  ${GREEN}✓${NC} $test_name: 通过"
        else
            echo -e "  ${RED}✗${NC} $test_name: 失败"
            all_passed=false
        fi
    done

    echo ""
    echo "=========================================================================="

    if [[ "$all_passed" == true ]]; then
        echo -e "${GREEN}✅ 所有测试通过！${NC}"
        echo ""
        echo "下一步:"
        echo "  1. 启动 AirSim 仿真服务器:"
        echo "     cd $ROOT_DIR/airsim_plugin"
        echo "     python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/extracted"
        echo ""
        echo "  2. 运行完整评估:"
        echo "     bash $ROOT_DIR/scripts/eval_qwen.sh"
        echo ""
        echo "  3. 计算评估指标:"
        echo "     bash $ROOT_DIR/scripts/metric.sh"
        echo ""
        echo "=========================================================================="
        return 0
    else
        echo -e "${RED}⚠️  部分测试失败${NC}"
        echo ""
        echo "失败的测试: ${failed_tests[*]}"
        echo ""
        echo "请根据错误信息修复问题后重试。"
        echo "=========================================================================="
        return 1
    fi
}

# 执行主函数
main
