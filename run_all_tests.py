"""
===============================================================================
TravelUAV Qwen3-VL 测试套件 - 集成版
===============================================================================

功能：运行所有 Qwen3-VL 相关测试
作者：Claude Code
创建时间：2026-01-16

测试内容：
1. GPU 环境测试 - 验证 Blackwell GPU (RTX PRO 6000) 支持
2. 轨迹模型测试 - 验证 VisionTrajectoryGenerator (阶段 2)
3. 集成测试 - 验证完整的两阶段推理流程
4. 性能对比测试 - 对比两阶段 vs 单阶段架构
===============================================================================
"""

import sys
import subprocess
import argparse
from pathlib import Path

# 颜色定义
class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    NC = '\033[0m'  # No Color

# 项目根目录
ROOT_DIR = Path("/home/yyx/TravelUAV")

# 测试配置
TESTS = {
    "gpu": {
        "name": "GPU 环境测试",
        "description": "验证 Blackwell GPU (RTX PRO 6000) 支持和 Qwen3-VL 模型",
        "script": ROOT_DIR / "test_gpu_mode.py",
    },
    "traj": {
        "name": "轨迹模型测试",
        "description": "验证 VisionTrajectoryGenerator (阶段 2) 的加载和基本功能",
        "script": ROOT_DIR / "test_traj_model.py",
    },
    "integration": {
        "name": "集成测试（两阶段架构）",
        "description": "验证完整的两阶段推理流程：Qwen3-VL + 专用轨迹模型",
        "script": ROOT_DIR / "test_qwen3vl_integration.py",
    },
    "performance": {
        "name": "性能对比测试",
        "description": "对比测试：两阶段 vs 单阶段架构",
        "script": ROOT_DIR / "test_performance_comparison.py",
    },
}


def print_header(text: str):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"{Colors.BLUE}{text}{Colors.NC}")
    print("=" * 70)


def print_success(text: str):
    """打印成功信息"""
    print(f"{Colors.GREEN}✓ {text}{Colors.NC}")


def print_error(text: str):
    """打印错误信息"""
    print(f"{Colors.RED}✗ {text}{Colors.NC}")


def print_warning(text: str):
    """打印警告信息"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.NC}")


def run_test(test_key: str, test_config: dict) -> bool:
    """运行单个测试"""
    print_header(f"测试: {test_config['name']}")
    print(f"{test_config['description']}\n")

    script_path = test_config['script']
    if not script_path.exists():
        print_error(f"测试脚本不存在: {script_path}")
        return False

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT_DIR),
            capture_output=False,  # 输出直接显示到终端
            text=True,
        )
        if result.returncode == 0:
            print_success(f"{test_config['name']} 通过")
            return True
        else:
            print_error(f"{test_config['name']} 失败 (退出码: {result.returncode})")
            return False
    except Exception as e:
        print_error(f"{test_config['name']} 执行失败: {e}")
        return False


def show_help():
    """显示帮助信息"""
    print_header("TravelUAV Qwen3-VL 测试套件")
    print("\n用法: python run_all_tests.py [选项]\n")
    print("选项:")
    print("  -g, --gpu          运行 GPU 环境测试")
    print("  -t, --traj         运行轨迹模型测试")
    print("  -i, --integration  运行集成测试（两阶段架构）")
    print("  -p, --performance  运行性能对比测试")
    print("  -a, --all          运行所有测试")
    print("  -h, --help         显示此帮助信息\n")
    print("示例:")
    print("  python run_all_tests.py -a               # 运行所有测试")
    print("  python run_all_tests.py -g -t            # 运行 GPU 和轨迹模型测试")
    print("  python run_all_tests.py --integration    # 仅运行集成测试\n")
    print("=" * 70)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TravelUAV Qwen3-VL 测试套件",
        add_help=False,
    )
    parser.add_argument("-g", "--gpu", action="store_true", help="运行 GPU 环境测试")
    parser.add_argument("-t", "--traj", action="store_true", help="运行轨迹模型测试")
    parser.add_argument("-i", "--integration", action="store_true", help="运行集成测试")
    parser.add_argument("-p", "--performance", action="store_true", help="运行性能对比测试")
    parser.add_argument("-a", "--all", action="store_true", help="运行所有测试")
    parser.add_argument("-h", "--help", action="store_true", help="显示帮助信息")

    args = parser.parse_args()

    if args.help:
        show_help()
        sys.exit(0)

    # 确定要运行的测试
    tests_to_run = []
    if args.all or not any([args.gpu, args.traj, args.integration, args.performance]):
        tests_to_run = list(TESTS.keys())
    else:
        if args.gpu:
            tests_to_run.append("gpu")
        if args.traj:
            tests_to_run.append("traj")
        if args.integration:
            tests_to_run.append("integration")
        if args.performance:
            tests_to_run.append("performance")

    # 打印测试配置
    print_header("TravelUAV Qwen3-VL 测试套件")
    print("\n测试配置:")
    for key, config in TESTS.items():
        status = f"{Colors.GREEN}✓{Colors.NC}" if key in tests_to_run else f"{Colors.RED}✗{Colors.NC}"
        print(f"  {status} {config['name']}")
    print("\n" + "=" * 70)

    # 运行测试
    results = {}
    failed_tests = []

    for test_key in tests_to_run:
        test_config = TESTS[test_key]
        success = run_test(test_key, test_config)
        results[test_key] = success
        if not success:
            failed_tests.append(test_config['name'])

    # 输出总结
    print_header("测试总结")
    print()
    all_passed = True
    for test_key, success in results.items():
        test_name = TESTS[test_key]['name']
        if success:
            print_success(f"{test_name}: 通过")
        else:
            print_error(f"{test_name}: 失败")
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print_success("所有测试通过！")
        print("\n下一步:")
        print("  1. 启动 AirSim 仿真服务器:")
        print(f"     cd {ROOT_DIR}/airsim_plugin")
        print("     python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/extracted")
        print("\n  2. 运行完整评估:")
        print(f"     bash {ROOT_DIR}/scripts/eval_qwen.sh")
        print("\n  3. 计算评估指标:")
        print(f"     bash {ROOT_DIR}/scripts/metric.sh")
        print("\n" + "=" * 70)
        sys.exit(0)
    else:
        print_warning("部分测试失败")
        print(f"\n失败的测试: {', '.join(failed_tests)}")
        print("\n请根据错误信息修复问题后重试。")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
