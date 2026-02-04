"""
TravelUAV Benchmark - HTTP Client 评测脚本
用于连接外部模型服务器进行评测，不依赖本地模型
"""
import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

# 将当前工作目录添加到系统路径
sys.path.append(str(Path(str(os.getcwd())).resolve()))

# --- 导入项目自定义模块 ---
from utils.logger import logger
from utils.utils import *
# 导入 HTTP 客户端
from src.model_wrapper.http_client import HttpClient
# 导入参数配置
from src.common.param import args, model_args, data_args
# 导入 AirSim 环境接口
from env_uav import AirVLNENV
# 导入助手模块
from assist import Assist
# 导入闭环评测工具类
from src.vlnce_src.closeloop_util import (
    EvalBatchState, BatchIterator, setup, CheckPort,
    initialize_env_eval, is_dist_avail_and_initialized
)


def eval_http(http_client: HttpClient, assist: Assist, eval_env: AirVLNENV, eval_save_dir: str):
    """
    HTTP客户端评测函数 - 完全不依赖本地模型

    :param http_client: HTTP客户端，用于连接外部模型服务器
    :param assist: 助手对象，用于生成提示
    :param eval_env: AirSim 环境对象
    :param eval_save_dir: 结果保存路径
    """
    # HTTP客户端设置为评估模式（实际无操作，仅为接口兼容）
    http_client.eval()

    # 初始化数据迭代器
    dataset = BatchIterator(eval_env)
    end_iter = len(dataset)
    pbar = tqdm.tqdm(total=end_iter)

    logger.info(f"🚀 Starting HTTP Client Evaluation (Total: {end_iter} episodes)")

    # --- 外层循环：遍历所有 Batch ---
    while True:
        # 1. 获取下一个 Mini-Batch
        env_batchs = eval_env.next_minibatch()

        if env_batchs is None:
            logger.info("✅ All episodes completed!")
            break

        # 2. 初始化 Batch 状态管理器
        batch_state = EvalBatchState(
            batch_size=eval_env.batch_size,
            env_batchs=env_batchs,
            env=eval_env,
            assist=assist
        )

        # 更新进度条
        pbar.update(n=eval_env.batch_size)

        # 3. 通知Server重置状态（新Episode开始）
        for i in range(eval_env.batch_size):
            http_client.reset(env_id=i, episode_id=f"batch_{eval_env.index_data}_{i}")

        # --- 内层循环：单步导航 ---
        for t in range(int(args.maxWaypoints) + 1):
            logger.info('Step: {} \t Completed: {} / {}'.format(
                t,
                int(eval_env.index_data) - int(eval_env.batch_size),
                end_iter
            ))

            # 4. 检查是否全部结束
            is_terminate = batch_state.check_batch_termination(t)
            if is_terminate:
                logger.info(f"Batch terminated at step {t}")
                break

            # 5. 获取助手提示
            assist_notices = batch_state.get_assist_notices()

            # 6. 调用HTTP客户端查询：发送观测，接收航点和停止信号
            refined_waypoints, predict_dones = http_client.query_batch(
                episodes=batch_state.episodes,
                target_positions=batch_state.target_positions,
                assist_notices=assist_notices
            )

            # 7. 更新停止信号
            batch_state.predict_dones = predict_dones

            # 8. 环境执行：将航点发送给 AirSim
            eval_env.makeActions(refined_waypoints)

            # 9. 获取新观测
            outputs = eval_env.get_obs()

            # 10. 更新状态
            batch_state.update_from_env_output(outputs)

            # 11. 计算指标
            batch_state.update_metric()

    try:
        pbar.close()
    except:
        pass

    logger.info("🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    import argparse

    # 解析命令行参数（只解析 HTTP 客户端特有的参数）
    parser = argparse.ArgumentParser(description="TravelUAV HTTP Client Evaluation")
    parser.add_argument(
        "--server_url",
        type=str,
        default="http://127.0.0.1:9009",
        help="External model server URL"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP request timeout in seconds"
    )
    # 使用 parse_known_args 允许其他参数传递给 HfArgumentParser
    cli_args, unknown = parser.parse_known_args()

    # 从 args 中获取配置路径
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    # 创建保存目录
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    # 基础设置（随机种子等）
    setup()

    # 检查 AirSim 端口
    assert CheckPort(), 'AirSim port connection error!'

    # 初始化评测环境
    logger.info("🔧 Initializing evaluation environment...")
    eval_env = initialize_env_eval(
        dataset_path=dataset_path,
        save_path=eval_save_path,
        eval_json_path=eval_json_path
    )

    # 如果是分布式环境，销毁进程组
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    # 初始化 HTTP 客户端（替代本地模型）
    logger.info(f"🌐 Initializing HTTP Client: {cli_args.server_url}")
    http_client = HttpClient(
        server_url=cli_args.server_url,
        timeout=cli_args.timeout
    )

    # 初始化助手
    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)
    logger.info(f"🤖 Assist setting: always_help={args.always_help}, use_gt={args.use_gt}")

    # 进入评测主循环
    eval_http(
        http_client=http_client,
        assist=assist,
        eval_env=eval_env,
        eval_save_dir=eval_save_path
    )

    # 清理环境
    eval_env.delete_VectorEnvUtil()
    logger.info("🧹 Environment cleaned up")
