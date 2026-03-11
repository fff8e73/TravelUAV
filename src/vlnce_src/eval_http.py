"""
TravelUAV Benchmark - HTTP Client 评测入口。

本文件负责把「AirSim 环境」与「外部 HTTP 模型服务」连接起来，
在不加载本地模型权重的情况下完成闭环评测。

核心流程：
1) 从评测集取 batch；
2) 将观测发送给 HTTP 服务，获取航点与 stop 信号；
3) 将航点下发到 AirSim 执行动作；
4) 回收新观测并更新 episode 状态；
5) 达到终止条件时保存轨迹并进入下一 batch。
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
# 导入闭环评测工具类
from src.vlnce_src.closeloop_util import (
    EvalBatchState, BatchIterator, setup, CheckPort,
    initialize_env_eval, is_dist_avail_and_initialized
)


def eval_http(http_client: HttpClient, eval_env: AirVLNENV, eval_save_dir: str):
    """
    HTTP 客户端评测主循环（不依赖本地模型推理）。

    :param http_client: HTTP客户端，用于连接外部模型服务器
    :param eval_env: AirSim 环境对象
    :param eval_save_dir: 结果保存路径（当前函数中主要由下游状态管理器使用）

    说明：
    - `episodes`/`collisions`/`target_positions` 由 `EvalBatchState` 统一维护；
    - 终止判定由 `check_batch_termination` 负责；
    - 指标更新由 `update_metric` 负责。
    """
    # 与本地模型接口对齐：显式切到 eval 模式。
    # 对 HTTP 客户端来说通常是空操作，但能保持调用约定一致。
    http_client.eval()

    # 统计评测集规模，用于进度条展示。
    dataset = BatchIterator(eval_env)
    end_iter = len(dataset)
    pbar = tqdm.tqdm(total=end_iter)

    logger.info(f"🚀 Starting HTTP Client Evaluation (Total: {end_iter} episodes)")

    # ---------------------------
    # 外层循环：按 batch 处理样本
    # ---------------------------
    while True:
        # 1) 拉取一个新的 batch；无数据时结束全量评测。
        env_batchs = eval_env.next_minibatch()

        if env_batchs is None:
            logger.info("✅ All episodes completed!")
            break

        # 2) 初始化当前 batch 的状态容器。
        #    这里会触发环境 reset，并初始化：
        #    - episodes（轨迹缓存）
        #    - dones/collisions/success 等标志位
        #    - target_positions 等评测上下文
        batch_state = EvalBatchState(
            batch_size=eval_env.batch_size,
            env_batchs=env_batchs,
            env=eval_env
        )

        # 按样本数推进总体进度。
        pbar.update(n=eval_env.batch_size)

        # 3) 通知外部服务：每个 env_id 开启新 episode。
        #    episode_id 绑定到当前 batch 索引，避免跨样本串状态。
        for i in range(eval_env.batch_size):
            http_client.reset(env_id=i, episode_id=f"batch_{eval_env.index_data}_{i}")

        # ---------------------------
        # 内层循环：按 step 执行导航
        # ---------------------------
        for t in range(int(args.maxWaypoints) + 1):
            logger.info('Step: {} \t Completed: {} / {}'.format(
                t,
                int(eval_env.index_data) - int(eval_env.batch_size),
                end_iter
            ))

            # 4) 检查 batch 是否全部结束（成功、超步数、碰撞策略触发等）。
            is_terminate = batch_state.check_batch_termination(t)
            if is_terminate:
                logger.info(f"Batch terminated at step {t}")
                break

            # 5) 调用外部服务推理：输入当前轨迹观测，输出下一段航点与 stop 预测。
            refined_waypoints, predict_dones = http_client.query_batch(
                episodes=batch_state.episodes,
                target_positions=batch_state.target_positions,
                collisions=batch_state.collisions
            )

            # 6) 写回模型 stop 预测，供后续 metric/终止逻辑使用。
            batch_state.predict_dones = predict_dones

            # 7) 将航点下发到 AirSim 执行动作。
            eval_env.makeActions(refined_waypoints)

            # 8) 回收执行后的观测（含 done/collision/oracle_success 等状态）。
            outputs = eval_env.get_obs()

            # 9) 更新 batch 状态缓存（轨迹、碰撞、距离等）。
            batch_state.update_from_env_output(outputs)

            # 10) 更新评测指标并可能设置 done。
            batch_state.update_metric()

    # 进度条关闭失败不影响主流程（兼容异常退出场景）。
    try:
        pbar.close()
    except:
        pass

    logger.info("🎉 Evaluation completed successfully!")


if __name__ == "__main__":
    import argparse

    # 仅解析 HTTP 客户端特有参数；其余参数由项目参数系统处理。
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
    # 允许未知参数透传，避免与上层 HfArgumentParser 冲突。
    cli_args, unknown = parser.parse_known_args()

    # 从全局 args 读取数据路径与结果路径。
    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    # 结果目录不存在则创建。
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    # 初始化随机种子、分布式上下文等运行环境。
    setup()

    # 启动前检查 AirSim 端口占用，防止多个任务冲突。
    assert CheckPort(), 'AirSim port connection error!'

    # 构建评测环境（加载评测集、场景管理器等）。
    logger.info("🔧 Initializing evaluation environment...")
    eval_env = initialize_env_eval(
        dataset_path=dataset_path,
        save_path=eval_save_path,
        eval_json_path=eval_json_path
    )

    # 本脚本按单进程评测；若外部误初始化了分布式，这里主动清理。
    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    # 显式关闭 DDP 标志，避免后续模块按分布式路径执行。
    args.DistributedDataParallel = False

    # 创建 HTTP 客户端（外部模型服务代理）。
    logger.info(f"🌐 Initializing HTTP Client: {cli_args.server_url}")
    http_client = HttpClient(
        server_url=cli_args.server_url,
        timeout=cli_args.timeout
    )

    # 进入主评测循环。
    eval_http(
        http_client=http_client,
        eval_env=eval_env,
        eval_save_dir=eval_save_path
    )

    # 主动释放向量化环境资源。
    eval_env.delete_VectorEnvUtil()
    logger.info("🧹 Environment cleaned up")
