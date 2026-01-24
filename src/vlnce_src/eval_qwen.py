"""
Qwen3-VL 评测脚本

适配 Qwen3-VL 的两阶段架构进行评测
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

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.qwen3vl_gpu_native import Qwen3VLGPUNativeWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized
from dataclasses import dataclass

@dataclass
class QwenModelArgs:
    """Qwen3-VL 模型参数"""
    model_path: str = None
    traj_model_path: str = None
    vision_tower: str = None
    image_processor: str = None
    use_4bit: bool = True

@dataclass
class QwenDataArgs:
    """Qwen3-VL 数据参数"""
    input_prompt: str = None
    refine_prompt: bool = True

def eval_qwen(model_wrapper: BaseModelWrapper, assist: Assist, eval_env: AirVLNENV, eval_save_dir):
    """使用 Qwen3-VL 进行评测"""
    model_wrapper.eval()

    with torch.no_grad():
        dataset = BatchIterator(eval_env)
        end_iter = len(dataset)
        pbar = tqdm.tqdm(total=end_iter)

        while True:
            env_batchs = eval_env.next_minibatch()
            if env_batchs is None:
                break
            batch_state = EvalBatchState(batch_size=eval_env.batch_size, env_batchs=env_batchs, env=eval_env, assist=assist)

            pbar.update(n=eval_env.batch_size)

            inputs, rot_to_targets = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions)

            for t in range(int(args.maxWaypoints) + 1):
                logger.info('Step: {} \t Completed: {} / {}'.format(t, int(eval_env.index_data)-int(eval_env.batch_size), end_iter))

                is_terminate = batch_state.check_batch_termination(t)
                if is_terminate:
                    break

                refined_waypoints = model_wrapper.run(inputs=inputs, episodes=batch_state.episodes, rot_to_targets=rot_to_targets)
                eval_env.makeActions(refined_waypoints)
                outputs = eval_env.get_obs()
                batch_state.update_from_env_output(outputs)

                batch_state.predict_dones = model_wrapper.predict_done(batch_state.episodes, batch_state.object_infos)

                batch_state.update_metric()

                assist_notices = batch_state.get_assist_notices()
                inputs, _ = model_wrapper.prepare_inputs(batch_state.episodes, batch_state.target_positions, assist_notices)

        try:
            pbar.close()
        except:
            pass


if __name__ == "__main__":

    eval_save_path = args.eval_save_path
    eval_json_path = args.eval_json_path
    dataset_path = args.dataset_path

    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)

    setup()

    assert CheckPort(), 'error port'

    eval_env = initialize_env_eval(dataset_path=dataset_path, save_path=eval_save_path, eval_json_path=eval_json_path)

    if is_dist_avail_and_initialized():
        torch.distributed.destroy_process_group()

    args.DistributedDataParallel = False

    # 准备 Qwen3-VL 模型参数
    qwen_model_args = QwenModelArgs(
        model_path=model_args.model_path,
        traj_model_path=model_args.traj_model_path,
        vision_tower=model_args.vision_tower,
        image_processor=model_args.image_processor,
        use_4bit=True
    )

    qwen_data_args = QwenDataArgs()

    # 加载 Qwen3-VL 模型（两阶段架构）
    print(f"正在加载 Qwen3-VL 模型...")
    print(f"  模型路径: {qwen_model_args.model_path}")
    print(f"  轨迹模型路径: {qwen_model_args.traj_model_path}")

    model_wrapper = Qwen3VLGPUNativeWrapper(
        model_args=qwen_model_args,
        data_args=qwen_data_args,
        use_traj_model=True
    )

    assist = Assist(always_help=args.always_help, use_gt=args.use_gt)

    print("Assist setting: always_help --", args.always_help, "    use_gt --", args.use_gt)

    eval_qwen(model_wrapper=model_wrapper,
              assist=assist,
              eval_env=eval_env,
              eval_save_dir=eval_save_path)

    eval_env.delete_VectorEnvUtil()
