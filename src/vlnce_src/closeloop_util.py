"""
闭环导航训练/评测工具集合。

职责：
- 环境初始化与端口检查；
- 轨迹、图像与元信息落盘；
- DAgger/Eval 两套 batch 状态机；
- 评测时距离与终止条件管理（含视频录制）。
"""


import json
import random
import shutil

import cv2
import numpy as np
from utils.utils import *
from src.common.param import args
import torch.backends.cudnn as cudnn
from src.vlnce_src.env_uav import AirVLNENV, RGB_FOLDER, DEPTH_FOLDER


# =======================
# 视频录制相关常量
# =======================
VIDEO_FPS = 10
VIDEO_CODEC = 'mp4v'
FRONT_CAMERA_INDEX = 0  # frontcamera 在 RGB_FOLDER 中的索引


class VideoRecorder:
    """视频录制器，用于录制无人机第一视角视频"""

    def __init__(self, save_path, fps=VIDEO_FPS):
        self.save_path = save_path
        self.fps = fps
        self.writer = None
        self.width = None
        self.height = None

    def _init_writer(self, width, height):
        self.width = int(width)
        self.height = int(height)
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_CODEC)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        self.writer = cv2.VideoWriter(self.save_path, fourcc, self.fps, (self.width, self.height))

    def write_frame(self, frame):
        """写入单帧"""
        if frame is None:
            return
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            return

        if self.writer is None:
            h, w = frame.shape[:2]
            self._init_writer(w, h)

        if self.writer is None or not self.writer.isOpened():
            return

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        self.writer.write(frame_bgr)

    def release(self):
        """释放资源"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def setup(dagger_it=0, manual_init_distributed_mode=False):
    """
    统一初始化运行时随机性与分布式上下文。

    - dagger_it 会参与 seed，避免不同迭代数据完全一致；
    - 若 manual_init_distributed_mode=False，则自动初始化分布式环境。
    """
    if not manual_init_distributed_mode:
        init_distributed_mode()

    seed = 100 + get_rank() + dagger_it
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = False

def CheckPort():
    """检查 DDP 主端口是否被占用。"""
    pid = FromPortGetPid(int(args.DDP_MASTER_PORT))
    if pid is not None:
        print('DDP_MASTER_PORT ({}) is being used'.format(args.DDP_MASTER_PORT))
        return False

    return True

def initialize_env(dataset_path, save_path, train_json_path, activate_maps=[]):
    """构建训练/采集环境。"""
    train_env = AirVLNENV(batch_size=args.batchSize, dataset_path=dataset_path, save_path=save_path, eval_json_path=train_json_path, activate_maps=activate_maps)
    return train_env

def initialize_env_eval(dataset_path, save_path, eval_json_path):
    """构建评测环境。"""
    train_env = AirVLNENV(
        batch_size=args.batchSize,
        dataset_path=dataset_path,
        save_path=save_path,
        eval_json_path=eval_json_path,
        activate_maps=args.activate_maps
    )
    return train_env

def save_to_dataset_dagger(episodes, path, dagger_it, teacher_after_collision_steps):
    """保存 DAgger 轨迹（图像+日志+元信息）。"""
    ori_path = path
    path_parts = ori_path.strip('/').split('/')
    map_name, seq_name = path_parts[-2], path_parts[-1]
    root_path = os.path.join(args.dagger_save_path, seq_name)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_path, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(root_path, 'dagger_info.json'), 'w') as f:
        json.dump({'teacher_after_collision_steps': teacher_after_collision_steps,
                   'map_name': map_name,
                   'seq_name': seq_name}, f)
        
def save_to_dataset_eval(episodes, path, ori_traj_dir):
    """保存 Eval 轨迹（图像+日志+原始轨迹引用）。"""
    root_path = os.path.join(path)
    if not os.path.exists(root_path):
        os.makedirs(root_path)
    folder_names = ['log'] + RGB_FOLDER + DEPTH_FOLDER
    for folder_name in folder_names:
        os.makedirs(os.path.join(root_path, folder_name), exist_ok=True)
    print(root_path)
    save_logs(episodes, root_path)
    save_images(episodes, root_path)

    ori_obj = os.path.join(ori_traj_dir, 'object_description.json')
    target_obj = os.path.join(root_path, 'object_description.json')
    shutil.copy2(ori_obj, target_obj)
    with open(os.path.join(path, 'ori_info.json'), 'w') as f:
        json.dump({'ori_traj_dir': ori_traj_dir}, f)

def save_logs(episodes, trajectory_dir):
    """将每帧传感器写入 log/*.json。"""
    save_dir = os.path.join(trajectory_dir, 'log')
    for idx, episode in enumerate(episodes):
        info = {'frame': idx, 'sensors': episode['sensors']}
        with open(os.path.join(save_dir, str(idx).zfill(6) + '.json'), 'w') as f:
            json.dump(info, f)

def save_images(episodes, trajectory_dir):
    """将每帧 RGB/Depth 图像按相机目录保存。"""
    for idx, episode in enumerate(episodes):
        if 'rgb' in episode:
            for cid, camera_name in enumerate(RGB_FOLDER):
                image = episode['rgb'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)
        if 'depth' in episode:
            for cid, camera_name in enumerate(DEPTH_FOLDER):
                image = episode['depth'][cid]
                cv2.imwrite(os.path.join(trajectory_dir, camera_name, str(idx).zfill(6) + '.png'), image)

def load_object_description():
    """加载 object_name -> 文本描述映射。"""
    object_desc_dict = dict()
    with open(args.object_name_json_path, 'r') as f:
        file = json.load(f)
        for item in file:
            object_desc_dict[item['object_name']] = item['object_desc']
    return object_desc_dict

def target_distance_increasing_for_10frames(lst):
    """
    判定最近 10 次距离是否单调不下降。

    注意：该函数把“持平”也视为满足条件（只要没有变小）。
    """
    if len(lst) < 10:
        return False
    sublist = lst[-10:]
    for i in range(1, len(sublist)):
        if sublist[i] < sublist[i - 1]:
            return False
    return True

class BatchIterator:
    """把 AirVLNENV 封装成可计数/可迭代的 batch 迭代器。"""
    def __init__(self, env: AirVLNENV):
        self.env = env
    
    def __len__(self):
        return len(self.env.data)
    
    def __next__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch
    
    def __iter__(self):
        batch = self.env.next_minibatch()
        if batch is None:
            raise StopIteration
        return batch

class DaggerBatchState:
    """DAgger 模式下的 batch 状态容器与终止控制器。"""
    def __init__(self, bs, env_batchs, train_env):
        self.bs = bs
        self.episodes = [[] for _ in range(bs)]
        self.train_env = train_env
        self.skips = [False] * bs
        self.dones = [False] * bs
        self.oracle_success = [False] * bs
        self.collisions = [False] * bs
        self.need_teacher = [False] * bs
        self.back_count = [dict() for _ in range(bs)]
        self.teacher_after_collision_steps = [[] for _ in range(bs)]
        self.envs_to_pause = []
        self.paths = [b['trajectory_dir'] for b in env_batchs]
        self.target_positions = [b['object_position'] for b in env_batchs]
        object_desc_dict = load_object_description()
        self.object_infos = [object_desc_dict.get(b['object']['asset_name'].replace("AA", "")) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        
    def update_from_env_output(self, outputs, check_collision_function=None):
        """
        合并环境输出到当前 batch 状态。

        - episodes 追加最新观测；
        - oracle_success 会强制将 done 置 True。
        """
        observations, dones, collisions, oracle_success = [list(x) for x in zip(*outputs)]
        if check_collision_function is not None:
            collisions, dones = check_collision_function(self.episodes, observations, collisions, dones)
        for i in range(self.bs):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            if oracle_success[i]:
                dones[i] = True
        self.oracle_success = oracle_success
        self.dones = dones
        self.collisions = collisions
        return
    
    
    def check_dagger_batch_termination(self, dagger_it):
        """
        检查 DAgger batch 是否终止，并按规则保存轨迹。

        碰撞轨迹会做截断，过短样本会直接跳过不保存。
        """
        for i in range(self.bs):
            ep = self.episodes[i]
            if not self.skips[i] and ((self.dones[i] and not self.collisions[i]) or (len(self.episodes[i]) >= args.maxWaypoints * 5 // 10 and self.collisions[i])):
                ori_path = self.paths[i]
                self.skips[i] = True
                if self.collisions[i]:
                    ep = ep[:-25]
                save_to_dataset_dagger(ep, ori_path, dagger_it, self.teacher_after_collision_steps[i])
            elif len(ep) < args.maxWaypoints * 5 // 10 and self.collisions[i] and not self.skips[i]: # the dagger is not long enough, so we don't save this data
                self.skips[i] = True
        if all(self.dones):
            return True # terminate
        return False 
    
    def dagger_step_back(self):
        """
        DAgger 回退策略：碰撞后回退到历史帧并切换 teacher 控制。
        """
        # if collisions without teacher action, return to last 2 frame and move with teacher action
        for i in range(self.bs):
            if self.dones[i] or i in self.envs_to_pause:
                continue
            # If no collision occurs or no teacher intervention is required, apply ModelWrapper control.
            # If a collision occurs and teacher intervention is required, the DAgger trajectory fails, and the training ends.
            # If current step is using teacher action, disable the teacher flag and apply ModelWrapper control.
            if not self.collisions[i] and self.need_teacher[i]:
                self.need_teacher[i] = False
            elif self.collisions[i] and not self.need_teacher[i]:
                if (len(self.episodes[i]) in self.back_count[i] and self.back_count[i][len(self.episodes[i])] > 3) or sum(self.back_count[i].values()) > 30:
                    continue
                else:
                    self.back_count[i][len(self.episodes[i])] = self.back_count[i].get(len(self.episodes[i]), 0) + 1
                    self.train_env.revert2frame(i)
                    self.need_teacher[i] = True
                    self.collisions[i] = False
                    # reset the done flag caused by collision
                    self.dones[i] = False
                    if len(self.episodes[i]) > 10:
                        self.episodes[i] = self.episodes[i][0:-10]
                    else:
                        self.episodes[i] = self.episodes[i][0:1]
                    assert len(self.episodes[i]) == len(self.train_env.sim_states[i].trajectory)
                    remove_index = 0
                    for teacher_after_collision_step in self.teacher_after_collision_steps[i][::-1]:
                        if teacher_after_collision_step >= len(self.episodes[i]):
                            remove_index -= 1
                    self.teacher_after_collision_steps[i] = self.teacher_after_collision_steps[i][0: (None if remove_index==0 else remove_index)]
                    self.teacher_after_collision_steps[i].append(len(self.episodes[i]))
                    
                    
class EvalBatchState:
    """Eval 模式下的 batch 状态容器（含视频录制与终止判定）。"""
    def __init__(self, batch_size, env_batchs, env):
        self.batch_size = batch_size
        self.eval_env = env
        self.episodes = [[] for _ in range(batch_size)]
        self.target_positions = [b['object_position'] for b in env_batchs]
        self.object_infos = [self._get_object_info(b) for b in env_batchs]
        self.trajs = [b['trajectory'] for b in env_batchs]
        self.ori_data_dirs = [b['trajectory_dir'] for b in env_batchs]
        self.dones = [False] * batch_size
        self.predict_dones = [False] * batch_size
        self.collisions = [False] * batch_size
        self.success = [False] * batch_size
        self.oracle_success = [False] * batch_size
        self.early_end = [False] * batch_size
        self.skips = [False] * batch_size
        self.distance_to_ends = [[] for _ in range(batch_size)]
        self.envs_to_pause = []

        # 初始化视频录制器
        self.video_recorders = [None] * batch_size
        self.video_paths = [None] * batch_size
        self.traj_names = [None] * batch_size

        self._initialize_batch_data()

    def _get_object_info(self, batch):
        object_desc_dict = self._load_object_description()
        return object_desc_dict.get(batch['object']['asset_name'].replace("AA", ""))

    def _load_object_description(self):
        with open(args.object_name_json_path, 'r') as f:
            return {item['object_name']: item['object_desc'] for item in json.load(f)}

    def _initialize_batch_data(self):
        """
        执行环境 reset，并初始化每个样本的 episode 缓存、距离序列与视频录制器。
        """
        outputs = self.eval_env.reset()
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]

        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            self.episodes[i].append(observations[i][-1])
            # 记录当前帧到目标距离，用于停机/退化判定。
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))

            # 初始化视频录制器并录制第一帧
            traj_name = self.ori_data_dirs[i].split('/')[-1]
            self.traj_names[i] = traj_name
            video_path = os.path.join(args.eval_save_path, '_tmp_videos', traj_name + '_fpv.mp4')
            self.video_paths[i] = video_path
            self.video_recorders[i] = VideoRecorder(video_path, fps=VIDEO_FPS)

            # 录制前视相机视频
            if self.video_recorders[i] is not None:
                rgb_record = observations[i][-1].get('rgb_record')
                if rgb_record is not None and len(rgb_record) > FRONT_CAMERA_INDEX:
                    front_image = rgb_record[FRONT_CAMERA_INDEX]
                    if front_image is not None:
                        self.video_recorders[i].write_frame(front_image)

    def _calculate_distance(self, observation, target_position):
        """计算当前位置到目标点的欧式距离。"""
        return np.linalg.norm(np.array(observation['sensors']['state']['position']) - np.array(target_position))

    def update_from_env_output(self, outputs):
        """
        用环境返回更新 episode，并追加最新距离。

        若最近 10 次距离单调不下降，则将样本标记为碰撞并结束。
        """
        observations, self.dones, self.collisions, self.oracle_success = [list(x) for x in zip(*outputs)]

        for i in range(self.batch_size):
            if i in self.envs_to_pause:
                continue
            for j in range(len(observations[i])):
                self.episodes[i].append(observations[i][j])

            # 录制前视相机视频：每个 step 写入两帧（中间索引帧 + 最后一帧）
            if self.video_recorders[i] is not None and len(observations[i]) > 0:
                mid_idx = len(observations[i]) // 2
                frame_candidates = [observations[i][mid_idx], observations[i][-1]]

                for obs_item in frame_candidates:
                    rgb_record = obs_item.get('rgb_record')
                    if rgb_record is not None and len(rgb_record) > FRONT_CAMERA_INDEX:
                        front_image = rgb_record[FRONT_CAMERA_INDEX]
                        if front_image is not None:
                            self.video_recorders[i].write_frame(front_image)

            # 记录当前帧到目标距离，用于停机/退化判定。
            self.distance_to_ends[i].append(self._calculate_distance(observations[i][-1], self.target_positions[i]))
            # 最近 10 帧距离不下降：认为陷入无效推进，提前终止该样本。
            if target_distance_increasing_for_10frames(self.distance_to_ends[i]):
                self.collisions[i] = True
                self.dones[i] = True

    def update_metric(self):
        """
        基于模型 stop 预测与距离阈值更新 success/early_end/done。
        """
        for i in range(self.batch_size):
            if self.dones[i]:
                continue
            if self.predict_dones[i] and not self.skips[i]:
                if self.distance_to_ends[i][-1] <= 20 and not self.early_end[i]:
                    self.success[i] = True
                elif self.distance_to_ends[i][-1] > 20:
                    self.early_end[i] = True
                if self.oracle_success[i] and self.early_end[i]:
                    self.dones[i] = True
                elif self.success[i]:
                    self.dones[i] = True
                    
    def check_batch_termination(self, t):
        """
        处理 batch 内已结束样本：保存轨迹、释放视频句柄，并返回 batch 是否全部完成。
        """
        for i in range(self.batch_size):
            # 达到最大步数时，强制标记为结束。
            if t == args.maxWaypoints:
                self.dones[i] = True
            if self.dones[i] and not self.skips[i]:
                self.envs_to_pause.append(i)
                prex = ''
                if self.success[i]:
                    prex = 'success_'
                    print(i, " has succeed!")
                elif self.oracle_success[i]:
                    prex = "oracle_"
                    print(i, " has oracle succeed!")
                new_traj_name = prex +  self.ori_data_dirs[i].split('/')[-1]
                new_traj_dir = os.path.join(args.eval_save_path, new_traj_name)
                save_to_dataset_eval(self.episodes[i], new_traj_dir, self.ori_data_dirs[i])
                self.skips[i] = True
                print(i, " has finished!")

                # 释放视频录制器
                if self.video_recorders[i] is not None:
                    self.video_recorders[i].release()
                    self.video_recorders[i] = None
                if self.video_paths[i] is not None and os.path.exists(self.video_paths[i]):
                    video_name = f"{self.traj_names[i] if self.traj_names[i] else 'trajectory'}.mp4"
                    final_video_path = os.path.join(new_traj_dir, video_name)
                    shutil.move(self.video_paths[i], final_video_path)
                    self.video_paths[i] = None
                    print(i, " video saved to ", final_video_path)
        return np.array(self.skips).all()