"""
TravelUAV HTTP Client
用于连接外部模型服务器的客户端
对齐 Isaac-Drone-Navigation-Benchmark 接口标准
"""
import requests
import json_numpy
import numpy as np
from typing import List, Dict, Any, Optional
from scipy.spatial.transform import Rotation as R

# 开启 numpy 序列化支持
json_numpy.patch()


class ActionBuffer:
    """
    每个 env_id 独立的动作缓冲区
    用于缓存 Server 返回的 [N,4] 动作序列，按需逐个消费
    """

    def __init__(self, k_exec_steps: int = 5, stop_threshold: float = 1e-5):
        """
        初始化动作缓冲区

        Args:
            k_exec_steps: 每批最多执行 K 步后强制重请求
            stop_threshold: 位移绝对值 < 此值视为 stop
        """
        self.action_queue: List[np.ndarray] = []  # 存放 [dx,dy,dz,dyaw] 的列表
        self.steps_since_inference: int = 0  # 当前批次已执行步数
        self.k_exec_steps: int = k_exec_steps
        self.stop_threshold: float = stop_threshold
        self.current_action_count: int = 0  # 记录当前存储的动作个数

    def need_inference(self) -> bool:
        """判断是否需要请求 server"""
        # 需要请求的情况：buffer 为空 或者 已执行 K 步
        return len(self.action_queue) == 0 or self.steps_since_inference >= self.k_exec_steps

    def push(self, actions: np.ndarray):
        """
        将 server 返回的 [N,4] 入队

        Args:
            actions: ndarray, shape [N, 4], 每行 [dx, dy, dz, dyaw]

        Raises:
            ValueError: 如果 Server 返回的动作个数 N 为 0
        """
        # 校验：Server 返回的动作个数不能为 0
        if actions.shape[0] == 0:
            raise ValueError("Server returned empty action list (N=0), this is an error")

        # 将每行动作转为列表并入队
        for i in range(actions.shape[0]):
            self.action_queue.append(actions[i])

        # 记录本次存入的动作个数
        self.current_action_count = actions.shape[0]
        self.steps_since_inference = 0

    def pop(self) -> np.ndarray:
        """
        取出一个动作

        Returns:
            action: np.ndarray [dx, dy, dz, dyaw]
        """
        if len(self.action_queue) == 0:
            # buffer 为空时返回零动作
            return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)

        action = self.action_queue.pop(0)
        self.steps_since_inference += 1
        # 递减动作计数
        self.current_action_count = max(0, self.current_action_count - 1)
        return action

    def is_stop(self, dx: float, dy: float, dz: float) -> bool:
        """
        停止判定：位移绝对值 < 阈值

        Args:
            dx, dy, dz: 动作的位移分量

        Returns:
            True if should stop
        """
        displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        return displacement < self.stop_threshold

    def reset(self):
        """新 episode 清空 buffer"""
        self.action_queue = []
        self.steps_since_inference = 0
        self.current_action_count = 0

    def get_action_count(self) -> int:
        """获取当前 buffer 中存储的动作个数"""
        return self.current_action_count


class HttpClient:
    """
    HTTP客户端 - 用于TravelUAV Benchmark连接外部模型Server
    对齐 A 标准接口：
    - 发送 A 标准 7+1 字段 obs
    - 接收 A 标准 [N,4] 相对动作
    - 使用 Buffer 机制缓存动作序列
    - 在 Client 端做坐标系逆转换
    """

    def __init__(self, server_url: str = "http://127.0.0.1:9009", timeout: int = 300,
                 k_exec_steps: int = 1, stop_threshold: float = 1e-5, batch_size: int = 1,
                 path_length: int = 5):
        """
        初始化HTTP客户端

        Args:
            server_url: 外部模型服务器地址
            timeout: 请求超时时间（秒）
            k_exec_steps: 每批动作执行多少步后重请求（默认1，每步都请求）
            stop_threshold: 停止阈值
            batch_size: 批大小，用于初始化 buffer
            path_length: 每步下发给仿真器的航点数量（默认5）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.k_exec_steps = k_exec_steps
        self.stop_threshold = stop_threshold
        self.batch_size = batch_size
        self.path_length = max(1, int(path_length))

        # 为每个 env_id 创建独立的 ActionBuffer
        self.buffers: Dict[int, ActionBuffer] = {
            i: ActionBuffer(k_exec_steps=k_exec_steps, stop_threshold=stop_threshold)
            for i in range(batch_size)
        }

        # 存储每个 env 的初始帧（用于计算相对位置）
        self.initial_frames: Dict[int, Dict] = {}

        # 存储每个 env 的 episode_id
        self.episode_ids: Dict[int, str] = {}

        print(f"🔗 [HttpClient] Connecting to server: {self.server_url}")
        print(f"   - k_exec_steps: {k_exec_steps}")
        print(f"   - stop_threshold: {stop_threshold}")
        print(f"   - batch_size: {batch_size}")
        print(f"   - path_length: {self.path_length}")

    def reset(self, env_id: int = 0, episode_id: str = None, **kwargs):
        """
        通知Server重置特定环境的状态，并重置本地 Buffer

        Args:
            env_id: 环境ID，最大索引为 batch_size-1
            episode_id: Episode ID，形式为batch_{eval_env.index_data}_{env_id}，其中eval_env.index_data是当前样本的索引
            **kwargs: 其他参数
        """
        # 重置本地 Buffer
        if env_id not in self.buffers:
            self.buffers[env_id] = ActionBuffer(
                k_exec_steps=self.k_exec_steps,
                stop_threshold=self.stop_threshold
            )
        else:
            self.buffers[env_id].reset()

        # 清除初始帧记录
        if env_id in self.initial_frames:
            del self.initial_frames[env_id]

        # 保存 episode_id
        if episode_id is not None:
            self.episode_ids[env_id] = episode_id

        # 通知 Server 重置
        payload = {"type": "reset", "env_id": env_id, **kwargs}
        try:
            resp = requests.post(
                f"{self.server_url}/reset",
                data=json_numpy.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=2.0
            )
            resp.raise_for_status()
            print(f"[HttpClient] Reset env_id={env_id} successfully")
        except Exception as e:
            print(f"[HttpClient] Reset warning: {e}")

    def _extract_observation(self, episode: List[Dict], collision: bool = False,
                             episode_id: str = None) -> Dict[str, Any]:
        """
        从episode中提取观测数据，构造 A 标准 obs (7+1 字段)

        A 标准字段:
        - rgb: [H,W,3] uint8
        - depth: [H,W] float32
        - instruction: str
        - step: int
        - compass: [1] float, 从 imu rotation 提取 yaw
        - gps: [2] float, 相对起点 [x,y]
        - collision: bool
        - episode_id: str (可选)

        Args:
            episode: 单个episode的历史观测序列
            collision: 是否碰撞
            episode_id: Episode ID（可选）

        Returns:
            A 标准 observation 字典
        """
        # 获取最新的观测帧和第一帧
        current_frame = episode[-1]
        first_frame = episode[0]

        obs = {}

        # 1. RGB图像（取第一个视角，即前视相机）
        # A 标准: [H, W, 3] uint8
        if 'rgb' in current_frame and len(current_frame['rgb']) > 0:
            rgb = current_frame['rgb'][0]
            # 确保是 3 通道
            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            obs['rgb'] = rgb

        # 2. 深度图
        # A 标准: [H, W] float32
        if 'depth' in current_frame and len(current_frame['depth']) > 0:
            obs['depth'] = current_frame['depth'][0]

        # 3. 自然语言指令
        if 'instruction' in current_frame:
            obs['instruction'] = current_frame['instruction']
        elif 'instruction' in first_frame:
            obs['instruction'] = first_frame['instruction']
        else:
            obs['instruction'] = "Navigate to the target"

        # 4. step: 当前步数
        obs['step'] = len(episode) - 1

        # 5. compass: 从 imu rotation 提取 yaw
        # A 标准: [1] float, 弧度
        if 'sensors' in current_frame and 'imu' in current_frame['sensors']:
            rot_matrix = np.array(current_frame['sensors']['imu']['rotation'])
            r = R.from_matrix(rot_matrix)
            euler = r.as_euler('zyx')
            yaw_rad = euler[0]
            obs['compass'] = np.array([yaw_rad], dtype=np.float32)

        # 6. gps: 相对起点位置
        # A 标准: [2] float, 相对起点 [x, y]
        if 'sensors' in current_frame and 'state' in current_frame['sensors']:
            current_pos = np.array(current_frame['sensors']['state']['position'])

            # 保存初始帧（如果还没有）
            env_id = id(episode) % 1000000  # 简单生成一个 env_id
            if env_id not in self.initial_frames:
                self.initial_frames[env_id] = first_frame

            initial_frame = self.initial_frames.get(env_id, first_frame)
            if 'sensors' in initial_frame and 'imu' in initial_frame['sensors']:
                initial_pos = np.array(initial_frame['sensors']['state']['position'])
                initial_rot = np.array(initial_frame['sensors']['imu']['rotation'])

                # 投影到起点局部坐标系
                rel_pos = initial_rot.T @ (current_pos - initial_pos)
                obs['gps'] = rel_pos[:2].astype(np.float32)  # 只取 x, y

        # 7. collision
        obs['collision'] = bool(collision)

        # 8. episode_id (可选)
        if episode_id is not None:
            obs['episode_id'] = episode_id

        return obs

    def _relative_action_to_waypoint(self, action: np.ndarray,
                                     current_pos: np.ndarray,
                                     current_rot: np.ndarray) -> np.ndarray:
        """
        将 A 标准的局部相对动作转换为世界坐标航点

        输入: action = [dx, dy, dz, dyaw], current_pos = [x,y,z], current_rot = [3,3]
        处理:
          1. 坐标系反转: A标准(+dx前,+dy左,+dz上) → AirSim
          2. delta_local = [dx_airsim, dy_airsim, dz_airsim]
          3. delta_world = current_rot @ delta_local
          4. waypoint_world = current_pos + delta_world
        输出: waypoint_world = [x, y, z] (世界坐标)

        注意：坐标系映射需要根据实际环境验证

        Args:
            action: np.ndarray [dx, dy, dz, dyaw]，A 标准局部坐标
            current_pos: 当前世界坐标位置 [x, y, z]
            current_rot: 当前旋转矩阵 [3, 3]

        Returns:
            waypoint: 世界坐标航点 [x, y, z]
        """
        dx, dy, dz, dyaw = action

        # 坐标系映射（需实际验证）
        # A 标准: +dx前, +dy左, +dz上, +dyaw逆时针
        # AirSim (NED): +x北, +y东, +z下
        # 映射关系：
        #   A dx (+前) → AirSim dx (北)
        #   A dy (+左) → AirSim dy (东, 取反因为+dy右)
        #   A dz (+上) → AirSim dz (下, 取反因为+z下)

        # 步骤1: 坐标系反转
        dx_airsim = dx
        dy_airsim = -dy  # A标准+左对应AirSim-右
        dz_airsim = -dz  # A标准+上对应AirSim-下

        delta_local = np.array([dx_airsim, dy_airsim, dz_airsim])

        # 步骤2: 转换到世界坐标
        current_rot = np.array(current_rot)
        delta_world = current_rot @ delta_local

        # 步骤3: 计算目标航点
        current_pos = np.array(current_pos)
        waypoint_world = current_pos + delta_world

        return waypoint_world

    def query_batch(self, episodes: List[List[Dict]], target_positions: List[np.ndarray],
                   collisions: List[bool] = None) -> tuple:
        """
        批量查询：发送 A 标准 obs，接收相对动作，转换为世界坐标航点

        Args:
            episodes: episode列表，每个episode是观测序列
            target_positions: 目标位置列表（保留参数，下游兼容）
            collisions: 碰撞状态列表（可选）

        Returns:
            (refined_waypoints, predict_dones)
            - refined_waypoints: List[List[np.ndarray]]，每个元素是递进航点序列 [[x, y, z], ...]（由最多 path_length 个动作生成）
            - predict_dones: List[bool]
        """
        batch_size = len(episodes)
        refined_waypoints = []
        predict_dones = []

        # 确保 buffer 数量足够
        for i in range(batch_size):
            if i not in self.buffers:
                self.buffers[i] = ActionBuffer(
                    k_exec_steps=self.k_exec_steps,
                    stop_threshold=self.stop_threshold
                )

        # 逐个处理每个 env
        for i in range(batch_size):
            buffer = self.buffers[i]

            # 获取碰撞信息
            collision = collisions[i] if collisions is not None else False

            # 1. 检查是否需要请求 server
            if buffer.need_inference():
                # 构造 A 标准 obs
                obs = self._extract_observation(episodes[i], collision,
                                                 episode_id=self.episode_ids.get(i))

                try:
                    # 发送请求
                    resp = requests.post(
                        f"{self.server_url}/act",
                        data=json_numpy.dumps(obs),
                        headers={"Content-Type": "application/json"},
                        timeout=self.timeout
                    )
                    resp.raise_for_status()

                    # 解析响应（A 标准 [N,4] 动作）
                    result = json_numpy.loads(resp.text)
                    actions = result.get("action", np.zeros((16, 4)))

                    # 存入 buffer
                    buffer.push(actions)

                except Exception as e:
                    print(f"[HttpClient] Query failed for batch {i}: {e}")
                    # fallback: 存入零动作
                    buffer.push(np.zeros((16, 4), dtype=np.float32))

            # 2. 从 buffer 弹出 action_count 个动作（由 Server 返回的 N 决定），生成递进航点序列
            action_count = buffer.get_action_count()
            action_path = [buffer.pop() for _ in range(action_count)]

            # 3. 停止判定 + 坐标系转换：遍历转换，遇到 stop 就停止下发
            current_frame = episodes[i][-1]
            if 'sensors' in current_frame and 'state' in current_frame['sensors']:
                current_pos = np.array(current_frame['sensors']['state']['position'])
                current_rot = np.eye(3)  # 默认单位矩阵

                if 'imu' in current_frame['sensors']:
                    current_rot = np.array(current_frame['sensors']['imu']['rotation'])

                waypoint_path = []
                simulated_pos = current_pos.copy()
                stop = False

                # 遍历所有动作，转换并添加到 waypoint_path
                for action in action_path:
                    # 检查是否应该停止
                    if buffer.is_stop(action[0], action[1], action[2]):
                        stop = True
                        # 停止遍历，剩余动作留在 buffer（不需要额外操作）
                        break

                    # 转换为世界坐标并添加
                    waypoint = self._relative_action_to_waypoint(action, simulated_pos, current_rot)
                    waypoint_path.append(waypoint)
                    simulated_pos = waypoint
            else:
                # fallback
                waypoint_path = []
                stop = False

            predict_dones.append(stop)
            refined_waypoints.append(waypoint_path)

        return refined_waypoints, predict_dones

    def eval(self):
        """
        兼容接口：设置为评估模式（对于HTTP客户端无实际作用）
        """
        pass
