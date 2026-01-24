# TravelUAV 评估分析完整文档

## 📋 目录

- [📊 评估统计与问题诊断](#评估统计与问题诊断)
- [🔍 详细问题分析](#详细问题分析)
- [⚠️ 评估进程中断分析](#评估进程中断分析)
- [🎯 任务完成的判断逻辑](#任务完成的判断逻辑)
- [📈 单个样本详细分析](#单个样本详细分析)
- [🔧 可能的原因](#可能的原因)
- [💡 解决方案](#解决方案)
- [🚀 快速开始](#快速开始)
- [📊 预期改进](#预期改进)
- [📝 总结](#总结)

---

## 📊 评估统计与问题诊断

### 评估统计

根据日志文件 `/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_20260117_160717.log`：

| 指标 | 数值 |
|------|------|
| 总样本数 | 1359 |
| 已完成样本 | 246 |
| 成功样本 (success_) | 0 |
| Oracle 成功样本 (oracle_) | 0 |
| 评估进度 | ~13% (186/1359) |
| 评估时间 | ~21 小时 |

### 距离分布分析

根据日志中的距离数据：

| 距离范围 | 次数 | 占比 |
|---------|------|------|
| < 20 米 | 0 | 0.0% |
| 20-50 米 | 0 | 0.0% |
| 50-100 米 | 25 | 5.8% |
| 100-200 米 | 147 | 33.9% |
| 200-300 米 | 168 | 38.7% |
| 300-400 米 | 94 | 21.7% |
| ≥ 400 米 | 0 | 0.0% |

**关键发现：**
- **最小距离：53.78 米**（远大于成功阈值 20 米）
- **平均距离：229.22 米**
- **0% 的样本**距离小于 20 米（成功阈值）
- **94.2% 的样本**距离大于 100 米

### 其他问题

- **碰撞次数**：74 次
- **路径规划失败**：频繁出现 `move on path api: stuck max len`

---

## 🔍 详细问题分析

### 1. **所有样本都没有成功完成**

从日志中可以看到：
- **0 个 `success_` 样本**：没有样本达到 `success` 条件
- **0 个 `oracle_` 样本**：没有样本达到 `oracle_success` 条件
- **246 个普通样本**：所有完成的样本都是普通结束（未成功）

### 2. **距离目标非常远**

根据日志中的距离数据：

| 距离范围 | 次数 | 占比 |
|---------|------|------|
| < 20 米 | 0 | 0.0% |
| 20-50 米 | 0 | 0.0% |
| 50-100 米 | 25 | 5.8% |
| 100-200 米 | 147 | 33.9% |
| 200-300 米 | 168 | 38.7% |
| 300-400 米 | 94 | 21.7% |
| ≥ 400 米 | 0 | 0.0% |

**关键发现：**
- **最小距离：53.78 米**（远大于成功阈值 20 米）
- **平均距离：229.22 米**
- **0% 的样本**距离小于 20 米（成功阈值）
- **94.2% 的样本**距离大于 100 米

### 3. **频繁触发 `move on path api: stuck max len`**

这个警告在日志中频繁出现，表明：
- 路径规划器无法生成有效路径
- 或者路径太长，超过最大限制
- 无人机无法沿路径飞行到目标

### 4. **频繁碰撞检测**

```
collision type: close
```

- 日志中检测到 **74 次碰撞**
- 碰撞检测基于深度图像差异
- 碰撞会导致任务提前终止

---

## ⚠️ 评估进程中断分析

### 中断时间线

根据日志文件 `/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_20260117_160717.log` 的最新 100 行：

```
05:48:03 - 第 1 次超时 (setPoses)
05:48:03 - 完成 2 个 episode (640878a4, 955faecf)
05:53:03 - 第 2 次超时 (setPoses) - 5 分钟后
05:58:04 - 第 3 次超时 (setPoses) - 5 分钟后
06:03:05 - 第 4 次超时 (setPoses) - 5 分钟后
06:08:06 - 第 5 次超时 (getSensorInfo) - 5 分钟后
06:08:06 - 程序崩溃 (TypeError)
```

**模式**: 每 5 分钟出现一次超时，持续 20 分钟后崩溃

### 进度状态

```
37%|███▋      | 504/1359 [62:00:14<105:11:06, 442.88s/it]
```

| 指标 | 数值 |
|------|------|
| 已完成 | 504 个 episode |
| 总共 | 1359 个 episode |
| 进度 | 37% |
| 已运行时间 | 62 分钟 |
| 预计剩余时间 | 105 分钟 |

### 错误详情

#### 错误 1: AirSim RPC 超时

```python
# AirVLNSimulatorClientTool.py:399
airsim_client.simSetKinematics(
    state, ignore_collision, vehicle_name
)
# ↓
msgpackrpc.error.TimeoutError: Request timed out
```

**原因**:
- AirSim 仿真器响应超时（默认超时时间可能为 5 分钟）
- 可能原因:
  - 仿真器负载过高
  - 网络连接问题
  - 仿真器崩溃或卡死
  - GPU 资源不足

#### 错误 2: 状态获取失败

```python
# AirVLNSimulatorClientTool.py:610
state_info = state_sensor.retrieve()
# ↓
airsim_client.getMultirotorState(vehicle_name)
# ↓
msgpackrpc.error.TimeoutError: Request timed out
```

**原因**:
- 无法获取无人机状态
- 仿真器可能已经无响应

#### 错误 3: NoneType 不可下标访问

```python
# env_uav.py:344
self.sim_states[cnt].trajectory = [state_info_results[index_1][index_2]]
# ↓
TypeError: 'NoneType' object is not subscriptable
```

**原因**:
- `state_info_results[index_1]` 返回 `None`
- 说明状态信息获取失败
- 由于之前的超时错误，状态数据为空

### 可能的原因

| 可能原因 | 描述 | 概率 |
|----------|------|------|
| **仿真器崩溃** | AirSim 进程异常退出 | ⭐⭐⭐⭐⭐ |
| **资源耗尽** | GPU/CPU/内存不足 | ⭐⭐⭐⭐ |
| **环境文件损坏** | 某个仿真环境文件异常 | ⭐⭐⭐⭐ |
| **网络问题** | RPC 连接中断 | ⭐⭐⭐ |
| **仿真器卡死** | 物理引擎卡住 | ⭐⭐⭐⭐ |
| **状态获取逻辑错误** | 异步状态获取失败 | ⭐⭐⭐ |
| **超时时间过短** | RPC 超时设置不合理 | ⭐⭐ |

### 解决方案

#### 方案 1: 检查 AirSim 仿真器状态

```bash
# 检查 AirSim 进程
ps aux | grep AirVLNSimulatorServerTool

# 检查端口 30000
netstat -tuln | grep 30000

# 查看 AirSim 日志
find /sim/data/TravelUAV_data/ -name "*.log" -type f | head -10
```

#### 方案 2: 重启 AirSim 仿真器

```bash
# 1. 停止当前的 AirSim 服务端
pkill -f AirVLNSimulatorServerTool

# 2. 重新启动 AirSim 服务端
cd /home/yyx/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs

# 3. 重新运行评估脚本
bash scripts/eval_qwen.sh
```

#### 方案 3: 检查系统资源

```bash
# 检查 GPU 显存
nvidia-smi

# 检查 CPU 和内存
htop

# 检查磁盘空间
df -h /sim/data/TravelUAV_data/
```

#### 方案 4: 检查环境文件

```bash
# 检查环境文件完整性
ls -lh /sim/data/TravelUAV_data/sim_envs/

# 检查特定环境（从日志看是 Carla_Town10HD）
ls -lh /sim/data/TravelUAV_data/sim_envs/Carla_Town10HD/
```

#### 方案 5: 增加超时时间

如果问题是超时时间过短，可以修改代码：

```python
# AirVLNSimulatorClientTool.py
# 增加 RPC 超时时间
self.client = airsim.MultirotorClient(ip="127.0.0.1", port=30000)
self.client.confirmConnection()
self.client.simLoadLevel('Carla_Town10HD')  # 显式加载环境
time.sleep(5)  # 等待环境加载完成
```

### 当前状态总结

#### ✅ 已完成

- ✅ 成功处理了 **504** 个 episode（37%）
- ✅ 模型推理正常工作
- ✅ 生成了评估结果文件

#### ❌ 遇到的问题

- ❌ AirSim 仿真器超时（5 次）
- ❌ 状态获取失败
- ❌ 程序崩溃（TypeError）

#### ⚠️ 可能的恢复方案

由于评估已经完成了 37%，可以尝试：

1. **从断点恢复**（如果代码支持）
2. **重新运行**（从头开始）
3. **检查并修复 AirSim 问题后继续**

### 立即操作

1. **检查 AirSim 仿真器状态**
   ```bash
   ps aux | grep AirVLNSimulatorServerTool
   ```

2. **查看 AirSim 日志**
   ```bash
   # 查找 AirSim 的日志文件
   find /sim/data/TravelUAV_data/ -name "*.log" -type f | head -10
   ```

3. **重启 AirSim 服务端**
   ```bash
   pkill -f AirVLNSimulatorServerTool
   cd /home/yyx/TravelUAV/airsim_plugin
   python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
   ```

### 后续操作

1. **重新运行评估**
   ```bash
   bash scripts/eval_qwen.sh
   ```

2. **监控运行状态**
   ```bash
   tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/*.log
   ```

3. **如果再次失败**
   - 检查 GPU 显存是否充足
   - 检查环境文件是否完整
   - 考虑减少 batch size

---

## 🎯 任务完成的判断逻辑

### 成功条件（`oracle_success`）

在 `env_uav.py:439` 中定义：

```python
if np.linalg.norm(np.array(waypoint) - np.array(self.batch[index]['object_position'])) < self.sim_states[index].SUCCESS_DISTANCE:
    self.sim_states[index].oracle_success = True
```

**成功条件：**
- 无人机位置与目标位置的距离 < 20 米
- 需要无人机实际到达目标附近

### 成功条件（`success`）

在 `closeloop_util.py:282-283` 中定义：

```python
if self.distance_to_ends[i][-1] <= 20 and not self.early_end[i]:
    self.success[i] = True
```

**成功条件：**
- 预测完成时，距离目标 ≤ 20 米
- 且没有提前结束

### 预测完成（`predict_done`）

在 `dino_monitor_online.py` 中定义：

```python
if len(target_detections) > 0:
    done = True
```

**完成条件：**
- GroundingDINO 检测到目标物体
- 且深度 < 18（目标在视野中且距离较近）

---

## 📈 单个样本详细分析

### 样本：004de33f-a963-4f7e-acc7-c517c3d0b6b0

**目标信息：**
- 物体名称：SM_TrafficCones_01
- 起始位置：[-404.723, -5.3125, -1.826]
- 结束位置：[-284.6335, -79.7535, -3.259]
- 目标位置：[-280.5669, -75.9224, 0.0205]

**距离分析：**
- 起始到目标：142.84 米
- 结束到目标：6.48 米（已接近目标！）

**轨迹分析：**
- 总帧数：34 帧
- 初始距离：290.66 米
- 最终距离：184.14 米
- 距离减少：106.51 米
- 达到成功阈值（20米）的帧数：0
- 向目标移动的比例：97.1%

**关键发现：**
- 无人机确实在向目标移动（97.1% 的帧都在减少距离）
- 但移动速度太慢，34 帧只减少了 106.51 米
- 平均每帧移动约 3.13 米
- 按此速度，需要约 93 帧才能到达 20 米范围内
- 但评估在 34 帧后就结束了

---

## 🔧 可能的原因

### 1. **模型问题**

- Qwen3-VL 生成的航点可能不准确
- 轨迹预测模型可能没有正确优化路径
- 两阶段架构的协调可能有问题

### 2. **路径规划问题**

- `move_on_path` API 无法处理长距离路径
- 路径规划器可能需要更短的目标
- 可能需要分段导航策略

### 3. **目标位置问题**

- 目标位置可能在不可达区域
- 目标位置可能被障碍物阻挡
- GPS 坐标转换可能有误差

### 4. **碰撞检测过于敏感**

- 深度图像差异阈值可能太低
- 导致正常移动也被判定为碰撞

### 5. **评估指标问题**

- `success` 和 `oracle_success` 的判断条件可能太严格
- 可能需要调整成功距离阈值

### 6. **最大步数不足**

- `maxWaypoints = 200` 可能不够
- 对于远距离目标（290 米），需要更多步数
- 建议增加到 500 或更高

---

## 💡 解决方案

### 方案 1：增加最大步数（推荐）

**修改文件**：`/home/yyx/TravelUAV/scripts/eval_qwen.sh`

**修改内容**：
```bash
# 第 93 行和第 132 行
--maxWaypoints 500  # 从 200 增加到 500
```

**使用方法**：
```bash
# 手动修改后运行
sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh
bash /home/yyx/TravelUAV/scripts/eval_qwen.sh
```

**预期效果**：
- 最多 2500 个航点（500 步 × 5 航点/步）
- 对于 229 米的平均距离，应该足够
- 成功率应该显著提升

### 方案 2：调整成功阈值

**修改文件**：`/home/yyx/TravelUAV/utils/env_utils_uav.py`

**修改内容**：
```python
# 第 26 行
self.SUCCESS_DISTANCE = 50  # 从 20 米增加到 50 米
```

**预期效果**：
- 更宽松的成功条件
- 更多样本能达到成功

### 方案 3：减少碰撞检测敏感度

**修改文件**：`/home/yyx/TravelUAV/src/vlnce_src/assist.py`

**修改内容**：
```python
# 第 58 行
if zero_cnt > 0.2 * current_episode[-1]['depth'][cid].size:  # 从 0.1 增加到 0.2
    close_collision = True

# 第 64 行
if np.all(diffs < 5):  # 从 3 增加到 5
    collision_type = 'tiny diff'

# 第 68 行
elif distance < 0.2:  # 从 0.1 增加到 0.2
    collision_type = 'distance'
```

**预期效果**：
- 减少误判的碰撞
- 让无人机有更多机会继续导航

### 方案 4：分段导航（高级）

**修改文件**：`/home/yyx/TravelUAV/src/vlnce_src/env_uav.py`

**修改思路**：
- 将长距离目标分解为多个中间点
- 每段距离 < 100 米
- 逐段导航到目标

**示例代码**：
```python
def makeActions(self, waypoints_list):
    # 检查距离是否过远
    for index, waypoints in enumerate(waypoints_list):
        target_pos = self.batch[index]['object_position']
        current_pos = self.sim_states[index].pose[0:3]
        distance = np.linalg.norm(np.array(target_pos) - np.array(current_pos))

        if distance > 100:
            # 使用中间点
            mid_point = (current_pos + target_pos) / 2
            waypoints_list[index] = [mid_point]
```

---

## 🚀 快速开始

### 1. 查看当前评估进度

```bash
# 查看当前运行的评估进程
ps aux | grep eval_qwen

# 查看日志
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_*.log
```

### 2. 修改 maxWaypoints 参数

在运行 `eval_qwen.sh` 之前，手动修改参数：

```bash
# 修改 maxWaypoints 从 200 到 500
sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh

# 验证修改
grep "maxWaypoints" /home/yyx/TravelUAV/scripts/eval_qwen.sh
```

### 3. 运行评估

```bash
# 激活环境
conda activate llamauav_sm_120

# 启动 AirSim 服务器（如果未启动）
cd /home/yyx/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/extracted &

# 运行评估
cd /home/yyx/TravelUAV
bash scripts/eval_qwen.sh
```

### 4. 监控进度

```bash
# 查看日志
tail -f /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_*.log

# 查看 GPU 使用情况
nvidia-smi

# 查看进程
ps aux | grep eval_qwen
```

---

## 📊 预期改进

### 增加最大步数后（maxWaypoints = 500）

| 指标 | 修改前 | 预期修改后 | 改进 |
|------|--------|-----------|------|
| 成功样本 | 0 | 50+ | ✅ 显著 |
| 平均距离 | 229 米 | < 100 米 | ✅ 显著 |
| 距离 < 20 米 | 0% | > 20% | ✅ 显著 |
| 评估时间 | ~21 小时 | ~50 小时 | ⚠️ 更长 |

### 调整成功阈值后（SUCCESS_DISTANCE = 50）

| 指标 | 修改前 | 预期修改后 | 改进 |
|------|--------|-----------|------|
| 成功样本 | 0 | 100+ | ✅ 显著 |
| 成功率 | 0% | > 10% | ✅ 显著 |

---

## 📝 总结

### 主要问题

1. **所有样本都没有成功完成**（0 个 success/oracle）
2. **无人机距离目标非常远**（平均 229 米）
3. **路径规划频繁失败**（`move on path stuck`）
4. **碰撞检测频繁触发**（74 次）

### 根本原因

- 路径规划器无法处理长距离导航
- 无人机移动速度太慢（平均 3.13 米/帧）
- 最大步数限制（200 步）不足以完成远距离目标
- 碰撞检测过于敏感

### 解决方案

1. **立即执行**：增加 `maxWaypoints` 到 500
2. **备选方案**：调整成功阈值到 50 米
3. **高级优化**：实现分段导航

### 预期改进

- 增加步数后，应该有更多样本能达到成功阈值
- 优化路径规划后，无人机移动速度应该更快
- 调整碰撞检测后，应该减少误判

---

## 📚 相关文件

### 日志文件

- **/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_20260117_160717.log** - 原始日志（最新 100 行显示中断原因）
- **/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/** - 评估结果（504 个样本，37% 进度）

### 代码文件

- **scripts/eval_qwen.sh** - 评估脚本（需要修改 maxWaypoints）
- **utils/env_utils_uav.py** - 成功阈值配置（SUCCESS_DISTANCE）
- **src/vlnce_src/assist.py** - 碰撞检测配置
- **src/vlnce_src/env_uav.py** - 路径规划和任务完成判断

### 分析文档

- **ANALYSIS_NO_SUCCESS.md** - 详细问题分析
- **SOLUTION_SUMMARY.md** - 解决方案总结
- **README_EVAL_ANALYSIS.md** - 文档导航
- **EVAL_ANALYSIS_FULL.md** - 本文档（合并版）

---

## 🔍 问题根源分析

### 1. 最大步数不足（核心问题）

```python
# env_uav.py:441
elif self.sim_states[index].step >= int(args.maxWaypoints):
    self.sim_states[index].is_end = True
```

- `maxWaypoints = 200`
- 每步 5 个航点，最多 1000 个航点
- 对于 229 米的平均距离，需要约 73 帧
- 但实际可能需要更多（考虑路径规划失败、碰撞等）

### 2. 路径规划失败

```
move on path api: stuck max len
```

- 路径规划器无法处理长距离目标
- 无人机可能卡在原地

### 3. 碰撞检测敏感

```
collision type: close
```

- 出现 74 次碰撞
- 深度图像差异阈值可能太低

### 4. 距离目标太远

- 94.2% 的样本距离 > 100 米
- 无人机移动速度慢（3.13 米/帧）
- 无法在有限步数内到达目标

---

## 📈 数据分析详情

### 距离分布统计

| 距离范围 | 次数 | 占比 | 问题 |
|---------|------|------|------|
| < 20 米 | 0 | 0.0% | ❌ 成功阈值 |
| 20-50 米 | 0 | 0.0% | ❌ 太远 |
| 50-100 米 | 25 | 5.8% | ⚠️ 较少 |
| 100-200 米 | 147 | 33.9% | ⚠️ 较多 |
| 200-300 米 | 168 | 38.7% | ❌ 主要问题 |
| 300-400 米 | 94 | 21.7% | ❌ 严重 |

### 单个样本分析（004de33f-a963-4f7e-acc7-c517c3d0b6b0）

```
起始距离：290.66 米
最终距离：184.14 米
距离减少：106.51 米
移动速度：3.13 米/帧
向目标移动比例：97.1%
达到成功阈值（20米）：0 帧
```

**结论**：无人机确实在向目标移动，但速度太慢，无法在 200 步内到达目标。

---

## 🎯 推荐操作顺序

### 第一步：增加最大步数（立即执行）

1. **修改配置**
   ```bash
   sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh
   ```

2. **运行评估**
   ```bash
   bash /home/yyx/TravelUAV/scripts/eval_qwen.sh
   ```

3. **等待完成**（预计需要 2-3 天）

4. **检查结果**
   ```bash
   # 查看成功样本
   ls /sim/data/TravelUAV_data/eval_closeloop/eval_qwen/ | grep "^success_" | wc -l
   ```

### 第二步：如果效果不明显

1. **调整成功阈值**
   ```bash
   # 修改 utils/env_utils_uav.py
   self.SUCCESS_DISTANCE = 50
   ```

2. **减少碰撞检测敏感度**
   ```bash
   # 修改 src/vlnce_src/assist.py
   # 调整阈值参数
   ```

3. **重新运行评估**

### 第三步：高级优化

1. **实现分段导航**
2. **优化路径规划算法**
3. **调整模型参数**

---

## 📊 对比分析

### 与 Gr00t 的对比

| 指标 | Gr00t | TravelUAV (Qwen3-VL) |
|------|-------|---------------------|
| 成功样本 | 未知 | 0 |
| 平均距离 | 未知 | 200+ 米 |
| 路径规划 | move_on_path | move_on_path |
| 完成检测 | DINO | DINO |

### 问题对比

| 问题 | Gr00t | TravelUAV |
|------|-------|-----------|
| `move on path stuck` | 可能有 | 频繁出现 |
| 碰撞检测 | 可能有 | 频繁出现 |
| 距离目标远 | 可能有 | 非常远 |

---

## 💡 快速修复指南

### 修改 maxWaypoints

```bash
# 1. 修改配置
sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh

# 2. 验证修改
grep "maxWaypoints" /home/yyx/TravelUAV/scripts/eval_qwen.sh

# 3. 运行评估
bash /home/yyx/TravelUAV/scripts/eval_qwen.sh
```

### 修改成功阈值

```bash
# 1. 修改配置
sed -i 's/self.SUCCESS_DISTANCE = 20/self.SUCCESS_DISTANCE = 50/g' /home/yyx/TravelUAV/utils/env_utils_uav.py

# 2. 验证修改
grep "SUCCESS_DISTANCE" /home/yyx/TravelUAV/utils/env_utils_uav.py
```

### 修改碰撞检测阈值

```bash
# 1. 修改 assist.py
sed -i 's/0\.1 \* current_episode/0.2 * current_episode/g' /home/yyx/TravelUAV/src/vlnce_src/assist.py
sed -i 's/diffs < 3/diffs < 5/g' /home/yyx/TravelUAV/src/vlnce_src/assist.py
sed -i 's/distance < 0\.1/distance < 0.2/g' /home/yyx/TravelUAV/src/vlnce_src/assist.py
```

---

## 📞 获取帮助

如果遇到问题：

1. **查看日志**：`/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/`
2. **检查环境**：确保 AirSim 服务器运行
3. **验证配置**：检查 `scripts/eval_qwen.sh` 中的参数
4. **查看文档**：本文件中的解决方案部分

---

## 🎯 总结

### 主要问题

1. **评估进程中断**（37% 进度，504/1359）
2. **AirSim 仿真器超时**（5 次超时，持续 20 分钟）
3. **状态获取失败**（TypeError: NoneType）
4. **最大步数不足**（200 步无法完成远距离导航）

### 根本原因

- **AirSim 仿真器问题**：仿真器崩溃或卡死，导致 RPC 超时
- **最大步数不足**：200 步无法完成平均 229 米的导航
- **资源问题**：可能 GPU/CPU/内存不足
- **环境文件问题**：Carla_Town10HD 环境可能异常

### 解决方案

#### 立即执行

1. **重启 AirSim 仿真器**
   ```bash
   pkill -f AirVLNSimulatorServerTool
   cd /home/yyx/TravelUAV/airsim_plugin
   python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
   ```

2. **增加最大步数**
   ```bash
   sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh
   ```

3. **重新运行评估**
   ```bash
   bash /home/yyx/TravelUAV/scripts/eval_qwen.sh
   ```

#### 备选方案

1. **检查系统资源**
   ```bash
   nvidia-smi  # 检查 GPU
   htop        # 检查 CPU/内存
   ```

2. **检查环境文件**
   ```bash
   ls -lh /sim/data/TravelUAV_data/sim_envs/Carla_Town10HD/
   ```

3. **减少 batch size**（如果资源不足）
   ```bash
   # 修改 eval_qwen.sh
   --batchSize 1  # 从 2 减少到 1
   ```

### 预期改进

| 指标 | 修改前 | 预期修改后 | 改进 |
|------|--------|-----------|------|
| 评估进度 | 37% (504/1359) | 100% (1359/1359) | ✅ 完成 |
| 成功样本 | 0 | 50+ | ✅ 显著 |
| 平均距离 | 229 米 | < 100 米 | ✅ 显著 |
| 评估时间 | 62 分钟（中断） | ~50 小时 | ⚠️ 更长 |

### 文档更新时间

- **2026-01-20**: 添加评估进程中断分析
- **2026-01-19**: 初始文档
