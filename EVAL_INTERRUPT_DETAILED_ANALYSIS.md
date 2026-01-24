# 评估进程中断详细分析

## 🚨 核心问题：AirSim 仿真器环境文件损坏

---

## 🔍 错误根因分析

### 1. **CarlaUE4 启动失败**

**错误日志来源**: `/sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/LinuxNoEditor/CarlaUE4/Saved/Logs/CarlaUE4.log`

**关键错误**:

```
[2026.01.19-05.12.57:749][  0]LogOutputDevice: Error: Ensure condition failed: VendorId != EGpuVendorId::Unknown
```

**原因**:
- Vulkan RHI 无法识别 GPU 厂商 ID
- 可能是 GPU 驱动问题或 Vulkan 配置错误

---

### 2. **资源文件缺失**

**错误**:
```
[2026.01.19-05.12.58:402][  0]LogStreaming: Error: Couldn't find file for package /Game/Carla/Static/GenericMaterials/Ground/SimpleRoad/CheapRoad
```

**缺失的资源**:
- `/Game/Carla/Static/GenericMaterials/Ground/SimpleRoad/CheapRoad`
- `/Game/Carla/Static/GenericMaterials/Ground/SimpleRoad/CheapSideWalkCurb`
- `/Game/Carla/Static/GenericMaterials/Ground/SimpleRoad/CheapSideWalk_00`
- `/Game/Carla/Static/GenericMaterials/Ground/SimpleRoad/CheapLaneMarking`
- `/Game/Carla/Static/GenericMaterials/WetPavement/WetPavement_Complex_Road_N2`
- `/Game/Carla/Static/GenericMaterials/WetPavement/WetPavement_Complex_Concrete`
- `/Game/Carla/Static/GenericMaterials/Ground/SideWalks/SidewalkN4/WetPavement_SidewalkN4`
- `/Game/Carla/Static/GenericMaterials/LaneMarking/Lanemarking`
- `/Carla/PostProcessingMaterials/DepthEffectMaterial_GLSL`

**原因**:
- Carla_Town10HD 环境文件不完整
- 可能是解压不完整或文件损坏

---

### 3. **深度相机材料缺失**

**错误**:
```
[2026.01.19-05.12.58:405][  0]Error: CDO Constructor (DepthCamera): Failed to find Material'/Carla/PostProcessingMaterials/DepthEffectMaterial_GLSL.DepthEffectMaterial_GLSL'
```

**影响**:
- 深度相机无法正常工作
- 碰撞检测依赖深度图像
- 导致 `getSensorInfo` 超时

---

## 📊 问题时间线

### 阶段 1: 启动阶段（05:12:57）

```
05:12:57 - CarlaUE4 启动
05:12:57 - Vulkan RHI 错误（GPU 厂商 ID 未知）
05:12:58 - 资源文件缺失错误（20+ 个文件）
05:12:58 - 深度相机材料缺失
```

**状态**: CarlaUE4 启动失败或运行异常

### 阶段 2: 评估运行阶段（05:48:03 - 06:08:06）

```
05:48:03 - 第 1 次超时 (setPoses)
05:53:03 - 第 2 次超时 (setPoses)
05:58:04 - 第 3 次超时 (setPoses)
06:03:05 - 第 4 次超时 (setPoses)
06:08:06 - 第 5 次超时 (getSensorInfo)
06:08:06 - 程序崩溃 (TypeError)
```

**状态**: AirSim 仿真器逐渐失去响应

### 阶段 3: 关闭阶段（06:08:09）

```
06:08:09 - CarlaUE4 正常关闭
```

**状态**: 仿真器进程退出

---

## 🎯 问题根源总结

### 根本原因 1: 环境文件不完整

**证据**:
1. **资源文件缺失**: 20+ 个材质和模型文件找不到
2. **深度相机失效**: 关键的 `DepthEffectMaterial_GLSL` 材料缺失
3. **环境文件大小异常**: Town10HD 目录只有 4KB（正常应该几百 MB）

**结论**: Carla_Town10HD 环境解压不完整或文件损坏

---

### 根本原因 2: GPU/Vulkan 配置问题

**证据**:
```
Ensure condition failed: VendorId != EGpuVendorId::Unknown
```

**可能原因**:
- Vulkan 驱动不兼容
- GPU 厂商 ID 无法识别
- NVIDIA RTX PRO 6000 的 Vulkan 支持问题

**影响**: 仿真器无法正常初始化图形渲染

---

### 根本原因 3: 资源加载失败连锁反应

**连锁反应**:
```
资源文件缺失
    ↓
材质加载失败
    ↓
深度相机失效
    ↓
getSensorInfo 超时
    ↓
状态获取失败
    ↓
TypeError: NoneType
    ↓
程序崩溃
```

---

## 🔧 解决方案

### 方案 1: 重新解压环境文件（推荐）

```bash
# 1. 停止所有相关进程
pkill -f eval_qwen.py
pkill -f AirVLNSimulatorServerTool
pkill -f CarlaUE4

# 2. 删除损坏的环境
rm -rf /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD

# 3. 重新解压环境文件
cd /sim/data/TravelUAV_data/sim_envs
unzip -o carla_town_envs.zip -d carla_town_envs/

# 4. 验证文件完整性
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/
```

**预期**:
- Town10HD 目录大小应该 > 100MB
- 包含完整的资源文件

---

### 方案 2: 检查并修复 Vulkan 配置

```bash
# 1. 检查 Vulkan 是否可用
vulkaninfo 2>&1 | head -20

# 2. 检查 NVIDIA 驱动
nvidia-smi

# 3. 设置 Vulkan 设备
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# 4. 验证 Vulkan 配置
vulkaninfo --summary
```

**预期输出**:
```
GPU: NVIDIA RTX PRO 6000
Vendor ID: 0x10de (NVIDIA)
```

---

### 方案 3: 使用不同的环境（临时方案）

如果 Town10HD 无法修复，可以使用其他环境：

```bash
# 检查可用环境
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/

# 使用 Town01（已验证可用）
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town01/
```

**修改评估脚本**:
```bash
# 修改 eval_qwen.sh 中的环境配置
# 将 Carla_Town10HD 改为 Carla_Town01
```

---

### 方案 4: 检查 AirSim 服务端配置

```bash
# 查看 AirSim 服务端启动参数
ps aux | grep AirVLNSimulatorServerTool

# 检查 root_path 是否正确
# 应该是: /sim/data/TravelUAV_data/sim_envs
```

**验证配置**:
```bash
# 检查 AirVLNSimulatorServerTool.py 中的 env_exec_path_dict
grep -A 20 "env_exec_path_dict" /home/yyx/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py
```

---

## 📋 详细检查步骤

### 步骤 1: 检查环境文件完整性

```bash
# 检查 Town10HD 目录结构
find /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/ -type f | wc -l

# 检查关键文件是否存在
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/LinuxNoEditor/CarlaUE4/Content/

# 检查材质文件
find /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/ -name "*CheapRoad*" -o -name "*DepthEffectMaterial*"
```

**预期**:
- 文件数量: > 1000 个
- 关键文件应该存在

---

### 步骤 2: 检查 Vulkan 配置

```bash
# 检查 Vulkan 驱动
ls -lh /usr/share/vulkan/icd.d/

# 检查 NVIDIA Vulkan 支持
nvidia-smi -q | grep "Vulkan"

# 测试 Vulkan
vulkaninfo 2>&1 | grep -E "(GPU|Vendor|Device)"
```

**预期**:
- 存在 `nvidia_icd.json`
- Vulkan 能识别 NVIDIA GPU

---

### 步骤 3: 检查 AirSim 服务端状态

```bash
# 查看服务端日志
cat /sim/data/TravelUAV_data/sim_envs/server.log

# 检查端口监听
netstat -tuln | grep 30000

# 检查进程
ps aux | grep AirVLNSimulatorServerTool
```

**预期**:
- 服务端正在监听 30000
- 无明显错误

---

### 步骤 4: 检查系统资源

```bash
# GPU 显存
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# CPU 和内存
htop

# 磁盘空间
df -h /sim/data/TravelUAV_data/
```

**预期**:
- GPU 显存: 至少 4GB 空闲
- 磁盘空间: 至少 20GB 空闲

---

## 🎯 修复优先级

### 优先级 1: 重新解压环境文件（必须）

**原因**: 资源文件缺失是导致超时的根本原因

**操作**:
```bash
cd /sim/data/TravelUAV_data/sim_envs
unzip -o carla_town_envs.zip -d carla_town_envs/
```

**验证**:
```bash
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/
# 应该看到大量文件，而不是只有 4KB
```

---

### 优先级 2: 检查 Vulkan 配置（必须）

**原因**: GPU 厂商 ID 未知会导致渲染失败

**操作**:
```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
vulkaninfo --summary
```

---

### 优先级 3: 重启 AirSim 服务端（必须）

**原因**: 确保服务端使用修复后的环境

**操作**:
```bash
pkill -f AirVLNSimulatorServerTool
cd /home/yyx/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs
```

---

### 优先级 4: 修改评估配置（可选）

**原因**: 增加步数以完成远距离导航

**操作**:
```bash
sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh
```

---

## 📊 预期修复效果

### 修复前

```
CarlaUE4 启动失败
    ↓
资源文件缺失
    ↓
深度相机失效
    ↓
超时错误 (5 次)
    ↓
程序崩溃
    ↓
进度: 37% (中断)
```

### 修复后

```
CarlaUE4 正常启动
    ↓
资源文件完整
    ↓
深度相机正常
    ↓
无超时错误
    ↓
程序正常运行
    ↓
进度: 100% (完成)
```

---

## 🧪 验证步骤

### 验证 1: 环境文件完整性

```bash
# 检查文件数量
find /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/ -type f | wc -l

# 预期: > 1000 个文件
```

### 验证 2: Vulkan 配置

```bash
# 检查 GPU 厂商
vulkaninfo 2>&1 | grep "vendorID"

# 预期: vendorID = 0x10de (NVIDIA)
```

### 验证 3: AirSim 服务端

```bash
# 检查服务端状态
ps aux | grep AirVLNSimulatorServerTool

# 预期: 进程正在运行
```

### 验证 4: 端口监听

```bash
# 检查端口
netstat -tuln | grep 30000

# 预期: LISTEN 状态
```

---

## 📝 修复记录

| 时间 | 操作 | 状态 |
|------|------|------|
| 2026-01-20 11:30 | 发现环境文件缺失 | ✅ |
| 2026-01-20 11:35 | 发现 Vulkan 配置问题 | ✅ |
| 2026-01-20 11:40 | 发现资源文件缺失 | ✅ |
| 2026-01-20 11:45 | 制定修复方案 | ✅ |
| 2026-01-20 11:50 | 等待用户执行修复 | ⏳ |

---

## 🎯 立即执行

### 执行顺序

```bash
# 1. 停止所有进程
pkill -f eval_qwen.py
pkill -f AirVLNSimulatorServerTool
pkill -f CarlaUE4

# 2. 重新解压环境文件
cd /sim/data/TravelUAV_data/sim_envs
unzip -o carla_town_envs.zip -d carla_town_envs/

# 3. 验证文件
ls -lh /sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/

# 4. 设置 Vulkan 环境
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json

# 5. 重启 AirSim 服务端
cd /home/yyx/TravelUAV/airsim_plugin
python AirVLNSimulatorServerTool.py --port 30000 --root_path /sim/data/TravelUAV_data/sim_envs

# 6. 修改评估配置（可选）
sed -i 's/--maxWaypoints 200/--maxWaypoints 500/g' /home/yyx/TravelUAV/scripts/eval_qwen.sh

# 7. 重新运行评估
cd /home/yyx/TravelUAV
bash scripts/eval_qwen.sh
```

---

## 📚 相关文件

### 日志文件

- **CarlaUE4 日志**: `/sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/LinuxNoEditor/CarlaUE4/Saved/Logs/CarlaUE4.log`
- **AirSim 服务端日志**: `/sim/data/TravelUAV_data/sim_envs/server.log`
- **评估日志**: `/sim/data/TravelUAV_data/eval_closeloop/eval_qwen/logs/eval_qwen_20260117_160717.log`

### 环境文件

- **环境压缩包**: `/sim/data/TravelUAV_data/sim_envs/carla_town_envs.zip`
- **解压目录**: `/sim/data/TravelUAV_data/sim_envs/carla_town_envs/`

### 代码文件

- **AirSim 服务端**: `/home/yyx/TravelUAV/airsim_plugin/AirVLNSimulatorServerTool.py`
- **评估脚本**: `/home/yyx/TravelUAV/scripts/eval_qwen.sh`

---

## 🎯 总结

### 核心问题

**Carla_Town10HD 环境文件不完整导致仿真器无法正常工作**

### 证据

1. ✅ **资源文件缺失**: 20+ 个材质文件找不到
2. ✅ **环境文件过小**: Town10HD 目录只有 4KB
3. ✅ **深度相机失效**: 关键材料缺失
4. ✅ **Vulkan 配置问题**: GPU 厂商 ID 未知

### 解决方案

1. **重新解压环境文件**（必须）
2. **检查 Vulkan 配置**（必须）
3. **重启 AirSim 服务端**（必须）
4. **增加最大步数**（可选）

### 预期效果

- ✅ 仿真器正常启动
- ✅ 无超时错误
- ✅ 评估完成 100%
- ✅ 成功样本 > 50

---

**分析时间**: 2026-01-20 11:50
**分析工具**: Claude Code
**日志文件**: `/sim/data/TravelUAV_data/sim_envs/carla_town_envs/Town10HD/LinuxNoEditor/CarlaUE4/Saved/Logs/CarlaUE4.log`
