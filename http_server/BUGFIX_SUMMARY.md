# HTTP 服务端 Bug 修复总结

## 📋 修复概览

| 阶段 | 问题 | 状态 |
|------|------|------|
| 阶段 1 | 图像处理 - list 类型未转换 | ✅ 已修复 |
| 阶段 2 | 模型推理 - 梯度问题 | ✅ 已修复 |
| 阶段 3 | 测试数据 - 尺寸异常 | ✅ 已修复 |
| 阶段 4 | 轨迹模型 - 归一化问题 | ⚠️ 需要验证 |

---

## 🐛 阶段 1: 图像处理 Bug

### 问题

```
TypeError: only a single or a list of entries is supported but got type=<class 'int'>
```

### 原因

客户端发送 JSON 时，numpy 数组被序列化为 list：
```python
# 客户端发送
{"rgb": [[0, 0, 0], ...]}  # list

# 服务端接收
obs["rgb"] = [[0, 0, 0], ...]  # Python list
```

服务端代码只处理了 numpy array，未处理 list：
```python
# 旧代码
if isinstance(rgb_array, np.ndarray):
    image = Image.fromarray(rgb_array).convert("RGB")
else:
    image = rgb_array  # ❌ 直接赋值 list
```

### 修复

```python
# 新代码
if isinstance(rgb_array, np.ndarray):
    image = Image.fromarray(rgb_array).convert("RGB")
elif isinstance(rgb_array, list):
    # ✅ 新增：处理 list 类型
    rgb_array = np.array(rgb_array, dtype=np.uint8)
    image = Image.fromarray(rgb_array).convert("RGB")
else:
    raise ValueError(f"Unsupported RGB type: {type(rgb_array)}")
```

**文件**: `http_server/server/server.py:211-223`

---

## 🐛 阶段 2: 梯度问题

### 问题

```
RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.
```

### 原因

轨迹模型输出的张量需要梯度计算，但尝试直接转换为 numpy：
```python
# 旧代码
refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
```

### 修复

```python
# 新代码
refined_waypoints = waypoints_traj.detach().cpu().to(dtype=torch.float32).numpy()
```

**文件**: `src/model_wrapper/qwen3vl_gpu_native.py:448`

**说明**:
- `.detach()`: 移除张量的梯度计算图
- `.cpu()`: 将张量移动到 CPU
- `.to(dtype=torch.float32)`: 转换为 float32
- `.numpy()`: 转换为 numpy 数组

---

## 🐛 阶段 3: 测试数据尺寸问题

### 问题

```
The channel dimension is ambiguous. Got image shape torch.Size([3, 1, 3])
```

### 原因

测试脚本使用了过小的图像尺寸：
```python
# 旧代码
"rgb": [[0, 0, 0]],  # 只有 1x3 个像素！
```

**预期**: `(256, 256, 3)` - (H, W, C)
**实际**: `(1, 3)` - (H, W) 但 H=1, W=3

### 修复

```python
# 新代码
"rgb": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8).tolist(),
```

**文件**: `http_server/tests/test_server.py:64`

---

## ⚠️ 阶段 4: 轨迹模型归一化问题（待验证）

### 问题

```
ValueError: mean must have 1 elements if it is an iterable, got 3
```

### 原因分析

这个错误在之前的测试中出现，但可能已经被修复：

1. **图像尺寸问题**（已修复）
   - 旧测试数据: 1x3 像素
   - 新测试数据: 256x256 像素
   - 这可能导致图像处理器行为不同

2. **CLIP 图像处理器配置**
   ```python
   clip_processor_path = "/home/yyx/TravelUAV/Model/LLaMA-UAV/llamavid-archive/processor/clip-patch14-224"
   image_processor = CLIPImageProcessor.from_pretrained(clip_processor_path)
   ```

   **检查结果**:
   - ✅ `image_mean`: [0.48145466, 0.4578275, 0.40821073] (3 个元素)
   - ✅ `image_std`: [0.26862954, 0.26130258, 0.27577711] (3 个元素)
   - ✅ 配置正确，支持 3 通道 RGB

### 验证步骤

重启服务端并运行测试：

```bash
# 终端 1（如果服务端已停止）
bash http_server/start_server.sh --no-4bit

# 终端 2
python http_server/tests/test_server.py
```

---

## 📝 已修复的文件

| 文件 | 修改内容 | 行数 |
|------|----------|------|
| `http_server/server/server.py` | 添加 list 类型处理 | 211-223 |
| `src/model_wrapper/qwen3vl_gpu_native.py` | 添加 `.detach()` | 448 |
| `http_server/tests/test_server.py` | 修复测试数据尺寸 | 64 |

---

## 🎯 测试验证

### 修复后的预期行为

```
============================================================
测试 3: act 端点（简单数据）
============================================================
✓ act 端点请求成功
  状态码: 200
  响应: {'actions': [1]}
  动作: [1]

============================================================
测试总结
============================================================
✓ 通过: 健康检查
✓ 通过: 根端点
✓ 通过: act 端点（简单）
✓ 通过: act 端点（完整）
✓ 通过: 客户端库
✓ 通过: 多次请求

总计: 6/6 测试通过
============================================================

🎉 所有测试通过！HTTP 服务端工作正常。
```

---

## 🔍 调试建议

如果仍然出现错误，请按以下步骤调试：

### 1. 检查图像尺寸

```python
# 在 server.py:224 后添加
print(f"RGB 数组形状: {rgb_array.shape}")
print(f"Image 尺寸: {image.size}")
```

### 2. 检查 episode 格式

```python
# 在 server.py:237 后添加
print(f"Episode RGB 类型: {type(episode[0]['rgb'][0])}")
print(f"Episode RGB 尺寸: {episode[0]['rgb'][0].size}")
```

### 3. 检查轨迹模型输入

```python
# 在 travel_util_clean.py:85 后添加
print(f"Images 形状: {images.shape}")
print(f"Images dtype: {images.dtype}")
```

---

## 📊 修复效果对比

### 修复前

```
客户端 → JSON → 服务端 → list → ❌ 错误
```

### 修复后

```
客户端 → JSON → 服务端 → list → ✅ 转换为 numpy → PIL Image
                                      ↓
                                  模型推理 → ✅ detach() → numpy
                                      ↓
                                  返回动作 [1]
```

---

## 🎯 下一步操作

### 1. 重启服务端

```bash
# 如果服务端还在运行，先停止
# 然后重新启动
bash http_server/start_server.sh --no-4bit
```

### 2. 运行测试

```bash
python http_server/tests/test_server.py
```

### 3. 验证结果

- ✅ 所有 6 个测试通过 → 修复完成
- ❌ 仍有错误 → 查看错误信息，继续调试

---

## 📝 修复记录

| 日期 | 阶段 | 问题 | 状态 |
|------|------|------|------|
| 2026-01-19 | 阶段 1 | 图像处理 - list 类型 | ✅ 已修复 |
| 2026-01-19 | 阶段 2 | 梯度问题 | ✅ 已修复 |
| 2026-01-19 | 阶段 3 | 测试数据尺寸 | ✅ 已修复 |
| 2026-01-19 | 阶段 4 | 归一化问题 | ⚠️ 待验证 |

---

**修复者**: Claude Code
**修复时间**: 2026-01-19
