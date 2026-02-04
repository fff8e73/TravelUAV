# TravelUAV 接口更新实施计划

## 📋 文档信息

- **创建日期**: 2026-02-04
- **版本**: v1.0
- **状态**: 待实施

---

## 🎯 更新目标

基于 Isaac-Drone-Navigation-Benchmark 的设计，对 TravelUAV HTTP 接口进行以下更新：

1. **动作格式变更**: 从绝对坐标改为相对位移
2. **框架迁移**: 从 Flask 迁移到 FastAPI
3. **坐标转换**: Client 端实现相对位移到世界坐标的转换

---

## 📊 需求分析

### 1. 动作格式变更

#### 当前格式（旧）
```python
# Server 返回绝对世界坐标
{
    "waypoints": [[x1, y1, z1, yaw1],
                  [x2, y2, z2, yaw2],
                  ...
                  [xN, yN, zN, yawN]]
}
```

#### 新格式（参考 Isaac-Drone-Navigation-Benchmark）
```python
# Server 返回相对位移
{
    "action": [[dx1, dy1, dz1, dyaw1],
               [dx2, dy2, dz2, dyaw2],
               ...
               [dxN, dyN, dzN, dyawN]]  # shape: [N, 4]
}
```

#### 坐标系定义
```
局部坐标系（相对于无人机当前位姿）:
- dx: 前后位移 (+前, -后)
- dy: 左右位移 (+左, -右)
- dz: 垂直位移 (+上, -下)
- dyaw: 偏航角增量 (弧度，逆时针为正)

注意事项:
1. 每一步的 action 是相对于上一步的相对位置
2. 第一步的 action 是相对于当前无人机位姿
3. Client 需要累积计算得到世界坐标
```

### 2. 框架迁移

#### Flask vs FastAPI 对比

| 特性 | Flask | FastAPI |
|------|-------|---------|
| 性能 | 同步，较慢 | 异步，快速 |
| 类型检查 | 无 | 内置 Pydantic |
| 文档生成 | 需手动 | 自动生成 Swagger/ReDoc |
| 异步支持 | 有限 | 原生支持 |
| 学习曲线 | 简单 | 中等 |

#### 迁移原因
1. **性能提升**: FastAPI 基于 Starlette 和 Pydantic，性能更优
2. **类型安全**: 自动验证请求/响应数据
3. **文档自动生成**: 访问 `/docs` 即可查看 API 文档
4. **现代化**: 支持 async/await，更适合 I/O 密集型任务

---

## 🏗️ 实施方案

### 阶段 1: 准备工作（1天）

#### 1.1 依赖更新
**文件**: `requirements_interface.txt`

```diff
# HTTP通信
- flask>=2.0.0
+ fastapi>=0.104.0
+ uvicorn[standard]>=0.24.0
json_numpy>=1.0.1
requests>=2.25.0
```

#### 1.2 创建测试环境
```bash
# 备份当前代码
cp -r src/model_wrapper src/model_wrapper.backup
cp -r server server.backup
```

---

### 阶段 2: Server 端改造（2-3天）

#### 2.1 创建新的 FastAPI Server

**文件**: `server/travel_model_server_v2.py`

核心改动：
1. 使用 FastAPI 替代 Flask
2. 返回格式改为 `[N, 4]` 的相对位移
3. 添加 Pydantic 数据模型验证

---

### 阶段 3: Client 端改造（2-3天）

#### 3.1 添加坐标转换功能

**文件**: `src/model_wrapper/http_client.py`

需要实现 `local_to_world()` 方法：
- 输入：局部坐标系的相对位移 `[N, 4]`
- 输出：世界坐标系的绝对位置 `[N, 4]`

转换步骤：
1. 获取当前无人机位姿（位置 + 四元数）
2. 将四元数转换为旋转矩阵
3. 对每一步相对位移：
   - 应用旋转矩阵转换到世界坐标系
   - 累加到当前位置
   - 更新偏航角
4. 返回世界坐标航点

---

### 阶段 4: 测试与验证（2天）

#### 4.1 单元测试

测试用例：
- 前进动作
- 转向动作
- 上升/下降动作
- 组合动作

#### 4.2 集成测试

使用少量 episodes 进行端到端测试

---

### 阶段 5: 部署与文档（1天）

#### 5.1 更新文档
- README.md ✅ 已完成
- README_INTERFACE.md
- API_REFERENCE.md

#### 5.2 更新脚本
- eval_http.sh
- 启动脚本

---

## 📅 时间表

| 阶段 | 任务 | 预计时间 |
|------|------|----------|
| 1 | 准备工作 | 1天 |
| 2 | Server 端改造 | 2-3天 |
| 3 | Client 端改造 | 2-3天 |
| 4 | 测试与验证 | 2天 |
| 5 | 部署与文档 | 1天 |
| **总计** | | **8-10天** |

---

## ⚠️ 风险与注意事项

### 1. 坐标系转换风险

**风险**: 坐标系定义不一致导致航点计算错误

**缓解措施**:
- 详细的单元测试覆盖各种场景
- 可视化工具验证转换结果
- 参考 Isaac-Drone-Navigation-Benchmark 的实现

### 2. 性能风险

**风险**: FastAPI 迁移可能引入新的性能问题

**缓解措施**:
- 性能基准测试对比
- 使用 async/await 优化 I/O 操作
- 监控内存和 CPU 使用

### 3. 兼容性风险

**风险**: 旧的 Server 实现无法与新 Client 兼容

**缓解措施**:
- 保留旧版本代码作为备份
- 提供版本协商机制
- 渐进式迁移，支持两种格式

---

## ✅ 验收标准

### 功能验收
- [ ] Server 能正确返回 `[N, 4]` 格式的相对位移
- [ ] Client 能正确将相对位移转换为世界坐标
- [ ] 单元测试全部通过
- [ ] 集成测试成功运行完整 episode

### 性能验收
- [ ] 单次推理延迟 < 100ms（不含模型推理时间）
- [ ] 批量推理吞吐量 >= 10 requests/s
- [ ] 内存占用无明显增加

### 文档验收
- [ ] API 文档完整（Swagger/ReDoc）
- [ ] README 更新完成
- [ ] 示例代码可运行

---

## 📚 参考资料

1. **Isaac-Drone-Navigation-Benchmark**
   - 动作格式定义
   - 坐标系转换实现

2. **FastAPI 官方文档**
   - https://fastapi.tiangolo.com/
   - 异步编程最佳实践

3. **Scipy Rotation**
   - https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.html
   - 四元数和欧拉角转换

---

**文档版本历史**:
- v1.0 (2026-02-04): 初始版本
