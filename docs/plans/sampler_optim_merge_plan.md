# 采样优化合入计划书（方案B）

## 1. 背景与目标
本计划用于将 `github_xllm_2` 中与采样相关的优化按“小步、可验证、可回滚”的方式，逐步合入到 `github_xllm` 的 `feat/sampler_optim` 分支，最终进入正式仓库。

## 2. 仓库与分支
- 源仓库（参考实现）：`/Users/maxiaolong.maxwell/Documents/code/github_xllm_2`
- 目标仓库（合入）：`/Users/maxiaolong.maxwell/Documents/code/github_xllm`
- 目标分支：`feat/sampler_optim`

## 3. 合入顺序（方案B）
按以下顺序分步合入，每步独立验证与回滚：
1) logit_softmax 自定义算子
2) TopK 缺失功能补齐（排序开关 + workspace/输出缓存）
3) sampler 优化
4) beamsearch 优化
5) sampler + beamsearch (+logits) CUDA Graph
6) NVTX 标识与开关

## 4. 关键决策与约束
- TopK 默认排序语义：保持现有行为（`sorted_by_value = true`）以避免语义变化。
- TopK 缓存策略：使用 wrapper（`air_topk_last_dim`）集中管理。
- Warp TopK（k<=32）与超大 K 兜底暂不关注，不作为本次合入范围。

## 5. 分步内容与验证

### 5.1 步骤1：logit_softmax 自定义算子
**目的**：引入自定义算子替代/优化现有路径，确保数值与性能收益。
**变更范围**：自定义算子实现文件 + 注册/编译接入。
**验证**：
- 数值一致性：与原实现对比输出差异（允许浮点容差）。
- 性能对比：同模型/同 batch/同长度下耗时对比。
**回滚**：恢复算子注册与调用路径至原实现。

### 5.2 步骤2：TopK 缺失功能补齐（重点）
**目的**：补齐 `sorted_by_value` 开关与缓存机制。
**拟引入能力**：
- `sorted_by_value` 参数（默认 true）
- workspace size 缓存（thread_local）
- workspace Tensor 缓存（按 device）
- 输出 Tensor 缓存（按 device/batch/k/dtype）
**变更范围**：
- 新增 `air_topk_last_dim.{h,cu}` wrapper
- `compute_topk_general` / sampler 调用替换
**验证**：
- `sorted_by_value=true` 时输出顺序与当前一致
- `sorted_by_value=false` 允许无序但 TopK 集合一致
- 重复调用减少 workspace size 查询与 cuda malloc
**回滚**：改回原 `compute_topk_general` 直接调用 `invokeTopkLastDim`。

### 5.3 步骤3：sampler 优化
**目的**：合入 sampler 优化路径与开关。
**变更范围**：sampler 逻辑、相关 flags、调用路径。
**验证**：
- 采样结果一致性（相同 seed/输入）
- 性能对比
**回滚**：关闭开关或恢复原逻辑。

### 5.4 步骤4：beamsearch 优化
**目的**：合入 beamsearch 计算路径优化。
**变更范围**：beamsearch 核心路径与相关 flags。
**验证**：
- 结果一致性（相同输入、beam_width）
- 性能对比
**回滚**：关闭开关或恢复原逻辑。

### 5.5 步骤5：sampler + beamsearch (+logits) CUDA Graph
**目的**：减少 launch 开销与同步，提升 decode 性能。
**变更范围**：graph capture/execute、切换逻辑与 flags。
**验证**：
- graph 开关前后性能对比
- 关键日志验证（capture/execute 行为）
**回滚**：关闭 graph 开关或回退 graph 代码。

### 5.6 步骤6：NVTX 标识与开关
**目的**：提升 Nsight Systems 可观察性。
**变更范围**：NVTX 包裹点 + flags。
**验证**：
- nsys 轨迹中出现预期的 range 名称
- 开关关闭时无额外 NVTX 开销
**回滚**：关闭开关或移除 NVTX 包裹。

## 6. 风险与对策
- **排序语义变化**：默认保持 `sorted_by_value=true`，避免行为变更。
- **缓存带来的资源占用**：采用 thread_local + 按需扩容策略，必要时可加上限或手动释放机制。
- **graph 捕获不稳定**：保留回退路径，必要时仅在 decode 启用。

## 7. 验收与交付
- 每步合入提交独立、可单独回退
- 给出基线性能与目标性能对比
- 关键开关与日志说明文档更新
