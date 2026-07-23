# OneRec MLA 落地改动清单 (feature/onerec-mla-v0.9.0)

> 本文档是研读阶段产出,供实施参考。基线: `release/v0.9.0` @ `8c94d568`,
> 子模块 `xllm_atb_layers` @ `ed3eac5`。所有 file:line 以该基线为准,实施前请复核。

## 0. 背景与既定决策

3B_mla 模型 (`/export/home/maxiaolong/models/moe_model/3B_mla`,root 权限需 sudo)
把 OneRec 的 attention 从 MHA 改成 MLA。改动覆盖三处:
- encoder(4层) self-attention
- decoder(12层) self-attention
- decoder(12层) cross-attention (EncDecAttention)

**每处 MLA 权重结构** (safetensors 探测确认,所有层 shape 一致):

| 权重 | shape | 说明 |
|------|-------|------|
| `q_proj.weight` | [2048, 2048] | Q 投影,输出 = n_heads(16) × d_kv(128) |
| `kv_a_proj.weight` | [512, 2048] | KV 低秩下投影,输出 = kv_lora_rank(512) |
| `kv_a_layernorm.weight` | [512] | 压缩后 RMSNorm (无 bias) |
| `kv_b_proj.weight` | [4096, 512] | KV 上投影,输出 = n_heads(16) × (nope128 + v128) |
| `o_proj.weight` | [2048, 2048] | 输出投影 |

**三个既定决策:**
1. 实现路径 = OneRec 内嵌 MLA 子图 (不复用 deepseekV2::LatentAttention)
2. decode KV cache = 存解压后 full K/V (16 head),XAttention op / FIA / cache select 零改动
3. MHA/MLA 区分 = 权重 key 自动探测 (config.json 无标志位)

**⚠️ 待验证假设 — kv_b_proj split 语义:**
`[4096,512]` = n_heads(16) × (nope_dim(128) + v_dim(128))。reshape 成
`[tokens, 16, 256]` 后在最后一维 split 成 K[128]/V[128] —— **per-head interleave,
NOT 简单前2048=K/后2048=V**。维度完美吻合 DeepSeek 标准,uniattention
`deepseekv3_attention.py:187-188` 佐证,但缺 3B_mla 训练侧代码直接确认。
**首次 correctness smoke 必须用真实输出验证此点。** 实现时把 split 做成清晰、
易改的独立函数。3B_mla 无 RoPE (qk_rope_head_dim=0,nope_dim = v_dim = d_kv = 128)。

---

## 1. 改动分层总览

```
config 层    onerec.h (REGISTER_MODEL_ARGS)  → 复用已有 MLA ModelArgs 字段
             ↓
探测层       npu_onerec_block_layer_impl.cpp::load_state_dict
             → 检测 kv_a_proj key 存在即 MLA,推断 kv_lora_rank
             ↓
参数层       block_layer.h (BlockLayerParam) + param_from_args
             → use_mla / kv_lora_rank / nope_dim / v_dim
             ↓
权重加载层   npu_onerec_block_layer_impl.cpp
             → 枚举槽位 + name→index 映射表 + merge (跳过 QKV pack)
             ↓
组图层       block_layer.cpp (AddSelfAttentionMLA / AddCrossAttentionMLA)
             → 复用 fusion_attention 的 MLA 子图或新写投影 helper
             ↓
消费端       XAttention op (self decode) / FIA (cross) — 零改动
```

---

## 2. 逐文件改动清单

### 2.1 `xllm/models/rec/onerec.h` — model_args (改动小)

`REGISTER_MODEL_ARGS(onerec, ...)` (当前 :230-274) 新增 MLA 字段加载。
ModelArgs 已有现成字段 (`model_args.h:136-140`),无需新增 PROPERTY,直接复用:

```cpp
// 参照 xllm/models/llm/deepseek_v2.h:237-241
LOAD_ARG_OR(qk_nope_head_dim, "qk_nope_head_dim", 0);   // 3B_mla config 无此字段→0,靠探测覆盖
LOAD_ARG_OR(v_head_dim, "v_head_dim", 0);
LOAD_ARG_OR(kv_lora_rank, "kv_lora_rank", 0);
LOAD_ARG_OR(q_lora_rank, "q_lora_rank", 0);             // 3B_mla q_proj 全秩 → 0
```

注意: 3B_mla config.json **没有**这些字段,`LOAD_ARG_OR` 会全部取默认 0。
真实值靠 §2.2 从权重 shape 推断后回填。

### 2.2 `xllm/core/layers/npu/npu_onerec_block_layer_impl.cpp` — 探测 + 权重加载 (改动大)

**(a) 枚举槽位** (当前 `OneRecBlockLayerTensorId` :107-209 / `OneRecMoeBlockLayerTensorId`)
为 MLA 新增槽位。推荐**复用现有 self/cross attn 的 Q/K/V/O 槽位语义**减少改动:
- `IN_Q_WEIGHT` ← q_proj (复用)
- `IN_K_WEIGHT` ← kv_a_proj (复用槽位,但语义变 = compressed 下投影)
- `IN_V_WEIGHT` ← kv_b_proj (复用槽位,语义变 = 上投影)
- `IN_SELF_ATTN_OUT_WEIGHT` ← o_proj (复用)
- **新增** `IN_KV_A_LAYERNORM_WEIGHT` (self) / `IN_CROSS_KV_A_LAYERNORM_WEIGHT` (cross)
  — 参照现有 layer_norm 槽位(只加载 .weight,无 bias)

⚠️ 若新增槽位使总数变化,需同步上调 `kOneRecWeightCountPerLayer=79` (:96) /
`kOneRecMoeWeightCountPerLayer=97` (:100),以及构造函数 resize (:701)、
init_node 绑定、variant_pack 分界。

**(b) name→index 映射表** (`kOneRecEncoderWeightMapping` :522-553 /
`kOneRecDecoderWeightMapping` :555-589) 新增 MLA key 条目:

```cpp
// self-attention (encoder + decoder)
{"layer.0.SelfAttention.q_proj.weight",         kInQWeight},
{"layer.0.SelfAttention.kv_a_proj.weight",      kInKWeight},
{"layer.0.SelfAttention.kv_a_layernorm.weight", kInKvALayernormWeight},  // 新槽
{"layer.0.SelfAttention.kv_b_proj.weight",      kInVWeight},
{"layer.0.SelfAttention.o_proj.weight",         kInSelfAttnOutWeight},
// cross-attention (decoder only, layer.1.EncDecAttention.*)
{"layer.1.EncDecAttention.q_proj.weight",         kInCrossQWeight},
{"layer.1.EncDecAttention.kv_a_proj.weight",      kInCrossKWeight},
{"layer.1.EncDecAttention.kv_a_layernorm.weight", kInCrossKvALayernormWeight}, // 新槽
{"layer.1.EncDecAttention.kv_b_proj.weight",      kInCrossVWeight},
{"layer.1.EncDecAttention.o_proj.weight",         kInCrossAttnOutWeight},
```
(同时保留现有 MHA 的 q/k/v/o 条目,两套并存,靠权重实际存在的 key 命中)

**(c) 探测 MLA** (在 `load_state_dict` :1016 内):
遍历 state_dict 时检测是否存在以 `kv_a_proj.weight` 结尾的 key。存在 → 置
`is_mla_ = true`,并从 `kv_a_proj` shape[0] 推断 `kv_lora_rank_`(=512),
从 `kv_b_proj` shape[0] / n_heads 推断 per-head (nope+v)。

**(d) merge_loaded_weights** (:936-958) — **最关键改动**:
当前无条件把 self-attn Q/K/V `torch::cat` pack 成一个 (:946-958)。
MLA 模式下**必须跳过这个 pack**:
```cpp
if (is_mla_) {
  // MLA: q_proj / kv_a_proj / kv_a_layernorm / kv_b_proj 各自独立槽位,不 pack
  // (kv_a_proj 输出512 与 q_proj 输出2048 维度不对齐,无法 cat)
} else {
  // 现有 MHA QKV pack 逻辑 (:946-958)
}
```
cross-attn 当前已经不 pack (:960-961),MLA 沿用即可。

**(e) verify_loaded_weights** (:830-888):
校验逻辑是"遍历映射表检查每个槽非占位"。MLA 映射项与实际权重一一对应即可,
逻辑不用改。但注意 kv_a_layernorm 是 1D [512],占位检测 (sizes[0]==1) 对
1D 张量的判断需确认不误伤。

### 2.3 `third_party/xllm_atb_layers/models/onerec/layer/block_layer.h` — 参数 (改动小)

`BlockLayerParam` (:52-110) 新增:
```cpp
bool use_mla = false;
int kvLoraRank = 512;
int qkNopeHeadDim = 128;   // = d_kv (无 rope)
int vHeadDim = 128;        // = d_kv
```
新增函数声明: `AddSelfAttentionMLA` / `AddCrossAttentionMLA` (或在 fusion 层)。

### 2.4 `third_party/xllm_atb_layers/models/onerec/layer/block_layer.cpp` — 组图 (改动大)

**(a) tensor candidates** (`GetOneRecLayerInTensorCandidates` :71 /
`GetOneRecLayerIntermediateTensorCandidates` :195):
- inTensor 新增 MLA 权重槽 (kv_a_layernorm 等,若不复用现有 weight_0/1/2 槽)
- intermediate 新增 `intermediate_compressed_kv`、`intermediate_compressed_kv_norm`

**(b) param 装配** (`SetSelfAttentionParam` :423 / `SetCrossAttentionParam` :530):
透传 `use_mla` / `kvLoraRank` / `nope_dim` / `v_dim` 到 FusionAttentionParam。

**(c) 组图分发** (`AddDecoderSelfAttention` :703 / `AddEncoderSelfAttention` :630 /
`AddCrossAttention` :790): `if (param.use_mla)` 分支走 MLA 子图。

**MLA 投影子图节点序列** (self & cross 共用模板):
```
1. RMSNorm(in_input)                    → normed        [现有 input_norm 复用]
2. Linear(normed, q_proj)               → intermediate_q
3. Linear(normed, kv_a_proj)            → compressed_kv  [tokens, 512]
4. RMSNorm(compressed_kv, kv_a_layernorm) → compressed_kv_norm
5. Linear(compressed_kv_norm, kv_b_proj)→ kv_combined    [tokens, 4096]
6. reshape [tokens,16,256] + split(-1,[128,128]) → intermediate_k, intermediate_v
7. [下游不变] self decode→XAttention op / cross→FIA
```

### 2.5 组图插入点二选一 (§研读结论)

| 方案 | 位置 | 取舍 |
|------|------|------|
| A. 通用 QKV split 加分支 | `fusion_attention.cpp:735` AddFAttnQKVLinearSplitNode / `qkv_linear_split.cpp:493-515` 加第4条 MLA 分支 | 触及所有用 QKVLinearSplit 的模型,有共享风险;且只覆盖 self-attn,cross 要另改 |
| **B. OneRec 内嵌 (推荐)** | `block_layer.cpp` 新写 MLA 投影 helper,绕开通用 split | 完全局部化,零风险波及其他模型;self/cross 都在 OneRec 内处理 |

决策已定 = **方案 B**。投影子图可封装成 OneRec-only 的 helper 函数
(可放 fusion_attention.cpp 内标注 OneRec-only,或直接在 block_layer.cpp)。

### 2.6 `xllm/core/runtime/rec_worker_impl.cpp` — KV cache shape (改动中)

decode 存解压后 full K/V (16 head),但当前 cache 分配用的是
`decoder_n_kv_heads`(=4,GQA 语义):
- `allocate_unshared_kv_caches` (:1104-1139) local_kv_heads (:1114)
- shared cache 分配 (:1349-1364) local_kv_heads (:1341)

MLA 下解压后是 16 head (= n_heads,非 4)。需在 MLA 模式用 `decoder_n_heads`
或从 kv_b_proj 推断的 head 数替代,否则 cache shape 与 attention op 期望不符。
**这是 xattention 侧唯一需要改的点** (op 本身不改,只是 cache 张量维度)。

---

## 3. 实施顺序建议

```
1. onerec.h: MLA args 加载 (LOAD_ARG_OR)                  → 编译通过
2. block_layer.h: BlockLayerParam 加字段                  → 编译通过
3. npu_..._impl.cpp: 探测 is_mla_ + kv_lora_rank 推断     → 日志验证探测命中
4. npu_..._impl.cpp: 枚举槽 + 映射表 + merge 跳过 pack    → 权重加载不报错
5. block_layer.cpp: MLA 投影子图 + param 装配             → 图构建成功
6. rec_worker_impl.cpp: KV cache shape 用 16 head         → decode 不越界
7. 单请求 correctness smoke (constrained=false)           → 验证 split 语义!
   对比训练侧/参考实现的 round0 top tokens + logprob
8. beam smoke → 全链路 → benchmark
```

**§7 是 kv_b_proj split 假设的验证闸口。** 若 round0 token 就分叉,优先怀疑
split 顺序 (per-head interleave vs block) 和 nope/v 维度切分点。

---

## 4. 关键风险清单

| 风险 | 位置 | 缓解 |
|------|------|------|
| kv_b_proj split 顺序错 | block_layer.cpp MLA 子图 | 首次 smoke 验证,split 做成易改独立函数 |
| KV cache head 数用错(4 vs 16) | rec_worker_impl.cpp | MLA 模式改用 n_heads |
| 权重槽位总数变化未同步常量 | npu_..._impl.cpp :96/:100 | 复用现有 Q/K/V/O 槽,只新增 kv_a_layernorm |
| encoder 也是 MLA 但无 KV cache | block_layer.cpp encoder 分支 | encoder self-attn 走 PA_ENCODER,无 cache,投影同理 |
| MHA 模型回归 | 全部 | 所有改动用 is_mla_/use_mla 分支,默认关闭保持 MHA 路径不变 |
```
