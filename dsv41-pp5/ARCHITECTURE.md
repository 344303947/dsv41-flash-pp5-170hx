# DSV4.1-Flash PP5 架构与版本特性 — v41-pp5-1m-20260914

## 一、版本特性（本版 vs 原始）

| 特性 | 原始状态 | 本版 |
|---|---|---|
| 专家部署 | 卸 11/9/9/9/10 GB 到 CPU（UVA 零拷贝，每 token 走 PCIe） | **全驻留 0 卸载**（staged Marlin repack） |
| 上下文 | 131072（128K） | **1048576（1M）**，896K token 实测通过 |
| 乱码 | 长上下文（>800 token）非确定性乱码 | 修复（影子 topk + eager break） |
| legacy 模式 | 无法启动（KV layout 断言 / group 越界） | 可启动 |
| P1 延迟 | 13.3s | **1.5s（~9x）** |
| prefill | ~100-1000 tok/s | **~4.6k tok/s（60K）/ ~2.3k（896K）** |
| KV 池 | 1.13M tokens @128K | **1.63M tokens @1M，并发 1.56x** |

## 二、模型架构（DeepSeek-V4.1-Flash）

```
40 层解码器（每层 ~7.4GB）
├── Dense 层 L0-1：标准 MLA 注意力
├── 压缩层 L2-39：CSA2 稀疏注意力（compress_ratio 1/2）
│   ├── kv_source_layer_ids = [2, 8, 14, 20]
│   │   （L3-7 读 L2 的压缩 KV/indexer K/topk；L9-13 读 L8；…；L21-39 读 L20）
│   ├── compressor：池化 + 门控 → 压缩 KV latent（512 dim）
│   └── indexer：Lightning Indexer（topk=2048，2 级候选过滤）
├── MoE：384 路由专家（MXFP4）+ 共享专家，Marlin 内核
├── mHC 超连接（hc_mult 流，tilelang 内核）
└── Engram（L1/L14）：n-gram hash 表 384M 行 × 256（FP8，203GB）

Checkpoint 510GB = FP8 dense 权重 + MXFP4 路由专家 + FP8 engram 表
```

## 三、并行架构：PP5 影子均衡（8,8,8,8,8）

### 为什么需要影子（问题）

v4.1 的压缩层必须读**同组 kv-source 层**的 cache，上游要求 source 与消费者同 rank。
40 层中 {20-39} 组占 147.8GB 权重，任何"合法"切分都会让某卡超载：

```
legacy 2,6,6,6,20：rank4 独扛 20 层 → 必须卸 102GB 专家 → decode 仅 10-16 t/s
```

### 影子方案（解法）

消费者 rank 本地实例化源层 attention 的**副本**（ShadowSource），从 PP 边界接收源层注意力输入 x（[T,5120]bf16 ≈10KB/token）**本地重算** cache：

```
rank0 (L0-7)     rank1 (L8-15)      rank2 (L16-23)      rank3 (L24-31)     rank4 (L32-39)
                                    ┌ shadow L14 ◄── x@L14 (捕获于 L14 attn 入口)
   …            L8 [源] ──L14 [源]──┤
                                    └ L20 [源] ──────►  ┌ shadow L20 ◄── x@L20
                                                        └ L20 [源] ──────►  ┌ shadow L20
                                                                             (relay)
```

影子重放链（与真实源层 bit-identical）：
`x → qr_kv 投影 → kv_score → compressor(压缩KV) → insert_cache → indexer K → indexer_op(topk)`

### 数据流（每步）

1. 源层 owner 在 attention 入口 tee 一份 x 到 `shadow_feed_buffer`（图内 copy）
2. PP 边界把 `shadow_x_{S}` 塞进 IntermediateTensors 发给下游（relay 链）
3. 消费者 rank 在本地层运行**前**先跑影子 `produce_kv_side_effects`（breakable-cudagraph eager 段）
4. 消费者层（L16-19 等）照常读共享 cache（topk_indices_buffer / compressed cache）——**消费者代码零改动**

## 四、显存布局（全驻留，5×A100-64G）

```
每卡 63.07 GiB：
├── 权重（含影子 attention）≈ 57-59 GiB
│   rank0 = 61.5GB 十进制 + embed + vision
│   rank4 = 60.7GB 十进制 + head + shadow20
├── KV pool：1.44 GiB（util 0.97）
├── 运行余量：~1.9 GiB（(1-0.97)×物理）
└── engram 表：不在 GPU（rank0/1 各 94.42 GiB 在 pinned CPU，UVA 随机读）

CPU RAM：503GB（engram pinned 189GiB + 页缓存）

engram 表用 `cudaHostAlloc` 按**精确字节数**分配（`_exact_pinned_tensor`，`envs.VLLM_ENGRAM_EXACT_PIN`）：
torch 的 pinned 分配器会把每次分配向上取整到 2 的幂并 pin 整块，91.55 GiB 的 weight
会实占 128 GiB（每 rank 白占 37.6 GiB）。精确分配后每 rank 94.42 GiB，两 rank 共省 75 GiB。
`cudaHostAlloc` 内存 `is_pinned()` 为真，故 UVA 视图路径不变。
```

**全驻留关键：staged Marlin repack**
vLLM 原版 repack 同时持有 raw+packed（~6.7GiB/层 transient）→ 全驻留差 0.3-1.6GiB OOM。
本版把 raw 先移到 pageable host RAM、清引用、逐专家打包 → GPU 峰值只剩 packed。

## 五、运行时栈

```
vllm-backport（fork, v0.13.1.dev6）
├── 并行：PP5 × TP1（heads=64/kv_heads=1 不可被 5 整除，TP5 不可行）
├── cudagraph：FULL_AND_PIECEWISE + breakable cudagraph
│   └── eager break 段：attention prep、影子重放、sparse indexer（host 分支不进图）
├── 量化内核：Marlin MXFP4（专家）/ MXFP8（dense）；A100 无 DeepGEMM → Triton 回退
├── KV cache：fp8_ds_mla（576B/token/层）+ BLHNC layout
├── 调度：V1 runner（engram lookback 需要）+ prefix caching + chunked prefill（2048）
└── 通信：NCCL Ring/Simple（PCIe Gen2 x16，无 NVLink）
```

## 六、本版对 fork 的关键改动（10 文件）

1. **影子源层**：`shadow_source.py`（新）+ `model.py` + `attention.py`
2. **乱码修复**：`produce_kv_side_effects` 加 eager break + 补 `indexer_op`（长上下文 topk）
3. **PP 修复**：KV layout 交集（`utils.py`）+ KV group 索引对齐（`kv_cache_utils.py`）
4. **全驻留**：staged repack（`marlin_utils_fp4.py` + `mxfp4.py`）
5. **1M 支持**：indexer gather cap + `PREFILL_CHUNK_SIZE` 预算自适应 + logits 上限
6. **每 rank 卸载预算**：`VLLM_CPU_OFFLOAD_GB_PER_RANK`（`envs.py` + `offloader/base.py`）

## 七、性能实测

| 场景 | 数据 |
|---|---|
| P1（短） | 1.5s（"42"） |
| P2（~110 token 生成） | 2.7s |
| P3（888 token prompt） | 7.8s |
| 60K prompt | 12.5s |
| 896K prompt | 388s（prefill ~2.3k tok/s） |
| decode 吞吐 | ~30-40 tok/s（日志 generation 29.6 t/s） |
| 确定性 | temperature=0 输出完全一致 |
