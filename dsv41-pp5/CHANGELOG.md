# DeepSeek-V4.1-Flash PP5 版本备份 — 2026-09-14

引擎：`/home/sean/works/vllm-backport`（v0.13.1.dev6）
环境：conda `vllm-v41`；硬件：5×A100-64G（PCIe Gen2 x16，无 NVLink），503GB RAM

## 2026-09-16 增量

### 版本标记

- **tag**: `v41-pp5-1m-pinmem-20260916`
- **HEAD**: `2b652f14dd`（新增 2 个提交：`362948e815` 启 `expandable_segments`、`2b652f14dd` engram 精确尺寸 pinned）

### 改动：engram 表「精确尺寸」pinned 分配（省 75 GiB CPU 内存）

| 文件 | 内容 |
|---|---|
| `vllm/models/deepseek_v4_1/common/engram.py` | 新增 `_exact_pinned_tensor`：`cudaHostAlloc(cudaHostAllocMapped)` 按精确字节数分配 + `torch.frombuffer` 零拷贝包成 CPU 张量；`__init__` 分配改走该路径，任一步失败自动回退 `pin_memory=True` |
| `vllm/envs.py` | 开关 `VLLM_ENGRAM_EXACT_PIN`（默认 1，设 0 即回退） |

**问题**：torch 的 pinned 分配器把每次分配向上取整到 2 的幂并 pin 整块 → 91.55 GiB 的
weight 实占 128 GiB、2.86 GiB 的 scales 实占 4 GiB，每个卸载 rank 浪费 37.6 GiB
（PP0+PP1 合计 75.2 GiB；`/proc/<pid>/mem` 核对：数据区之后 36.446/36.443 与 1.139 GiB 全为 0）。

**为何不需要改 C++**：`cudaHostAlloc` 得到的内存 `is_pinned()` 为真（在 `torch.cuda.init()`
之后由驱动查询得出），因此现成的 UVA 视图路径 `get_accelerator_view_from_cpu_tensor`
仍走零拷贝分支，不复制、不需重编。若 `is_pinned()` 为假，该 helper 会静默复制一整份
（等于 +94 GiB/rank），所以代码里做了显式校验并在失败时回退。

### 验收（2026-09-16，端口 9004 实测）

| 门禁 | 结果 |
|---|---|
| 字节指纹（两 rank weight/scale sha256） | 与改动前**逐字节相同** ✓ |
| CPU pinned（`Shmem`/`free shared`） | 264.3 → **189.1 GiB**（省 75.2 GiB）✓ |
| `MemAvailable` | 211 → **288 GiB** ✓ |
| 每 rank 块占用 | 128+4=132.00 → **91.55+2.86=94.42 GiB** ✓ |
| KV 池 / 显存 | 1,816,965 tokens（不变）✓ |
| decode / prefill | 41.06 → 41.10 tok/s；2826 → 2832 tok/s（±0.3%，噪声内）✓ |
| 模型加载 | PP0 362→327s、PP1 378→344s（更快）✓ |
| 启动 argv / 关键 env | 逐字一致 ✓ |

注：该模型 reasoning 文本本身 run-to-run 非确定（同实例重复 3 次亦互不相同，`content` 全同），
故以**字节指纹**而非文案作精度判据。

**回退**：`VLLM_ENGRAM_EXACT_PIN=0` 重启即回到 torch pinned 分配器；
`~/dsv41-pinmem-rollback.sh` 回退代码（`git revert 2b652f14dd`）。
改造档案（基线数值/门禁脚本/回退脚本）：`~/works/dsv41-pinmem-a-20260916/`。

## 版本标记（2026-09-14）

- **tag**: `v41-pp5-1m-20260914`
- **HEAD**: `693b4da135`（含 2 个新提交：`3dee16caa7` 影子+修复+staged、`693b4da135` 1M workspace 自适应）
- **完整仓库备份**: `vllm-backport.bundle`（git bundle，含全部历史与 tag；`git clone` 即可恢复）

## 当前运行配置（已验证）

```
PP5(8,8,8,8,8) + 影子均衡 + 全驻留(offload 0) + gpu-util 0.97
+ maxlen 1048576(1M) + VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
```

- KV 池 1,632,827 tokens；1M 并发 1.56x；**896K token prompt 实测通过**（388s）
- P1 1.5s / P2 2.7s / P3 7.8s（对比原始卸载版 ~10x 提速）

## 本版本包含的全部改动（9 改 + 1 新增）

### 一、影子源层（fork 定制，跨 rank KV 共享）

| 文件 | 内容 |
|---|---|
| `vllm/models/deepseek_v4_1/shadow_source.py`（**新增**） | ShadowSource：消费者 rank 本地复刻源层 attention，从 PP 边界接收源层注意力输入 x 重算压缩 KV/indexer K/topk |
| `vllm/models/deepseek_v4_1/nvidia/model.py` | `_plan_shadow_sources` 规划/实例化影子；DecoderLayer 捕获 x 到 `shadow_feed_buffer`；PP 边界转发/中继 `shadow_x_{S}`；权重加载别名重定向 |
| `vllm/models/deepseek_v4_1/attention.py` | `produce_kv_side_effects`：重算 KV 侧效。**修复**：加 `@eager_break_during_capture`（cache 写入/topk 不进图）+ 补调 `indexer_op`（长上下文 topk 计算）|
| `vllm/model_executor/offloader/base.py` + `vllm/envs.py` | `VLLM_CPU_OFFLOAD_GB_PER_RANK`：每 PP rank 独立卸载预算 |

### 二、诊断修复（乱码 + 启动失败）

| 文件 | 问题 → 修复 |
|---|---|
| `vllm/v1/attention/backends/utils.py` | legacy 2,6,6,6,20 分区 rank0 无压缩层导致 KV layout 断言失败 → 不一致时取交集并按首选数排序 |
| `vllm/v1/core/kv_cache_utils.py` | ① `_project_kv_cache_groups_to_worker`：恢复保留空 group（worker/scheduler group 索引对齐）；② `get_kv_cache_config_from_groups`：tensor 生成按 `layer_names` 过滤（空 group 不再为其他 rank 的层分配） |

### 三、全驻留优化（参考 zebgop-ops/dsv41flash-pp）

| 文件 | 内容 |
|---|---|
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py` | 新增 `_repack_marlin_experts_staged` + `prepare_moe_mxfp4_layer_for_marlin_staged`：repack 走 host RAM（pageable），消除 raw+packed 双份 GPU 峰值（~6.7GiB/层）|
| `vllm/model_executor/layers/quantization/mxfp4.py` | `_setup_kernel` Marlin 分支走 staged；`process_weights_after_loading` 清除局部引用（否则 pin 住 raw）|
| `vllm/v1/attention/backends/mla/indexer.py` | `get_max_prefill_buffer_size` cap 到 `min(max_model_len, 131072) * 40`：1M 上下文下 workspace 从 5.16GiB 降到 0.64GiB（prefill 切更多 chunk）|

## 当前最优配置（start_vllm_v41_pp5.sh 默认）

```
PP5(8,8,8,8,8) + 影子均衡 + 全驻留(offload 0) + gpu-util 0.98 + maxlen 1048576(1M)
```

## 性能实测（shadow+cudagraph）

| 指标 | 原始(卸10GB/卡) | 全驻留 |
|---|---|---|
| P1 | 13.3s | **1.4s** |
| P2 | 27.8s | **2.7s** |
| P4 | 65.7s | **6.2s** |
| prefill | ~100-1000 tok/s | **~4.6k tok/s** |
| KV 池 | 11.15 GiB | 1.78 GiB (1,134,755 tokens) |

- 模型原生上限 1048576（config: YaRN ×16, original 65536）
- 131072 上下文并发 8.66x；512K 并发 2.16x；1M 并发 1.08x

## 恢复方法

```bash
# 方式一：打补丁（在 vllm-backport 根目录）
cd /home/sean/works/vllm-backport
git apply /path/to/patches/all-changes.patch
cp /path/to/new-files/shadow_source.py vllm/models/deepseek_v4_1/

# 方式二：直接覆盖完整文件（full-files/ 下文件名用 _ 代替 /）
#   例：vllm_v1_core_kv_cache_utils.py → vllm/v1/core/kv_cache_utils.py
```

## 已知限制 / 待验证

- `--dspark`（投机解码）未在最新配置下实测（rank4 卸 9GB）
- 1M 上下文仅完成启动验证，未跑 1M 长 prompt 端到端（60K 已验证）
- legacy `--legacy-offload` 模式仍是旧配置（102/110GB 卸载），未适配新优化
- engram 表（189 GiB = 数据本身的字节数，分配器取整浪费已于 2026-09-16 消除）仍在
  pinned RAM（UVA，不可回收）；如需再省可参考 dsv41flash-pp 的 NVMe 方案
