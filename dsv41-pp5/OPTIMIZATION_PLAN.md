# DeepSeek-V4.1-Flash 性能优化记录（最终版）

- 日期：2026-09-28 / 29
- 引擎：`/home/sean/works/vllm-backport`（wtdcode/vllm-backport 系，base HEAD `d63af5a472`）
- 硬件：6×A100-64G（SM80，170HX 改，PCIe Gen2，无 NVLink）
- 模型：`/home/sean/models/DeepSeek-V4.1-Flash`（MXFP4 专家 + FP8 dense + 203GB Engram）
- 部署：PP6(7,7,7,7,7,5) 影子源层均衡 + DSpark x5，`vllm-v41` conda env
- 服务由 `fan-console.service`（管理台，端口 9100）拉起

---

## 1. 结论摘要（TL;DR）

**已采纳并设为默认（有实测收益）：**

| 改动 | 效果 |
|---|---|
| `cudagraph capture [1,2,4,8,16]`（原 8） | 并发 4：聚合 +38%、单流 decode +60%；单请求 decode 61.5→68.5（+11%） |
| `--long-prefill-token-threshold 2048` | 短请求 TTFT 10.6→4.2s（−60%）；长请求自身也更块（13.1→11.7s） |
| `NCCL_P2P_DISABLE=1` | prefill +7–10%（长 prefill 20.1→18.0s），decode 持平 |
| `--max-num-seqs 16` | 与 capture 上限匹配 |

**已证伪 / 不可行（勿重复尝试）：**

- P0 开关：`VLLM_SPARSE_RAGGED_FAST_SCAN`、`VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` → 噪声内无收益。
- P1-A decode topk ragged 跨层复用（上游 vllm #57434）→ 无收益（PP 每卡层少）。
- dspark `num_speculative_tokens`：3 与 5 无差异；7 非法（须为 n_predict=5 的倍数）。
- 提高 prefill kv-gather workspace（`PREFILL_CHUNK_SIZE`）→ 1M 上下文 KV 不足，OOM。
- PP bubble 不存在；async scheduling 本就默认开启；MoE/Marlin 已近 SM80 roofline。

**关键发现：**

- **并发扩展性差**：单流 decode 61.5（N=1）→ 20.8（N=4）→ 14.9（N=16）；聚合仅 ~2.8×。根因是 **MoE 稀疏性**（每 token 激活不同专家，batch 增大无法摊薄权重读取）+ prefill/decode 调度抢占。
- **prefill 撞 185W 功耗墙**（`sw_power_cap=Active`，SM 1455→1185MHz，−18%）；decode 不撞（70–131W）。
- 当前 vLLM 与 GPU 矿工 `SRBMiner` 共存（1812 MiB/卡，抢 SM/带宽）——是性能的一部分隐藏成本；GPU4 余量仅 ~3GB。

---

## 2. 最终默认配置（定稿，脚本与 fan-console 已同步）

**脚本 `/home/sean/start_vllm_v41.sh` 默认：**

```
PP6(7,7,7,7,7,5) · max-model-len 1048576 · gpu-memory-utilization 0.89
max-num-batched-tokens 8096
max-num-seqs 16
cudagraph capture [1,2,4,8,16]            # CG16=1
long-prefill-token-threshold 2048          # LONG_PREFILL=2048
DSpark x5                                  # NUM_SPEC_TOKENS=5
NCCL_P2P_DISABLE=1                         # NCCL_P2P=0
```

**回退开关：** `--cg8` / `--long-prefill 0` / `--spec-tokens 5` / `--nccl-p2p` / `--max-seqs N`。

**fan-console `backend/vllm.py` 表单默认（已同步）：** `pp6 / port9004 / maxlen1048576 / gpu_util空(=脚本0.89) / max_batched8096 / max_seqs16 / cg16=16 / long_prefill=2048 / spec=auto`；`_build_cmd` 支持 `--cg16/--cg8` 与 `--long-prefill`。

---

## 3. 已采纳优化的实测数据

### 3.1 cg16（CUDA Graph 覆盖 8→16）
| 并发 N | cg8 聚合/decode | cg16 聚合/decode |
|---|---|---|
| 1 | 38.3 / 61.5 | 35.1 / 65.5 |
| 4 | 50.6 / 20.8 | **70.0 / 33.3** |
| 8 | 80.9 / 19.1 | 81.6 / 20.0 |
| 16 | 107.5 / 14.9 | 94.9 / 12.6 |

单请求 decode：cg8 61.5 → cg16 68.5。KV 池不变、启动无副作用。

### 3.2 long-prefill-token-threshold 2048（混合压测，均无矿工）
| 指标 | 阈值 0 | 阈值 2048 |
|---|---|---|
| 短请求 TTFT（1 长 60k + 4 短） | 10.6–11.5 s | **4.1–4.2 s（−60%）** |
| 长请求自身 TTFT | 13.1 s (4550 tok/s) | **11.4–12.0 s (5021–5287 tok/s)** |

### 3.3 NCCL_P2P_DISABLE=1
| 指标 | 默认（P2P 开） | P2P_DISABLE |
|---|---|---|
| 长 prefill 100k | 19.8–20.4 s | **18.0–18.1 s（+10%）** |
| 24k prefill | 4.82 s | **4.50 s（+7%）** |
| 单请求 decode | 68.5 | ~68（持平） |

### 3.4 DSpark vs Plain（均 cg16）
| 项 | dspark | plain |
|---|---|---|
| 单请求 decode | **65.5** | 44.2 |
| N=16 聚合 | 94.9 | **112.4** |

低并发保留 dspark（+1.5×）；高并发可选 `--plain`（未多轮确认）。

---

## 4. 已证伪 / 不适用（实测结论）

| 方向 | 结论 |
|---|---|
| `VLLM_SPARSE_RAGGED_FAST_SCAN=1` | 噪声内，无收益 |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` | 无收益且挤小 KV（1,384,528→1,343,537） |
| P1-A decode ragged 跨层复用（#57434） | 无收益（PP 每卡 5–7 层，复用面小） |
| P2-A MLA bucket 单组（`VLLM_DSV4_MLA_SINGLE_GROUP`） | 未验证容量收益，代码未保留（默认关） |
| `--spec-tokens 3` | 与 5 无差异 |
| `--spec-tokens 7` | 非法：`must be divisible by n_predict=5` |
| 提高 prefill workspace（`VLLM_V41_PREFILL_WS_GB` 2.2/4） | KV 不足，启动 OOM（1M 固有代价） |
| `mnb 16384` | 启动 OOM（矿工共存余量不足） |
| PP rank 流水 bubble | 不存在（decode 时 6 卡 sm 同时 100%） |
| async scheduling | 本就默认开启（DSpark 在支持名单） |
| torch.compile / inductor 整体 | 同硬件已证伪（见 `benchmarks/kernels/dsv4_sm80_refutations.md`） |

### 4.1 Level1Techs「驱动注册表 P2P 修复」方案评估（2026-09-29）

论坛方案：用驱动侧注册表修复解决 CMP 170HX 的 P2P 可达性死局（`nvidia-smi topo` 显示 NODE/GNS、`cudaDeviceCanAccessPeer=0`），配 `NCCL_P2P_LEVEL=SYS` + `--disable-custom-all-reduce`，W4A16 TP2 prefill 2500→4400 tok/s。

对本机的评估：
- **本机 P2P 已完全可用**：`cudaDeviceCanAccessPeer` 6×6 全 1；`nvidia-smi topo -p2p r/w/p` 全 **OK**（无 GNS/NODE）。
- 本机已应用同类驱动侧修复：`/etc/modprobe.d/cmp-pcie-gen2.conf` 的
  `NVreg_RegistryDwords="RmForceEnableGen2=1;RMPcieLinkSpeed=0x1;RMForceStaticBar1=1;RMPcieP2PType=1"`
  + `cmpunlocker` 补丁（0011/0015/0016）。→ **论坛的"死局修复"对本机不适用（已完成）**，不会有 1.85×（那是死局→可用的提升）。
- 本机 P2P 健康基线是 **Gen2 x16 ~6.7 GB/s 单向 / ~13 GB/s 双向**（PCIe Gen2 硬件上限，非注册表可解）。
- **`NCCL_P2P_LEVEL=SYS` 实测无收益**：长 prefill 19.2–20.7s（≈ 默认 P2P 19.8–20.4s），不及 `P2P_DISABLE` 的 18.0s；decode 受矿工波动干扰无法定性。
- `--disable-custom-all-reduce` 本机脚本**一直已设**。
- 结论：**不采用 SYS，保持 `NCCL_P2P_DISABLE=1`**。
- 附注：矿工 SRBMiner 周期性启停会显著干扰测量（decode 可从 ~68 掉到 ~26–33 tok/s），对比须核对矿工状态。


---

## 5. 未测 / 后续方向（截至 2026-09-29）

1. **矿工对 vLLM 的影响量化**（停/启 SRBMiner 各测一轮）。
2. **PP5 vs PP6**（少一段 PP 边界）。
3. **`mnb 12288`**（16384 OOM，12288 未完成）。
4. **`gpu-util` 提高**（0.89→0.92+，需停矿工释放余量）。
5. **chunked-prefill 其它旋钮**（已采用 long-prefill 2048）。
6. **Engram 运行时开销**（估算占 prefill 2.6–4.4%，未精确 profiling）。
7. **PP 段负载均衡**（engram/shadow 使各段耗时不等）。
8. **NCCL_P2P_LEVEL**（已用 DISABLE，LEVEL=SYS 未试）。
9. **inductor 融合 pass**（`fuse_attn_quant` 等默认关）。
10. **量化格式** MXFP4→GPTQ-int4（离线转换，重）。
11. **更长上下文 prefill**（256k/512k/1M，仅测到 100k）。
12. **P1-B 调度 livelock 修复**（未纳入）。

---

## 6. 备份 / 还原 / 回退

- 一键还原：`/home/sean/backups/vllm_v41_opt_restore_20260928_200626/restore.sh`（`--dry-run` 预览 / `--hard` 含 git 回退）。
- tag：`pre-opt-20260928`。
- fan-console 备份：`/tmp/opencode/vllm.py.bak2`（`backend/vllm.py`）。

> 仓库工作树当前仅 `vllm/models/deepseek_v4_1/common/engram.py`（历史未提交改动）+ 本文档待提交；所有实验性代码改动（P0/P1-A/P2-A/P2-B/C）已还原/未纳入。

---

## 7. 测量脚本（已归档 `dsv41-pp5/bench/`）

| 脚本 | 用途 |
|---|---|
| `bench_v41.py ctx_words gen [port]` | 单请求冷 prefill + decode（随机 nonce 防前缀命中） |
| `bench_concurrent.py N ctx gen` | N 并发聚合吞吐 / TTFT / 单流 decode |
| `bench_mixed.py N long_ctx short_ctx` | 1 长 prefill + N 短请求（验证 long-prefill 效果） |
| `bench_steady.py N ctx gen` | 稳态重叠窗口 decode（排除 prefill 干扰） |
| `test_p2b_bound.py` | P2-B 越界边界回归（GPU） |
| `test_p1a_topk_cache.py` | P1-A 缓存语义回归（GPU） |

测量纪律：不引单次 ITL；必防前缀命中；同进程内 A/B；比值不跨调用比绝对值；停损阈值单改动 <2% 回退。

---

## 8. 关键命令备忘

```bash
# dry-run 查看最终参数
/home/sean/start_vllm_v41.sh --dry-run

# 启动（默认配置）
nohup bash /home/sean/start_vllm_v41.sh --bg >/tmp/v41.log 2>&1 &

# 停止（fan-console 管理下，root 进程）
pids=$( { pgrep -f "DeepSeek-V4.1-Flash"; pgrep -f "VLLM::"; } | sort -u )
sudo kill -9 $pids        # 等待显存释放后再启动

# 回退某单项
./start_vllm_v41.sh --cg8 --bg           # capture 回 8
./start_vllm_v41.sh --long-prefill 0 --bg # 长 prefill 不限
./start_vllm_v41.sh --nccl-p2p --bg      # 恢复 NCCL P2P
./start_vllm_v41.sh --spec-tokens 5 --bg # 默认即 5
```
