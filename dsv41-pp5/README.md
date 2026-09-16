# dsv41-flash-pp5-170hx

DeepSeek-V4.1-Flash（510GB：MXFP4 路由专家 + FP8 dense + 203GB Engram 表）在
**5×A100-64G**（PCIe Gen2 x16，无 NVLink）上的 PP5 全驻留部署，基于
[`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport) fork 加本地补丁。

本仓库是**修改后的完整引擎源码**（含两个本地提交），项目文档与启动脚本在
[`dsv41-pp5/`](dsv41-pp5/)。

## 关键特性

- **全驻留 5 卡**：专家 0 卸载（staged Marlin repack，消除 raw+packed 双份峰值）
- **精确尺寸 pinned**：engram 表（2×94.4 GiB）用 `cudaHostAlloc` 精确分配，绕开
  torch pinned 分配器的「2 的幂向上取整」—— CPU pinned 占用 **264 → 189 GiB**（省
  75 GiB，可用内存 211 → 288 GiB），表字节与改动前逐字节一致；`VLLM_ENGRAM_EXACT_PIN=0` 可回退
- **1M 上下文**：maxlen 1048576，896K token prompt 实测通过
- **影子源层**：PP 切分落在 v4.1 kv 共享组内时，消费者 rank 本地重算源层压缩
  KV / indexer K / top-k，使 8,8,8,8,8 均分（而非被组约束逼到 2,6,6,6,20）
- 乱码修复（长上下文非确定性特殊 token 泄漏）、legacy 模式 KV group 修复

## 性能（5×A100，全驻留）

| 场景 | 数据 |
|---|---|
| P1（短问答） | 1.5s |
| P2（~110 token 生成） | 2.7s |
| 60K prompt prefill | 12.5s（~4.6k tok/s） |
| 896K prompt | 388s（~2.3k tok/s） |
| KV 池 | 1,632,827 tokens @ util 0.97 |
| 1M 并发 | 1.56x |

对比原始卸载配置（卸 10GB/卡）：P1 13.3s → 1.5s（~9x）。

## CPU 内存（503GB 机器）

engram 表卸载到 pinned 主存的占用，直方图为「精确尺寸分配」改造前后（2026-09-16 实测）：

| 指标 | 改造前 | 改造后 |
|---|---|---|
| pinned（`meminfo Shmem` / `free shared`） | 264.3 GiB | **189.1 GiB** |
| 每 rank 块占用（weight + scales） | 128 + 4 = 132.00 GiB | **91.55 + 2.86 = 94.42 GiB** |
| `MemAvailable` | 211 GiB | **288 GiB** |
| 模型加载耗时（PP0 / PP1） | 362 / 378 s | 327 / 344 s |
| KV 池 | 1,816,965 tokens | 1,816,965 tokens（不变） |
| decode / prefill | 41.06 tok/s / 2826 tok/s | 41.10 tok/s / 2832 tok/s |

两个 rank 的 engram 表 sha256 在改造前后**逐字节一致**（零精度损失）；同实例重复测试
显示 reasoning 文本本身 run-to-run 非确定（`content` 100% 一致），故不以文案比对作精度判据。

注意：pinned 内存**不可回收也不可换出**（swap 仅 8 GiB），省下的是 OOM 余量而非可回收内存。

## 快速开始

```bash
# 依赖：conda env vllm-v41（编译好的本仓库）、模型 /model/DeepSeek-V4.1-Flash
cp dsv41-pp5/start_vllm_ds4.1.sh ~/
~/start_vllm_ds4.1.sh --bg        # 端口 9004，后台启动，日志在 ~/logs/
```

启动参数（脚本内可改）：`--port`（默认 9004）、`--maxlen`（默认 1048576）、
`--gpu-util`（默认 0.97）、`--offload-ranks`（默认 0,0,0,0,0 全驻留）、
`--dspark`（投机解码，未实测）、`--legacy-offload`（旧 2,6,6,6,20 模式）。

## 目录

```
vllm/                     修改后的引擎源码（含 shadow_source.py 等）
dsv41-pp5/
├── ARCHITECTURE.md       架构详解（模型/并行/影子/显存/运行时）
├── CHANGELOG.md          全部改动清单 + 性能数据 + 恢复方法
├── all-changes.patch     相对上游的完整 diff
├── start_vllm_ds4.1.sh   启动脚本（端口 9004）
└── start_vllm_v41_pp5.sh 启动脚本（端口 9010，原版）
```

## 本地提交

- `3dee16caa7` 影子源层 + PP/KV 修复 + staged Marlin 全驻留
- `693b4da135` 1M 上下文的 prefill gather workspace 自适应
- `362948e815` 启动脚本开 `expandable_segments`（消除加载期碎片 OOM）
- `2b652f14dd` engram 表精确尺寸 pinned 分配（省 75 GiB CPU 内存）

tag：`v41-pp5-1m-pinmem-20260916`（上一版 `v41-pp5-1m-20260914`）

回退：`VLLM_ENGRAM_EXACT_PIN=0` 重启即回到 torch pinned 分配器；
`~/dsv41-pinmem-rollback.sh` 可回退代码。改造档案见 `~/works/dsv41-pinmem-a-20260916/`。
