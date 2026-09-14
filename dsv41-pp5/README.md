# dsv41-flash-pp5-170hx

DeepSeek-V4.1-Flash（510GB：MXFP4 路由专家 + FP8 dense + 203GB Engram 表）在
**5×A100-64G**（PCIe Gen2 x16，无 NVLink）上的 PP5 全驻留部署，基于
[`wtdcode/vllm-backport`](https://github.com/wtdcode/vllm-backport) fork 加本地补丁。

本仓库是**修改后的完整引擎源码**（含两个本地提交），项目文档与启动脚本在
[`dsv41-pp5/`](dsv41-pp5/)。

## 关键特性

- **全驻留 5 卡**：专家 0 卸载（staged Marlin repack，消除 raw+packed 双份峰值）
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

tag：`v41-pp5-1m-20260914`
