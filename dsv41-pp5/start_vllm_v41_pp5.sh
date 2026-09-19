#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash 一键启动（影子源层均衡 PP5/PP6 · 纯显存模式）
#   ◆ 引擎: wtdcode/vllm-backport 本地编译 + 影子源层补丁（conda env vllm-v41）
#   ◆ 模型: /model/DeepSeek-V4.1-Flash （Native FP4 + Engram/ngram）
#   ◆ 并行: --pp 5（默认）切分 8,8,8,8,8 ｜ --pp 6 切分 7,7,7,7,7,5，段数即
#           --pipeline-parallel-size，配合 --tensor-parallel-size 1
#           （heads=64 / kv_heads=1 不能被 PP 整除，TP 不可行）
#   ◆ ngram: --engram-config '{"cpu_offload":true}' 把 ~203GB engram 表卸到钉住内存
#            经 UVA 读取（layers.1@rank0 / layers.14@rank2 各 ~101GB，PP6）
#   ◆ 权重:  PP5 每 rank ~60-61.5GB；PP6 每 rank 6-7 层 ≈48GB，余量大幅变多
#
# ★ 影子源层（本 fork 定制，打破 KV 共享组墙）:
#   v4.1 压缩层必须读取 KV 源层（kv_source_layer_ids=[2,8,14,20]）的压缩 KV /
#   indexer K / candidate，上游要求同 rank 同组 —— 原生只允许 2,6,6,6,20 切分
#   （{20-39} 20 层组 147.8GB 独占一卡，必须卸 ~110GB 专家 → decode 仅 10-16 t/s）。
#   本 fork 让消费 rank 用源层 prefix 实例化"影子 attention"，从 PP 边界接收
#   源层注意力输入 x（[T,5120]bf16 ≈10KB/token）本地精确重算 KV。影子计划完全按
#   PP 段数 + VLLM_PP_LAYER_PARTITION 自动推导（不写死段数）：
#     PP5(8,8,8,8,8): rank2 影子14、rank3/4 影子20
#     PP6(7,7,7,7,7,5): rank1 影子2、rank3/4/5 影子20
#   代码: vllm/models/deepseek_v4_1/{shadow_source.py,nvidia/model.py,attention.py}
#
# ★ PCIe 链路分布（启动时会打印 rank→链路表并给建议）:
#   本机 6 卡实测: GPU0/4/5 = Gen2 x16，GPU2/3 = Gen2 x8，GPU1 = Gen2 x4。
#   NCCL P2P 实测带宽 ≈ min(两端链路): x16↔x16 6.7GB/s，x16↔x8 3.3，x16↔x4 1.7。
#   （GPU1-5 间 torch 裸拷贝 400-5000GB/s 是矿卡 BAR1 P2P 映射的缓存假象，
#    NCCL 不走该路径，故仍以链路宽度为准。）
#   两条准则（脚本只提示，不自动改卡序）:
#     1) 最窄链路的卡放【最后一段】—— 两端段各只有 1 个 PP 边界（中间段 2 个），
#        且最后一段层数最少；层 1（engram）被层连续性钉死在第一段，故 x4 卡不能当
#        第一段（DSpark 的草稿模型/aux 层 37-39 也必须在最后一段）。
#     2) engram 层（L1/L14）尽量放 x16 卡 —— engram 表经 UVA 走 PCIe 随机读。
#   PP6 默认 --gpus 0,5,4,3,2,1: rank0/1/2 吃满三张 x16（engram L1/L14 + 2 份
#     流量的 0-1 边界），x4 卡落末段；各段实测 6.7/6.7/3.3/3.3/1.7 GB/s。
#   PP5 默认 --gpus 0,4,5,3,2: 跳过 x4 卡，rank0/1 吃 x16（engram L1/L14），
#     末段 x8；各段实测 6.7/6.7/3.3/3.3 GB/s。
#
# ★ DSpark（投机解码）: num_speculative_tokens=5（= 模型 dspark_block_size）。
#   PP6 默认开启（--plain 关闭）；PP5 保持默认关闭（--dspark 开启）。
#   PP>1 需要 v4.1 模型声明 supports_aux_hidden_states_over_pp（本 fork 已加），
#   否则加载期报 "does not support dspark with pipeline parallelism"。
#
# 显存预算（PP5 8,8,8,8,8，每层 7.4GB，卡 63GiB≈67.7GB）:
#   rank0(L0-7)   = 59.2 + embed1.3 + vision1.0 ≈ 61.5GB
#   rank1(L8-15)  = 59.2
#   rank2(L16-23) = 59.2 + 影子14 ≈ 59.4
#   rank3(L24-31) = 59.2 + 影子20 ≈ 59.4
#   rank4(L32-39) = 59.2 + head1.3 + 影子20 ≈ 60.7
#   实测 2026-09-14：移植 marlin_staged（repack 走 host RAM，无 raw+packed 双份）后
#   全驻留(0,0,0,0,0)成功，util 0.97 下 KV 1.44GiB，速度 ~10x，prefill ~4.6k tok/s。
#   --dspark 时 rank4 需再放 mtp ~8.4GiB → 卸 9GB（未实测）。
#   显存预算（PP6 7,7,7,7,7,5，每层 ≈6.9GiB；GPU 按默认 --gpus 0,5,4,3,2,1）:
#   rank0 GPU0 L0-6  ≈48.3+embed+vision/engramL1   rank1 GPU5 L7-13 ≈48.3+影子2
#   rank2 GPU4 L14-20≈48.5(engramL14)              rank3 GPU3 L21-27≈48.5(影子20)
#   rank4 GPU2 L28-34≈48.5(影子20)
#   rank5 GPU1 L35-39≈34.5+head1.2+影子0.2+dspark8.4 ≈44.3（x4 卡放这里）
#   中间段空余 ≈12GiB → KV 池（取各段最小值）远大于 PP5。
#
# ★ PP6 显存临界（2026-09-19 实测）: util 0.97 下 rank2(GPU4) 运行期 OOM。
#   rank2 权重最大（51.9GiB）且含 engram L14，util 0.97 时 KV 池 7.7GiB、非 KV 余量
#   仅 ~1.9GiB；长 prefill 时每层 attention 的 q_out（[T,64,512]bf16，T=8096 时
#   ~530MB）+ sparse indexer 的 fp8 logits（~142MB）+ 运行期 Triton JIT 编译
#   （warmup 只覆盖 tokens=16 的小 shape，长 prefill 首次触发）叠加，CUDA 仅剩
#   19MB 时 new_empty 失败（expandable_segments mapping failed OOM）。
#   → PP6 默认 gpu-util 降为 0.94：KV 池仍有 ~5.7GiB（≈3M tokens），非 KV 余量
#     ~3.8GiB，足以覆盖上述尖峰。如需更大 KV 可显式 --gpu-util 0.96 并先验证长
#     prefill（会重现 OOM）；PP5 不受影响，仍默认 0.97。
#
# 用法: ./start_vllm_v41_pp5.sh [选项]
#   --pp N          流水线并行段数：5（默认，5 卡）或 6（6 卡）
#   --layers L       手动指定切分（逗号分隔，和必须 40，段数=PP，末段≥3 含层 37-39）
#   --port P         API 端口（默认 9004）
#   --gpus LIST      使用的卡，逗号分隔（默认已按 PCIe/NCCL 实测最优排好：
#                    PP6 → 0,5,4,3,2,1 ｜ PP5 → 0,4,5,3,2）
#                    顺序即 PP rank 顺序 —— 最窄链路的那张请放末尾
#   --maxlen N       上下文长度（默认 1048576=1M，模型原生上限）
#   --gpu-util N     gpu_memory_utilization（默认 PP5=0.97 / PP6=0.94，见显存临界说明）
#   --dspark         启用 DSpark 投机解码 x5（PP6 默认开启）
#   --plain          关闭 DSpark（PP5 默认）
#   --offload-gb N   各 rank 专家卸载 GB（0=纯显存）
#   --offload-ranks L 逐 rank 卸载预算（逗号分隔，必须 PP 个值）
#   --max-batched N  max-num-batched-tokens（默认 2048）
#   --max-seqs N     max-num-seqs（默认 8）
#   --piecewise      cudagraph 用 PIECEWISE（比 FULL_AND_PIECEWISE 省显存，decode 略慢）
#   --eager          enforce-eager 关 cudagraph（诊断用，排除图捕获干扰）
#   --legacy-offload 退回旧模式: 切分 2,6,6,6,20 + rank4 卸 102/110GB（仅 PP5）
#   --dry-run        只打印链路表 + env + 完整命令行后退出（不查卡、不启动）
#   --bg             后台运行（默认前台显示日志）
#   --force          启动前强制清理本端口的旧实例（默认绝不杀任何进程）
# 日志:   ~/logs/vllm_v41_pp${PP}_*.log
# 地址:   http://0.0.0.0:$PORT/v1
# =============================================================================
set -euo pipefail

# ---- 固定约定 -----------------------------------------------------------------
ENV_NAME=vllm-v41
VLLM_ROOT=/home/sean/works/vllm-backport
# 模型权重：优先用快盘副本（/home 所在 aigo P7000Y：随机读 1.7GB/s+），
# 根盘的 Yottamstear 盘随机读仅 ~18MB/s，加载期会拖慢几十倍；快盘副本
# 由 /models/models/DeepSeek-V4.1-Flash（HDD 备份）复制而来。
if [ -d /home/sean/models/DeepSeek-V4.1-Flash ]; then
  MODEL_DIR=/home/sean/models/DeepSeek-V4.1-Flash
else
  MODEL_DIR=/model/DeepSeek-V4.1-Flash
fi
API_KEY='sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807'
LOG_DIR="$HOME/logs"
NUM_LAYERS=40

# ---- 可配置项 -----------------------------------------------------------------
PORT=9004
PP=5
GPUS=""
LAYERS=""
MAXLEN=1048576
# 未指定时按 PP 落定：PP5 用 0.97（实测稳定），PP6 用 0.94（见下方 OOM 说明）。
GPU_UTIL=""
# 每 rank 专家 UVA 卸载预算（GB，逗号分隔，按 PP rank；--cpu-offload-gb 取其最大值激活 UVA）。
# 实测校准(2026-09-14, PCIe Gen2 + staged Marlin repack)：全驻留(0,...,0)已可行：
# 移植 marlin_staged 后加载期无 raw+packed 双份峰值，KV 1.44GiB（util 0.97），
# P1 13.3s→1.4s、P4 65.7s→6.2s（~10x），prefill ~4.6k tok/s，60K 长上下文正常。
# 如需更大 KV/并发可上调卸载量换取 KV 空间（PCIe 很慢，优先调 --layers 均分）。
OFFLOAD_RANKS=""
OFFLOAD_GB_ARG=""
OFFLOAD_SET=0
PP_PARTITION=""
MAX_BATCHED=8096
MAX_SEQS=8
PIECEWISE=0
EAGER=0
# 投机解码三态: ""=未指定（按 PP 落定） / plain / dspark
SPEC_MODE=""
LEGACY=0
FG=1
FORCE=0
DRY=0
NAME=DeepSeek-V4.1-Flash

while [ $# -gt 0 ]; do
  case "$1" in
    --pp|-pp)      PP="$2"; shift 2 ;;
    --layers)      LAYERS="$2"; shift 2 ;;
    --port)        PORT="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --maxlen)      MAXLEN="$2"; shift 2 ;;
    --gpu-util)    GPU_UTIL="$2"; shift 2 ;;
    --offload-gb)    OFFLOAD_GB_ARG="$2"; OFFLOAD_SET=1; shift 2 ;;
    --offload-ranks) OFFLOAD_RANKS="$2"; OFFLOAD_SET=1; shift 2 ;;
    --max-batched) MAX_BATCHED="$2"; shift 2 ;;
    --max-seqs)    MAX_SEQS="$2"; shift 2 ;;
    --piecewise)     PIECEWISE=1; shift ;;
    --eager)         EAGER=1; shift ;;
    --dspark)        SPEC_MODE=dspark; shift ;;
    --plain)         SPEC_MODE=plain; shift ;;
    --legacy-offload) LEGACY=1; shift ;;
    --dry-run)     DRY=1; shift ;;
    --bg)          FG=0; shift ;;
    --force)       FORCE=1; shift ;;
    -h|--help|-help)
      sed -n '2,/^set -euo pipefail/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

log() { echo -e "$(date '+%F %T') [INFO]  $*"; }
die() { echo -e "$(date '+%F %T') [ERROR] $*" >&2; exit 1; }

# ---- 0. PP 预设与切分校验 ------------------------------------------------------
case "$PP" in
  5) PP_DEFAULT="8,8,8,8,8"; GPUS_DEFAULT="0,4,5,3,2"; GPU_UTIL_DEFAULT=0.97 ;;
  6) PP_DEFAULT="7,7,7,7,7,5"; GPUS_DEFAULT="0,5,4,3,2,1"; GPU_UTIL_DEFAULT=0.94 ;;
  *) die "--pp 只支持 5 或 6（当前 --pp $PP）" ;;
esac
[ -n "$GPUS" ] || GPUS="$GPUS_DEFAULT"
[ -n "$GPU_UTIL" ] || GPU_UTIL="$GPU_UTIL_DEFAULT"

if [ "$LEGACY" = "1" ]; then
  [ "$PP" = "5" ] || die "--legacy-offload 只对 --pp 5 有意义（它靠 2,6,6,6,20 把 L20-39 整组放末卡）"
  [ -z "$LAYERS" ] || die "--legacy-offload 与 --layers 互斥"
  PP_PARTITION="2,6,6,6,20"
  MODE_DESC=旧模式卸载
elif [ -n "$LAYERS" ]; then
  PP_PARTITION="$LAYERS"
  MODE_DESC=自定义切分
else
  PP_PARTITION="$PP_DEFAULT"
  MODE_DESC=影子均衡
fi

IFS=',' read -ra _PL <<< "$PP_PARTITION"
[ "${#_PL[@]}" -eq "$PP" ] || die "切分段数 ${#_PL[@]} != PP=$PP（$PP_PARTITION）"
_sum=0
for _p in "${_PL[@]}"; do _sum=$(( _sum + _p )); done
[ "$_sum" -eq "$NUM_LAYERS" ] || die "切分总和 $_sum != $NUM_LAYERS（$PP_PARTITION）"
[ "${_PL[$((PP - 1))]}" -ge 3 ] || die "最后一段只有 ${_PL[$((PP - 1))]} 层；DSpark 的 aux 层 37,38,39 必须全在末段 → 末段至少 3 层"

# 各段层区间（影子/engram/链路提示用）
declare -a SEG_START SEG_END
_s=0
for _i in "${!_PL[@]}"; do
  SEG_START[$_i]=$_s
  SEG_END[$_i]=$(( _s + _PL[_i] - 1 ))
  _s=$(( _s + _PL[_i] ))
done

# ---- 0b. 投机解码与卸载预算（按 PP 落定）---------------------------------------
[ -n "$SPEC_MODE" ] || { if [ "$PP" = "6" ]; then SPEC_MODE=dspark; else SPEC_MODE=plain; fi; }
PLAIN=1; [ "$SPEC_MODE" = "dspark" ] && PLAIN=0

if [ -n "$OFFLOAD_GB_ARG" ]; then
  _repeated="$OFFLOAD_GB_ARG"
  for _i in $(seq 2 "$PP"); do _repeated="$_repeated,$OFFLOAD_GB_ARG"; done
  OFFLOAD_RANKS="$_repeated"
  OFFLOAD_SET=1
fi
if [ "$OFFLOAD_SET" = "0" ]; then
  if [ "$LEGACY" = "1" ]; then
    if [ "$PLAIN" = "1" ]; then OFFLOAD_RANKS="0,0,0,0,102"; else OFFLOAD_RANKS="0,0,0,0,110"; fi
  elif [ "$PLAIN" = "0" ] && [ "$PP" = "5" ]; then
    # dspark 未在 PP5 全驻留下实测：rank4 加 mtp 的 ~8.4GiB，卸 9GB 腾空间。
    OFFLOAD_RANKS="0,0,0,0,9"
  else
    # PP6 每段只 6-7 层（末段 5 层 + dspark），实测余量足够 → 全驻留。
    # 若加载期 OOM，用 --offload-ranks 0,0,0,0,0,4 给末段腾 4GB。
    _zeros="0"
    for _i in $(seq 2 "$PP"); do _zeros="$_zeros,0"; done
    OFFLOAD_RANKS="$_zeros"
  fi
fi
IFS=',' read -ra _OR <<< "$OFFLOAD_RANKS"
[ "${#_OR[@]}" -eq "$PP" ] || die "卸载预算 ${#_OR[@]} 个值 != PP=$PP（$OFFLOAD_RANKS）；缺值的 rank 会静默回落到全局最大值"
# 全局值 = 各 rank 预算最大值（仅用于激活 UVA 后端；实际按 rank 覆盖）
OFFLOAD_GB=0
for _o in "${_OR[@]}"; do
  [ "$_o" -gt "$OFFLOAD_GB" ] 2>/dev/null && OFFLOAD_GB="$_o"
done

# ---- 0c. 卡数校验（PP 必须恰好 PP 张）-----------------------------------------
IFS=',' read -ra GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -ne "$PP" ]; then
  die "PP$PP 需要恰好 $PP 张 GPU，当前 --gpus='$GPUS' 给了 ${#GPU_IDS[@]} 张"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then die "未找到 nvidia-smi"; fi

# ---- 0d. PCIe 链路表（rank 顺序 = --gpus 顺序）---------------------------------
declare -a LINK_G LINK_W
for _i in "${!GPU_IDS[@]}"; do
  _g="${GPU_IDS[$_i]}"
  _gen=""; _width=0
  if ! read -r _gen _width < <(nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current \
      --format=csv,noheader,nounits -i "$_g" 2>/dev/null | tr -d ','); then
    _gen="?"; _width=0
  fi
  # 卡不存在时 nvidia-smi 会把错误文本写到 stdout（"No devices were found"）
  case "$_gen" in ''|*[!0-9]*) _gen="?" ;; esac
  case "$_width" in ''|*[!0-9]*) _width=0 ;; esac
  LINK_G[$_i]="$_gen"; LINK_W[$_i]="$_width"
done

# 影子源层计划（与 vllm/models/deepseek_v4_1/nvidia/model.py:_plan_shadow_sources 同规则）
SRC=(2 8 14 20)
declare -A SRC_UPPER
for _i in "${!SRC[@]}"; do
  if [ $(( _i + 1 )) -lt "${#SRC[@]}" ]; then SRC_UPPER[${SRC[$_i]}]=$(( SRC[_i + 1] - 1 )); else SRC_UPPER[${SRC[$_i]}]=$(( NUM_LAYERS - 1 )); fi
done
_shadows_of() {  # $1 = rank，输出该 rank 需托管的影子源层
  local r=$1 s=${SEG_START[$1]} out="" S
  for S in "${SRC[@]}"; do
    if [ "$S" -lt "$s" ] && [ "$s" -le "${SRC_UPPER[$S]}" ]; then out="$out $S"; fi
  done
  echo "${out# }"
}

echo "——————————————————————————————————————————————————————————————————————"
printf '  %-4s %-5s %-12s %-9s %s\n' rank gpu pcie layers 备注
for _i in "${!GPU_IDS[@]}"; do
  _note=""
  for _e in 1 14; do
    if [ "${SEG_START[$_i]}" -le "$_e" ] && [ "$_e" -le "${SEG_END[$_i]}" ]; then _note="$_note engram-L$_e"; fi
  done
  _sh="$(_shadows_of "$_i")"
  [ -n "$_sh" ] && _note="$_note 影子$(echo "$_sh" | tr ' ' '/')"
  [ "$_i" -eq $(( PP - 1 )) ] && [ "$PLAIN" = "0" ] && _note="$_note mtp/dspark"
  [ "$_i" -eq 0 ] && _note="$_note embed/vision"
  [ "$_i" -eq $(( PP - 1 )) ] && _note="$_note head"
  printf '  %-4s %-5s gen%-2s x%-6s L%-8s%s\n' "$_i" "${GPU_IDS[$_i]}" \
    "${LINK_G[$_i]}" "${LINK_W[$_i]}" "${SEG_START[$_i]}-${SEG_END[$_i]}" "$_note"
done

# 告警 1: 最窄链路的卡不在最后一段
_min_i=0; _maxw=0; _narrow=0
for _i in "${!LINK_W[@]}"; do
  [ "${LINK_W[$_i]}" -lt "${LINK_W[$_min_i]}" ] && _min_i=$_i
  [ "${LINK_W[$_i]}" -gt "$_maxw" ] && _maxw="${LINK_W[$_i]}"
done
for _w in "${LINK_W[@]}"; do [ "$_w" -lt "$_maxw" ] && _narrow=1; done
if [ "$_narrow" = "1" ] && [ "$_min_i" -ne $(( PP - 1 )) ]; then
  echo "  ⚠ GPU ${GPU_IDS[$_min_i]}(x${LINK_W[$_min_i]}, rank$_min_i) 是链路最窄的卡，但它不在最后一段。"
  echo "    中间段要收发两次、末段只收一次，且末段层数最少；建议把它放到 --gpus 末尾。"
fi
# 告警 2: engram 层落在窄链路卡上（engram 表经 UVA 走 PCIe 随机读）
for _i in "${!GPU_IDS[@]}"; do
  for _e in 1 14; do
    if [ "${SEG_START[$_i]}" -le "$_e" ] && [ "$_e" -le "${SEG_END[$_i]}" ] \
       && [ "${LINK_W[$_i]}" -gt 0 ] && [ "${LINK_W[$_i]}" -lt "$_maxw" ]; then
      echo "  ⚠ engram L$_e 在 rank$_i（GPU ${GPU_IDS[$_i]}，x${LINK_W[$_i]}），不是最宽链路(x$_maxw)。"
      echo "    engram 表 ~101GB 常驻主机内存、每 token 经 UVA 随机读；建议把 x$_maxw 的卡换到 rank$_i。"
    fi
  done
done
echo "——————————————————————————————————————————————————————————————————————"

# ---- 1. 预检：卡存在性与显存占用（默认只警告退出，绝不代杀进程）----------------
for _i in "${!GPU_IDS[@]}"; do
  if ! nvidia-smi --query-gpu=index --format=csv,noheader -i "${GPU_IDS[$_i]}" >/dev/null 2>&1; then
    if [ "$DRY" = "1" ]; then
      log "[dry-run] GPU ${GPU_IDS[$_i]} 查询失败（PP6 需要 6 张卡）"
    else
      die "GPU ${GPU_IDS[$_i]} 不存在或不可查询（PP$PP 需要 $PP 张卡）"
    fi
  fi
done
BUSY=""
if [ "$DRY" = "0" ]; then
  for g in "${GPU_IDS[@]}"; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
    if [ -n "$used" ] && [ "$used" -gt 4096 ]; then
      BUSY="$BUSY $g(${used}MiB)"
    fi
  done
fi
if [ -n "$BUSY" ]; then
  echo "——————————————————————————————————————————————————————————————————————"
  echo "  GPU ${GPU_IDS[*]} 中有卡显存已被占用:${BUSY}"
  echo "  PP$PP 需要这 $PP 张卡都基本空闲，否则会 OOM。默认【不代劳终止任何进程】。"
  echo "  请先手动停止占用卡片的旧实例（如 GPU 0-4 上的旧 v41 实例）后重试；"
  echo "  确认无误可加 --force（仅清理本端口 $PORT 的旧 vllm 实例）。"
  echo "——————————————————————————————————————————————————————————————————————"
  [ "$FORCE" = "1" ] || exit 1
fi

# ---- 1b. 清 page cache（消除加载期 direct reclaim 干扰）------------------------
# 6 个 rank 并行加载 + 两个 engram rank 各 94GB 冷读会把 ~500GB 内存逼到上限，
# 内核 direct reclaim 反复回收刚读入的页（实测 PSI full 3.9%、有效读速掉到
# ~60MB/s，单卡 engram 加载 30 分钟+）。启动前清掉无用缓存可显著缓解。
# 需要免密 sudo，不可用则跳过（不影响启动，只是可能慢）。
if [ "$DRY" = "0" ]; then
  if sudo -n sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null; then
    log "已清理 page cache（减少加载期内存回收干扰）"
  else
    log "跳过 page cache 清理（sudo 不可用）"
  fi
fi

# ---- 2. 环境 -------------------------------------------------------------------
if [ "$DRY" = "0" ]; then
  source /home/sean/miniconda3/etc/profile.d/conda.sh
  conda activate "$ENV_NAME"
  cd "$VLLM_ROOT"     # 必须 cd 到 repo 根，避免父目录 vllm/ 命名空间包遮蔽
  python -c "import vllm" >/dev/null 2>&1 || die "env $ENV_NAME 中 import vllm 失败，请先完成编译（见 build_vllm_v41.sh）"
fi

# ---- 3. 环境变量 ----------------------------------------------------------------
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPUS"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HF_HUB_OFFLINE=1
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
# 稀疏 indexer 的 logits 临时张量上限（默认 512MB）。全驻留下显存余量小，
# 长序列 prefill 的 chunk logits 会 OOM；128MB 让 chunk 切得更细（稍慢但稳）。
export VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128
# 全驻留显存临界（加载期 20MB 分配曾差 1MB 碎片 OOM），开可扩展段消除碎片
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
# PP 切分（影子模式下均分合法；--legacy-offload 退回 2,6,6,6,20）
export VLLM_PP_LAYER_PARTITION="$PP_PARTITION"
# 每 rank 专家卸载预算（Gen2 + repack 峰校准，见 OFFLOAD_RANKS 注释），必须 PP 个值
export VLLM_CPU_OFFLOAD_GB_PER_RANK="$OFFLOAD_RANKS"

# ---- 4. 清理本端口旧实例（仅 --force）-----------------------------------------
if [ "$FORCE" = "1" ] && [ "$DRY" = "0" ]; then
  pkill -f "vllm.entrypoints.openai.api_server.*--port $PORT" 2>/dev/null || true
  sleep 2
fi

# ---- 5. 组装参数 ----------------------------------------------------------------
if [ "$PIECEWISE" = "1" ]; then
  COMPILE_CFG='{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
else
  COMPILE_CFG='{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8],"max_cudagraph_capture_size":8}'
fi
if [ "$PLAIN" = "1" ]; then
  SPEC_DESC="(无投机)"
else
  SPEC_DESC="+ DSpark(x5)"
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/vllm_v41_pp${PP}_$(date +%Y%m%d_%H%M%S).log"

log "启动 DeepSeek-V4.1-Flash  PP${PP}($PP_PARTITION·$MODE_DESC) $SPEC_DESC  GPU=${GPU_IDS[*]}  端口=$PORT  maxlen=$MAXLEN  gpu-util=$GPU_UTIL  专家offload=${OFFLOAD_RANKS}GB/rank"
log "engram cpu_offload=ON(ngram→内存)  cudagraph=$([ "$PIECEWISE" = 1 ] && echo PIECEWISE || echo FULL_AND_PIECEWISE)  日志=$LOG_FILE"

CMD=(
  vllm serve "$MODEL_DIR"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$NAME"
  --pipeline-parallel-size "$PP"
  --tensor-parallel-size 1
  --max-model-len "$MAXLEN"
  --max-num-batched-tokens "$MAX_BATCHED"
  --max-num-seqs "$MAX_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --kv-cache-dtype fp8_ds_mla
  --trust-remote-code
  --engram-config '{"cpu_offload":true}'
)
if [ "$OFFLOAD_GB" != "0" ]; then
  # 注意: --cpu-offload-params 是 nargs=* 裸词列表，必须写 experts（不带 JSON 花括号），
  # 否则整个 '{"experts"}' 被当成一个词、匹配不到任何参数、卸载不生效。
  CMD+=( --cpu-offload-gb "$OFFLOAD_GB" --cpu-offload-params experts )
fi
CMD+=(
  --enable-prefix-caching
  --disable-custom-all-reduce
  --compilation-config "$COMPILE_CFG"
  --tokenizer-mode deepseek_v41
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
  --api-key "$API_KEY"
)
[ "$EAGER" = "1" ] && CMD+=( --enforce-eager )
[ "$PLAIN" = "0" ] && CMD+=( --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":5,\"use_local_argmax_reduction\":true}" )

if [ "$DRY" = "1" ]; then
  echo "VLLM_PP_LAYER_PARTITION=$PP_PARTITION"
  echo "VLLM_CPU_OFFLOAD_GB_PER_RANK=$OFFLOAD_RANKS"
  echo "CUDA_VISIBLE_DEVICES=$GPUS"
  printf ' %q' "${CMD[@]}"; echo
  exit 0
fi

# ---- 6. 启动 -------------------------------------------------------------------
if [ "$FG" = "1" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo $! > "$LOG_FILE.pid"
  log "已后台启动 PID=$(cat "$LOG_FILE.pid")，跟踪: tail -f $LOG_FILE"
fi
