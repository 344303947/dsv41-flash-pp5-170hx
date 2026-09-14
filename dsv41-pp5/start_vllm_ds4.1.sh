#!/usr/bin/env bash
# =============================================================================
# DeepSeek-V4.1-Flash @ 5x A100-64G 一键启动（影子源层均衡 PP5 · 纯显存模式）
#   ◆ 引擎: wtdcode/vllm-backport 本地编译 + 影子源层补丁（conda env vllm-v41）
#   ◆ 模型: /models/models/DeepSeek-V4.1-Flash （Native FP4 + Engram/ngram）
#   ◆ 并行: PP5 均分 8,8,8,8,8（--pipeline-parallel-size 5 --tensor-parallel-size 1；
#           heads=64 / kv_heads=1 不能被 5 整除，TP5 不可行）
#   ◆ ngram: --engram-config '{"cpu_offload":true}' 把 ~203GB engram 表卸到钉住内存
#            经 UVA 读取（layers.1@rank0 / layers.14@rank1 各 ~101GB）
#   ◆ 权重:  其余 307GB 全部驻留显存（每 rank ~60-61.5GB < 67.7GB）
#
# ★ 影子源层（本 fork 定制，打破 KV 共享组墙）:
#   v4.1 压缩层必须读取 KV 源层（kv_source_layer_ids=[2,8,14,20]）的压缩 KV /
#   indexer K / candidate，上游要求同 rank 同组 —— 原生只允许 2,6,6,6,20 切分
#   （{20-39} 20 层组 147.8GB 独占一卡，必须卸 ~110GB 专家 → decode 仅 10-16 t/s）。
#   本 fork 让消费 rank 用源层 prefix 实例化"影子 attention"，从 PP 边界接收
#   源层注意力输入 x（[T,5120]bf16 ≈10KB/token）本地精确重算 KV:
#     rank1 捕获 x@L14 → rank2 影子14（供 L16-19）
#     rank2 捕获 x@L20 → rank3 影子20（供 L24-31）→ rank4 影子20（供 L32-39）
#   代码: vllm/models/deepseek_v4_1/{shadow_source.py,nvidia/model.py,attention.py}
#   代价: 每步跨边界多传一个小张量 + 影子重算压缩/indexer（无 MoE）→ 近零开销。
#
# 显存预算（8,8,8,8,8，每层 7.4GB，卡 63GiB≈67.7GB）:
#   rank0(L0-7)   = 59.2 + embed1.3 + vision1.0 ≈ 61.5GB
#   rank1(L8-15)  = 59.2
#   rank2(L16-23) = 59.2 + 影子14 ≈ 59.4
#   rank3(L24-31) = 59.2 + 影子20 ≈ 59.4
#   rank4(L32-39) = 59.2 + head1.3 + 影子20 ≈ 60.7
#   实测 2026-09-14：移植 marlin_staged（repack 走 host RAM，无 raw+packed 双份）后
#   全驻留(0,0,0,0,0)成功，util 0.98 下 KV 1.78GiB，速度 ~10x（P1 13.3s→1.4s），
#   prefill ~4.6k tok/s，60K 长上下文正常。--dspark 时 rank4 需再放 mtp ~8.4GiB
#   → 卸 9GB（未实测）。
#
# 用法: ./start_vllm_v41_pp5.sh [选项]
#   --port P         API 端口（默认 9004）
#   --gpus LIST      使用的卡，逗号分隔（默认 0,1,2,3,4；须恰好 5 张）
#   --maxlen N       上下文长度（默认 1048576=1M，模型原生上限；KV 池 1,134,755 tokens 刚好覆盖）
#   --gpu-util N     gpu_memory_utilization（默认 0.95）
#   --dspark         启用 DSpark 投机解码（默认关闭；rank4 自动多卸 mtp 的 ~8GB 专家）
#   --plain          （兼容保留）关闭 DSpark —— 现在就是默认
#   --offload-gb N   手动指定 rank4 专家卸载 GB（0=纯显存）
#   --max-batched N  max-num-batched-tokens（默认 8192）
#   --max-seqs N     max-num-seqs（默认 8）
#   --piecewise      cudagraph 用 PIECEWISE（比 FULL_AND_PIECEWISE 省显存，decode 略慢）
#   --eager          enforce-eager 关 cudagraph（诊断用，排除图捕获干扰）
#   --legacy-offload 退回旧模式: 切分 2,6,6,6,20 + rank4 卸 110GB（不用影子，排障用）
#   --bg             后台运行（默认前台显示日志）
#   --force          启动前强制清理本端口的旧实例（默认绝不杀任何进程）
# 日志:   ~/logs/vllm_v41_pp5_*.log
# 地址:   http://0.0.0.0:$PORT/v1
# =============================================================================
set -euo pipefail

# ---- 固定约定 -----------------------------------------------------------------
ENV_NAME=vllm-v41
VLLM_ROOT=/home/sean/works/vllm-backport
MODEL_DIR=/model/DeepSeek-V4.1-Flash
API_KEY='sk-gRSilwwHpck1glDE9a40A435EcB04353957444F4Ad836807'
LOG_DIR="$HOME/logs"

# ---- 可配置项 -----------------------------------------------------------------
PORT=9004
GPUS="0,1,2,3,4"
MAXLEN=1048576
GPU_UTIL=0.97
# 每 rank 专家 UVA 卸载预算（GB，逗号分隔，按 PP rank；--cpu-offload-gb 取其最大值激活 UVA）。
# 实测校准(2026-09-14, PCIe Gen2 x16 + staged Marlin repack)：
#   全驻留(0,0,0,0,0)已可行：移植 marlin_staged 后加载期无 raw+packed 双份峰值，
#   KV cache 1.78GiB（util 0.98），P1 13.3s→1.4s、P4 65.7s→6.2s（~10x），
#   prefill ~4.6k tok/s，60K 长上下文正常。
#   如需更大 KV/并发（长上下文>70K 或多并发），可上调卸载量换取 KV 空间。
OFFLOAD_RANKS="0,0,0,0,0"
OFFLOAD_SET=0
PP_PARTITION="8,8,8,8,8"
MAX_BATCHED=2048
MAX_SEQS=8
PIECEWISE=0
EAGER=0
PLAIN=1
LEGACY=0
FG=1
FORCE=0
NAME=DeepSeek-V4.1-Flash

while [ $# -gt 0 ]; do
  case "$1" in
    --port)        PORT="$2"; shift 2 ;;
    --gpus)        GPUS="$2"; shift 2 ;;
    --maxlen)      MAXLEN="$2"; shift 2 ;;
    --gpu-util)    GPU_UTIL="$2"; shift 2 ;;
    --offload-gb)    OFFLOAD_RANKS="$2,$2,$2,$2,$2"; OFFLOAD_SET=1; shift 2 ;;
    --offload-ranks) OFFLOAD_RANKS="$2"; OFFLOAD_SET=1; shift 2 ;;
    --max-batched) MAX_BATCHED="$2"; shift 2 ;;
    --max-seqs)    MAX_SEQS="$2"; shift 2 ;;
    --piecewise)     PIECEWISE=1; shift ;;
    --eager)         EAGER=1; shift ;;
    --dspark)        PLAIN=0; shift ;;
    --plain)         PLAIN=1; shift ;;
    --legacy-offload) LEGACY=1; shift ;;
    --bg)          FG=0; shift ;;
    --force)       FORCE=1; shift ;;
    -h|--help|-help)
      sed -n '2,52p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# ---- 模式自适应（未显式指定 offload 时决定预算）----------------------------------
# --dspark: rank4 再加 ~8GB(mtp 权重不能卸)，总 18；--legacy-offload: 旧 102/110。
if [ "$LEGACY" = "1" ]; then
  PP_PARTITION="2,6,6,6,20"
  if [ "$OFFLOAD_SET" = "0" ]; then
    if [ "$PLAIN" = "1" ]; then OFFLOAD_RANKS="0,0,0,0,102"; else OFFLOAD_RANKS="0,0,0,0,110"; fi
  fi
elif [ "$OFFLOAD_SET" = "0" ] && [ "$PLAIN" = "0" ]; then
  # dspark 未在全驻留下实测：rank4 加 mtp 的 ~8.4GiB，卸 9GB 腾空间。
  OFFLOAD_RANKS="0,0,0,0,9"
fi
# 全局值 = 各 rank 预算最大值（仅用于激活 UVA 后端；实际按 rank 覆盖）
OFFLOAD_GB=0
IFS=',' read -ra _OR <<< "$OFFLOAD_RANKS"
for _o in "${_OR[@]}"; do
  [ "$_o" -gt "$OFFLOAD_GB" ] 2>/dev/null && OFFLOAD_GB="$_o"
done

log() { echo -e "$(date '+%F %T') [INFO]  $*"; }
die() { echo -e "$(date '+%F %T') [ERROR] $*" >&2; exit 1; }

# ---- 0. 卡数校验（PP5 必须恰好 5 张）------------------------------------------
IFS=',' read -ra GPU_IDS <<< "$GPUS"
if [ "${#GPU_IDS[@]}" -ne 5 ]; then
  die "PP5 需要恰好 5 张 GPU，当前 --gpus='$GPUS' 给了 ${#GPU_IDS[@]} 张"
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then die "未找到 nvidia-smi"; fi

# ---- 1. 预检：目标卡显存占用（默认只警告退出，绝不代杀进程）---------------------
BUSY=""
for g in "${GPU_IDS[@]}"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$g" 2>/dev/null || echo 0)
  if [ -n "$used" ] && [ "$used" -gt 4096 ]; then
    BUSY="$BUSY $g(${used}MiB)"
  fi
done
if [ -n "$BUSY" ]; then
  echo "——————————————————————————————————————————————————————————————————————"
  echo "  GPU ${GPU_IDS[*]} 中有卡显存已被占用:${BUSY}"
  echo "  PP5 需要这 5 张卡都基本空闲，否则会 OOM。默认【不代劳终止任何进程】。"
  echo "  请先手动停止占用卡片的旧实例（如 GPU 2 上的 Qwen3.8-27B）后重试；"
  echo "  确认无误可加 --force（仅清理本端口 $PORT 的旧 vllm 实例）。"
  echo "——————————————————————————————————————————————————————————————————————"
  [ "$FORCE" = "1" ] || exit 1
fi

# ---- 2. 环境 -------------------------------------------------------------------
source /home/sean/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
cd "$VLLM_ROOT"     # 必须 cd 到 repo 根，避免父目录 vllm/ 命名空间包遮蔽
python -c "import vllm" >/dev/null 2>&1 || die "env $ENV_NAME 中 import vllm 失败，请先完成编译（见 build_vllm_v41.sh）"

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
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
# PP 切分（影子模式下均分合法；--legacy-offload 退回 2,6,6,6,20）
export VLLM_PP_LAYER_PARTITION="$PP_PARTITION"
# 每 rank 专家卸载预算（Gen2 + repack 峰校准，见 OFFLOAD_RANKS 注释）
export VLLM_CPU_OFFLOAD_GB_PER_RANK="$OFFLOAD_RANKS"

# ---- 4. 清理本端口旧实例（仅 --force）-----------------------------------------
if [ "$FORCE" = "1" ]; then
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
LOG_FILE="$LOG_DIR/vllm_v41_pp5_$(date +%Y%m%d_%H%M%S).log"

MODE_DESC=影子均衡; [ "$LEGACY" = "1" ] && MODE_DESC=旧模式卸载
log "启动 DeepSeek-V4.1-Flash  PP5($PP_PARTITION·$MODE_DESC) $SPEC_DESC  GPU=${GPU_IDS[*]}  端口=$PORT  maxlen=$MAXLEN  gpu-util=$GPU_UTIL  专家offload=${OFFLOAD_RANKS}GB/rank"
log "engram cpu_offload=ON(ngram→内存)  cudagraph=$([ "$PIECEWISE" = 1 ] && echo PIECEWISE || echo FULL_AND_PIECEWISE)  日志=$LOG_FILE"

CMD=(
  vllm serve "$MODEL_DIR"
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$NAME"
  --pipeline-parallel-size 5
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

# ---- 6. 启动 -------------------------------------------------------------------
if [ "$FG" = "1" ]; then
  "${CMD[@]}" 2>&1 | tee "$LOG_FILE"
else
  nohup "${CMD[@]}" >"$LOG_FILE" 2>&1 &
  echo $! > "$LOG_FILE.pid"
  log "已后台启动 PID=$(cat "$LOG_FILE.pid")，跟踪: tail -f $LOG_FILE"
fi
