# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4.1 sparse MLA attention for SM8x (Ampere: A100/A800/A6000).

Reuses the ROCm Triton sparse-MLA implementation wholesale, exactly like the
V4 Ampere layer: its kernels, ragged metadata builders and bf16 o_proj
reference path are plain Triton/torch (the aiter-only preshuffle GEMMs and the
fused aiter norm+quant self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV41ROCMAiterMLAAttention,
)
from vllm.models.deepseek_v4_1.ampere.prefill_metadata import (
    combine_topk_swa_indices,
)
from vllm.platforms.interface import DeviceCapability


class DeepseekV41AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV41"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV41AmpereMLAAttention(DeepseekV41ROCMAiterMLAAttention):
    """SM8x DeepSeek V4.1 attention: ROCm Triton path on CUDA Ampere."""

    backend_cls = DeepseekV41AmpereMLASparseBackend

    @staticmethod
    def _combine_prefill_indices(*args, **kwargs):
        # Bound live eager intermediates after removing Torch's implicit syncs.
        if not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream().synchronize()
        return combine_topk_swa_indices(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The ROCm layer routes wo_a through the ordinary MXFP8 linear kernel
        # (is_bmm=False) because its bf16 einsum dequantizes the raw fp8 weight
        # itself. On CUDA that ordinary list is Marlin, whose repack would turn
        # the weight into packed garbage for the einsum. Keep the grouped-BMM
        # kernel list instead: on SM8x it resolves to the emulation kernel,
        # which dequantizes wo_a to a plain bf16 [g*r, d] weight at load time;
        # _get_cached_wo_a_bf16 then only views it (no second dequant).
        self.wo_a.is_bmm = True
