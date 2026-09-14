# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cross-rank shadow replica of a v4.1 kv-source attention layer.

When a PP split lands inside a v4.1 kv-sharing group, consumer layers on one
rank need the compressed-KV / indexer-K / candidate caches that a kv-source
layer publishes from another rank. Shipping that (large, growing) cache across
the wire every step is prohibitive, so the consumer rank instead hosts a
*shadow*: a replica of the source layer's attention that rebuilds those caches
locally from the source layer's *attention input* — a per-step activation of
just ``[num_tokens, hidden_size]``, cheap to forward across PP boundaries.

The input is captured mid-layer on the owner rank (after the hyperconnection
pre-mix and any Engram injection), so the shadow's replay — including
Engram-conditioned sources like layer 14 — reproduces the exact attention
input the source layer sees. The shadow reuses the source attention's own
``_run_parallel_input_projections`` / ``compressor`` / ``indexer`` (see
``DeepseekV4Attention.produce_kv_side_effects``), making the rebuilt rows
bit-identical to what the real source layer would have written. Only the
attention subtree is replicated: the source layer's MoE experts, FFN and
Engram tables stay on the owner rank.

The shadow is instantiated under the *source layer's own prefix*, so it
registers in this rank's ``static_forward_context`` under exactly the name the
consumers already look up (``compressed_cache_prefix``, the indexer
``k_cache_prefix``, candidate/topk buffers included). No consumer-side code
changes: their existing lookups resolve to the shadow's caches, and the
``__init__`` guards that raise ``NotImplementedError`` for cross-rank sources
are satisfied.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class ShadowSource(nn.Module):
    """Replica of one kv-source layer's attention that only produces KV.

    Args:
        vllm_config: engine config.
        source_layer_prefix: full prefix of the source decoder layer, e.g.
            ``"model.layers.20"``. The shadow's attention is built under
            ``f"{source_layer_prefix}.attn"`` so consumers that look the
            source up by that name resolve to this replica.
        topk_indices_buffer / candidate_block_buffer / aux_stream_list: the
            same shared objects the real layers on this rank use, so the
            shadow's published topk/candidates land in the buffers the
            consumers read.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        source_layer_prefix: str,
        topk_indices_buffer: torch.Tensor | None = None,
        candidate_block_buffer: torch.Tensor | None = None,
        aux_stream_list: list | None = None,
    ) -> None:
        super().__init__()
        # Lazy import: model.py imports this module, so importing it at module
        # scope here would be circular. By call time model.py is fully loaded.
        from vllm.models.deepseek_v4_1.nvidia.model import _select_dsv4_attn_cls

        attn_cls = _select_dsv4_attn_cls(vllm_config)
        self.source_layer_prefix = source_layer_prefix
        self.attn = attn_cls(
            vllm_config,
            prefix=f"{source_layer_prefix}.attn",
            topk_indices_buffer=topk_indices_buffer,
            aux_stream_list=aux_stream_list,
            candidate_block_buffer=candidate_block_buffer,
        )
        logger.info(
            "Built ShadowSource for kv-source layer '%s' on this rank",
            source_layer_prefix,
        )

    def forward(self, attn_input: torch.Tensor, positions: torch.Tensor) -> None:
        """Rebuild the source layer's KV caches from its attention input.

        ``attn_input`` must be the exact tensor the source layer feeds into its
        attention (post hyperconnection pre-mix / Engram injection / attn RMSNorm
        path up to the attention entry), routed here across the PP boundary.
        """
        self.attn.produce_kv_side_effects(attn_input, positions)
