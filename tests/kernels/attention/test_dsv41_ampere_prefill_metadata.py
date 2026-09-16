# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The Ampere device-side prefill index constructor must match the Torch one.

``DeepseekV41AmpereMLAAttention`` overrides ``_combine_prefill_indices`` with
``ampere/prefill_metadata.py`` to drop the Torch indexing path's implicit host
syncs. The rows and lengths it writes must stay identical to the
``combine_topk_swa_indices`` fallback in ``amd/rocm.py``.
"""

import pytest
import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    combine_topk_swa_indices as torch_combine,
)
from vllm.models.deepseek_v4_1.ampere.prefill_metadata import (
    combine_topk_swa_indices as device_combine,
)
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="device-side combine requires CUDA/ROCm",
)


@pytest.mark.parametrize(
    ("window_size", "compress_ratio", "topk", "M", "N", "query_lens", "seq_lens"),
    [
        (128, 4, 64, 8, 4096, [5, 7, 11], [300, 1024, 2048]),
        (64, 8, 32, 4, 2048, [1], [77]),
        (128, 4, 64, 8, 4096, [2, 2, 2, 2], [10, 20, 30, 40]),
    ],
)
def test_device_prefill_indices_match_torch(
    window_size, compress_ratio, topk, M, N, query_lens, seq_lens
):
    torch.manual_seed(0)
    device = "cuda"
    query_start_loc = [0]
    for length in query_lens:
        query_start_loc.append(query_start_loc[-1] + length)
    num_tokens = query_start_loc[-1]
    query_start_loc = torch.tensor(query_start_loc, dtype=torch.int32, device=device)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    gather_lens = seq_lens_t.clone()
    topk_indices = torch.randint(
        0, N, (num_tokens, topk), dtype=torch.int32, device=device
    )

    args = (
        topk_indices,
        query_start_loc,
        seq_lens_t,
        gather_lens,
        window_size,
        compress_ratio,
        topk,
        M,
        N,
    )
    expected_indices, expected_lens = torch_combine(*args)
    indices, lens = device_combine(*args)
    assert torch.equal(lens, expected_lens)
    assert torch.equal(indices, expected_indices)
