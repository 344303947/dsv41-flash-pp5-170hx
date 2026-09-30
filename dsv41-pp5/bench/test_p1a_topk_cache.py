"""P1-A 回归：decode ragged topk 跨层复用语义。

- 同一 index source 的多个消费者：第二次起命中缓存（返回同一对象）。
- source 自己运行时清空缓存并重算。
用未绑定方法 + SimpleNamespace 直接驱动，避免实例化整个 attention。
"""

import types

import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    DeepseekV41ROCMAiterMLAAttention as A,
)

dev = "cuda"
PREFIX = "model.layers.2.attn"


def make_self(ratio, source, ctx_key=PREFIX, prefix="model.layers.3.attn"):
    s = types.SimpleNamespace()
    s.prefix = prefix
    s.compress_ratio = ratio
    s._index_source_prefix = ctx_key
    s._topk_ragged_cache = {}
    s._static_forward_context = {ctx_key: source}
    s.topk_indices_buffer = torch.tensor(
        [[0, 4, 7, -1], [8, -1, -1, -1]], dtype=torch.int32, device=dev
    )
    return s


source = types.SimpleNamespace(_topk_ragged_cache={})
swa = types.SimpleNamespace(
    token_to_req_indices=torch.tensor([0, 1], dtype=torch.int32, device=dev),
    is_valid_token=torch.tensor([True, True], device=dev),
)
attn = types.SimpleNamespace(
    block_table=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=dev),
    block_size=4,
)

# 消费者 1：计算并缓存
c1 = make_self(2, source)
r1 = A._decode_topk_ragged(c1, swa, attn, 2, 2)
assert source._topk_ragged_cache.get(2) is not None

# 消费者 2（同 source 同 ratio）：命中缓存，返回同一对象
c2 = make_self(2, source, prefix="model.layers.4.attn")
r2 = A._decode_topk_ragged(c2, swa, attn, 2, 2)
assert r2 is r1, "consumer must reuse the source's cached ragged metadata"
print("cross-layer reuse OK (same object)")

# source 自己运行：清空缓存并重算（新对象）
src_self = make_self(2, None, ctx_key=PREFIX, prefix=PREFIX)
src_self._static_forward_context = {PREFIX: src_self}
src_self._topk_ragged_cache = {2: r1}  # 假装上一步遗留
r3 = A._decode_topk_ragged(src_self, swa, attn, 2, 2)
assert r3 is not r1, "source run must rebuild, not reuse stale entry"
assert src_self._topk_ragged_cache[2] is r3
print("source invalidation+rebuild OK")

# 不同 ratio 各自缓存（source 上按 ratio 分槽）
c_ratio1 = make_self(1, source, prefix="model.layers.6.attn")
A._decode_topk_ragged(c_ratio1, swa, attn, 2, 2)
assert 1 in source._topk_ragged_cache and 2 in source._topk_ragged_cache
print("per-ratio cache slots OK")
print("P1-A regression PASSED")
