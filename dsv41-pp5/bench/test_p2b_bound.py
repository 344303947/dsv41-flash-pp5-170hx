"""P2-B 回归：block_table 越界 load 加 stride 边界后，
in-range 映射与手工期望一致，且 stale 大索引不再崩溃。"""

import torch

from vllm.models.deepseek_v4_1.amd.rocm import (
    compute_global_topk_ragged_indices_and_indptr as build,
)

dev = "cuda"
block_size = 4
# [num_reqs=2, width=3]
block_table = torch.tensor(
    [[10, 11, 12], [20, 21, 22]], dtype=torch.int32, device=dev
)
token_to_req = torch.tensor([0, 1], dtype=torch.int32, device=dev)
is_valid = torch.tensor([True, True], device=dev)

# row0 local [0,4,7,-1] -> slots [40,44,47]; row1 local [8,-1..] -> [88]
ok_topk = torch.tensor(
    [[0, 4, 7, -1], [8, -1, -1, -1]], dtype=torch.int32, device=dev
)
out, indptr, lens = build(ok_topk, token_to_req, block_table, block_size, is_valid)
row0 = out[indptr[0] : indptr[1]].tolist()
row1 = out[indptr[1] : indptr[2]].tolist()
assert row0 == [40, 44, 47], row0
assert row1 == [88], row1
assert lens.tolist() == [3, 1], lens.tolist()
print("in-range mapping OK:", row0, row1, lens.tolist())

# stale 大索引：block_indices=250000 >= width=3 -> 必须被 mask 掉、不得越界读。
bad_topk = torch.tensor(
    [[1000000, -1, -1, -1], [0, -1, -1, -1]], dtype=torch.int32, device=dev
)
out2, indptr2, lens2 = build(bad_topk, token_to_req, block_table, block_size, is_valid)
torch.cuda.synchronize()
print("stale index survived, out2:", out2.tolist(), "lens2:", lens2.tolist())

# stale 索引下的 in-range 邻居仍正确（row1 local 0 -> 20*4+0=80）
assert out2[indptr2[1] : indptr2[2]].tolist() == [80]
print("P2-B regression PASSED")
