"""Dump Triton IR/PTX for chosen chunk_scaled_dot_kkt configs and summarize the wgmma pattern.

Usage (env must be set before triton compiles; use a fresh cache dir):
  TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=OUT TRITON_CACHE_DIR=FRESH python dump_kkt_ir.py MODULE "BK,num_warps,num_stages" ...
"""
import importlib
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import triton

mod = importlib.import_module(sys.argv[1])
wanted = [tuple(int(x) for x in a.split(",")) for a in sys.argv[2:]]
T, H, K, BT = 4096, 32, 128, 64
g_ = torch.Generator(device="cuda").manual_seed(0)
r = lambda *s: torch.randn(*s, device="cuda", generator=g_)
k = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
beta = torch.sigmoid(r(1, T, H)).bfloat16()
g = (F.logsigmoid(r(1, T, H)) / 8).view(1, T // BT, BT, H).cumsum(2).view(1, T, H).float()
tuner = mod.chunk_scaled_dot_kkt_fwd_kernel
while not (hasattr(tuner, "configs") and hasattr(tuner, "cache")):
    tuner = tuner.fn
configs = list(tuner.configs)
dump = Path(os.environ["TRITON_DUMP_DIR"])
for bk, nw, ns in wanted:
    cfg = next(c for c in configs if c.kwargs.get("BK") == bk and c.num_warps == nw and c.num_stages == ns)
    before = set(dump.rglob("*.ptx")) if dump.exists() else set()
    tuner.configs = [cfg]
    tuner.cache.clear()
    mod.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
    torch.cuda.synchronize()
    new = sorted(set(dump.rglob("*.ptx")) - before)
    for ptx_path in new:
        ptx = ptx_path.read_text()
        ttgir = next(ptx_path.parent.glob("*.ttgir"), None)
        tg = ttgir.read_text() if ttgir else ""
        wg = re.findall(r"wgmma\.mma_async[^\n]*", ptx)
        reg_a = [w for w in wg if re.search(r"\{\s*%r\d+", w.split(",", 1)[1] if "," in w else "")]
        waits = re.findall(r"wgmma\.wait_group\.sync\.aligned\s+(\d+)", ptx)
        print(f"BK={bk} num_warps={nw} num_stages={ns}: {ptx_path.parent.name}")
        print(f"  ptx wgmma.mma_async: {len(wg)} (A from registers: {len(reg_a)}), mma.sync: {len(re.findall(r'mma.sync.aligned', ptx))}, "
              f"wgmma.wait_group args: {sorted(set(waits))}")
        print(f"  ttgir: warp_group_dot={tg.count('warp_group_dot')} is_async={len(re.findall(r'isAsync = true', tg))} "
              f"warp_group_dot_wait={tg.count('warp_group_dot_wait')} local_alloc={tg.count('local_alloc')} scf.for={tg.count('scf.for')}")
        if wg:
            print("  first wgmma:", wg[0][:160])
tuner.configs = configs
tuner.cache.clear()
