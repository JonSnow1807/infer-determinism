"""Run one pinned kkt config and save its output, to compare assemblers (run once per ptxas setting).

  python ptxas_numerics.py MODULE BK NUM_WARPS NUM_STAGES OUT.pt
"""
import importlib
import subprocess
import sys

import torch
import torch.nn.functional as F
import triton

mod = importlib.import_module(sys.argv[1])
bk, nw, ns, out = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
T, H, K, BT = 4096, 32, 128, 64
g_ = torch.Generator(device="cuda").manual_seed(0)
r = lambda *s: torch.randn(*s, device="cuda", generator=g_)
k = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
beta = torch.sigmoid(r(1, T, H)).bfloat16()
g = (F.logsigmoid(r(1, T, H)) / 8).view(1, T // BT, BT, H).cumsum(2).view(1, T, H).float()
tuner = mod.chunk_scaled_dot_kkt_fwd_kernel
while not (hasattr(tuner, "configs") and hasattr(tuner, "cache")):
    tuner = tuner.fn
tuner.configs = [c for c in tuner.configs if c.kwargs.get("BK") == bk and c.num_warps == nw and c.num_stages == ns]
A = mod.chunk_scaled_dot_kkt_fwd(k=k, g=g, beta=beta, cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
n = T // BT
kc = k.float().view(1, n, BT, H, K).permute(0, 1, 3, 2, 4)
ref = kc @ kc.transpose(-1, -2)
gc = g.view(1, n, BT, H).permute(0, 1, 3, 2)
ref = ref * torch.exp(gc[..., :, None] - gc[..., None, :]) * beta.float().view(1, n, BT, H).permute(0, 1, 3, 2)[..., :, None]
ref = ref.masked_fill(~torch.ones(BT, BT, dtype=torch.bool, device="cuda").tril(-1), 0.0).permute(0, 1, 3, 2, 4).reshape(1, T, H, BT)
from triton.backends.nvidia.compiler import get_ptxas
tool = get_ptxas(90) if get_ptxas.__code__.co_argcount else get_ptxas()
ver = subprocess.run([tool.path, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
rel = ((A - ref).norm() / ref.norm()).item()
print(f"triton {triton.__version__} ptxas {ver} {sys.argv[1].rsplit('.', 1)[-1]} BK={bk} nw={nw} ns={ns}: rel_err={rel:.3e}")
torch.save({"A": A.cpu(), "ref": ref.cpu(), "ptxas": ver}, out)
