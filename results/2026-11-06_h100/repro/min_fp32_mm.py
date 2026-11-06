# Minimal repro: fp32 torch.mm / F.linear row result depends on batch size M under vLLM batch-invariant mode (H100/SM90).
import os
os.environ["VLLM_BATCH_INVARIANT"] = "1"
import torch
from vllm.model_executor.determinism.batch_invariant import init_batch_invariance, matmul_batch_invariant
init_batch_invariance()
print("fp32 precision:", torch.backends.cuda.matmul.fp32_precision, "CUBLAS_WORKSPACE_CONFIG:", os.environ.get("CUBLAS_WORKSPACE_CONFIG"))
K = N = 4096
g = torch.Generator("cuda").manual_seed(0)
w32 = torch.randn(N, K, device="cuda", generator=g) * 0.02
x32 = torch.randn(64, K, device="cuda", generator=g)
for name, x, w in (("fp32", x32, w32), ("bf16", x32.bfloat16(), w32.bfloat16())):
    ref = torch.mm(x[:1], w.t())
    for M in (2, 3, 8, 31, 32, 33, 64):
        out = torch.mm(x[:M], w.t())[:1]
        print(f"{name} mm M=1 vs M={M}: equal={torch.equal(ref, out)} maxabs={(ref.double()-out.double()).abs().max().item():.3e} n_diff={(ref!=out).sum().item()}")
# vLLM's own Triton kernel, called explicitly, for contrast
ref = matmul_batch_invariant(x32[:1], w32.t())
print("fp32 triton matmul_batch_invariant M=1 vs M=64 equal:", torch.equal(ref, matmul_batch_invariant(x32[:64], w32.t())[:1]))
