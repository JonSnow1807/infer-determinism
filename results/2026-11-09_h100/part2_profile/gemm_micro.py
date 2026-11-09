"""cuBLAS (torch.mm, VLLM_BATCH_INVARIANT unset semantics) vs vLLM Triton batch-invariant GEMM, Qwen3-8B shapes."""
import json, sys, torch
import vllm.model_executor.determinism.batch_invariant as bi
bi.enable_batch_invariant_mode  # import check
try: bi.resolve_tuned_matmul_configs()
except Exception as e: print("tuned cfg:", e)
shapes = {"qkv(6144x4096)": (6144, 4096), "o(4096x4096)": (4096, 4096), "gate_up(24576x4096)": (24576, 4096),
          "down(4096x12288)": (4096, 12288), "lm_head(151936x4096)": (151936, 4096)}
def t(fn, n=200):
    for _ in range(20): fn()
    torch.cuda.synchronize(); s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n): fn()
    e.record(); torch.cuda.synchronize(); return 1000 * s.elapsed_time(e) / n
out = []
for name, (N, K) in shapes.items():
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    for M in (1, 8, 16, 32, 64, 128, 256, 512):
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        c = t(lambda: torch.mm(x, w.t())); b = t(lambda: bi.matmul_persistent(x, w.t()))
        out.append(dict(shape=name, M=M, N=N, K=K, cublas_us=c, triton_bi_us=b, ratio=b / c))
        print(f"{name:22s} M={M:4d} cublas {c:8.1f}us  triton-BI {b:8.1f}us  x{b/c:.2f}", flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
