"""vllm#65385's repro_kkt_issue.py, with import fallbacks (argv module paths, vLLM >= 0.33 path, then upstream FLA).

Unchanged logic: for every autotune config of chunk_scaled_dot_kkt_fwd_kernel, run the kernel twice
on identical inputs and compare with a float32 reference (T=4096, H=32, K=128, BT=64).
"""
import importlib

import torch
import torch.nn.functional as F
import triton

import sys

M, SOURCE = None, None
for path in tuple(sys.argv[1:]) + ("vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt",
             "vllm.model_executor.layers.fla.ops.chunk_scaled_dot_kkt",
             "fla.ops.common.chunk_scaled_dot_kkt"):
    try:
        M, SOURCE = importlib.import_module(path), path
        break
    except Exception as exc:  # noqa: BLE001
        print(f"import {path} failed: {type(exc).__name__}: {exc}")
assert M is not None, "no chunk_scaled_dot_kkt module importable"

DEV = "cuda"
triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device=DEV, dtype=torch.int8))
T, H, K, BT = 4096, 32, 128, 64


def make_inputs(seed=0):
    g_ = torch.Generator(device=DEV).manual_seed(seed)
    r = lambda *s: torch.randn(*s, device=DEV, generator=g_)
    k = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
    beta = torch.sigmoid(r(1, T, H)).bfloat16()
    g = F.logsigmoid(r(1, T, H)) / 8
    g = g.view(1, T // BT, BT, H).cumsum(2).view(1, T, H).float()
    return k, g, beta


def ref_kkt(k, g, beta):
    n = T // BT
    kc = k.float().view(1, n, BT, H, K).permute(0, 1, 3, 2, 4)
    A = kc @ kc.transpose(-1, -2)
    gc = g.view(1, n, BT, H).permute(0, 1, 3, 2)
    A = A * torch.exp(gc[..., :, None] - gc[..., None, :])
    A = A * beta.float().view(1, n, BT, H).permute(0, 1, 3, 2)[..., :, None]
    strict_lower = torch.ones(BT, BT, dtype=torch.bool, device=DEV).tril(-1)
    A = A.masked_fill(~strict_lower, 0.0)
    return A.permute(0, 1, 3, 2, 4).reshape(1, T, H, BT)


def find_autotuner(obj):
    for _ in range(4):
        if hasattr(obj, "configs") and hasattr(obj, "cache"):
            return obj
        obj = getattr(obj, "fn", None)
        if obj is None:
            break
    raise RuntimeError("autotuner not found")


def main():
    print("torch %s | triton %s | %s | %s (sm%d%d)" % (
        torch.__version__, triton.__version__, SOURCE, torch.cuda.get_device_name(),
        *torch.cuda.get_device_capability()))
    k, g, beta = make_inputs()
    ref = ref_kkt(k, g, beta)
    tuner = find_autotuner(M.chunk_scaled_dot_kkt_fwd_kernel)
    configs = list(tuner.configs)
    print("%d autotune configs\n" % len(configs))
    n_bad = 0
    for cfg in configs:
        tuner.configs = [cfg]
        tuner.cache.clear()
        run = lambda: M.chunk_scaled_dot_kkt_fwd(k=k.clone(), g=g.clone(), beta=beta.clone(),
                                                 cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
        a1, a2 = run(), run()
        torch.cuda.synchronize()
        det = torch.equal(a1, a2)
        rel = ((a1 - ref).norm() / ref.norm()).item()
        bad = (not det) or rel > 1e-2
        n_bad += bad
        name = "BK=%-3s num_warps=%d num_stages=%d" % (cfg.kwargs.get("BK"), cfg.num_warps, cfg.num_stages)
        print("%-36s deterministic=%-5s rel_err_vs_fp32_ref=%.2e %s" % (name, det, rel, "<-- BAD" if bad else ""))
    tuner.configs = configs
    tuner.cache.clear()
    print("\n%d / %d configs are wrong" % (n_bad, len(configs)))


if __name__ == "__main__":
    main()
