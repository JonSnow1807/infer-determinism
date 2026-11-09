"""Minimal Triton-only repro of the chunk_scaled_dot_kkt (vllm#65385) failure on sm90.

Pattern: inside a K loop, the dot's A operand is computed in registers (`x * s[:, None]`),
so Triton lowers it to an async wgmma with A in registers, pipelined across iterations.
Control: same math with the scale applied to the accumulator after the loop (A from smem).

  python min_wgmma_rega_race.py [--runs 20]
Prints, per variant and config, how many of the runs differ bitwise from the first run
and the worst relative error against a float64 reference.
"""
import argparse
import subprocess

import torch
import triton
import triton.language as tl


@triton.jit
def scaled_gram_rega(X, S, Out, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o_t = pid * BT + tl.arange(0, BT)
    s = tl.load(S + o_t)
    acc = tl.zeros([BT, BT], dtype=tl.float32)
    for i in range(tl.cdiv(K, BK)):
        o_k = i * BK + tl.arange(0, BK)
        x = tl.load(X + o_t[:, None] * K + o_k[None, :])
        xs = x * s[:, None]  # A operand computed in registers inside the loop
        acc += tl.dot(xs, tl.trans(x))
    o = tl.arange(0, BT)
    tl.store(Out + pid * BT * BT + o[:, None] * BT + o[None, :], acc)


@triton.jit
def scaled_gram_smem(X, S, Out, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o_t = pid * BT + tl.arange(0, BT)
    s = tl.load(S + o_t)
    acc = tl.zeros([BT, BT], dtype=tl.float32)
    for i in range(tl.cdiv(K, BK)):
        o_k = i * BK + tl.arange(0, BK)
        x = tl.load(X + o_t[:, None] * K + o_k[None, :])
        acc += tl.dot(x, tl.trans(x))
    acc = acc * s[:, None].to(tl.float32)  # scale applied after the loop
    o = tl.arange(0, BT)
    tl.store(Out + pid * BT * BT + o[:, None] * BT + o[None, :], acc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--rows", type=int, default=131072)
    args = ap.parse_args()
    try:
        from triton.backends.nvidia.compiler import get_ptxas
        tool = get_ptxas(90) if get_ptxas.__code__.co_argcount else get_ptxas()
        ptxas = subprocess.run([tool.path, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception as exc:  # noqa: BLE001
        ptxas = f"unknown ({exc})"
    print(f"torch {torch.__version__} | triton {triton.__version__} | ptxas {ptxas} | {torch.cuda.get_device_name()}")
    BT, K = 64, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(args.rows, K, device="cuda", generator=g), dim=-1).bfloat16()
    s = torch.sigmoid(torch.randn(args.rows, device="cuda", generator=g)).bfloat16()
    n = args.rows // BT
    xb = x.double().view(n, BT, K)
    ref = (xb @ xb.transpose(1, 2)) * s.double().view(n, BT, 1)
    for name, kern in (("A in registers", scaled_gram_rega), ("scale after loop", scaled_gram_smem)):
        for bk, nw, ns in ((64, 4, 2), (64, 4, 3), (32, 4, 3), (64, 8, 2), (64, 2, 2), (128, 4, 2)):
            outs = []
            for _ in range(args.runs):
                out = torch.full((n, BT, BT), float("nan"), device="cuda")
                kern[(n,)](x, s, out, K=K, BT=BT, BK=bk, num_warps=nw, num_stages=ns)
                outs.append(out)
            torch.cuda.synchronize()
            differ = sum(not torch.equal(outs[0].view(torch.int32), o.view(torch.int32)) for o in outs[1:])
            rel = max(((o.double() - ref).norm() / ref.norm()).item() for o in outs)
            print(f"{name:17s} BK={bk:<3d} num_warps={nw} num_stages={ns}: {differ:2d}/{args.runs - 1} runs differ from run 0, "
                  f"worst rel_err={rel:.2e}{'  <-- NONDETERMINISTIC' if differ else ''}")


if __name__ == "__main__":
    main()
