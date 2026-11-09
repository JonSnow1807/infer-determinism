"""Discriminating experiments for min_wgmma_rega_race.py (BK=64, num_warps=4, num_stages=2).

variant "sepB": A = x*s from registers as before, but the B operand comes from a separately
loaded pointer (same data), so A's and B's shared-memory tiles are different allocations.
Run under TRITON_KERNEL_OVERRIDE to test an edited TTGIR (see min_race_override.sh).
"""
import sys

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
        xs = x * s[:, None]
        acc += tl.dot(xs, tl.trans(x))
    o = tl.arange(0, BT)
    tl.store(Out + pid * BT * BT + o[:, None] * BT + o[None, :], acc)


@triton.jit
def scaled_gram_rega_sepB(X, Y, S, Out, K: tl.constexpr, BT: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    o_t = pid * BT + tl.arange(0, BT)
    s = tl.load(S + o_t)
    acc = tl.zeros([BT, BT], dtype=tl.float32)
    for i in range(tl.cdiv(K, BK)):
        o_k = i * BK + tl.arange(0, BK)
        x = tl.load(X + o_t[:, None] * K + o_k[None, :])
        y = tl.load(Y + o_t[:, None] * K + o_k[None, :])
        acc += tl.dot(x * s[:, None], tl.trans(y))
    o = tl.arange(0, BT)
    tl.store(Out + pid * BT * BT + o[:, None] * BT + o[None, :], acc)


def main(which: str, runs: int = 20, rows: int = 131072):
    BT, K = 64, 128
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.nn.functional.normalize(torch.randn(rows, K, device="cuda", generator=g), dim=-1).bfloat16()
    y = x.clone()
    s = torch.sigmoid(torch.randn(rows, device="cuda", generator=g)).bfloat16()
    n = rows // BT
    xb = x.double().view(n, BT, K)
    ref = (xb @ xb.transpose(1, 2)) * s.double().view(n, BT, 1)
    outs = []
    for _ in range(runs):
        out = torch.full((n, BT, BT), float("nan"), device="cuda")
        if which == "sepB":
            scaled_gram_rega_sepB[(n,)](x, y, s, out, K=K, BT=BT, BK=64, num_warps=4, num_stages=2)
        else:
            scaled_gram_rega[(n,)](x, s, out, K=K, BT=BT, BK=64, num_warps=4, num_stages=2)
        outs.append(out)
    torch.cuda.synchronize()
    differ = sum(not torch.equal(outs[0].view(torch.int32), o.view(torch.int32)) for o in outs[1:])
    rel = max(((o.double() - ref).norm() / ref.norm()).item() for o in outs)
    print(f"triton {triton.__version__} {which:6s} BK=64 num_warps=4 num_stages=2: {differ}/{runs - 1} runs differ, worst rel_err={rel:.2e}")


if __name__ == "__main__":
    main(sys.argv[1])
