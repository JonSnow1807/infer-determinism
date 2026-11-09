"""Scenarios for the autotune-config sweep, with float32 references.

Each scenario builds fixed inputs (seeded, on CUDA) and calls one library entry
point; the sweep engine captures every autotuned Triton launch underneath it.
References are plain PyTorch float32 and are deliberately simple loops over
chunks or tokens; they are oracles for the sweep, not fast implementations.

Import paths: vLLM 0.33 vendors flash-linear-attention under
``vllm.third_party.flash_linear_attention``; older releases (e.g. the 0.19.1
of vllm#65385) used ``vllm.model_executor.layers.fla``. Upstream FLA is the
``fla`` package (fla-core).
"""
from __future__ import annotations

import importlib
import importlib.metadata
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F

from .autotune_sweep import Launch


class SkipScenario(Exception):
    """Library or device not available for this scenario."""


@dataclass
class Built:
    run: Callable[[], Any]
    e2e_reference: Callable[[Any], Mapping[str, torch.Tensor]] | None = None
    e2e_outputs: Callable[[Any], Mapping[str, torch.Tensor]] | None = None
    meta: dict[str, object] = field(default_factory=dict)
    # Called before every capture: clears caches that would route launches around
    # Autotuner.run (vLLM's mamba launch_autotuned keeps its own config cache).
    reset: Callable[[], None] = lambda: None


@dataclass
class Scenario:
    name: str
    description: str
    build: Callable[[], Built]
    references: dict[str, Callable[[Launch], Mapping[str, torch.Tensor]]] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    e2e_tol: float = 2e-2


def _import(*paths: str):
    errors = []
    for path in paths:
        try:
            return importlib.import_module(path)
        except Exception as exc:  # ImportError, or a library failing at import time
            errors.append(f"{path}: {type(exc).__name__}: {exc}")
    raise SkipScenario("; ".join(errors))


def _cuda() -> str:
    if not torch.cuda.is_available():
        raise SkipScenario("CUDA is required")
    try:
        import triton  # noqa: F401
        triton.set_allocator(lambda size, alignment, stream: torch.empty(size, device="cuda", dtype=torch.int8))
    except ImportError as exc:
        raise SkipScenario(f"triton is missing: {exc}") from exc
    except AttributeError:
        pass  # older triton without set_allocator
    return "cuda"


def _gen(seed: int, device: str) -> Callable[..., torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    return lambda *shape: torch.randn(*shape, device=device, generator=g)


def _segments(T: int, B: int, cu: torch.Tensor | None) -> list[tuple[int, int, int]]:
    if cu is None:
        return [(b, 0, T) for b in range(B)]
    c = [int(x) for x in cu.tolist()]
    return [(0, c[i], c[i + 1]) for i in range(len(c) - 1)]


# ---------------------------------------------------------------------------
# Kernel-level references (inputs are the captured launch arguments)


def ref_kkt_exp2(launch: Launch) -> dict[str, torch.Tensor]:
    """Upstream FLA >= 0.6 keeps g in log2 space (chunk_local_cumsum scale=RCP_LN2) and gates with exp2."""
    return ref_kkt(launch, base2=True)


def ref_kkt(launch: Launch, base2: bool = False) -> dict[str, torch.Tensor]:
    """A[t,h,j] = beta[t,h] * exp(g[t,h]-g[s,h]) * <k[t], k[s]>, s = chunk start + j < t."""
    a = launch.inputs()
    k, beta, g = a["k"].float(), a["beta"].float(), a.get("g")
    cu = a.get("cu_seqlens")
    B, T, Hk, K = k.shape
    H = beta.shape[-1]
    BT = int(a["BT"])
    k = k.repeat_interleave(H // Hk, dim=2)
    out = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)
    for b, bos, eos in _segments(T, B, cu):
        for c0 in range(bos, eos, BT):
            c1 = min(c0 + BT, eos)
            n = c1 - c0
            kc = k[b, c0:c1]  # [n, H, K]
            A = torch.einsum("ihk,jhk->hij", kc * beta[b, c0:c1, :, None], kc)
            if g is not None:
                gc = g[b, c0:c1].float().T  # [H, n]
                A = A * (torch.exp2 if base2 else torch.exp)(gc[:, :, None] - gc[:, None, :])
            A = A * torch.ones(n, n, device=k.device).tril(-1)
            out[b, c0:c1, :, :n] = A.permute(1, 0, 2)
    return {"A": out}


def ref_solve_tril(launch: Launch) -> dict[str, torch.Tensor]:
    """(I + A)^-1 per chunk; A strictly lower triangular, [B, T, H, BT]."""
    a = launch.inputs()
    A, cu = a["A"].float(), a.get("cu_seqlens")
    B, T, H, BT = A.shape
    out = torch.zeros_like(A)
    for b, bos, eos in _segments(T, B, cu):
        for c0 in range(bos, eos, BT):
            c1 = min(c0 + BT, eos)
            n = c1 - c0
            blk = A[b, c0:c1, :, :n].permute(1, 0, 2).double()  # [H, n, n]
            eye = torch.eye(n, device=A.device, dtype=torch.float64).expand(H, n, n)
            inv = torch.linalg.solve_triangular(eye + blk.tril(-1), eye, upper=False)
            out[b, c0:c1, :, :n] = inv.permute(1, 0, 2).float()
    return {"Ai": out}


def ref_chunk_o(launch: Launch) -> dict[str, torch.Tensor]:
    """GDN output: scale * (q h^T * exp(g) + tril(q k^T * exp(g_i - g_j)) v) per chunk."""
    a = launch.inputs()
    q, k, v, h, g = (a[n].float() if a.get(n) is not None else None for n in ("q", "k", "v", "h", "g"))
    cu, scale = a.get("cu_seqlens"), float(a["scale"])
    B, T, Hk, K = q.shape
    H, V, BT = v.shape[2], v.shape[3], int(a["BT"])
    q, k = q.repeat_interleave(H // Hk, dim=2), k.repeat_interleave(H // Hk, dim=2)
    out = torch.zeros(B, T, H, V, device=q.device, dtype=torch.float32)
    chunk = 0
    for b, bos, eos in _segments(T, B, cu):
        for c0 in range(bos, eos, BT):
            c1 = min(c0 + BT, eos)
            n = c1 - c0
            tg = chunk if cu is not None else b * math.ceil(T / BT) + (c0 - bos) // BT
            hq = h.reshape(-1, H, V, K)[tg]  # [H, V, K]
            qc, kc, vc = q[b, c0:c1].transpose(0, 1), k[b, c0:c1].transpose(0, 1), v[b, c0:c1].transpose(0, 1)
            o = torch.einsum("hik,hvk->hiv", qc, hq)
            A = torch.einsum("hik,hjk->hij", qc, kc)
            if g is not None:
                gc = g[b, c0:c1].T
                o = o * torch.exp(gc)[:, :, None]
                A = A * torch.exp(gc[:, :, None] - gc[:, None, :])
            A = A * torch.ones(n, n, device=q.device).tril()
            out[b, c0:c1] = ((o + A @ vc) * scale).transpose(0, 1)
            chunk += 1
    return {"o": out}


def ref_l2norm(launch: Launch) -> dict[str, torch.Tensor]:
    a = launch.inputs()
    x = a["x"].float()
    eps = float(a.get("eps", 1e-6))
    D = int(a["D"]) if "D" in a else x.shape[-1]
    x2 = x.reshape(-1, D)
    return {"y": (x2 / torch.sqrt((x2 * x2).sum(-1, keepdim=True) + eps)).reshape(a["y"].shape)}


def ref_cumsum_scalar(launch: Launch) -> dict[str, torch.Tensor]:
    a = launch.inputs()
    s, cu, BT = a["s"].float(), a.get("cu_seqlens"), int(a["BT"])
    if a.get("HEAD_FIRST") or a.get("REVERSE"):
        raise ValueError("reference covers HEAD_FIRST=False, REVERSE=False only")
    B, T = s.shape[0], s.shape[1]
    out = torch.zeros_like(s)
    for b, bos, eos in _segments(T, B, cu):
        for c0 in range(bos, eos, BT):
            c1 = min(c0 + BT, eos)
            out[b, c0:c1] = s[b, c0:c1].cumsum(0)
    scale = a.get("scale")
    if isinstance(scale, float) and a.get("HAS_SCALE"):
        out = out * scale
    return {"o": out}


# ---------------------------------------------------------------------------
# End-to-end references (sequential recurrences in float32)


def naive_delta_rule(q, k, v, g, beta, scale, h0, cu, *, per_dim_gate: bool, state_vk: bool, l2norm: bool):
    """Gated delta rule over tokens. Returns (o [1,T,H,V], final [N,H,*,*]).

    State per head is S[v, k] when state_vk else S[k, v]; g is natural-log decay,
    scalar per head ([1,T,H]) or per key dim ([1,T,H,K]) when per_dim_gate.
    """
    q, k, v, g, beta = q.float(), k.float(), v.float(), g.float(), beta.float()
    if l2norm:
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    _, T, Hk, K = q.shape
    H, V = v.shape[2], v.shape[3]
    q, k = q.repeat_interleave(H // Hk, dim=2), k.repeat_interleave(H // Hk, dim=2)
    out = torch.zeros(1, T, H, V, device=q.device)
    finals = []
    for n, (_, bos, eos) in enumerate(_segments(T, 1, cu)):
        S = h0[n].float().clone() if h0 is not None else torch.zeros(H, V, K, device=q.device)
        if not state_vk:
            S = S.transpose(-1, -2).contiguous()  # stored [K, V] -> work in [V, K]
        for t in range(bos, eos):
            decay = torch.exp(g[0, t])  # [H] or [H, K]
            S = S * (decay[:, None, :] if per_dim_gate else decay[:, None, None])
            kt, vt, qt = k[0, t], v[0, t], q[0, t]
            u = beta[0, t][:, None] * (vt - torch.einsum("hvk,hk->hv", S, kt))
            S = S + u[:, :, None] * kt[:, None, :]
            out[0, t] = torch.einsum("hvk,hk->hv", S, qt * scale)
        finals.append(S if state_vk else S.transpose(-1, -2))
    return out, torch.stack(finals)


def naive_ssd(x, dt, A, Bm, Cm, D, dt_bias, h0, cu, dt_softplus=True):
    """Sequential Mamba2 scan; x [T,nh,hd], B/C [T,ng,ds]. Returns (y, final [N,nh,hd,ds])."""
    x, dt, Bm, Cm = x.float(), dt.float(), Bm.float(), Cm.float()
    T, nh, hd = x.shape
    ng = Bm.shape[1]
    Bm, Cm = Bm.repeat_interleave(nh // ng, dim=1), Cm.repeat_interleave(nh // ng, dim=1)
    dt = dt + dt_bias.float()[None] if dt_bias is not None else dt
    dt = F.softplus(dt) if dt_softplus else dt
    y = torch.zeros(T, nh, hd, device=x.device)
    finals = []
    for n, (_, bos, eos) in enumerate(_segments(T, 1, cu)):
        h = h0[n].float().clone() if h0 is not None else torch.zeros(nh, hd, Bm.shape[-1], device=x.device)
        for t in range(bos, eos):
            dA = torch.exp(dt[t] * A.float())  # [nh]
            h = h * dA[:, None, None] + (dt[t][:, None, None] * x[t][:, :, None]) * Bm[t][:, None, :]
            y[t] = torch.einsum("hds,hs->hd", h, Cm[t])
            if D is not None:
                y[t] = y[t] + (D.float()[:, None] if D.dim() == 1 else D.float()) * x[t]
        finals.append(h)
    return y, torch.stack(finals)


# ---------------------------------------------------------------------------
# Scenario builders

VLLM_FLA = ("vllm.third_party.flash_linear_attention.ops", "vllm.model_executor.layers.fla.ops")
SEQLENS = (1000, 1600, 1496)  # three sequences, 4096 tokens, none a multiple of 64


def _vllm_fla(sub: str):
    return _import(*(f"{p}.{sub}" for p in VLLM_FLA))


def _cu(seqlens, device) -> torch.Tensor:
    return torch.tensor([0, *torch.tensor(seqlens).cumsum(0).tolist()], device=device, dtype=torch.int32)


def build_kkt_issue(kind: str = "vllm", K: int = 128) -> Built:
    """The exact inputs of vllm#65385's repro_kkt_issue.py (T=4096, H=32, K=128, BT=64).

    kind="fla" feeds the same natural-log g to upstream FLA, whose kernel gates with exp2,
    so it computes a different (but self-consistent) function; it is checked against
    ref_kkt_exp2, not compared numerically with the vLLM kernel.
    """
    dev = _cuda()
    if kind == "vllm":
        mod = _vllm_fla("chunk_scaled_dot_kkt")
        fwd = mod.chunk_scaled_dot_kkt_fwd
    elif kind == "vllm0191":  # v0.19.1 kernel source installed beside the 0.33 ops (scripts/vendored)
        fwd = _vllm_fla("chunk_scaled_dot_kkt_v0191").chunk_scaled_dot_kkt_fwd
    elif kind == "betaafter":  # experiment: beta applied after the K loop (scripts/vendored)
        fwd = _vllm_fla("chunk_scaled_dot_kkt_betaafter").chunk_scaled_dot_kkt_fwd
    else:
        fwd = _import("fla.ops.common.chunk_scaled_dot_kkt").chunk_scaled_dot_kkt_fwd
    T, H, BT = 4096, 32, 64
    r = _gen(0, dev)
    k = F.normalize(r(1, T, H, K), dim=-1).bfloat16()
    beta = torch.sigmoid(r(1, T, H)).bfloat16()
    g = F.logsigmoid(r(1, T, H)) / 8
    g = g.view(1, T // BT, BT, H).cumsum(2).view(1, T, H).float()

    def run():
        return fwd(k=k.clone(), g=g.clone(), beta=beta.clone(), cu_seqlens=None, chunk_size=BT, output_dtype=torch.float32)
    return Built(run, meta={"T": T, "H": H, "K": K, "BT": BT, "source": kind, "module": getattr(fwd, "__module__", "?")})


def _gdn_inputs(dev: str, T: int, Hk: int, H: int, K: int, V: int, seqlens, seed: int = 7):
    r = _gen(seed, dev)
    q = r(1, T, Hk, K).bfloat16()
    k = r(1, T, Hk, K).bfloat16()
    v = r(1, T, H, V).bfloat16()
    g = (F.logsigmoid(r(1, T, H)) / 8).float()
    beta = torch.sigmoid(r(1, T, H)).bfloat16()
    N = len(seqlens) if seqlens else 1
    h0 = (r(N, H, V, K) * 0.1).float()
    cu = _cu(seqlens, dev) if seqlens else None
    return q, k, v, g, beta, h0, cu


def build_gdn(kind: str, seqlens, Hk: int, H: int) -> Built:
    dev = _cuda()
    T, K, V = 4096, 128, 128
    q, k, v, g, beta, h0, cu = _gdn_inputs(dev, T, Hk, H, K, V, seqlens)
    scale = K ** -0.5
    if kind == "vllm":
        fn = _vllm_fla("chunk").chunk_gated_delta_rule
        state_vk = True

        def run():
            return fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=h0.clone(), output_final_state=True,
                      cu_seqlens=cu, use_qk_l2norm_in_kernel=True)
    else:
        fn = _import("fla.ops.gated_delta_rule").chunk_gated_delta_rule
        state_vk = False
        h0_kv = h0.transpose(-1, -2).contiguous()

        def run():
            return fn(q=q, k=k, v=v, g=g, beta=beta, scale=scale, initial_state=h0_kv.clone(), output_final_state=True,
                      cu_seqlens=cu.long() if cu is not None else None, use_qk_l2norm_in_kernel=True)

    def reference(_out):
        o, final = naive_delta_rule(q, k, v, g, beta, scale, h0, cu, per_dim_gate=False, state_vk=True, l2norm=True)
        return {"o": o, "final_state": final if state_vk else final.transpose(-1, -2)}

    def outputs(out):
        return {"o": out[0], "final_state": out[1]}
    return Built(run, reference, outputs, {"T": T, "Hk": Hk, "H": H, "K": K, "V": V, "seqlens": list(seqlens or []),
                                           "source": kind, "module": fn.__module__})


def build_kda(kind: str, seqlens) -> Built:
    dev = _cuda()
    T, H, K, V = 4096, 32, 128, 128
    if kind == "vllm":
        fn = _vllm_fla("kda").chunk_kda
    else:
        fn = _import("vllm.models.kimi_k3.nvidia.ops.third_party.kda").chunk_kda
    r = _gen(11, dev)
    q, k, v = r(1, T, H, K).bfloat16(), r(1, T, H, K).bfloat16(), r(1, T, H, V).bfloat16()
    g = (F.logsigmoid(r(1, T, H, K)) / 8).float()
    beta = torch.sigmoid(r(1, T, H)).float()
    N = len(seqlens)
    h0 = (r(N, H, V, K) * 0.1).float()
    cu = _cu(seqlens, dev)
    scale = K ** -0.5

    def run():
        return fn(q=q, k=k, v=v.clone(), g=g, beta=beta, scale=scale, initial_state=h0.clone(), output_final_state=True,
                  use_qk_l2norm_in_kernel=True, cu_seqlens=cu)

    def reference(_out):
        o, final = naive_delta_rule(q, k, v, g, beta, scale, h0, cu, per_dim_gate=True, state_vk=True, l2norm=True)
        return {"o": o, "final_state": final}

    def outputs(out):
        return {"o": out[0], "final_state": out[1]}
    return Built(run, reference, outputs, {"T": T, "H": H, "K": K, "V": V, "seqlens": list(seqlens), "source": kind,
                                           "module": fn.__module__})


def build_mamba2(seqlens) -> Built:
    dev = _cuda()
    combined = _import("vllm.model_executor.layers.mamba.ops.ssd_combined")
    meta_mod = _import("vllm.v1.attention.backends.mamba2_attn")
    T = sum(seqlens)
    nheads, headdim, ngroups, dstate, chunk = 64, 64, 8, 128, 256
    r = _gen(13, dev)
    x = r(T, nheads, headdim).bfloat16()
    dt = (r(T, nheads) * 0.5 - 2.0).float()
    A = -torch.exp(r(nheads) * 0.5).float()
    Bm, Cm = r(T, ngroups, dstate).bfloat16(), r(T, ngroups, dstate).bfloat16()
    D = torch.ones(nheads, device=dev, dtype=torch.float32)
    dt_bias = (r(nheads) * 0.1).float()
    h0 = (r(len(seqlens), nheads, headdim, dstate) * 0.1).float()
    cu = _cu(seqlens, dev)
    cu_chunk, last_chunk, seq_idx = meta_mod.compute_varlen_chunk_metadata(cu, chunk)

    def run():
        out = torch.empty_like(x)
        final = combined.mamba_chunk_scan_combined_varlen(
            x, dt, A, Bm, Cm, chunk, cu_seqlens=cu, cu_chunk_seqlens=cu_chunk, last_chunk_indices=last_chunk,
            seq_idx=seq_idx, out=out, D=D, dt_bias=dt_bias, initial_states=h0.clone(), dt_softplus=True,
            state_dtype=torch.float32)
        return out, final

    def reference(_out):
        y, final = naive_ssd(x, dt, A, Bm, Cm, D, dt_bias, h0, cu)
        return {"out": y, "final_state": final}

    def outputs(out):
        return {"out": out[0], "final_state": out[1]}

    helpers = _import("vllm.model_executor.layers.mamba.ops.triton_helpers")

    def reset():
        getattr(helpers, "_autotuned_configs", {}).clear()
    return Built(run, reference, outputs, {"T": T, "nheads": nheads, "headdim": headdim, "ngroups": ngroups,
                                           "dstate": dstate, "chunk_size": chunk, "seqlens": list(seqlens)}, reset)


GDN_REFS = {
    "chunk_scaled_dot_kkt_fwd_kernel": ref_kkt,
    "merge_16x16_to_64x64_inverse_kernel": ref_solve_tril,
    "chunk_fwd_kernel_o": ref_chunk_o,
    "l2norm_fwd_kernel": ref_l2norm,
    "l2norm_fwd_kernel1": ref_l2norm,
    "chunk_local_cumsum_scalar_kernel": ref_cumsum_scalar,
}

SCENARIOS: dict[str, Scenario] = {s.name: s for s in (
    Scenario("kkt_issue65385", "vLLM vendored chunk_scaled_dot_kkt_fwd on the vllm#65385 inputs (positive control)",
             lambda: build_kkt_issue("vllm"), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt}),
    Scenario("kkt_k256", "vLLM vendored chunk_scaled_dot_kkt_fwd with head dim K=256 (BK=128 also loops), otherwise vllm#65385 inputs",
             lambda: build_kkt_issue("vllm", K=256), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt}),
    Scenario("kkt_k256_betaafter", "EXPERIMENT: beta-after-loop kkt with K=256",
             lambda: build_kkt_issue("betaafter", K=256), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt}),
    Scenario("kkt_issue65385_v0191", "vLLM v0.19.1 chunk_scaled_dot_kkt source (the version vllm#65385 used) on its inputs",
             lambda: build_kkt_issue("vllm0191"), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt}),
    Scenario("kkt_issue65385_betaafter", "EXPERIMENT: vLLM 0.33 kkt with beta applied after the K loop, vllm#65385 inputs",
             lambda: build_kkt_issue("betaafter"), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt}),
    Scenario("kkt_issue65385_fla", "upstream FLA chunk_scaled_dot_kkt_fwd on the vllm#65385 inputs (g in log2 space)",
             lambda: build_kkt_issue("fla"), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt_exp2}),
    Scenario("gdn_vllm_fixed", "vLLM vendored chunk_gated_delta_rule, B=1 T=4096 H=Hk=32 K=V=128, l2norm, h0",
             lambda: build_gdn("vllm", None, 32, 32), dict(GDN_REFS)),
    Scenario("gdn_vllm_varlen", "vLLM vendored chunk_gated_delta_rule, 3 seqs (1000/1600/1496), Hk=16 H=32 (Qwen3-Next TP1)",
             lambda: build_gdn("vllm", SEQLENS, 16, 32), dict(GDN_REFS)),
    Scenario("gdn_fla_varlen", "upstream FLA chunk_gated_delta_rule, same inputs as gdn_vllm_varlen",
             lambda: build_gdn("fla", SEQLENS, 16, 32), {"chunk_scaled_dot_kkt_fwd_kernel": ref_kkt_exp2}),
    Scenario("kda_vllm_varlen", "vLLM vendored chunk_kda (Kimi Linear), 3 seqs, H=32 K=V=128, l2norm, h0",
             lambda: build_kda("vllm", SEQLENS), {"l2norm_fwd_kernel": ref_l2norm, "l2norm_fwd_kernel1": ref_l2norm}),
    Scenario("kda_kimik3_varlen", "vLLM Kimi-K3 NVIDIA chunk_kda copy, same inputs as kda_vllm_varlen",
             lambda: build_kda("kimik3", SEQLENS), {"l2norm_fwd_kernel": ref_l2norm, "l2norm_fwd_kernel1": ref_l2norm}),
    Scenario("mamba2_ssd_varlen", "vLLM Mamba2 SSD mamba_chunk_scan_combined_varlen, 3 seqs, 64 heads x 64, 8 groups, dstate 128, chunk 256",
             lambda: build_mamba2(SEQLENS)),
)}


# ---------------------------------------------------------------------------
# Driver


def sweep_environment() -> dict[str, object]:
    from .runner import environment
    env = environment()
    versions = dict(env.get("versions", {}))  # type: ignore[arg-type]
    for dist in ("triton", "fla-core", "flash-linear-attention", "nvidia-cuda-nvcc-cu13", "nvidia-cuda-runtime"):
        try:
            versions[dist] = importlib.metadata.version(dist)
        except Exception:
            versions.setdefault(dist, None)
    try:
        import triton
        versions["triton_imported"] = f"{triton.__version__} @ {triton.__file__}"
    except Exception as exc:
        versions["triton_imported"] = f"unavailable: {exc}"
    try:  # the assembler Triton hands sm_90 (TRITON_PTXAS_PATH / bundled ptxas or ptxas-blackwell)
        import subprocess
        from triton.backends.nvidia.compiler import get_ptxas
        tool = get_ptxas(90) if get_ptxas.__code__.co_argcount else get_ptxas()
        ver = subprocess.run([tool.path, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
        versions["ptxas_sm90"] = f"{tool.path} ({ver})"
    except Exception as exc:
        versions["ptxas_sm90"] = f"unknown: {type(exc).__name__}: {exc}"
    env["versions"] = versions
    import os
    env["env"] = {**dict(env.get("env", {})),  # type: ignore[arg-type]
                  **{k: os.environ.get(k) for k in ("TRITON_CACHE_DIR", "PYTHONPATH", "VLLM_TRITON_FORCE_FIRST_CONFIG",
                                                    "TRITON_PTXAS_PATH", "TRITON_PTXAS_BLACKWELL_PATH",
                                                    "FLA_USE_CUDA_GRAPH", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE")}}
    if torch.cuda.is_available():
        env["capability"] = list(torch.cuda.get_device_capability(0))
    env["type"] = "sweep_metadata"
    return env


def run_sweep(names, out, *, repeats: int = 3, kernels: str | None = None, e2e: bool = True, sweep: bool = True,
              e2e_repeats: int = 3, poison: bool = True, append: bool = False) -> list[dict[str, object]]:
    import json
    import re
    import time
    from pathlib import Path

    from .autotune_sweep import capture, launch_record, rel_err, sweep_launch, fast_compare

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with out.open("a" if append else "w") as file:
        def emit(rows):
            for row in rows:
                records.append(row)
                file.write(json.dumps(row, default=str, allow_nan=True) + "\n")
            file.flush()

        emit([sweep_environment()])
        for name in names:
            scenario = SCENARIOS[name]
            start = time.perf_counter()
            try:
                built = scenario.build()
            except SkipScenario as exc:
                emit([{"type": "sweep_skip", "scenario": name, "reason": str(exc)}])
                continue
            except Exception as exc:
                emit([{"type": "sweep_error", "scenario": name, "stage": "build", "error": f"{type(exc).__name__}: {exc}"}])
                continue
            emit([{"type": "sweep_scenario", "scenario": name, "description": scenario.description, "meta": built.meta}])
            try:
                if e2e and built.e2e_reference is not None:
                    emit(_e2e(name, built, "autotune", e2e_repeats, scenario.e2e_tol))
                if sweep:
                    built.reset()
                    cap = capture(built.run, name, mode="first", dedupe=False)
                    if not cap.launches:
                        emit([{"type": "sweep_error", "scenario": name, "stage": "capture",
                               "error": "no autotuned launch captured (cache bypass or no autotuned kernels)"}])
                    if built.e2e_reference is not None:
                        emit(_e2e(name, built, "first", 1, scenario.e2e_tol, cap))
                    for launch in cap.launches:
                        if kernels and not re.search(kernels, launch.kernel):
                            continue
                        emit([launch_record(launch)])
                        emit(sweep_launch(launch, repeats=repeats, reference=scenario.references.get(launch.kernel),
                                          tol=scenario.tolerances.get(launch.kernel, 1e-2), poison_check=poison))
                        launch.pre.clear()
                        launch.post.clear()
                        torch.cuda.empty_cache()
                    del cap
            except Exception as exc:
                import traceback
                emit([{"type": "sweep_error", "scenario": name, "stage": "run", "error": f"{type(exc).__name__}: {exc}",
                       "traceback": traceback.format_exc()[-4000:]}])
            emit([{"type": "sweep_scenario_done", "scenario": name, "seconds": round(time.perf_counter() - start, 1)}])
            torch.cuda.empty_cache()
    return records


def _e2e(name: str, built: Built, mode: str, repeats: int, tol: float, cap=None) -> list[dict[str, object]]:
    from .autotune_sweep import capture, fast_compare, rel_err

    if cap is None:
        built.reset()
        cap = capture(built.run, name, mode=mode, dedupe=True)
    first = {k: v.detach().clone() for k, v in built.e2e_outputs(cap.output).items()}
    diffs: dict[str, int] = {}
    for _ in range(max(0, repeats - 1)):
        again = built.e2e_outputs(built.run())
        torch.cuda.synchronize()
        for k, v in again.items():
            d = fast_compare(first[k], v)
            diffs[k] = max(diffs.get(k, 0), d.differing_elements)
    ref = built.e2e_reference(cap.output)
    errs = {k: rel_err(first[k], ref[k].to(first[k].device)) for k in first if k in ref}
    return [{"type": "sweep_e2e", "scenario": name, "mode": mode, "picked": cap.picked, "rel_err": errs,
             "tolerance": tol, "repeats": repeats, "repeat_differing_elements": diffs,
             "deterministic": not any(diffs.values()), "nonfinite": {k: int((~torch.isfinite(v.float())).sum().item()) for k, v in first.items()},
             "flags": (["wrong_vs_ref"] if any(e > tol for e in errs.values()) else []) + (["nondeterministic"] if any(diffs.values()) else [])}]
