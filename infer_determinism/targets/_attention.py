"""CPU-safe attention case descriptions and capability checks.

Also embedded in standalone reproducers by runner._script.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass

import torch

from ..runner import SkipCase


@dataclass(frozen=True)
class AttentionConfig:
    num_q_heads: int = 4
    num_kv_heads: int = 4
    head_dim: int = 64
    window: int = 0  # Total visible tokens, including the current token.
    fp8: bool = False

    @property
    def window_left(self) -> int:
        return self.window - 1 if self.window else -1

    @property
    def label(self) -> str:
        return (f"q{self.num_q_heads}_kv{self.num_kv_heads}_d{self.head_dim}"
                + (f"/window{self.window}" if self.window else "")
                + ("/fp8_e4m3" if self.fp8 else ""))

    def shapes(self) -> dict[str, object]:
        return {"num_q_heads": self.num_q_heads, "num_kv_heads": self.num_kv_heads,
                "D": self.head_dim, "window": self.window,
                "kv_dtype": "float8_e4m3fn" if self.fp8 else "bfloat16",
                **({"k_scale": K_SCALE, "v_scale": V_SCALE} if self.fp8 else {})}


# Fixed per-tensor dequantization scales; never computed over a batch.
Q_SCALE, K_SCALE, V_SCALE = 0.5, 0.25, 0.125
GQA_HEADS = ((32, 8), (64, 8), (8, 1))


def attention_configs():
    """(config, full_only) matrix shared by FlashInfer and vLLM FA."""
    for dim in (128, 256):
        for hq, hkv in GQA_HEADS:
            yield AttentionConfig(hq, hkv, dim), (dim != 128 or hq != 32)
    for window in (1024, 4096):
        for dim in (128, 256):
            yield AttentionConfig(32, 8, dim, window), (window != 1024 or dim != 128)
    for hq, hkv in GQA_HEADS:
        yield AttentionConfig(hq, hkv, 128, fp8=True), hq != 32
    # Interaction probes; no claim that every build supports these kernels.
    yield AttentionConfig(32, 8, 256, fp8=True), True
    yield AttentionConfig(32, 8, 128, 4096, fp8=True), True


def require_keywords(fn, names, context: str) -> None:
    """Do not silently drop a requested feature on an older API."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return  # C bindings may not expose a signature; validate at call time.
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return
    missing = sorted(set(names) - set(params))
    if missing:
        raise SkipCase(f"{context}: installed API lacks keyword(s) {', '.join(missing)}")


def attention_call(fn, *args, context: str, **kwargs):
    """Classify explicit capability rejections, never arbitrary CUDA failures."""
    try:
        return fn(*args, **kwargs)
    except (TypeError, RuntimeError, NotImplementedError, ValueError) as exc:
        msg = str(exc).lower()
        # Recognized backend/build rejections. Shape/stride/scale assertions,
        # JIT failures and arbitrary errors mentioning FP8 remain errors.
        unsupported = (
            isinstance(exc, TypeError) and "unexpected keyword argument" in msg
            or "no available kernel" in msg
            or msg.startswith("this flash attention build does not support ")
            or msg.startswith("flashattention forward only supports head dimension at most ")
            or re.match(r"flashattention (?:on ampere/ada cards )?only supports? fp16", msg)
            or re.fullmatch(r"unsupported head_dim[ :]+\d+\.?", msg)
        )
        if unsupported:
            raise SkipCase(f"{context}: {type(exc).__name__}: {exc}") from exc
        raise


def fp8_dtype():
    dtype = getattr(torch, "float8_e4m3fn", None)
    if dtype is None:
        raise SkipCase("FP8 KV requires torch.float8_e4m3fn")
    return dtype


def quantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    # Scale convention: real value ~= stored FP8 value * scale.
    limit = torch.finfo(fp8_dtype()).max
    return (tensor.float() / scale).clamp(-limit, limit).to(fp8_dtype())


def replay_graph(graph, output: torch.Tensor) -> None:
    """Poison all active outputs so a missing graph write cannot look equal."""
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("CUDA graph replay produced nonfinite or unwritten active output")
