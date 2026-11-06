"""vLLM fused expert kernels with tied router logits."""
from __future__ import annotations

import importlib

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase


def _api():
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for vLLM MoE")
    for path in ("vllm.model_executor.layers.fused_moe.fused_moe", "vllm.model_executor.layers.fused_moe"):
        try:
            module = importlib.import_module(path)
            if callable(getattr(module, "fused_experts", None)):
                return module
        except ImportError:
            continue
    raise SkipCase("vLLM fused_experts is unavailable")


def _run(spec: BatchSpec) -> torch.Tensor:
    module = _api()
    x = torch.stack([seeded((64,), spec, i, torch.bfloat16, salt=101) for i in range(spec.batch_size)])
    logits = torch.stack([seeded((4,), spec, i, torch.float32, salt=103) for i in range(spec.batch_size)])
    # All four probe experts tie. Position and batch size must not change tie handling.
    logits[spec.probe_index] = 0
    ids = torch.topk(logits, 2, dim=-1).indices.to(torch.int32)
    weights = torch.full((spec.batch_size, 2), 0.5, device="cuda", dtype=torch.float32)
    w1 = seeded((4, 128, 64), spec, spec.probe_index, torch.bfloat16, salt=107)
    w2 = seeded((4, 64, 64), spec, spec.probe_index, torch.bfloat16, salt=109)
    return module.fused_experts(x, w1, w2, weights, ids)[spec.probe_index].clone()


def cases() -> list[Case]:
    _api()
    return [Case("moe", "fused_experts/topk_ties", "bfloat16", {"experts": 4, "topk": 2, "hidden": 64, "intermediate": 64}, _run)]
