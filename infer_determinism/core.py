"""Small, import-safe target API and bitwise comparison machinery."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class BatchSpec:
    batch_size: int = 1
    probe_index: int = 0
    seed: int = 1729
    other_kv_length: int = 32
    probe_kv_length: int = 32
    chunk_size: int = 0
    block_size: int = 16
    num_splits: int = 0
    fixed_split_size: int = 0
    disable_split_kv: bool = False
    tf32: bool = False
    graph_capture_kv_length: int = 0

    def __post_init__(self) -> None:
        if self.batch_size < 1 or not 0 <= self.probe_index < self.batch_size:
            raise ValueError("probe_index must be inside a nonempty batch")

    def json(self) -> dict[str, int | bool]:
        return asdict(self)


@dataclass
class Case:
    target: str
    name: str
    dtype: str
    shapes: Mapping[str, object]
    run: Callable[[BatchSpec], torch.Tensor]
    perturbations: tuple[str, ...] = ("repeat", "batch", "position")
    repro_source: str = ""
    baseline: BatchSpec = field(default_factory=BatchSpec)
    allowed_block_sizes: tuple[int, ...] = (8, 16, 32, 64)


@dataclass(frozen=True)
class Difference:
    equal: bool
    differing_elements: int
    max_abs_diff: float | str
    max_ulp_distance: int


def _bits(t: torch.Tensor) -> torch.Tensor:
    """Same-width integer view; compares NaN payloads and signed zero too."""
    if t.dtype in (torch.float16, torch.bfloat16, torch.int16):
        return t.contiguous().view(torch.int16)
    if t.dtype in (torch.float32, torch.int32):
        return t.contiguous().view(torch.int32)
    if t.dtype in (torch.float64, torch.int64):
        return t.contiguous().view(torch.int64)
    if t.dtype in (torch.int8, torch.uint8, torch.bool):
        return t.contiguous().view(torch.uint8)
    raise TypeError(f"unsupported comparison dtype: {t.dtype}")


def compare(reference: torch.Tensor, candidate: torch.Tensor) -> Difference:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(f"probe output shape/dtype changed: {reference.shape}/{reference.dtype} vs {candidate.shape}/{candidate.dtype}")
    a, b = reference.detach().cpu(), candidate.detach().cpu()
    ab, bb = _bits(a), _bits(b)
    diff = ab != bb
    count = int(diff.sum().item())
    if not count:
        return Difference(True, 0, 0.0, 0)
    absdiff = (a.double() - b.double()).abs()
    # NaN payload differences have no meaningful finite absolute error.
    finite = absdiff[torch.isfinite(absdiff)]
    max_abs: float | str = ("inf" if torch.isinf(absdiff).any() else
                            float(finite.max().item()) if finite.numel() else "nan")
    width = ab.element_size() * 8
    mask = (1 << width) - 1
    sign = 1 << (width - 1)
    av = [int(x) & mask for x in ab.reshape(-1).tolist()]
    bv = [int(x) & mask for x in bb.reshape(-1).tolist()]
    def ordered(v: int) -> int:
        return ((~v) & mask) if v & sign else (v | sign)
    ulp = max(abs(ordered(x) - ordered(y)) for x, y in zip(av, bv))
    return Difference(False, count, max_abs, ulp)


def seeded(shape: tuple[int, ...], spec: BatchSpec, row: int, dtype: torch.dtype, device: str = "cuda", salt: int = 0) -> torch.Tensor:
    """Probe data is stable across batch positions; other rows vary by slot."""
    g = torch.Generator(device="cpu").manual_seed(spec.seed + salt + (0 if row == spec.probe_index else 10000 + row))
    return torch.randn(shape, generator=g, dtype=torch.float32).to(device=device, dtype=dtype)


def probe_specs(case: Case, quick: bool = False) -> list[tuple[str, BatchSpec]]:
    base = case.baseline
    sizes = (1, 2, 3, 8, 16) if quick else (1, 2, 3, 7, 8, 15, 16, 31, 64, 128, 257)
    specs: list[tuple[str, BatchSpec]] = []
    if "repeat" in case.perturbations:
        specs += [("repeat", base) for _ in range(2 if quick else 5)]
    if "batch" in case.perturbations:
        specs += [("batch", replace(base, batch_size=n, probe_index=0)) for n in sizes if n != base.batch_size]
    if "tile_m" in case.perturbations:
        ms = (31, 32, 33, 64, 65) if quick else (31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257)
        specs += [("tile_m", replace(base, batch_size=n, probe_index=0)) for n in ms]
    if "big_m" in case.perturbations:
        specs += [("big_m", replace(base, batch_size=n, probe_index=0)) for n in ((512, 1024) if quick else (500, 512, 1000, 1024, 2048, 4096))]
    if "position" in case.perturbations:
        specs += [("position", replace(base, batch_size=n, probe_index=p)) for n in sizes if n > 1 for p in sorted({n // 2, n - 1})]
    if "other_kv_length" in case.perturbations:
        specs += [("other_kv_length", replace(base, batch_size=3, probe_index=1, other_kv_length=k)) for k in ((1, 65) if quick else (1, 7, 31, 33, 65, 129, 257, 1025, 3000))]
    if "chunk_size" in case.perturbations:
        specs += [("chunk_size", replace(base, chunk_size=k)) for k in ((1,) if quick else (1, 2, 7))]
    if "block_size" in case.perturbations:
        specs += [("block_size", replace(base, block_size=k)) for k in ((8, 32) if quick else (8, 16, 32, 64)) if k != base.block_size and k in case.allowed_block_sizes]
    if "num_splits" in case.perturbations:
        specs += [("num_splits", replace(base, num_splits=k)) for k in ((1, 2) if quick else (1, 2, 4, 8)) if k != base.num_splits]
    if "fixed_split_size" in case.perturbations:
        specs += [("fixed_split_size", replace(base, fixed_split_size=k)) for k in ((32, 128) if quick else (16, 32, 128, 256, 512)) if k != base.fixed_split_size]
    if "tf32" in case.perturbations:
        specs.append(("tf32", replace(base, tf32=not base.tf32)))
    if "graph_replay" in case.perturbations:
        specs.append(("graph_replay", replace(base, batch_size=3, probe_index=0,
                                              graph_capture_kv_length=max(1, base.probe_kv_length // 2))))
    return specs
