"""CPU fixtures used by harness self-tests; not part of default GPU suite."""
from __future__ import annotations

import torch

from ..core import BatchSpec, Case


def _invariant(spec: BatchSpec) -> torch.Tensor:
    return torch.tensor([1.0, -0.0, 3.5], dtype=torch.float32)


def _noninvariant(spec: BatchSpec) -> torch.Tensor:
    x = torch.tensor([1e20, -1e20, 1.0], dtype=torch.float32)
    if spec.batch_size == 1:
        value = (x[0] + x[1]) + x[2]
    else:
        value = x[0] + (x[1] + x[2])
    return value.reshape(1)


REPRO = '''
def run(spec):
    x = torch.tensor([1e20, -1e20, 1.0], dtype=torch.float32)
    if spec["batch_size"] == 1:
        return ((x[0] + x[1]) + x[2]).reshape(1)
    return (x[0] + (x[1] + x[2])).reshape(1)
a, b = run(REFERENCE), run(CANDIDATE)
print("bitwise equal:", torch.equal(a.view(torch.int32), b.view(torch.int32)))
print("reference:", a.item(), "candidate:", b.item())
'''


def cases() -> list[Case]:
    return [
        Case("fake", "invariant", "float32", {"D": 3}, _invariant),
        Case("fake", "noninvariant", "float32", {"D": 3}, _noninvariant, repro_source=REPRO),
    ]
