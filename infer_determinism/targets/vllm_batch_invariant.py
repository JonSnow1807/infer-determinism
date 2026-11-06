"""Direct probes of installed vLLM batch-invariant operators."""
from __future__ import annotations

import importlib
import os

import torch

from ..core import BatchSpec, Case, seeded
from ..runner import SkipCase

PATHS = ("vllm.model_executor.determinism.batch_invariant", "vllm.model_executor.layers.batch_invariant")


def _module():
    if not torch.cuda.is_available():
        raise SkipCase("CUDA is required for vLLM batch-invariant ops")
    if os.environ.get("VLLM_BATCH_INVARIANT") != "1":
        raise SkipCase("set VLLM_BATCH_INVARIANT=1 before importing vLLM")
    for path in PATHS:
        try:
            return importlib.import_module(path)
        except ImportError:
            continue
    raise SkipCase(f"no batch-invariant module at {PATHS}")


def _run(spec: BatchSpec, op: str, dtype: torch.dtype) -> torch.Tensor:
    module = _module()
    x = torch.stack([seeded((128,), spec, i, dtype, salt=51) for i in range(spec.batch_size)])
    weight = seeded((128,), spec, spec.probe_index, dtype, salt=53)
    if op == "matmul":
        fn = getattr(module, "matmul_batch_invariant", None) or getattr(module, "linear_batch_invariant", None)
        if fn is None:
            raise SkipCase("installed vLLM exposes no batch-invariant matmul/linear entry point")
        matrix = seeded((128, 128), spec, spec.probe_index, dtype, salt=59)
        y = fn(x, matrix if fn.__name__ == "matmul_batch_invariant" else matrix.T)
    elif op == "log_softmax":
        fn = getattr(module, "log_softmax", None)
        if fn is None:
            raise SkipCase("installed vLLM has no log_softmax")
        y = fn(x, dim=-1)
    elif op == "mean":
        fn = getattr(module, "mean_dim", None)
        if fn is None:
            raise SkipCase("installed vLLM has no mean_dim")
        y = fn(x, dim=-1)
    else:
        fn = getattr(module, "rms_norm_batch_invariant", None) or getattr(module, "rms_norm", None)
        if fn is None:
            raise SkipCase("installed vLLM has no RMS norm entry point")
        y = fn(x, weight, eps=1e-6)
    return y[spec.probe_index].clone()


REPRO = '''
import importlib, os
os.environ["VLLM_BATCH_INVARIANT"]="1"
def run(spec):
    module = next((importlib.import_module(p) for p in ("vllm.model_executor.determinism.batch_invariant", "vllm.model_executor.layers.batch_invariant") if importlib.util.find_spec(p)), None)
    op,dtype_name=CASE_NAME.split("/")
    dtype=getattr(torch,dtype_name)
    def seeded(shape,row,salt):
        g=torch.Generator(device="cpu").manual_seed(spec["seed"]+salt+(0 if row==spec["probe_index"] else 10000+row))
        return torch.randn(shape,generator=g).to(device="cuda",dtype=dtype)
    x=torch.stack([seeded((128,),i,51) for i in range(spec["batch_size"])])
    w=seeded((128,),spec["probe_index"],53)
    if op=="matmul":
        fn=getattr(module,"matmul_batch_invariant",None) or getattr(module,"linear_batch_invariant")
        matrix=seeded((128,128),spec["probe_index"],59)
        y=fn(x,matrix if fn.__name__=="matmul_batch_invariant" else matrix.T)
    elif op=="log_softmax": y=module.log_softmax(x,dim=-1)
    elif op=="mean": y=module.mean_dim(x,dim=-1)
    else: y=(getattr(module,"rms_norm_batch_invariant",None) or module.rms_norm)(x,w,eps=1e-6)
    return y[spec["probe_index"]].clone()
a,b=run(REFERENCE),run(CANDIDATE)
bits=torch.int16 if a.element_size()==2 else torch.int32
print("bitwise equal:",torch.equal(a.contiguous().view(bits),b.contiguous().view(bits)))
print("max abs diff:",(a.float()-b.float()).abs().max().item())
'''


def cases() -> list[Case]:
    module = _module()
    ops = ("matmul", "log_softmax", "mean", "rms_norm")
    return [Case("vllm_batch_invariant", f"{op}/{str(dtype).split('.')[-1]}", str(dtype), {"hidden": 128}, lambda spec, o=op, d=dtype: _run(spec, o, d), repro_source=REPRO) for op in ops for dtype in (torch.bfloat16, torch.float16, torch.float32)]
