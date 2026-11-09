"""CPU contract tests: layouts, feature forwarding, graph metadata, and repros."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import replace
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from infer_determinism.core import BatchSpec, Case, compare, probe_specs, seeded
from infer_determinism.runner import SkipCase, _script, run_case
from infer_determinism.targets import flashinfer as fi, torch_sdpa as sdpa, vllm_flash_attn as fa
from infer_determinism.targets._attention import (
    AttentionConfig, GQA_HEADS, K_SCALE, V_SCALE, attention_call, quantize_fp8,
    replay_graph, require_keywords,
)


@pytest.fixture
def cpu_cuda(monkeypatch):
    """Exercise adapter Python paths with CPU tensors and a recording graph."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for name in ("tensor", "arange", "zeros", "empty", "full"):
        original = getattr(torch, name)
        def allocate(*args, _fn=original, **kwargs):
            if kwargs.get("device") == "cuda":
                kwargs["device"] = "cpu"
            if args == (1024 * 1024 * 1024,):  # Workspace is unused by fake kernels.
                args = (1,)
            return _fn(*args, **kwargs)
        monkeypatch.setattr(torch, name, allocate)
    for module in (fi, fa, sdpa):
        monkeypatch.setattr(module, "seeded", lambda shape, spec, row, dtype, salt=0:
                            seeded(shape, spec, row, dtype, device="cpu", salt=salt))
    state = SimpleNamespace(capturing=False, replaying=False, callback=None, events=[])
    class Graph:
        def replay(self):
            state.events.append("replay")
            state.replaying = True
            state.callback()
            state.replaying = False
    @contextmanager
    def capture(graph):
        state.capturing = True
        state.events.append("capture")
        yield
        state.capturing = False
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Graph)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    return state


def test_matrix_and_quick_budget(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(fi, "_library", lambda: SimpleNamespace())
    monkeypatch.setattr(fa, "_api", lambda: SimpleNamespace())
    for module in (sdpa, fi, fa):
        cases = module.cases()
        assert len({c.name for c in cases}) == len(cases)
        new = [c for c in cases if "num_q_heads" in c.shapes]
        assert {(c.shapes["num_q_heads"], c.shapes["num_kv_heads"]) for c in new} == set(GQA_HEADS)
        for c in new:
            if c.shapes["D"] == 256 or c.shapes["window"] == 4096:
                assert c.shapes["full_only"]
            if c.shapes["window"]:
                assert c.baseline.probe_kv_length > c.shapes["window"]
                assert {"batch", "position", "graph_replay"} <= set(c.perturbations)
                specs = [s for p, s in probe_specs(c, True) if p == "graph_replay"]
                assert {s.probe_index for s in specs} == {0, 2}
                assert all(s.graph_capture_kv_length < c.shapes["window"] for s in specs)
        assert len([c for c in new if not c.shapes["full_only"]]) == {sdpa: 4, fi: 6, fa: 5}[module]
        if module is not sdpa:
            for kind in (("batch_decode", "batch_prefill") if module is fi else ("FA2", "FA3")):
                covered = {(c.shapes["num_q_heads"], c.shapes["num_kv_heads"], c.shapes["D"])
                           for c in new if kind in c.name and not c.shapes["window"] and c.shapes["kv_dtype"] == "bfloat16"}
                assert covered == {(hq, hk, d) for hq, hk in GQA_HEADS for d in (128, 256)}


def test_full_only_does_not_call_run():
    def fail(spec):
        pytest.fail("full-only case was executed")
    case = Case("fixture", "large", "bf16", {"full_only": True}, fail)
    assert run_case(case, quick=True)[0]["reason"].startswith("full-only")
    case.perturbations = ("big_m",)
    assert probe_specs(case, quick=True) == []
    assert probe_specs(case, quick=False)


def test_capability_checks_preserve_real_errors():
    def old_api(q):
        return q
    with pytest.raises(SkipCase, match="window_left"):
        require_keywords(old_api, ("window_left",), "fixture")
    def rejects():
        raise RuntimeError("This flash attention build does not support FP8.")
    with pytest.raises(SkipCase, match="does not support FP8"):
        attention_call(rejects, context="FA3")
    def broken():
        raise RuntimeError("CUDA illegal memory access")
    with pytest.raises(RuntimeError, match="illegal memory"):
        attention_call(broken, context="FA3")
    def malformed_scale():
        raise RuntimeError("FP8 requires contiguous scale tensors")
    with pytest.raises(RuntimeError, match="contiguous scale"):
        attention_call(malformed_scale, context="FA3")
    with pytest.raises(SkipCase, match="unexpected keyword"):
        attention_call(old_api, 1, window_left=4, context="old API")


@pytest.mark.parametrize("fp8", [False, True])
def test_gqa_paging_keeps_probe_bits(cpu_cuda, fp8):
    config = AttentionConfig(32, 8, 128, fp8=fp8)
    base = BatchSpec(probe_kv_length=35, other_kv_length=19)
    batched = replace(base, batch_size=3, probe_index=2)
    aq, ak, av = fi._inputs(base, 8, config)
    bq, bk, bv = fi._inputs(batched, 8, config)
    assert aq[0].shape == (8, 32, 128)
    assert ak[0].shape == (35, 8, 128)
    assert torch.equal(aq[0], bq[2])
    for left, right in ((ak[0], bk[2]), (av[0], bv[2])):
        assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))
    cache, ptr, idx, tails = fi._paged(bk, bv, 16)
    assert cache.shape == (7, 2, 16, 8, 128)
    assert ptr.tolist() == [0, 2, 4, 7]
    assert tails.tolist() == [3, 3, 3]
    restored = cache[4:7, 0].reshape(-1, 8, 128)[:35]
    assert torch.equal(restored.view(torch.uint8), ak[0].view(torch.uint8))
    sp, si, st = fi._short_metadata(ptr, idx, [19, 19, 35], 16, 17)
    assert sp.tolist() == [0, 2, 4, 6]
    assert si.tolist() == [0, 1, 2, 3, 4, 5]
    assert st.tolist() == [1, 1, 1]
    # Shrinking an earlier request must not move the later request's pages.
    sp, si, st = fi._short_metadata(torch.tensor([0, 3, 5]), torch.arange(5), [35, 19], 16, 17)
    assert sp.tolist() == [0, 2, 4]
    assert si.tolist() == [0, 1, 3, 4]
    if fp8:
        original = seeded((35, 8, 128), base, 0, torch.bfloat16, device="cpu", salt=73)
        assert torch.equal(ak[0].view(torch.uint8), quantize_fp8(original, K_SCALE).view(torch.uint8))


@pytest.mark.parametrize("kind", ["batch_decode", "batch_prefill"])
def test_flashinfer_feature_forwarding_and_graph_replan(cpu_cuda, monkeypatch, kind):
    state = cpu_cuda
    calls = []
    class Wrapper:
        def __init__(self, workspace, layout, **kwargs):
            self.buffers = kwargs
            assert torch.count_nonzero(workspace) == 0
            assert layout == "NHD"
            assert kwargs["backend"] == "fa2"
            calls.append(self)
        def plan(self, *args, **kwargs):
            assert not state.capturing
            decode = kind == "batch_decode"
            p, idx, tails, hq, hk, dim, page = args if decode else args[1:]
            assert (hq, hk, dim, page) == (32, 8, 128, 16)
            assert kwargs["window_left"] == 1023
            assert kwargs["kv_data_type"] == torch.float8_e4m3fn
            assert kwargs["q_data_type"] == torch.bfloat16
            assert kwargs["disable_split_kv"]
            if not decode:
                assert kwargs["causal"]
            self.lengths = ((p[1:] - p[:-1] - 1) * page + tails).tolist()
            self.metadata = p.clone(), idx.clone(), tails.clone()
            state.events.append(tuple(self.lengths))
        def run(self, q, cache, *, k_scale, v_scale, out=None):
            assert (k_scale, v_scale) == (K_SCALE, V_SCALE)
            assert cache.dtype == torch.float8_e4m3fn
            assert q.dtype == torch.bfloat16
            if state.capturing:
                state.callback = lambda: self.run(q, cache, k_scale=k_scale, v_scale=v_scale, out=out)
            if state.replaying:
                assert self.lengths == [32, 32, 1152]
                assert self.metadata[1].tolist() == list(range(76))
            result = q.clone()  # Kernel arithmetic is intentionally not simulated.
            if out is not None:
                out.copy_(result)
                return out
            return result
    monkeypatch.setattr(fi, "_library", lambda: SimpleNamespace(
        BatchDecodeWithPagedKVCacheWrapper=Wrapper, BatchPrefillWithPagedKVCacheWrapper=Wrapper))
    config = AttentionConfig(32, 8, 128, 1024, True)
    base = BatchSpec(probe_kv_length=1152, disable_split_kv=True)
    a = fi._run(base, kind, True, True, config)
    b = fi._run(replace(base, batch_size=3, probe_index=2, graph_capture_kv_length=576), kind, True, True, config)
    assert compare(a, b).equal
    assert state.events[-5:] == [(32, 32, 1152), (32, 32, 576), "capture", (32, 32, 1152), "replay"]
    kwargs = calls[-1].buffers
    suffix = "buffer" if kind == "batch_decode" else "buf"
    assert kwargs["use_cuda_graph"]
    assert kwargs[f"paged_kv_indices_{suffix}"].numel() == 76
    if kind == "batch_prefill":
        assert kwargs["qo_indptr_buf"].tolist() == [0, 8, 16, 24]


@pytest.mark.parametrize("version,window,fp8", [(3, 1024, False), (3, 1024, True), (2, 1024, False), (2, 0, False)])
def test_fa_descales_scheduler_and_graph_metadata(cpu_cuda, monkeypatch, version, window, fp8):
    state = cpu_cuda
    snapshots = []
    def kernel(**kwargs):
        q, out = kwargs["q"], kwargs["out"]
        if window:
            assert kwargs["window_size"] == (1023, 0)
            assert kwargs["causal"]
        else:
            assert "window_size" not in kwargs
        assert kwargs["num_splits"] == 1
        assert q.shape[1:] == (32, 128)
        assert kwargs["k"].shape[2:] == (8, 128)
        assert out.dtype == torch.bfloat16
        if fp8:
            assert q.dtype == kwargs["k"].dtype == kwargs["v"].dtype == torch.float8_e4m3fn
            assert kwargs["k_descale"].shape == (q.shape[0], 8)
            assert torch.all(kwargs["k_descale"] == K_SCALE)
            assert torch.all(kwargs["v_descale"] == V_SCALE)
        snapshots.append((kwargs["cu_seqlens_q"].tolist(), kwargs["seqused_k"].tolist()))
        if state.capturing:
            state.callback = lambda: kernel(**kwargs)
        out.copy_(q.to(torch.bfloat16))
        return out
    def schedule(batch, qlen, klen, hq, hk, dim, used, **kwargs):
        assert (hq, hk, dim) == (32, 8, 128)
        assert kwargs["window_size"] == (1023, 0)
        assert kwargs["causal"]
        assert kwargs["qkv_dtype"] == (torch.float8_e4m3fn if fp8 else torch.bfloat16)
        state.events.append("schedule")
        return torch.zeros(1)
    monkeypatch.setattr(fa, "_api", lambda: SimpleNamespace(
        flash_attn_varlen_func=kernel, get_scheduler_metadata=schedule))
    config = AttentionConfig(32, 8, 128, window, fp8)
    base = BatchSpec(probe_kv_length=1152, num_splits=1)
    a = fa._run(base, version, config)
    b = fa._run(replace(base, batch_size=3, probe_index=2, graph_capture_kv_length=576), version, config)
    assert compare(a, b).equal
    capture_qptr = [0, 1, 1, 1] if version == 3 or window else [0, 1, 2, 3]
    assert snapshots[-2:] == [(capture_qptr, [32, 32, 576]), ([0, 1, 2, 3], [32, 32, 1152])]
    assert state.events.count("schedule") == (1 if version == 3 else 0)  # No stale capture-time scheduler.


def test_sdpa_uses_native_gqa(cpu_cuda, monkeypatch):
    import torch.nn.attention
    monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda backend: nullcontext())
    def kernel(q, k, v, *, enable_gqa, is_causal):
        assert enable_gqa and is_causal
        assert q.shape == (3, 64, 8, 128)
        assert k.shape == v.shape == (3, 8, 32, 128)
        return q
    monkeypatch.setattr(sdpa.F, "scaled_dot_product_attention", kernel)
    result = sdpa._run(BatchSpec(batch_size=3, probe_index=2), torch.bfloat16, "MATH", AttentionConfig(64, 8, 128))
    assert result.shape == (64, 8, 128)


@pytest.mark.parametrize("module", [sdpa, fi, fa])
def test_new_repro_is_self_contained(cpu_cuda, monkeypatch, module, tmp_path):
    monkeypatch.setattr(fi, "_library", lambda: SimpleNamespace())
    monkeypatch.setattr(fa, "_api", lambda: SimpleNamespace())
    case = next(c for c in module.cases() if c.shapes.get("num_q_heads") == 32)
    script = _script(case, case.baseline.json(), replace(case.baseline, batch_size=3).json(), {"env": {}})
    assert "from ." not in script
    compile(script, "standalone.py", "exec")
    # Execute the embedded declarations and reconstruct cases without GPUs/deps.
    declarations = script.split("\ncase = next(c for c in cases() if c.name == CASE_NAME)")[0]
    namespace = {"__name__": "__main__"}
    exec(declarations, namespace)
    if module is fi:
        namespace["_library"] = lambda: SimpleNamespace()
    if module is fa:
        namespace["_api"] = lambda: SimpleNamespace()
    recreated = next(c for c in namespace["cases"]() if c.name == case.name)
    assert recreated.shapes == case.shapes
    assert recreated.baseline.json() == case.baseline.json()
    # Run declarations and case reconstruction in an isolated process outside
    # the checkout. No infer_determinism installation or optional library.
    path = tmp_path / "repro_declarations.py"
    path.write_text(declarations + '''
from types import SimpleNamespace
torch.cuda.is_available = lambda: True
_api = _library = lambda: SimpleNamespace()
case = next(c for c in cases() if c.name == CASE_NAME)
assert case.shapes['num_q_heads'] == 32
print(case.name)
''')
    result = subprocess.run([sys.executable, str(path)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert case.name in result.stdout


def test_graph_reference_is_eager_final_batch():
    def run(spec):
        # Deliberately batch-dependent, but graph replay itself is invariant.
        return torch.tensor([float(spec.batch_size + spec.probe_index)])
    case = Case("fixture", "graph", "float32", {}, run, ("graph_replay",))
    records = run_case(case, quick=True)
    assert records[-1]["type"] == "pass"
    for record in records[:-1]:
        assert record["reference"]["batch_size"] == record["config"]["batch_size"] == 3
        assert record["reference"]["probe_index"] == record["config"]["probe_index"]
        assert record["reference"]["graph_capture_kv_length"] == 0


def test_graph_rejects_unwritten_output(cpu_cuda):
    output = torch.zeros((3, 32, 128), dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="nonfinite or unwritten"):
        replay_graph(SimpleNamespace(replay=lambda: None), output)
