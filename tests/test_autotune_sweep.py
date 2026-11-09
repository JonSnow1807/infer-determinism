"""CPU tests for the autotune-config sweep engine, using a Triton-like stand-in."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from infer_determinism import autotune_sweep as sw
from infer_determinism.core import compare
from infer_determinism.report import sweep_markdown


class FakeConfig:
    def __init__(self, kwargs, num_warps=4, num_stages=2):
        self.kwargs, self.num_warps, self.num_stages, self.num_ctas, self.maxnreg, self.pre_hook = kwargs, num_warps, num_stages, 1, None, None

    def all_kwargs(self):
        return {**self.kwargs, "num_warps": self.num_warps, "num_stages": self.num_stages}


class FakeKernel:
    """Stands in for a JITFunction: behaviour depends on the MODE meta-parameter."""

    def __init__(self, body):
        self.body = body
        self.__name__ = body.__name__

    def run(self, *args, grid=None, warmup=False, num_warps=4, num_stages=2, **kwargs):
        if kwargs.get("MODE") == "invalid":
            raise RuntimeError("out of resource: shared memory")
        self.body(*args, **kwargs)


class FakeAutotuner:
    """Same contract as triton.runtime.autotuner.Autotuner.run for one or many configs."""

    def __init__(self, fn, arg_names, configs, pick=0):
        self.fn, self.base_fn, self.arg_names, self.configs, self.cache, self.pick = fn, fn, arg_names, configs, {}, pick
        self.keys = []

    def run(self, *args, **kwargs):
        config = self.configs[self.pick] if len(self.configs) > 1 else self.configs[0]
        self.best_config = config
        return self.fn.run(*args, **kwargs, **config.all_kwargs())


def scale_kernel(x, out, N, MODE="ok", B=0):
    if MODE == "racy":
        out.copy_(x * 2 + torch.randn_like(x) * 1e-3)
    elif MODE == "wrong":
        out.copy_(x * 3)
    elif MODE == "partial":
        out[: N // 2] = x[: N // 2] * 2
    else:
        out.copy_(x * 2)


def add_kernel(a, b, N, MODE="ok"):
    b.add_(a)  # in-place: reads and writes b


CONFIGS = [FakeConfig({"MODE": "ok", "B": 32}, 2), FakeConfig({"MODE": "racy"}, 4), FakeConfig({"MODE": "wrong"}, 8),
           FakeConfig({"MODE": "invalid"}, 8), FakeConfig({"MODE": "partial"}, 4), FakeConfig({"MODE": "ok", "B": 64}, 4)]


@pytest.fixture
def tuners(monkeypatch):
    monkeypatch.setattr(sw, "default_autotuner_class", lambda: FakeAutotuner)
    monkeypatch.setattr(sw, "INVALID_CONFIG_ERRORS", (RuntimeError,))
    t1 = FakeAutotuner(FakeKernel(scale_kernel), ["x", "out", "N"], list(CONFIGS), pick=1)
    t2 = FakeAutotuner(FakeKernel(add_kernel), ["a", "b", "N"], [FakeConfig({"MODE": "ok"}, w) for w in (2, 4)])
    return t1, t2


def scenario(t1, t2):
    x = torch.arange(8, dtype=torch.float32)
    def run():
        out = torch.empty(8)
        t1.run(x, out, 8, grid=(1,), warmup=False)
        acc = torch.ones(8)
        t2.run(out, acc, N=8, grid=(1,), warmup=False)
        t1.run(x, torch.empty(8), 8, grid=(1,), warmup=False)  # same signature: deduplicated
        return acc
    return run


def test_capture_first_mode_records_outputs_and_dedupes(tuners):
    t1, t2 = tuners
    cap = sw.capture(scenario(t1, t2), "fake", mode="first")
    assert [l.kernel for l in cap.launches] == ["scale_kernel", "add_kernel"]
    first, second = cap.launches
    assert first.capture_config == 0 and set(first.outputs) == {"out"}
    assert set(second.outputs) == {"b"}  # in-place accumulator detected as output
    assert torch.equal(cap.output, torch.arange(8.) * 2 + 1)  # first config, not the racy autotune pick
    assert cap.picked["scale_kernel"] == ["B=32 MODE=ok num_warps=2 num_stages=2"] * 2
    assert t1.configs == CONFIGS and t1.cache == {}


def test_capture_autotune_mode_uses_tuner_choice(tuners):
    t1, t2 = tuners
    cap = sw.capture(scenario(t1, t2), "fake", mode="autotune")
    assert cap.launches[0].capture_config == 1


def test_sweep_flags_racy_wrong_invalid_and_counts_unwritten(tuners):
    t1, t2 = tuners
    cap = sw.capture(scenario(t1, t2), "fake", mode="first")
    ref = lambda launch: {"out": launch.inputs()["x"].float() * 2}
    rows = sw.sweep_launch(cap.launches[0], repeats=3, reference=ref)
    by = {r["config_index"]: r for r in rows if r["type"] == "sweep_config"}
    assert by[0]["flags"] == [] and by[5]["flags"] == []
    assert "nondeterministic" in by[1]["flags"] and by[1]["repeat_differing_elements"] > 0
    assert {"wrong_vs_ref", "outlier_vs_consensus"} <= set(by[2]["flags"]) and by[2]["deterministic"]
    assert by[3]["status"] == "invalid"
    assert by[4]["poison_differing"]["out"] == 4  # half the output never written
    summary = rows[-1]
    assert summary["type"] == "sweep_summary" and summary["invalid"] == 1
    assert summary["nondeterministic"] == [1] and 2 in summary["flagged"]


def test_replay_restores_inplace_inputs(tuners):
    t1, t2 = tuners
    cap = sw.capture(scenario(t1, t2), "fake", mode="first")
    rows = sw.sweep_launch(cap.launches[1], repeats=3, poison_check=False)
    assert all(r["deterministic"] and not r["flags"] for r in rows if r["type"] == "sweep_config")


def test_aliasing_preserved_across_materialize(monkeypatch):
    monkeypatch.setattr(sw, "INVALID_CONFIG_ERRORS", (RuntimeError,))
    def body(src, dst, N, MODE="ok"):
        dst.copy_(src + 1)  # dst aliases the second half of src's storage
    tuner = FakeAutotuner(FakeKernel(body), ["src", "dst", "N"], [FakeConfig({}, 2), FakeConfig({}, 4)])
    base = torch.zeros(8)
    cap = sw.capture(lambda: tuner.run(base[:4], base[4:], 4), "alias", autotuner_cls=FakeAutotuner)
    launch = cap.launches[0]
    args, _, views = launch.materialize()
    assert args[0].untyped_storage().data_ptr() == args[1].untyped_storage().data_ptr()
    assert args[1].storage_offset() == 4


def test_fast_compare_matches_core_compare():
    g = torch.Generator().manual_seed(0)
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        a = torch.randn(257, generator=g).to(dtype)
        b = a.clone()
        b[3] = -b[3]
        b[100] = float("nan")
        slow, fast = compare(a, b), sw.fast_compare(a, b)
        assert (slow.equal, slow.differing_elements, slow.max_ulp_distance) == (fast.equal, fast.differing_elements, fast.max_ulp_distance)
        assert slow.max_abs_diff == fast.max_abs_diff
    assert sw.fast_compare(torch.tensor([0.0]), torch.tensor([-0.0])).differing_elements == 1


def test_rel_err_marks_new_nonfinite_as_inf():
    assert sw.rel_err(torch.tensor([1.0, float("nan")]), torch.tensor([1.0, 2.0])) == float("inf")
    assert sw.rel_err(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0])) == 0.0


def test_sweep_report_renders(tuners, tmp_path):
    t1, t2 = tuners
    cap = sw.capture(scenario(t1, t2), "fake", mode="first")
    rows = [{"type": "sweep_metadata", "gpu": "fake", "capability": [9, 0], "versions": {}},
            sw.launch_record(cap.launches[0])] + sw.sweep_launch(cap.launches[0], repeats=2)
    path = tmp_path / "s.jsonl"
    path.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    text = sweep_markdown(path)
    assert "scale_kernel" in text and "MODE=racy" in text and "nondeterministic" in text


def test_kernel_references_on_cpu():
    from infer_determinism import sweep_scenarios as sc
    torch.manual_seed(0)
    T, H, K, BT = 130, 2, 8, 64
    k = torch.nn.functional.normalize(torch.randn(1, T, H, K), dim=-1)
    beta, g = torch.rand(1, T, H), -torch.rand(1, T, H).cumsum(1) * 0.01
    launch = SimpleNamespace(inputs=lambda: {"k": k, "beta": beta, "g": g, "cu_seqlens": torch.tensor([0, 70, 130]), "BT": BT})
    A = sc.ref_kkt(launch)["A"]
    t, s = 75, 72  # second sequence starts at 70; same chunk, s < t
    expect = beta[0, t, 1] * torch.exp(g[0, t, 1] - g[0, s, 1]) * (k[0, t, 1] @ k[0, s, 1])
    assert torch.allclose(A[0, t, 1, s - 70], expect)
    assert A[0, t, 1, t - 70] == 0 and A[0, 69, 0, :].count_nonzero() <= 63
    inv = sc.ref_solve_tril(SimpleNamespace(inputs=lambda: {"A": A, "cu_seqlens": torch.tensor([0, 70, 130])}))["Ai"]
    blk, iblk = A[0, 70:130, 0, :60], inv[0, 70:130, 0, :60]
    assert torch.allclose((torch.eye(60) + blk) @ iblk, torch.eye(60), atol=1e-4)


def test_naive_delta_rule_matches_chunk_reference_shapes():
    from infer_determinism import sweep_scenarios as sc
    torch.manual_seed(1)
    T, Hk, H, K, V = 10, 1, 2, 4, 3
    q, k, v = torch.randn(1, T, Hk, K), torch.randn(1, T, Hk, K), torch.randn(1, T, H, V)
    g, beta = -torch.rand(1, T, H), torch.rand(1, T, H)
    h0 = torch.randn(2, H, V, K)
    o, final = sc.naive_delta_rule(q, k, v, g, beta, 0.5, h0, torch.tensor([0, 4, 10]), per_dim_gate=False, state_vk=True, l2norm=True)
    assert o.shape == (1, T, H, V) and final.shape == (2, H, V, K)
    # Sequence 2 starts from h0[1], independent of sequence 1.
    o2, _ = sc.naive_delta_rule(q[:, 4:], k[:, 4:], v[:, 4:], g[:, 4:], beta[:, 4:], 0.5, h0[1:], None, per_dim_gate=False, state_vk=True, l2norm=True)
    assert torch.allclose(o[:, 4:], o2)


def test_naive_ssd_shapes_and_independence():
    from infer_determinism import sweep_scenarios as sc
    torch.manual_seed(2)
    T, nh, hd, ng, ds = 9, 4, 3, 2, 5
    x, dt, A = torch.randn(T, nh, hd), torch.randn(T, nh), -torch.rand(nh)
    Bm, Cm, D, bias = torch.randn(T, ng, ds), torch.randn(T, ng, ds), torch.ones(nh), torch.zeros(nh)
    h0 = torch.randn(2, nh, hd, ds)
    y, final = sc.naive_ssd(x, dt, A, Bm, Cm, D, bias, h0, torch.tensor([0, 5, 9]))
    y2, _ = sc.naive_ssd(x[5:], dt[5:], A, Bm[5:], Cm[5:], D, bias, h0[1:], torch.tensor([0, 4]))
    assert final.shape == (2, nh, hd, ds) and torch.allclose(y[5:], y2)


def test_scenarios_skip_without_cuda():
    from infer_determinism import sweep_scenarios as sc
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    for scenario in sc.SCENARIOS.values():
        with pytest.raises(sc.SkipScenario):
            scenario.build()


def test_reference_declared_output_checked_even_if_capture_left_it_unchanged(monkeypatch):
    monkeypatch.setattr(sw, "INVALID_CONFIG_ERRORS", (RuntimeError,))
    def body(x, out, N, MODE="ok"):
        if MODE == "ok":
            out.copy_(x)  # x is zeros, out starts as zeros: capture sees no change
        else:
            out.fill_(1.0)
    tuner = FakeAutotuner(FakeKernel(body), ["x", "out", "N"], [FakeConfig({"MODE": "ok"}, 2), FakeConfig({"MODE": "bad"}, 4)])
    cap = sw.capture(lambda: tuner.run(torch.zeros(4), torch.zeros(4), 4), "unchanged", autotuner_cls=FakeAutotuner)
    assert cap.launches[0].outputs == {}
    rows = sw.sweep_launch(cap.launches[0], repeats=2, reference=lambda l: {"out": torch.zeros(4)})
    by = {r["config_index"]: r for r in rows if r["type"] == "sweep_config"}
    assert by[0]["flags"] == [] and "wrong_vs_ref" in by[1]["flags"]


def test_written_nonfinite_flagged_even_when_all_configs_agree(monkeypatch):
    monkeypatch.setattr(sw, "INVALID_CONFIG_ERRORS", (RuntimeError,))
    def body(x, out, N, MODE="ok"):
        out.copy_(x / 0.0 * 0.0)  # NaN everywhere, deterministically
    tuner = FakeAutotuner(FakeKernel(body), ["x", "out", "N"], [FakeConfig({}, w) for w in (1, 2, 4)])
    cap = sw.capture(lambda: tuner.run(torch.ones(4), torch.zeros(4), 4), "nan", autotuner_cls=FakeAutotuner)
    rows = sw.sweep_launch(cap.launches[0], repeats=2)
    assert all("nonfinite" in r["flags"] for r in rows if r["type"] == "sweep_config")


def test_stray_writes_flagged(monkeypatch):
    monkeypatch.setattr(sw, "INVALID_CONFIG_ERRORS", (RuntimeError,))
    def body(x, out, N, MODE="ok"):
        out.copy_(x + 1)
        if MODE == "stray":
            x.zero_()  # clobbers an input the capture config only read
    tuner = FakeAutotuner(FakeKernel(body), ["x", "out", "N"], [FakeConfig({"MODE": "ok"}, 2), FakeConfig({"MODE": "stray"}, 4)])
    cap = sw.capture(lambda: tuner.run(torch.ones(4), torch.zeros(4), 4), "stray", autotuner_cls=FakeAutotuner)
    rows = sw.sweep_launch(cap.launches[0], repeats=2)
    by = {r["config_index"]: r for r in rows if r["type"] == "sweep_config"}
    assert by[0]["flags"] == [] and by[1]["flags"] == ["stray_writes"] and by[1]["stray_writes"] == {"x": 4}
