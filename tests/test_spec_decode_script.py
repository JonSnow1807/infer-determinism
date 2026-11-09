"""CPU checks for the pure comparison helpers of scripts/spec_decode_bi.py."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("spec_decode_bi", Path(__file__).parents[1] / "scripts" / "spec_decode_bi.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_diff_reports_first_token_and_logprob_bits():
    a = {"token_ids": [1, 2, 3, 4], "logprobs": [-0.1, -0.2, -0.3, -0.4]}
    same = mod.diff(a, dict(a))
    assert same["tokens_equal"] and same["logprob_bits_equal_prefix"] and same["first_token_diff"] is None
    b = {"token_ids": [1, 2, 9, 4], "logprobs": [-0.1, -0.2000001, -0.3, -0.4]}
    d = mod.diff(a, b)
    assert not d["tokens_equal"] and d["first_token_diff"] == 2
    assert d["logprob_first_diff"] == 1 and not d["logprob_bits_equal_prefix"]
    shorter = mod.diff(a, {"token_ids": [1, 2], "logprobs": None})
    assert shorter["first_token_diff"] == 2 and "logprob_first_diff" not in shorter


def test_fillers_are_deterministic_and_varied():
    f1, f2 = mod.filler_prompts(64), mod.filler_prompts(64)
    assert f1 == f2 and len({m for _, m in f1}) > 10
