"""Speculative decoding under VLLM_BATCH_INVARIANT=1.

Claim tested: with batch invariance on and greedy sampling, a request's output
(token ids and returned logprobs) is bitwise the same whether it runs alone or
batched with other requests, for a fixed speculative-decoding configuration.
This is not the spec-vs-non-spec question of vllm#54506 (that comparison is
written out as context only, by compare_methods()).

One process per method (the environment must be set before vLLM is imported):
    python scripts/spec_decode_bi.py --method none   --out results/spec_none.json
    python scripts/spec_decode_bi.py --method ngram  --out results/spec_ngram.json
    python scripts/spec_decode_bi.py --method eagle3 --out results/spec_eagle3.json
    python scripts/spec_decode_bi.py --compare results/spec_*.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
import time

PROBES = [
    # Repetitive code: gives the n-gram proposer matches to draft from.
    "Complete the Python module. Keep the same style for every function.\n\n"
    "def add_one(values):\n    result = []\n    for value in values:\n        result.append(value + 1)\n    return result\n\n"
    "def add_two(values):\n    result = []\n    for value in values:\n        result.append(value + 2)\n    return result\n\n"
    "def add_three(values):\n",
    "Write a detailed story about a lighthouse keeper who discovers that the light has been sending messages "
    "to ships in a code nobody has used for a hundred years. Describe the first message in full.",
    "List the first 40 elements of the periodic table as 'number. name (symbol)', one per line, then explain the "
    "pattern in the electron configurations of the first 20.",
]


def filler_prompts(n: int, seed: int = 1234) -> list[tuple[str, int]]:
    rng = random.Random(seed)
    topics = ["the history of the printing press", "how a heat pump works", "the rules of chess",
              "photosynthesis", "the stock market", "volcanoes", "the immune system", "TCP congestion control",
              "medieval castles", "black holes", "the French revolution", "compilers"]
    out = []
    for i in range(n):
        words = " ".join(rng.choice(["background", "detail", "context", "notes", "aside"]) for _ in range(rng.randint(0, 600)))
        out.append((f"{words}\nExplain {rng.choice(topics)} to a student. Part {i}.", 16 + (i * 37) % 200))
    return out


def bits(values: list[float]) -> list[int]:
    return [struct.unpack("<I", struct.pack("<f", v))[0] for v in values]


def extract(output) -> dict[str, object]:
    comp = output.outputs[0]
    lps = None
    if getattr(comp, "logprobs", None):
        lps = [float(step[tok].logprob) for step, tok in zip(comp.logprobs, comp.token_ids)]
    return {"token_ids": list(comp.token_ids), "logprobs": lps, "text": comp.text[:200]}


def diff(a: dict, b: dict) -> dict[str, object]:
    ta, tb = a["token_ids"], b["token_ids"]
    first = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None)
    if first is None and len(ta) != len(tb):
        first = min(len(ta), len(tb))
    res: dict[str, object] = {"tokens_equal": ta == tb, "first_token_diff": first, "len": [len(ta), len(tb)]}
    if a["logprobs"] is not None and b["logprobs"] is not None:
        n = min(len(a["logprobs"]), len(b["logprobs"]), first if first is not None else 10**9)
        la, lb = a["logprobs"][:n], b["logprobs"][:n]
        res["logprob_bits_equal_prefix"] = bits(la) == bits(lb)
        res["logprob_max_abs_diff_prefix"] = max((abs(x - y) for x, y in zip(la, lb)), default=0.0)
        res["logprob_first_diff"] = next((i for i, (x, y) in enumerate(zip(bits(la), bits(lb))) if x != y), None)
    return res


def run(args) -> dict[str, object]:
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    import vllm
    from vllm import LLM, SamplingParams

    spec = None
    if args.method == "ngram":
        spec = {"method": "ngram", "num_speculative_tokens": args.k, "prompt_lookup_max": 5, "prompt_lookup_min": 2}
    elif args.method == "eagle3":
        spec = {"method": "eagle3", "model": args.draft, "num_speculative_tokens": args.k}
    elif args.method == "mtp":
        spec = {"method": "mtp", "num_speculative_tokens": args.k}
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    llm = LLM(model=args.model, speculative_config=spec, attention_config={"backend": args.backend},
              max_num_seqs=max(sizes), max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_mem,
              seed=0, enforce_eager=args.eager)
    want_lp = 1
    params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens, logprobs=want_lp)
    try:
        llm.generate([PROBES[0]], SamplingParams(temperature=0.0, max_tokens=4, logprobs=want_lp), use_tqdm=False)
    except Exception as exc:  # logprobs unsupported with this spec method: tokens only
        print(f"logprobs unavailable: {exc}", file=sys.stderr)
        want_lp = None
        params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    fillers = filler_prompts(max(sizes))
    results: list[dict[str, object]] = []
    outputs: dict[str, object] = {}
    t0 = time.time()
    for p, probe in enumerate(PROBES):
        alone = extract(llm.generate([probe], params, use_tqdm=False)[0])
        again = extract(llm.generate([probe], params, use_tqdm=False)[0])
        outputs[f"probe{p}"] = alone
        results.append({"probe": p, "batch_size": 1, "position": 0, "compare": "repeat_alone", **diff(alone, again)})
        for bs in sizes:
            if bs == 1:
                continue
            for pos in sorted({0, bs // 2, bs - 1}):
                prompts = [f for f, _ in fillers[: bs - 1]]
                plist = [SamplingParams(temperature=0.0, max_tokens=m, logprobs=want_lp) for _, m in fillers[: bs - 1]]
                prompts.insert(pos, probe)
                plist.insert(pos, params)
                outs = llm.generate(prompts, plist, use_tqdm=False)
                got = extract(outs[pos])
                results.append({"probe": p, "batch_size": bs, "position": pos, "compare": "batched_vs_alone", **diff(alone, got)})
    metrics = {}
    try:
        for m in llm.get_metrics():
            if "spec_decode" in m.name:
                metrics[m.name] = getattr(m, "value", None) if hasattr(m, "value") else getattr(m, "values", None)
    except Exception as exc:
        metrics["error"] = str(exc)
    return {"vllm": vllm.__version__, "model": args.model, "method": args.method, "speculative_config": spec,
            "backend": args.backend, "eager": args.eager, "batch_sizes": sizes, "max_tokens": args.max_tokens,
            "logprobs": want_lp is not None, "seconds": round(time.time() - t0, 1), "spec_metrics": metrics,
            "env": {k: os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "CUBLAS_WORKSPACE_CONFIG")},
            "all_equal": all(r["tokens_equal"] and r.get("logprob_bits_equal_prefix", True) for r in results),
            "results": results, "outputs": outputs}


def compare_methods(paths: list[str]) -> None:
    runs = [json.load(open(p)) for p in paths]
    base = next((r for r in runs if r["method"] == "none"), None)
    for r in runs:
        bad = [x for x in r["results"] if not (x["tokens_equal"] and x.get("logprob_bits_equal_prefix", True))]
        print(f"{r['method']:7s} vllm {r['vllm']} backend {r['backend']} eager={r['eager']} logprobs={r['logprobs']}: "
              f"{len(r['results']) - len(bad)}/{len(r['results'])} comparisons bitwise equal")
        for x in bad[:10]:
            print("   DIFF", {k: x[k] for k in ("probe", "batch_size", "position", "compare", "first_token_diff", "len")},
                  {k: x.get(k) for k in ("logprob_first_diff", "logprob_max_abs_diff_prefix")})
        if base is not None and r is not base:
            for k, out in r["outputs"].items():
                d = diff(base["outputs"][k], out)
                print(f"   context only (vllm#54506): {k} {r['method']} vs none: tokens_equal={d['tokens_equal']} "
                      f"first_diff={d['first_token_diff']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["none", "ngram", "eagle3", "mtp"], default="none")
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--draft", default="RedHatAI/Qwen3-8B-speculator.eagle3")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--backend", default="FLASH_ATTN")
    ap.add_argument("--batch-sizes", default="1,2,8,32,64")
    ap.add_argument("--max-tokens", type=int, default=160)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs="+", default=None)
    args = ap.parse_args()
    if args.compare:
        compare_methods(args.compare)
        return
    result = run(args)
    text = json.dumps(result, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(json.dumps({k: v for k, v in result.items() if k not in ("results", "outputs")}, indent=1))
    print("all_equal:", result["all_equal"])


if __name__ == "__main__":
    main()
