"""Decode throughput + kernel profile for vLLM, default vs VLLM_BATCH_INVARIANT=1.
usage: VLLM_BATCH_INVARIANT={0,1} python profile_decode.py OUT.json BATCH [--eager]
"""
import json, os, sys, time, statistics, collections
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams

out_path, B = sys.argv[1], int(sys.argv[2])
eager = "--eager" in sys.argv
IN, OUT = 512, 256
shapes = collections.Counter()
if os.environ.get("VLLM_BATCH_INVARIANT") == "1":
    import vllm.model_executor.determinism.batch_invariant as bi
    orig = bi.mm_batch_invariant
    def rec(a, b, bias=None):
        shapes[(a.shape[0], b.shape[1], a.shape[1])] += 1
        with torch.profiler.record_function(f"BI_MM M={a.shape[0]} N={b.shape[1]} K={a.shape[1]}"):
            return orig(a, b, bias=bias)
    if eager:
        bi.mm_batch_invariant = rec

llm = LLM("Qwen/Qwen3-8B", dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.40,
          enable_prefix_caching=False, seed=0, enforce_eager=eager)
g = torch.Generator().manual_seed(0)
prompts = [{"prompt_token_ids": torch.randint(1000, 100000, (IN,), generator=g).tolist()} for _ in range(B)]
def gen(n):
    sp = SamplingParams(temperature=0, max_tokens=n, ignore_eos=True)
    torch.cuda.synchronize(); t = time.perf_counter()
    llm.generate(prompts, sp, use_tqdm=False)
    torch.cuda.synchronize(); return time.perf_counter() - t
res = {"batch": B, "in": IN, "out": OUT, "eager": eager, "invariant": os.environ.get("VLLM_BATCH_INVARIANT"),
       "torch": torch.__version__}
if not eager:
    gen(8); gen(1)
    t1 = [gen(1) for _ in range(3)]; tn = [gen(OUT) for _ in range(3)]
    res.update(t_prefill_s=statistics.median(t1), t_total_s=statistics.median(tn), all_t1=t1, all_tn=tn)
    res["decode_tok_per_s"] = B * (OUT - 1) / (res["t_total_s"] - res["t_prefill_s"])
    res["decode_step_ms"] = 1000 * (res["t_total_s"] - res["t_prefill_s"]) / (OUT - 1)
    res["total_tok_per_s"] = B * OUT / res["t_total_s"]
# profile decode steps via manual stepping
eng = llm.llm_engine
sp = SamplingParams(temperature=0, max_tokens=OUT, ignore_eos=True)
for i, p in enumerate(prompts):
    eng.add_request(f"r{i}", p, sp)
steps = 0
while True:  # run until all prompts prefilled and a few decode steps done
    outs = eng.step(); steps += 1
    if steps >= 3 and all(len(o.outputs[0].token_ids) >= 2 for o in outs) and len(outs) == B: break
for _ in range(3): eng.step()
torch.cuda.synchronize()
NSTEP = 8
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for _ in range(NSTEP): eng.step()
    torch.cuda.synchronize()
rows = []
for e in prof.key_averages():
    ct = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
    selft = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    rows.append({"name": e.key, "count": e.count, "device_us_total": ct, "self_device_us": selft})
res["profile_steps"] = NSTEP
res["kernels"] = sorted([r for r in rows if r["self_device_us"] > 0], key=lambda r: -r["self_device_us"])[:60]
res["bi_mm_ranges"] = sorted([r for r in rows if r["name"].startswith("BI_MM")], key=lambda r: -r["device_us_total"])
res["mm_shapes_seen"] = {f"M={k[0]} N={k[1]} K={k[2]}": v for k, v in shapes.items()}
json.dump(res, open(out_path, "w"), indent=1)
print("DONE", {k: res[k] for k in res if k.startswith(("decode", "total", "t_"))})
