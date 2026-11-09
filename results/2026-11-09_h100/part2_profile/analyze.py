import json
def cat(n):
    l=n.lower()
    if any(x in l for x in ("flashattn","flash::","fmha","fwdcombine")): return "attention"
    if "matmul_kernel_persistent" in l: return "triton_BI_gemm"
    if any(x in l for x in ("nvjet","xmma","gemm","cublas","gemv","splitk")): return "cublas_gemm"
    if "rms_norm" in l or "rmsnorm" in l: return "rmsnorm"
    return "other"
for rep,pre in (("rep1",""),("rep2","rep2/")):
  for B in (32,128):
    for inv in (0,1):
        d=json.load(open(f"{pre}inv{inv}_b{B}.json"))
        c={};tot=0
        for k in d["kernels"]:
            n=k["name"]
            if n.startswith(("BI_MM","aten","cuda","Memcpy","Memset","_vllm","_C::","Runtime")) : continue
            c[cat(n)]=c.get(cat(n),0)+k["self_device_us"]; tot+=k["self_device_us"]
        s=d["profile_steps"]
        print(f"{rep} B={B} inv={inv}: decode {d['decode_tok_per_s']:.0f} tok/s, {d['decode_step_ms']:.2f} ms/step; kernel-sum {tot/s/1000:.2f} ms/step; "+", ".join(f"{k} {v/s/1000:.2f}ms ({100*v/tot:.0f}%)" for k,v in sorted(c.items(),key=lambda x:-x[1])))
