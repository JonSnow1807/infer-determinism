# H100 NVL run 3: 2026-11-09

Run 3 tested three things that runs 1 and 2 did not cover:

1. An autotune-config sweep, a new harness feature (`python -m infer_determinism sweep`). It covers the Triton kernels vLLM vendors for linear attention and state-space models.
2. Speculative decoding under `VLLM_BATCH_INVARIANT=1`.
3. FlashInfer FA3 attention with vLLM's batch-invariant plan arguments. trtllm-gen attention was out of reach on this GPU.

No new bug turned up in the vLLM kernels outside the known one. The known `chunk_scaled_dot_kkt` failure of [vllm#65385](https://github.com/vllm-project/vllm/issues/65385) served as the positive control, and it reproduced. Tracing it down produced one finding that does not appear to be reported anywhere: a Triton code-generation race on Hopper. It reproduces with a 15-line kernel on Triton 3.6.0 through 3.9.0 and on `main`, with both the old and the fixed `ptxas`. Details are in Finding 1.

## Environment

| | |
|---|---|
| GPU | 1x NVIDIA H100 NVL 94 GB (sm90), driver 580.159.03, Vast.ai instance 59948100 (Ireland) |
| Image | `nvcr.io/nvidia/pytorch:26.08-py3` (CUDA 13.4.1 toolkit), fresh venv, `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`, `PIP_CONSTRAINT` unset |
| Python packages | torch 2.14.0+cu130, vLLM 0.33.0 (tag commit 4ba787ec), FlashInfer 0.7.2.post1, fla-core 0.6.0 (tag 413ad2c6), Triton 3.8.0 (vLLM's pin) |
| Other Triton builds | 3.6.0, 3.7.1, 3.9.0 (PyPI wheels on `PYTHONPATH`), `main` at dddf5617 (built from source on the box) |
| `ptxas` used for sm90 | The image exports `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` (13.4.1), which overrides the bundled `ptxas` of Triton < 3.9. Runs named `*_bundledptxas` unset it and use Triton's own (12.8 for 3.6.0/3.7.1, 12.9 for 3.8.0), as a plain `pip install vllm` does. Triton 3.9.0 and `main` always use their bundled `ptxas-blackwell` 13.4 for sm90. |

Full freeze: [`pipfreeze.txt`](pipfreeze.txt); [`versions.txt`](versions.txt). The harness commit that ran is 3299727, plus the scenario additions committed with these results; [`logs/`](logs/) has every job's console output and the driver scripts.

## 1. Autotune-config sweep

### Method

A scenario calls a library entry point once on fixed seeded inputs. Every `Autotuner.run` underneath is captured, with storage-level snapshots of its arguments. Each captured launch is then replayed once per config in its autotune list, on identical input bytes. The sweep flags a config when:

- its output is not bitwise identical over 3 to 8 replays;
- it disagrees with a float32 reference by more than 1e-2 relative (Frobenius norm), where a reference exists;
- it disagrees with the elementwise median of the other configs by more than 1e-2 relative;
- it writes nonfinite values;
- it writes an argument that the capture config left untouched.

Each scenario also runs end to end with normal autotuning: 3 calls in one process, compared with a float32 sequential recurrence.

### Coverage

The sweep covered 37 captured launches and 495 configs per Triton/ptxas setting, in seven settings: 3.6.0, 3.7.1 and 3.8.0, each with the image's 13.4.1 and the bundled `ptxas`, plus 3.9.0. That is 3,465 config sweeps in total.

| Scenario | Kernels swept (autotuned launches) |
|---|---|
| `gdn_vllm_fixed`, `gdn_vllm_varlen`: vLLM vendored `chunk_gated_delta_rule` (T=4096, K=V=128; varlen 1000/1600/1496 tokens with 16 QK / 32 V heads, i.e. Qwen3-Next at TP=1) | `chunk_local_cumsum_scalar`, `chunk_scaled_dot_kkt_fwd`, `merge_16x16_to_64x64_inverse`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o` |
| `kda_vllm_varlen`: vLLM vendored `chunk_kda` (Kimi Linear) | `chunk_local_cumsum_vector`, `chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter` / `_intra_sub_intra`, `merge_16x16_to_64x64_inverse`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` (gk path), `chunk_gla_fwd_kernel_o` |
| `kda_kimik3_varlen`: vLLM's Kimi-K3 NVIDIA `chunk_kda` copy | `chunk_local_cumsum_vector`, `chunk_kda_fwd_kernel_intra_token_parallel`, `chunk_kda_fwd_kernel_inter_solve_fused`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_gla_fwd_kernel_o` |
| `mamba2_ssd_varlen`: vLLM `mamba_chunk_scan_combined_varlen` (64 heads x 64, 8 groups, d_state 128, chunk 256) | `_chunk_cumsum_fwd`, `_chunk_state_fwd`, `_state_passing_fwd`, `_bmm_chunk_fwd`, `_chunk_scan_fwd` |
| `gdn_fla_varlen`: upstream FLA 0.6.0 `chunk_gated_delta_rule`, same inputs | `l2norm_fwd` (q and k), `chunk_local_cumsum_scalar`, `chunk_gated_delta_rule_fwd_kkt_solve`, `recompute_w_u_fwd`, `chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o` |

vLLM's own `l2norm` path (`l2norm_fwd_kernel2`) is not autotuned, so it was not captured.

### Results

- **Only `chunk_scaled_dot_kkt_fwd_kernel` failed, in every setting.** Every failing config has `num_warps >= 4` and `BK < K`.
- **The other 35 vLLM and upstream-FLA launches passed.** No config was nondeterministic, wrong, or nonfinite. This includes KDA's `chunk_gla_fwd_kernel_o`, which also computes its dot's left operand in registers inside a K loop.
- **End to end, every scenario stayed within 4.4e-3 of its float32 recurrence**, and was bitwise identical over 3 calls.
- **On this H100 NVL, normal autotuning always picked a safe kkt config** (`BK=64 num_warps=2`, 14 runs). So the vllm#65385 failure needs a GPU or timing where autotuning lands on a bad config, as it did on the reporter's H20.

Sweep reports: [`sweep_triton3.6.0.md`](sweep_triton3.6.0.md), [`sweep_triton3.6.0_bundledptxas.md`](sweep_triton3.6.0_bundledptxas.md), [`sweep_triton3.7.1.md`](sweep_triton3.7.1.md), [`sweep_triton3.7.1_bundledptxas.md`](sweep_triton3.7.1_bundledptxas.md), [`sweep_triton3.8.0_gdn.md`](sweep_triton3.8.0_gdn.md) + [`_kda`](sweep_triton3.8.0_kda.md) + [`_other`](sweep_triton3.8.0_other.md), [`sweep_triton3.8.0_bundledptxas.md`](sweep_triton3.8.0_bundledptxas.md), [`sweep_triton3.9.0.md`](sweep_triton3.9.0.md).

### Positive control: vllm#65385 (`chunk_scaled_dot_kkt_fwd`, T=4096, H=32, K=128, BT=64)

The sweep reproduces the reported failure class in every setting. Nondeterministic configs out of 27, over 5 replays each, with the worst relative error against the float32 reference:

| Kernel source | 3.6.0 | 3.7.1 | 3.8.0 | 3.9.0 | main |
|---|---|---|---|---|---|
| vLLM 0.33 vendored, `ptxas` 13.4 | 2 (2.5e-3) | 8 (3.2e-2) | 11 (3.2e-2) | 9 (5.3e-2) | 10 (6.4e-2) |
| vLLM 0.33 vendored, bundled `ptxas` (12.8 / 12.8 / 12.9) | 7 (1.8e-2) | 9 (3.2e-2) | 6 (3.4e-2) | n/a | n/a |
| vLLM 0.19.1 source (the reporter's), `ptxas` 13.4 | 2 (2.9e-3) | 10 (4.1e-2) | 10 (3.4e-2) | does not compile (block pointers removed in 3.9) | n/a |
| vLLM 0.19.1 source, bundled `ptxas` | 5 (1.8e-2) | 9 (2.3e-2) | 10 (2.5e-2) | n/a | n/a |
| Upstream FLA 0.6.0 (beta applied after the K loop) | 0 | 0 | 0 | 0 | n/a |

The issue's own script, which runs each config twice, agrees config for config where it detects anything: [`control_issue_script_*.txt`](.). The magnitude on H100 NVL is smaller than the reporter's H20 numbers (relative error 0.2 to 0.7). The worst cases here reach 2e-2 to 6e-2, with the corrupted elements' values changing sign or magnitude (maximum ordered-ULP distance about 2e9).

Two harness slips, kept in the data for traceability:

- `control_triton{3.6.0,3.7.1,3.8.0,3.9.0}.md` flags all 27 upstream-FLA configs as `wrong_vs_ref`. My first reference used `exp` where FLA 0.6.0 uses `exp2`. With the corrected reference (`ref_kkt_exp2`, the `control_v0191_*` and `*_bundledptxas` files), FLA is at 2.3e-7 with 0 flags.
- An earlier beta-after-loop run had the image's TF32 override active for the float32 reference. It was rerun and overwritten; the `control_betaafter_*` files are the clean rerun.

**Head dimension 256** ([`control_k256_*.md`](.)): with K=256, all six `BK=128, num_warps >= 4` configs also fail, with up to 1.2e-1 relative error. The count is 15 of 27 on 3.8.0 (both `ptxas`) and 3.9.0, and 16 of 27 on `main`. The open workaround [vllm#65927](https://github.com/vllm-project/vllm/pull/65927) prunes `BK < 128 and num_warps >= 4`. That keeps these configs, so it is complete only for K <= 128. The condition that matches every observation is `BK < K and num_warps >= 4`.

### Finding 1: Triton races on shared memory when one loaded tile feeds both operands of a pipelined Hopper wgmma

Minimal repro: [`scripts/min_wgmma_rega_race.py`](../../scripts/min_wgmma_rega_race.py), with no vLLM involved. In a K loop it computes `acc += tl.dot(x * s[:, None], tl.trans(x))` on a tile `x` loaded once per iteration. Output on this box ([`min_rega_race.txt`](min_rega_race.txt), [`min_rega_race_triton_main.txt`](min_rega_race_triton_main.txt)), 20 runs per config, BT=64, K=128:

| Triton / `ptxas` | BK=64 nw=4 ns=2 | BK=64 nw=4 ns=3 | BK=32 nw=4 ns=3 | BK=64 nw=8 ns=2 | nw=2 or BK=K | control: scale after loop |
|---|---|---|---|---|---|---|
| 3.6.0 / 13.4 | ok | ok | ok | 2/19 differ | ok | ok (all) |
| 3.6.0 / 12.8 | 19/19 differ, 4.6e-3 | 19/19 | ok | 19/19 | ok | ok |
| 3.7.1 / 13.4 and / 12.8 | 19/19, up to 2.4e-2 | 19/19 | 19/19 | 19/19 | ok | ok |
| 3.8.0 / 13.4 and / 12.9 | 19/19, up to 2.6e-2 | 19/19 | 19/19 | 19/19 | ok | ok |
| 3.9.0 / 13.4 | 19/19, 3.4e-2 | 19/19 | 19/19 | 19/19 | ok | ok |
| main dddf5617 / 13.4 | 19/19, 2.6e-2 | 19/19 | 19/19 | 19/19 | ok | ok |

"ok" means 0 of 19 runs differ from run 0 and relative error is at most 9.8e-4. The scale-after-loop control is 1.7e-7.

**What the IR shows** ([`irdump/minrace/t390/`](irdump/minrace/t390/)), Triton 3.9.0, BK=64, num_warps=4, num_stages=2:

- **Failing kernel.** The TTGIR allocates a single shared-memory stage for `x` (`ttg.local_alloc : memdesc<1x64x64xbf16>`). It loads the left operand from that buffer into registers (`local_load`), and issues two `warp_group_dot {isAsync = true}` whose right operand is a view of the same buffer, each followed by `warp_group_dot_wait {pendings = 1}`. It then issues the next iteration's `async_copy_global_to_local` into that same buffer, while the last wgmma may still be reading it. In the SASS, `LDGSTS [R4]` follows `WARPGROUP.DEPBAR.LE gsb0, 0x1` with no wait for the pending HGMMA group.
- **Control kernel** (left operand from shared memory). It gets a 2-stage buffer (`memdesc<2x64x64xbf16>`), so the next copy goes to the other stage.

**Experiments run** ([`logs/override.log`](logs/override.log), [`irdump/override/`](irdump/override/)):

1. I edited the failing kernel's TTGIR, changing both in-loop `warp_group_dot_wait {pendings = 1}` to `pendings = 0`, and loaded it through `TRITON_KERNEL_OVERRIDE`. Result: 0 of 19 runs differ, relative error 9.8e-4.
2. I kept the register left operand and `pendings = 1`, but loaded the right operand from a separate pointer to the same data, so the two operands occupy different shared-memory buffers (`sepB` in [`scripts/min_race_experiments.py`](../../scripts/min_race_experiments.py)). Result: 0 of 19 runs differ.
3. I applied the scale after the loop, so both operands come from shared memory. Result: deterministic everywhere, including all 27 kkt configs (below).

**Conclusion, verified by those experiments:** the nondeterminism needs a wgmma left pending (`pendings = 1`) together with a single-buffered tile that is both the register LHS source and the RHS shared-memory operand. Experiment 2 keeps register-LHS wgmma with `pendings = 1` and is clean, so the register read-after-write hazard of [triton#9433](https://github.com/triton-lang/triton/issues/9433) does not explain this case. That issue was closed after ptxas 13.3; its last comment expects 3.9.0 to be correct by default.

**Hypothesis, not verified:** the fault is a write-after-read race on shared memory. The pipeliner gives the tile a single stage, because it has a register user, while the WGMMA pipelining still leaves the dot reading it asynchronously across the next async copy. Which pass should own the fix is also unverified.

**Not found reported:** I searched triton-lang/triton issues and PRs for wgmma register-operand, pipelining, wait_group, WAR and nondeterministic-dot terms on 2026-11-09. Related but different reports: triton#9433 (register RAW hazard). Reports of the same Hopper symptom class handled per kernel, which I did not root-cause against this mechanism: vllm#61718 (Mamba2 chunk-scan config filter), vllm#61906 (LoRA expand), state-spaces/mamba#1061, and pytorch/helion#4368 ("sm90 register-LHS dot race", fixed in Helion's code generation).

**Kernel-side fix tested** ([`scripts/vendored/chunk_scaled_dot_kkt_betaafter.py`](../../scripts/vendored/chunk_scaled_dot_kkt_betaafter.py)): vLLM 0.33's kkt kernel with `beta` applied to the fp32 accumulator after the K loop. That is the upstream FLA 0.6.0 structure, and the arithmetic is unchanged except that `k * beta` is no longer rounded to bf16.

- Result: 0 of 27 configs flagged on 3.6.0, 3.7.1, 3.8.0, 3.9.0 and `main`, with both `ptxas` builds, at K=128 and K=256, over 5 to 8 replays.
- Relative error against float32 drops from 1.66e-3 to 2.5e-7 (K=256: 3.8e-7). Most of that improvement comes from removing the bf16 rounding of `k * beta`, not from removing the race.
- [`irdump/betaafter/`](irdump/betaafter/) shows both operands from shared memory, still with `wait_group 1`.

Unlike the config pruning of vllm#65927, this keeps all 27 configs.

## 2. Speculative decoding under VLLM_BATCH_INVARIANT=1

[`scripts/spec_decode_bi.py`](../../scripts/spec_decode_bi.py) ran Qwen/Qwen3-8B (bf16, TP=1, CUDA graphs on, `CUBLAS_WORKSPACE_CONFIG=:4096:8`) with greedy decoding, 160 tokens, logprobs=1.

- **Comparisons.** Each of 3 probe prompts ran alone twice, then inside batches of 2, 8, 32 and 64 at the first, middle and last position. The other requests had prompt lengths of up to about 600 words and 16 to 215 output tokens, so batch composition changed during decoding.
- **Configurations.** No speculation; n-gram (k=3, prompt lookup 2 to 5); and EAGLE3 (`RedHatAI/Qwen3-8B-speculator.eagle3`, k=3) on both the FLASH_ATTN and TRITON_ATTN backends.
- **Result: all 36 comparisons were bitwise equal** (token ids and logprob bits) in all four configurations.
- **Speculation was active:** n-gram accepted 46,581 of 188,313 draft tokens; EAGLE3 accepted 68,965 of 122,289.
- **Context only (vllm#54506):** spec versus non-spec gave identical tokens for all probes on FLASH_ATTN. The TRITON_ATTN EAGLE3 run against the FLASH_ATTN baseline differs at token 5 of probe 1, a cross-backend comparison that makes no invariance claim.

Files: [`spec_none.json`](spec_none.json), [`spec2_ngram.json`](spec2_ngram.json), [`spec2_eagle3.json`](spec2_eagle3.json), [`spec2_eagle3_triton_attn.json`](spec2_eagle3_triton_attn.json), [`spec2_compare.txt`](spec2_compare.txt). The first n-gram/EAGLE3 pass ran with stats disabled (`spec_*.json`) and gave the same 36/36 result.

## 3. FlashInfer FA3 and trtllm-gen

On H100, vLLM's FlashInfer backend builds its prefill wrapper with `backend="auto"`. FlashInfer resolves that to `fa3` for bf16. In FlashInfer 0.7.2.post1 (`flashinfer/prefill.py`), the batch-invariance arguments `fixed_split_size` and `disable_split_kv` are forwarded only to `fa2` plans, so vLLM's invariant mode passes them to FA3 without effect.

The new `run3/` cases ran FA3 exactly that way. They used vLLM's batch-invariant plan arguments (prefill `fixed_split_size=4096`, decode 2048, `disable_split_kv=True`), checked that `plan()` resolved to `fa3`, and also ran an explicit `fa3` control.

- **Shapes:** 4/4 D64; GQA/MQA 32/8, 64/8, 8/1 at D128/D256; windows 1024/4096; FP8 Q+KV (FA3 requires FP8 Q); FP8 with a window; and FA3 decode in bf16 and FP8.
- **Perturbations:** repeat, batch, position, companion KV length, page size, chunked prefill, and CUDA-graph replay against eager.
- **Result: 21 cases, 1,065 checks, 0 divergences, 0 errors** ([`flashinfer_run3.md`](flashinfer_run3.md)). FA3 in this setup does not need the split controls it ignores.
- **trtllm-gen** ships SM100/SM103 cubins only. vLLM also disables TRTLLM and XQA attention when `VLLM_BATCH_INVARIANT=1` (`supports_trtllm_attention`). The case records an explicit skip; testing it would need a B200.

## Recommended scope for a TP=2 all-reduce test

The single-GPU results do not justify renting 2 GPUs. My reading of vLLM 0.33.0's source, not tested on hardware:

- **TP=2 always uses the 1-stage custom all-reduce**, which sums ranks in a fixed order, so there is nothing size-dependent to test at TP=2.
- **At TP=4 and 8,** the size-based switch to 2-stage summed partitions starting at the owner's rank. [vllm#57042](https://github.com/vllm-project/vllm/pull/57042), merged on main 2026-11-04 after the 0.33.0 tag, changed that to absolute rank order. It is already known and fixed upstream.

## Cost

Instance 59948100 (H100 NVL, $3.44/h listed) was up from 20:24:47Z to 21:01:56Z, 37 minutes. Vast invoices: GPU 0.508 h at $3.387/h = $1.72, storage $0.03, upload $0.08, total about $1.84. The box was destroyed and confirmed gone from `vastai show instances`.
