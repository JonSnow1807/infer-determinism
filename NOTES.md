# Harness notes

## `vllm_attention_layer`: deferred for run 2

No target with this name is registered for run 2: construction without a model is too dependent on version-specific worker state to validate in this CPU environment. Raw FA and FlashInfer results do not cover vLLM's backend selection, native metadata construction, cache update, or forward-context execution.

A pinned, eager, single-backend layer probe is feasible, but a faithful test of both FLASH_ATTN and FLASHINFER under `VLLM_BATCH_INVARIANT=1` needs more than constructing an implementation object:

- The selector consumes active model, cache, attention and platform configuration. Directly importing an implementation would bypass the selection being tested.
- Both metadata builders consume an attention KV-cache specification, registered layer names, a `VllmConfig`, device and scheduler limits. FlashInfer additionally creates wrappers/workspaces and tracks graph capacity. Head counts, scales, sliding-window configuration and layouts are resolved from layers/configuration.
- Real forward execution needs nonempty attention metadata, populated KV cache, block tables, query prefix sums, sequence lengths, valid slot mappings, and the matching forward context. Graph execution must update the native builder's persistent buffers, not replace them with raw-kernel metadata.

These are observable pitfalls rather than hypothetical dependencies: the inspected [FlashAttention backend](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py) returns zeros when metadata is absent (profiling), and the [attention layer's cache-update path](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/attention/attention.py) skips writes if the layer's slot mapping is absent. An equality-only mock layer could therefore pass without doing the requested work. The [FlashInfer builder](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flashinfer.py) has its own configuration and graph wrapper lifecycle. These internals are version-sensitive; neither library is installed in this CPU environment.

The native FA invariant policy also uses the configured model maximum length and disables AOT scheduling, in addition to selecting one split. The raw adapter's `num_splits=1` is only one of those controls. This distinction matters when interpreting a sliding-window divergence near [#65587](https://github.com/vllm-project/vllm/issues/65587).

For a later native target, pin the H100 wheel/source revision, create an entirely local tiny model configuration and registered attention layer, let vLLM select each requested backend, build real metadata/cache state, and validate one eager output against a masked mathematical reference before testing invariance. Verify nonzero finite outputs, cache writes and the selected implementation, then exercise the native capture/replay metadata lifecycle. Run each backend in a fresh process with the environment set before importing vLLM. This requires GPU validation; it should not be advertised as native coverage until those checks run.
