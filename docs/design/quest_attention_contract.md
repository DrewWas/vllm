# QuEST Attention Implementation Contract

## Baseline

- vLLM version: 0.25.0
- Model runner: V1 only (`VLLM_USE_V2_MODEL_RUNNER=0` is required)
- Baseline commit: 702f4814fe54fabff350d43cb753ae3e47c0c276
- Model: meta-llama/Llama-3.1-8B-Instruct
- GPU: NVIDIA H100 80GB HBM3
- Weight dtype: BF16
- KV-cache dtype: BF16
- Tensor parallelism: 1
- Maximum active sequences: 1
- CUDA graphs: disabled
- Prefix caching: disabled
- Preemption: excluded from the initial implementation
- Speculative decoding: disabled

## Initial QuEST scope

QuEST is used only during autoregressive decode.

Prefill remains unchanged and uses the existing dense FlashAttention path.

Transformer layers 0 and 1 remain dense during decode.

Transformer layers 2 through 31 use QuEST sparse decode attention.

The full K and V cache remains allocated in HBM. QuEST does not evict,
delete, compress, or offload ordinary KV data in the initial implementation.

The optimization target is reduced HBM traffic during decode, not reduced
KV-cache capacity.

## Page layout

- Logical QuEST page size: 16 tokens
- QuEST pages align with vLLM KV-cache blocks
- No separate QuEST paging allocator is introduced
- Existing vLLM block tables and slot mappings remain authoritative

Every physical page receives sidecar key metadata:

- elementwise minimum key vector
- elementwise maximum key vector
- valid-token count
- initialization state

Metadata is stored independently for every:

- transformer layer
- physical KV block
- KV head
- key dimension

## Model dimensions

- Transformer layers: 32
- Query heads: 32
- KV heads: 8
- Head dimension: 128
- Query heads per KV head: 4

The query-to-KV-head mapping is:

    kv_head = query_head // 4

Page selection is performed independently for each query head.

## Page scoring

For query q and page p, the QuEST page score is:

    score(p, q) =
        sum_d max(
            q[d] * page_min_key[p, d],
            q[d] * page_max_key[p, d]
        )

The highest-scoring pages are selected subject to the configured budget.

## Budget

The backend supports:

- fixed page-count budgets
- percentage-based page budgets
- 100% budget for dense-equivalence testing

Initial evaluation budgets:

- 100%
- 10%
- 5%
- 2%
- 1%

The selected page count must never be zero for a nonempty sequence.

The current partially filled page must be eligible for selection.

## Attention semantics

The sparse path performs ordinary scaled dot-product softmax attention over
tokens contained in the selected pages.

The initial implementation must preserve:

- RoPE-transformed key semantics
- GQA head mapping
- causal masking
- partial-page masking
- ordinary value aggregation
- output shape and dtype
- existing KV write behavior

No gathered dense copy of the entire KV cache may be created.

## Backend architecture

QuEST is implemented as a separate vLLM attention backend.

The backend delegates:

- prefill to the existing dense implementation
- layers 0 and 1 decode to dense attention
- 100%-budget decode to the dense-compatible reference path
- sparse decode in later layers to QuEST

The existing FlashAttention backend is not directly modified unless a small,
general integration hook is required.

## Implementation order

1. Backend registration and configuration
2. Dense delegation
3. 100%-budget equivalence path
4. Sidecar page metadata allocation
5. Metadata update and validation
6. Page scoring reference implementation
7. GPU page scoring and top-k
8. Correctness-first Triton sparse attention
9. Profiling and optimization

## Correctness gates

The implementation cannot advance past a phase until its gate passes.

- Dense backend selection remains unchanged when QuEST is disabled
- QuEST 100% budget matches dense BF16 output within an agreed tolerance
- Page metadata matches an offline recomputation
- GPU page scores match a CPU reference
- Selected page IDs match the reference selector
- Sparse attention matches an offline selected-page implementation
- No unselected full K/V page is read by the final sparse kernel
- Existing KV data remains intact

## Excluded from the initial implementation

- CPU KV offload
- KV eviction
- KV compression or quantization
- Multiple concurrent requests
- continuous batching
- prefix caching
- request preemption
- speculative decoding
- tensor parallelism
- pipeline parallelism
- CUDA graph capture
- custom CUDA or C++ kernels

These features are added only after single-request correctness is established.

## Confirmed vLLM integration surface

The initial implementation adds a named `QUEST` attention backend.

Expected new files:

- `vllm/v1/attention/backends/quest.py`
- `vllm/v1/attention/ops/quest/__init__.py`
- `vllm/v1/attention/ops/quest/page_score.py`
- `vllm/v1/attention/ops/quest/sparse_attention.py`

Expected initial modification:

- `vllm/v1/attention/backends/registry.py`

The QuEST backend reuses:

- FlashAttention KV-cache layout
- FlashAttention metadata builder
- existing block tables
- existing slot mappings
- existing KV-write behavior

The initial implementation does not modify:

- the scheduler
- `KVCacheManager`
- the Llama model
- block-table allocation
- FlashAttention source

The backend identifies the transformer layer from:

    layer.layer_name

and extracts its numerical layer index using vLLM's existing
`extract_layer_index()` helper.

Decode policy:

    layers 0–1: dense FlashAttention
    layers 2–31: QuEST when sparse mode is enabled
