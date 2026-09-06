# [Spec Decode][V1] Support per-request effective proposal lengths for adaptive speculative decoding (#48202)

## Description

This PR implements reference support for **per-request effective proposal lengths in adaptive speculative decoding**, addressing **RFC #48202** and **Issue #36657**.

### Motivation
In current speculative decoding setups with draft models (Eagle, DraftModel, MTP, etc.), the speculation depth $K$ is statically uniform across all requests in a batch. In heterogeneous or multi-turn workloads:
- For low-confidence ("cold") requests, draft tokens beyond the point of divergence are destined to be rejected by the target model, needlessly consuming verification memory bandwidth and compute.
- Uniform drafting prevents the scheduler and target verifier from saving work on difficult prompts.

### Proposed Changes
1. **`SpeculativeConfig` (`vllm/config/speculative.py`)**:
   - Added `draft_token_acceptance_threshold: float | None = None` to specify the minimum draft confidence threshold (in `[0.0, 1.0]`).
   - Added `draft_token_acceptance_mode: AdaptiveProposalMode = "token_threshold"` supporting:
     - `"token_threshold"` (default): Evaluates marginal per-token probabilities ($p_i \ge \tau$). Preserves strong prefix chains without compounding geometric decay.
     - `"cumulative"`: Evaluates cumulative joint prefix survival probability ($\prod_{j=1}^i p_j \ge \tau$). Note: decays geometrically with depth (e.g. $0.75^7 \approx 0.13$) and is aggressive by construction.
     - `"cudagraph_aligned"`: Joint prefix survival snapped up to the nearest CUDA graph bucket boundary for zero marginal verification overhead.
   - Added `draft_token_acceptance_cudagraph_buckets: list[int] | None = None` to allow explicitly configuring target CUDA graph buckets.
   - Added `uses_adaptive_proposal_length() -> bool` helper.
   - Enforced mutual exclusion with `use_local_argmax_reduction` in `_verify_args()` to prevent missing logits errors.

2. **`vllm.v1.spec_decode.utils` (`vllm/v1/spec_decode/utils.py`)**:
   - Implemented `compute_adaptive_valid_draft_tokens(confidences, threshold, mode="token_threshold", cudagraph_buckets=None)` to compute per-request valid draft counts and boolean mask using vectorized GPU logic without host-device synchronization.
   - Promoted `confidences` to `float32` and clamped to `[0.0, 1.0]` to guarantee IEEE 754 precision and prevent `bfloat16`/`float16` rounding drift or underflow during cumulative probability multiplications.
   - Mode `"cudagraph_aligned"` employs pure scalar broadcasting in `torch.where` to avoid runtime GPU memory allocations during bucket snapping.

3. **`LLMBaseProposer` (`vllm/v1/spec_decode/llm_base_proposer.py`)**:
   - Unified logits and confidence computation in `_sample_draft_tokens_with_confidence()`: for greedy draft generation, computes `compute_logits()` strictly once and extracts both token IDs and confidence probabilities simultaneously from `probs.max(dim=-1)`, eliminating redundant GEMM passes.
   - Implemented `_resolve_cudagraph_buckets()` to dynamically resolve real graph buckets from `speculative_config.draft_token_acceptance_cudagraph_buckets` or `compilation_config.cudagraph_capture_sizes`, passing them through to both `compute_adaptive_valid_draft_tokens()` call sites.
   - Computed `num_valid_draft_tokens` tensor (`[batch_size]`) and masked invalid draft tokens with `-1` using in-place `masked_fill_`.
   - Exposed `take_last_num_valid_draft_tokens()`.

4. **`GPUModelRunner` (`vllm/v1/worker/gpu_model_runner.py`)**:
   - Enabled async D2H buffer (`_num_valid_draft_tokens_cpu`), copy stream, and CUDA event when `uses_adaptive_proposal_length()` is active.
   - Gathered valid counts from drafter and transferred via non-blocking stream.
   - Leveraged existing `update_scheduler_for_invalid_drafts` to trim `scheduled_spec_decode_tokens` and token counts per request in-place before target model execution.
   - Sanitized draft tokens in `_get_draft_token_ids_cpu()` to filter out negative masked tokens (`[t for t in tokens if t >= 0]`), protecting CPU scheduler token accounting and grammar validation (guided decoding) from invalid token IDs.

5. **Testing (`tests/v1/spec_decode/test_adaptive_proposal_length.py`)**:
   - Added 18 comprehensive unit tests covering:
     - Valid and invalid threshold / mode configuration bounds.
     - Default mode (`token_threshold`) preserving strong chains without geometric decay.
     - Validation of `draft_token_acceptance_cudagraph_buckets` (positive values, non-empty).
     - Proposer resolution hierarchy of `cudagraph_buckets` across speculative and compilation configs.
     - Mutual exclusion between adaptive thresholding and `use_local_argmax_reduction`.
     - Strict validation rejecting unsupported speculative methods (`ngram`, `medusa`, `mlp_speculator`, `suffix`).
     - Correctness of `"token_threshold"`, `"cumulative"`, and `"cudagraph_aligned"` modes.
     - `bfloat16` and `float16` precision stability preventing cumulative underflow / truncation drift.
     - Device tensor caching and zero runtime GPU allocations during bucket snapping.
     - Graceful fallback for empty, negative, or out-of-range CUDA graph buckets.
     - Extreme boundary thresholds ($\tau = 0.0$, $\tau = 1.0$) and all-zero probability matrices.
     - Single draft token edge case ($K = 1$).
     - Multi-request heterogeneous batch with varying points of divergence.
     - Zero-overhead single-pass logits execution during greedy drafting.
     - Scheduler trimming and batch token recount across multiple requests.
     - CPU draft token sanitization removing negative mask tokens.

---

## Test Plan
- Run unit tests:
  ```bash
  python3 -m unittest tests/v1/spec_decode/test_adaptive_proposal_length.py
  ```
- Run speculative decoding regression suite:
  ```bash
  pytest tests/v1/spec_decode/test_llm_base_proposer.py
  ```

## Performance Impact
- Zero regression when `draft_token_acceptance_threshold` is `None` (standard fixed-$K$ execution path unchanged).
- Prevents redundant verification passes on low-confidence draft tokens using asynchronous device-to-host synchronization.
