# [Spec Decode][V1] Support per-request effective proposal lengths for adaptive speculative decoding (#48202)

## Description

This PR implements reference support for **per-request effective proposal lengths in adaptive speculative decoding**, addressing **RFC #48202** and **Issue #36657**.

### Motivation
In current speculative decoding setups with draft models (Eagle, DraftModel, MTP, etc.), the speculation depth $K$ is statically uniform across all requests in a batch. In heterogeneous or multi-turn workloads:
- For low-confidence ("cold") requests, draft tokens beyond the point of divergence are destined to be rejected by the target model, needlessly consuming verification memory bandwidth and compute.
- Uniform drafting prevents the scheduler and target verifier from saving work on difficult prompts.

### Proposed Changes
1. **`SpeculativeConfig`**:
   - Added `draft_token_acceptance_threshold: float | None = None` to specify the minimum draft confidence threshold (in `[0.0, 1.0]`).
   - Added `uses_adaptive_proposal_length() -> bool` helper.
   - Added validation enforcing threshold bounds.

2. **`vllm.v1.spec_decode.utils`**:
   - Added `compute_adaptive_valid_draft_tokens(confidences, threshold)` to compute per-request valid draft counts and boolean mask using vectorized cumulative product logic on GPU without host-device synchronization:
     $$\text{valid\_k}_i = \sum_{s=0}^{K-1} \prod_{j=0}^{s} \mathbb{I}(p_{i, j} \ge \text{threshold})$$

3. **`LLMBaseProposer` (`vllm.v1.spec_decode.llm_base_proposer`)**:
   - Extended draft sampling (`_sample_draft_tokens_with_confidence`) to track top-1 / sampled token confidence across draft steps when thresholding is enabled.
   - Computed `num_valid_draft_tokens` tensor (`[batch_size]`) and masked invalid draft tokens with `-1`.
   - Exposed `take_last_num_valid_draft_tokens()`.

4. **`GPUModelRunner` (`vllm.v1.worker.gpu_model_runner`)**:
   - Enabled async D2H buffer (`_num_valid_draft_tokens_cpu`), copy stream, and event when `uses_adaptive_proposal_length()` is active.
   - Gathered valid counts from drafter and copied via non-blocking stream.
   - Leveraged existing `update_scheduler_for_invalid_drafts` to trim `scheduled_spec_decode_tokens` and token counts per request in-place before target model execution.

5. **Testing**:
   - Added comprehensive tests in `tests/v1/spec_decode/test_adaptive_proposal_length.py` testing config bounds, truncation masking, and scheduler trimming.

---

## Test Plan
- Run unit tests:
  ```bash
  pytest tests/v1/spec_decode/test_adaptive_proposal_length.py
  ```
- Run speculative decoding regression suite:
  ```bash
  pytest tests/v1/spec_decode/test_llm_base_proposer.py
  pytest tests/v1/spec_decode/test_dynamic_sd.py
  ```

## Benchmarks & Performance
- Zero regression when `draft_token_acceptance_threshold` is `None` (standard fixed-K execution path).
- Trims rejected tails on device with non-blocking async D2H synchronization, preventing verification overhead on low-confidence branches.
