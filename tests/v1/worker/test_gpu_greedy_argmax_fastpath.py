# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the all-greedy torch.argmax fast path in the GPU sampler."""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
MAX_NUM_REQS = 4


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler(return_sampling_mask: bool = False) -> Sampler:
    req_states = RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return Sampler(
        max_num_reqs=MAX_NUM_REQS,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
        req_states=req_states,
        reasoning_config=MockReasoningConfig(),
        return_sampling_mask=return_sampling_mask,
    )


def _sample(sampler: Sampler, logits: torch.Tensor, return_logprobs: bool):
    num_reqs = logits.shape[0]
    idx_mapping_np = np.arange(num_reqs, dtype=np.int32)
    idx_mapping = torch.from_numpy(idx_mapping_np).to(DEVICE, torch.int64)
    return sampler.sample(
        logits,
        expanded_idx_mapping=idx_mapping,
        idx_mapping=idx_mapping,
        idx_mapping_np=idx_mapping_np,
        pos=torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        input_ids=torch.zeros(num_reqs, dtype=torch.int64, device=DEVICE),
        expanded_local_pos=torch.zeros(num_reqs, dtype=torch.int32, device=DEVICE),
        return_logprobs=return_logprobs,
    )


def test_all_greedy_detects_batch_composition():
    sampler = _make_sampler()
    sampler.add_request(0, 1, SamplingParams(temperature=0.0))
    sampler.add_request(1, 1, SamplingParams(temperature=0.0))
    sampler.add_request(2, 1, SamplingParams(temperature=0.7))
    states = sampler.sampling_states

    assert states.all_greedy(np.array([0, 1], dtype=np.int32))
    assert not states.all_greedy(np.array([0, 2], dtype=np.int32))
    assert not states.all_greedy(np.array([2], dtype=np.int32))
    # An empty batch is not "all greedy": there is nothing to fast-path, and
    # np.all() on an empty array would otherwise report True.
    assert not states.all_greedy(np.array([], dtype=np.int32))


def test_greedy_fast_path_matches_gumbel_path():
    """The fast path must pick the same token as gumbel_sample() at temp 0."""
    torch.manual_seed(0)
    logits = torch.randn(3, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)

    greedy = _make_sampler()
    for i in range(3):
        greedy.add_request(i, 1, SamplingParams(temperature=0.0))
    fast_sampled, _ = _sample(greedy, logits.clone(), return_logprobs=False)

    assert torch.equal(fast_sampled, logits.argmax(dim=-1))
    assert fast_sampled.dtype == torch.int64


def test_greedy_batches_carry_no_top_k_or_top_p():
    """SamplingParams strips top-k/top-p from a greedy request, so an all-greedy
    batch never reaches the masking branch of the fast path at all."""
    params = SamplingParams(temperature=0.0, top_k=4, top_p=0.9)
    assert params.top_k == 0
    assert params.top_p == 1.0

    sampler = _make_sampler()
    for i in range(2):
        sampler.add_request(i, 1, params)
    idx_mapping_np = np.arange(2, dtype=np.int32)
    top_k, top_p = sampler.sampling_states.get_top_k_top_p(
        torch.from_numpy(idx_mapping_np).to(DEVICE, torch.int64), idx_mapping_np
    )
    assert top_k is None
    assert top_p is None


def test_fast_path_filters_logits_when_a_mask_consumer_needs_them():
    """The fast path skips top-k/top-p for sampling, since masking cannot move
    the argmax, but processed_logits is also returned -- to logprobs, and to
    SamplingMaskTensors when return_sampling_mask is set. Guarding on
    return_logprobs alone would hand back unfiltered logits there.

    Reaching this through SamplingParams is impossible today (see
    test_greedy_batches_carry_no_top_k_or_top_p), so the state is set directly.
    The guard is defensive: it keeps the fast path and the gumbel path
    returning the same tensor if that normalization ever changes.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)

    sampler = _make_sampler(return_sampling_mask=True)
    for i in range(2):
        sampler.add_request(i, 1, SamplingParams(temperature=0.0))
    # Re-introduce top_k behind SamplingParams' back.
    sampler.sampling_states.top_k.np[:2] = 4
    sampler.sampling_states.top_k.copy_to_uva()

    sampled, processed_logits = _sample(sampler, logits.clone(), return_logprobs=False)

    kept = torch.isfinite(processed_logits).sum(dim=-1)
    assert torch.equal(kept, torch.full_like(kept, 4))
    assert torch.equal(sampled, logits.argmax(dim=-1))
