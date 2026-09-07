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


def test_greedy_fast_path_filters_logits_for_sampling_mask():
    """processed_logits feeds SamplingMaskTensors even without logprobs.

    Regression test: the fast path may only skip top-k/top-p when nothing
    downstream consumes processed_logits. torch.argmax is unaffected by the
    masking, but the returned tensor is not.
    """
    torch.manual_seed(0)
    logits = torch.randn(2, VOCAB_SIZE, dtype=torch.float32, device=DEVICE)

    sampler = _make_sampler(return_sampling_mask=True)
    for i in range(2):
        sampler.add_request(i, 1, SamplingParams(temperature=0.0, top_k=4))
    sampled, processed_logits = _sample(sampler, logits.clone(), return_logprobs=False)

    # top_k=4 must have masked everything outside the top 4 logits per row.
    kept = torch.isfinite(processed_logits).sum(dim=-1)
    assert torch.equal(kept, torch.full_like(kept, 4))
    # ...and the argmax is unchanged by that masking.
    assert torch.equal(sampled, logits.argmax(dim=-1))
