# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the pure-decode num_sampled/num_rejected bypass in the GPU sampler."""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler tests", allow_module_level=True)

from vllm.v1.worker.gpu.input_batch import get_num_sampled_and_rejected
from vllm.v1.worker.gpu.sample.sampler import Sampler
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
MAX_NUM_REQS = 8


class MockReasoningConfig:
    reasoning_start_token_ids = [90]
    reasoning_end_token_ids = [91]
    natural_reasoning_end_token_ids = [91]


def _make_sampler() -> Sampler:
    req_states = RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=1024,
        max_num_batched_tokens=64,
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
    )


def _run_kernel(seq_lens: list[int], prefill_lens: list[int]):
    """Call the real kernel the way Sampler.__call__ does."""
    num_reqs = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEVICE)
    return get_num_sampled_and_rejected(
        seq_lens_t.new_ones(num_reqs),
        seq_lens_t,
        # One logit per request, as in the non-speculative sampling path.
        torch.arange(num_reqs + 1, dtype=torch.int32, device=DEVICE),
        torch.arange(num_reqs, dtype=torch.int64, device=DEVICE),
        torch.tensor(prefill_lens, dtype=torch.int32, device=DEVICE),
    )


@pytest.mark.parametrize("num_reqs", [1, 3, MAX_NUM_REQS])
def test_preallocated_decode_buffers_match_kernel(num_reqs: int):
    """The buffers handed out when has_prefill is False must equal what the
    kernel would have produced for a batch where every request has finished
    prefill (seq_len >= prefill_len).

    This compares the sampler's real state against the real kernel, so a wrong
    dtype, a wrong fill value or a wrong slice length all fail here.
    """
    sampler = _make_sampler()
    # seq_len >= prefill_len for every request, including the seq_len == prefill_len
    # boundary, which is the case the kernel's strict `<` comparison hinges on.
    seq_lens = [10 + 7 * i for i in range(num_reqs)]
    prefill_lens = [10 + 7 * i if i % 2 else 5 for i in range(num_reqs)]

    kernel_sampled, kernel_rejected = _run_kernel(seq_lens, prefill_lens)
    bypass_sampled = sampler._decode_num_sampled[:num_reqs]
    bypass_rejected = sampler._decode_num_rejected[:num_reqs]

    assert torch.equal(bypass_sampled, kernel_sampled)
    assert torch.equal(bypass_rejected, kernel_rejected)
    assert bypass_sampled.dtype == kernel_sampled.dtype
    assert bypass_rejected.dtype == kernel_rejected.dtype


def test_decode_buffers_are_shared_views():
    """The buffers are reused across steps, so consumers must treat them as
    read-only. Pin that down: successive slices alias one allocation."""
    sampler = _make_sampler()
    a = sampler._decode_num_sampled[:2]
    b = sampler._decode_num_sampled[:5]

    assert a.data_ptr() == b.data_ptr()
    assert a.data_ptr() == sampler._decode_num_sampled.data_ptr()


def test_chunked_prefill_still_goes_through_the_kernel():
    """The bypass is only valid when has_prefill is False. When a request is
    still prefilling the kernel must zero its sampled count -- which the
    constant buffers cannot express."""
    # req 0 decodes (15 >= 10); reqs 1 and 2 are mid-prefill.
    kernel_sampled, kernel_rejected = _run_kernel([15, 10, 25], [10, 20, 50])

    assert torch.equal(
        kernel_sampled, torch.tensor([1, 0, 0], dtype=torch.int32, device=DEVICE)
    )
    assert torch.equal(
        kernel_rejected, torch.tensor([0, 0, 0], dtype=torch.int32, device=DEVICE)
    )
