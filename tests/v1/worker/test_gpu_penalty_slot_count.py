# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PenaltiesState.num_penalty_slots gates the per-token bin-count update.

_post_update_kernel increments output_bin_counts for every sampled token. Only
the penalties kernel reads those counts, so when no slot holds a request that
uses penalties the model runner passes None and the kernel skips the
read-modify-write.

The count must be maintained as a delta: PenaltiesState has no remove_request,
so slots are recycled by overwriting them in add_request. A plain increment
would never come back down.
"""

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA required for sampler state tests", allow_module_level=True)

from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.sample.penalties import PenaltiesState
from vllm.v1.worker.gpu.states import RequestState

DEVICE = torch.device("cuda")
VOCAB_SIZE = 128
MAX_NUM_REQS = 4

NO_PENALTY = SamplingParams()
FREQ_PENALTY = SamplingParams(frequency_penalty=0.5)
PRESENCE_PENALTY = SamplingParams(presence_penalty=0.5)


def _make_state() -> PenaltiesState:
    req_states = RequestState(
        max_num_reqs=MAX_NUM_REQS,
        max_model_len=64,
        max_num_batched_tokens=16,
        num_speculative_steps=1,
        vocab_size=VOCAB_SIZE,
        device=DEVICE,
    )
    return PenaltiesState(req_states)


def test_starts_at_zero():
    assert _make_state().num_penalty_slots == 0


@pytest.mark.parametrize("params", [FREQ_PENALTY, PRESENCE_PENALTY])
def test_counts_requests_that_use_penalties(params: SamplingParams):
    state = _make_state()
    state.add_request(0, params)
    assert state.num_penalty_slots == 1
    state.add_request(1, params)
    assert state.num_penalty_slots == 2


def test_plain_requests_do_not_count():
    state = _make_state()
    for req_idx in range(MAX_NUM_REQS):
        state.add_request(req_idx, NO_PENALTY)
    assert state.num_penalty_slots == 0


def test_slot_reuse_decrements():
    """The regression this exists for: a recycled slot must give its count back."""
    state = _make_state()
    state.add_request(0, FREQ_PENALTY)
    assert state.num_penalty_slots == 1

    # A new request lands in the same slot and does not use penalties.
    state.add_request(0, NO_PENALTY)
    assert state.num_penalty_slots == 0
    assert not state.use_penalty[0]


def test_count_tracks_use_penalty_exactly():
    """Whatever the sequence of slot assignments, the scalar must agree with
    the array it summarizes."""
    state = _make_state()
    script = [
        (0, FREQ_PENALTY),
        (1, NO_PENALTY),
        (2, PRESENCE_PENALTY),
        (0, NO_PENALTY),
        (3, FREQ_PENALTY),
        (2, FREQ_PENALTY),
        (3, NO_PENALTY),
        (1, FREQ_PENALTY),
    ]
    for req_idx, params in script:
        state.add_request(req_idx, params)
        assert state.num_penalty_slots == int(state.use_penalty.sum()), script
