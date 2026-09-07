# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler.schedule() builds the per-step skipped-waiting queue only when the
waiting loop can actually run.

The queue is a pure scratch object: it collects requests that were skipped in
this pass so they can be re-queued ahead of older skipped items. In
steady-state decode nothing is waiting, the loop body never executes, and the
allocation is dead. These tests pin down that the object is not built then, and
that skipping still behaves the same when it is.
"""

from unittest.mock import patch

from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import RequestStatus

QUEUE_FACTORY = "vllm.v1.core.sched.scheduler.create_request_queue"


def test_no_queue_built_during_steady_state_decode():
    """Nothing waiting and nothing skipped: no scratch queue is created."""
    scheduler = create_scheduler()
    requests = create_requests(num_requests=2, num_tokens=4)
    for request in requests:
        scheduler.add_request(request)

    # Drain the waiting queue: this step admits both requests.
    scheduler.schedule()
    assert not scheduler.waiting
    assert not scheduler.skipped_waiting

    # The next step is pure decode.
    with patch(QUEUE_FACTORY) as factory:
        scheduler.schedule()
    factory.assert_not_called()


def test_queue_built_when_requests_are_waiting():
    """With work in the waiting queue the loop runs, so the queue is built."""
    scheduler = create_scheduler()
    for request in create_requests(num_requests=1, num_tokens=4):
        scheduler.add_request(request)
    assert scheduler.waiting

    with patch(QUEUE_FACTORY, side_effect=create_request_queue) as factory:
        scheduler.schedule()
    factory.assert_called_once()


def test_skipped_requests_are_requeued():
    """End-to-end behavior the scratch queue exists for: a request that cannot
    be scheduled this pass lands in skipped_waiting rather than being lost."""
    # One slot only, so the second request cannot be admitted.
    scheduler = create_scheduler(max_num_seqs=1)
    requests = create_requests(num_requests=2, num_tokens=4)
    for request in requests:
        scheduler.add_request(request)

    scheduler.schedule()

    scheduled = {r.request_id for r in scheduler.running}
    assert len(scheduled) == 1
    still_pending = len(scheduler.waiting) + len(scheduler.skipped_waiting)
    assert still_pending == 1
    for request in requests:
        assert request.status != RequestStatus.FINISHED_ABORTED
