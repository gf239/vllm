# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for per-request effective proposal lengths in adaptive speculative
decoding (RFC #48202)."""

import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

if "openai_harmony" not in sys.modules:
    sys.modules["openai_harmony"] = MagicMock()

import torch

from vllm.config.speculative import SpeculativeConfig
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.spec_decode.ngram_proposer_gpu import update_scheduler_for_invalid_drafts
from vllm.v1.spec_decode.utils import compute_adaptive_valid_draft_tokens


class TestAdaptiveProposalLength(unittest.TestCase):
    def test_speculative_config_acceptance_threshold_validation(self):
        """Test validation of draft_token_acceptance_threshold in SpeculativeConfig."""
        # Valid values
        for val in [0.0, 0.25, 0.5, 0.8, 1.0]:
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = val
            cfg.draft_token_acceptance_mode = "cumulative"
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()
            self.assertTrue(cfg.uses_adaptive_proposal_length())

        # Unset / None
        cfg = SpeculativeConfig.__new__(SpeculativeConfig)
        cfg.draft_token_acceptance_threshold = None
        cfg.draft_token_acceptance_mode = "cumulative"
        cfg.tensor_parallel_size = None
        cfg.num_speculative_tokens = 3
        cfg.rejection_sample_method = "standard"
        cfg.synthetic_acceptance_rates = None
        cfg.synthetic_acceptance_length = None
        cfg.draft_model_config = None
        cfg.use_heterogeneous_vocab = False
        cfg._verify_args()
        self.assertFalse(cfg.uses_adaptive_proposal_length())

        # Out of bounds (< 0.0)
        with self.assertRaises(ValueError):
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = -0.1
            cfg.draft_token_acceptance_mode = "cumulative"
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()

        # Out of bounds (> 1.0)
        with self.assertRaises(ValueError):
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = 1.05
            cfg.draft_token_acceptance_mode = "cumulative"
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()

        # Incompatible with use_local_argmax_reduction
        with self.assertRaises(ValueError):
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = 0.5
            cfg.draft_token_acceptance_mode = "cumulative"
            cfg.use_local_argmax_reduction = True
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()

    def test_speculative_config_mode_validation(self):
        """Test validation of draft_token_acceptance_mode in SpeculativeConfig."""
        for valid_mode in ["token_threshold", "cumulative", "cudagraph_aligned"]:
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = 0.5
            cfg.draft_token_acceptance_mode = valid_mode
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()
            self.assertEqual(cfg.draft_token_acceptance_mode, valid_mode)

        # Invalid mode
        with self.assertRaises(ValueError):
            cfg = SpeculativeConfig.__new__(SpeculativeConfig)
            cfg.draft_token_acceptance_threshold = 0.5
            cfg.draft_token_acceptance_mode = "invalid_mode"
            cfg.tensor_parallel_size = None
            cfg.num_speculative_tokens = 3
            cfg.rejection_sample_method = "standard"
            cfg.synthetic_acceptance_rates = None
            cfg.synthetic_acceptance_length = None
            cfg.draft_model_config = None
            cfg.use_heterogeneous_vocab = False
            cfg._verify_args()

    def test_compute_adaptive_valid_draft_tokens_token_threshold_mode(self):
        """Test mode='token_threshold' (per-token marginal probability)."""
        # Batch of 3 requests, K = 3, threshold = 0.6
        # Req 0: [0.9, 0.8, 0.7] -> All >= 0.6 -> valid_k = 3
        # Req 1: [0.8, 0.4, 0.9] -> Token 1 drops -> valid_k = 1
        # Req 2: [0.3, 0.9, 0.9] -> Token 0 drops -> valid_k = 0
        confidences = torch.tensor(
            [
                [0.9, 0.8, 0.7],
                [0.8, 0.4, 0.9],
                [0.3, 0.9, 0.9],
            ],
            dtype=torch.float32,
        )
        num_valid, valid_mask = compute_adaptive_valid_draft_tokens(
            confidences, threshold=0.6, mode="token_threshold"
        )

        self.assertEqual(num_valid.tolist(), [3, 1, 0])
        self.assertEqual(
            valid_mask.tolist(),
            [
                [True, True, True],
                [True, False, False],
                [False, False, False],
            ],
        )

    def test_compute_adaptive_valid_draft_tokens_cumulative_mode(self):
        """Test mode='cumulative' (joint prefix survival probability)."""
        # Batch of 3 requests, K = 3, threshold = 0.6
        # Req 0: [0.9, 0.8, 0.7] -> cumprod: [0.9, 0.72, 0.504] -> valid_k = 2
        # Req 1: [0.8, 0.4, 0.9] -> cumprod: [0.8, 0.32, 0.288] -> valid_k = 1
        # Req 2: [0.3, 0.9, 0.9] -> cumprod: [0.3, 0.27, 0.243] -> valid_k = 0
        confidences = torch.tensor(
            [
                [0.9, 0.8, 0.7],
                [0.8, 0.4, 0.9],
                [0.3, 0.9, 0.9],
            ],
            dtype=torch.float32,
        )
        num_valid, valid_mask = compute_adaptive_valid_draft_tokens(
            confidences, threshold=0.6, mode="cumulative"
        )

        self.assertEqual(num_valid.tolist(), [2, 1, 0])
        self.assertEqual(
            valid_mask.tolist(),
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
            ],
        )

    def test_compute_adaptive_valid_draft_tokens_cudagraph_aligned_mode(self):
        """Test mode='cudagraph_aligned' (cumulative count snapped up to nearest
        CUDA graph bucket)."""
        # Batch of 3 requests, K = 3, threshold = 0.6, buckets = [2, 3]
        # Base cumulative counts are [2, 1, 0]
        # Req 0: base_k = 2 -> snaps to bucket 2
        # Req 1: base_k = 1 -> snaps to bucket 2
        # Req 2: base_k = 0 -> stays 0
        confidences = torch.tensor(
            [
                [0.9, 0.8, 0.7],
                [0.8, 0.4, 0.9],
                [0.3, 0.9, 0.9],
            ],
            dtype=torch.float32,
        )
        num_valid, valid_mask = compute_adaptive_valid_draft_tokens(
            confidences,
            threshold=0.6,
            mode="cudagraph_aligned",
            cudagraph_buckets=[2, 3],
        )

        self.assertEqual(num_valid.tolist(), [2, 2, 0])
        self.assertEqual(
            valid_mask.tolist(),
            [
                [True, True, False],
                [True, True, False],
                [False, False, False],
            ],
        )

    def test_compute_adaptive_valid_draft_tokens_edge_cases(self):
        """Test empty inputs and invalid mode handling."""
        # Empty tensor
        empty_conf = torch.empty((0, 3), dtype=torch.float32)
        num_valid, valid_mask = compute_adaptive_valid_draft_tokens(
            empty_conf, threshold=0.6, mode="cumulative"
        )
        self.assertEqual(num_valid.numel(), 0)
        self.assertEqual(valid_mask.numel(), 0)

        # Invalid mode
        conf = torch.tensor([[0.9, 0.8]], dtype=torch.float32)
        with self.assertRaises(ValueError):
            compute_adaptive_valid_draft_tokens(
                conf, threshold=0.6, mode="nonexistent_mode"
            )

    def test_update_scheduler_for_invalid_drafts(self):
        """Test scheduler trimming for variable proposal lengths across requests."""
        num_valid_event = MagicMock()
        # Mock CPU buffer returning valid counts: req_0 -> 3, req_1 -> 1, req_2 -> 0
        mock_cpu_counts = [
            SimpleNamespace(item=lambda: 3),
            SimpleNamespace(item=lambda: 1),
            SimpleNamespace(item=lambda: 0),
        ]

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=SimpleNamespace(req_ids=["req_0", "req_1", "req_2"]),
            num_scheduled_tokens={"req_0": 4, "req_1": 4, "req_2": 4},
            total_num_scheduled_tokens=12,
            scheduled_spec_decode_tokens={
                "req_0": [101, 102, 103],
                "req_1": [201, 202, 203],
                "req_2": [301, 302, 303],
            },
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0],
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
        )

        req_id_to_index = {"req_0": 0, "req_1": 1, "req_2": 2}

        update_scheduler_for_invalid_drafts(
            num_valid_draft_tokens_event=num_valid_event,
            num_valid_draft_tokens_cpu=mock_cpu_counts,
            scheduler_output=scheduler_output,
            req_id_to_index=req_id_to_index,
        )

        num_valid_event.synchronize.assert_called_once()

        # Req 0 kept all 3 tokens
        self.assertEqual(
            scheduler_output.scheduled_spec_decode_tokens["req_0"], [101, 102, 103]
        )
        self.assertEqual(scheduler_output.num_scheduled_tokens["req_0"], 4)

        # Req 1 trimmed to 1 token
        self.assertEqual(scheduler_output.scheduled_spec_decode_tokens["req_1"], [201])
        self.assertEqual(scheduler_output.num_scheduled_tokens["req_1"], 2)

        # Req 2 has 0 valid tokens -> removed from spec_decode dict
        self.assertNotIn("req_2", scheduler_output.scheduled_spec_decode_tokens)
        self.assertEqual(scheduler_output.num_scheduled_tokens["req_2"], 1)

        # Total tokens trimmed = (3-3) + (3-1) + (3-0) = 0 + 2 + 3 = 5 tokens trimmed
        # 12 - 5 = 7 tokens remaining
        self.assertEqual(scheduler_output.total_num_scheduled_tokens, 7)

    def test_sample_draft_tokens_with_confidence(self):
        """Test draft token sampling with confidence and ensure single logits
        evaluation."""
        from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

        proposer = SpecDecodeBaseProposer.__new__(SpecDecodeBaseProposer)
        proposer.draft_token_acceptance_threshold = 0.5
        proposer._enable_probabilistic_draft_probs = False
        proposer.use_heterogeneous_vocab = False
        proposer.vocab_mapping = None
        proposer.use_local_argmax_reduction = False

        mock_model = MagicMock()
        mock_model.compute_logits.return_value = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float32
        )
        proposer.model = mock_model

        sampling_metadata = SimpleNamespace(all_greedy=True)
        hidden_states = torch.empty((1, 16), dtype=torch.float32)

        draft_token_ids, draft_probs, confidences = (
            proposer._sample_draft_tokens_with_confidence(
                hidden_states, sampling_metadata
            )
        )

        mock_model.compute_logits.assert_called_once()
        self.assertEqual(draft_token_ids.tolist(), [2])
        self.assertIsNone(draft_probs)
        expected_prob = torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=-1)[2].item()
        self.assertAlmostEqual(confidences.item(), expected_prob, places=4)

        # When threshold is None, confidence tracking is bypassed
        proposer.draft_token_acceptance_threshold = None
        mock_model.reset_mock()
        mock_model.compute_logits.return_value = torch.tensor(
            [[1.0, 2.0, 3.0]], dtype=torch.float32
        )
        draft_token_ids, draft_probs, confidences = (
            proposer._sample_draft_tokens_with_confidence(
                hidden_states, sampling_metadata
            )
        )
        self.assertIsNone(confidences)
        self.assertEqual(draft_token_ids.tolist(), [2])

    def test_get_draft_token_ids_cpu_strips_negative_tokens(self):
        """Test that _get_draft_token_ids_cpu strips negative (masked) tokens so that
        the CPU scheduler and grammar verifier never encounter invalid -1 IDs."""
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        runner = GPUModelRunner.__new__(GPUModelRunner)
        runner.draft_token_ids_event = MagicMock()

        # Case 1: draft_token_ids is a torch.Tensor on CPU buffer with -1 padding
        runner._draft_token_ids = torch.tensor(
            [[101, 102, -1], [201, -1, -1]], dtype=torch.int64
        )
        runner._draft_token_req_ids = ["req_0", "req_1"]
        runner.draft_token_ids_cpu = torch.tensor(
            [[101, 102, -1], [201, -1, -1]], dtype=torch.int64
        )
        draft_tokens, req_ids = runner._get_draft_token_ids_cpu()
        runner.draft_token_ids_event.synchronize.assert_called_once()
        self.assertEqual(req_ids, ["req_0", "req_1"])
        self.assertEqual(draft_tokens, [[101, 102], [201]])

        # Case 2: draft_token_ids is a python list with -1 values
        runner._draft_token_ids = [[301, -1], [-1, -1]]
        runner.input_batch = SimpleNamespace(req_ids=["req_2", "req_3"])
        draft_tokens_list, req_ids_list = runner._get_draft_token_ids_cpu()
        self.assertEqual(req_ids_list, ["req_2", "req_3"])
        self.assertEqual(draft_tokens_list, [[301], []])


if __name__ == "__main__":
    unittest.main()
