# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for FlashAttentionImpl._get_descales.

Descales are only valid for an FP8 KV cache; for an unquantized cache the
backend must pass None instead of expanding layer._{q,k,v}_scale.
"""

import torch

from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

NUM_KV_HEADS = 4
NUM_SEQS = 3


class _StubLayer:
    """Stands in for an Attention layer, which only contributes the scales."""

    def __init__(self):
        self._q_scale = torch.tensor(0.5)
        self._k_scale = torch.tensor(2.0)
        self._v_scale = torch.tensor(4.0)


def _make_impl(kv_cache_dtype: str, supports_quant_query_input: bool):
    # FlashAttentionImpl.__init__ probes the platform and the compiled FA
    # extension; _get_descales only reads these three attributes.
    impl = FlashAttentionImpl.__new__(FlashAttentionImpl)
    impl.kv_cache_dtype = kv_cache_dtype
    impl.num_kv_heads = NUM_KV_HEADS
    impl.supports_quant_query_input = supports_quant_query_input
    return impl


def test_unquantized_cache_yields_no_descales():
    for kv_cache_dtype in ("auto", "bfloat16", "float16"):
        impl = _make_impl(kv_cache_dtype, supports_quant_query_input=True)
        assert impl._get_descales(_StubLayer(), num_seqs=NUM_SEQS) == (
            None,
            None,
            None,
        )


def test_quantized_cache_expands_scales():
    layer = _StubLayer()
    impl = _make_impl("fp8", supports_quant_query_input=True)

    q_descale, k_descale, v_descale = impl._get_descales(layer, num_seqs=NUM_SEQS)

    for descale, scale in (
        (q_descale, layer._q_scale),
        (k_descale, layer._k_scale),
        (v_descale, layer._v_scale),
    ):
        assert descale is not None
        assert descale.shape == (NUM_SEQS, NUM_KV_HEADS)
        # expand() must stay a broadcast view, not a materialized copy.
        assert descale.data_ptr() == scale.data_ptr()
        assert torch.equal(descale, scale.expand(NUM_SEQS, NUM_KV_HEADS))


def test_quantized_cache_drops_query_descale_when_unsupported():
    impl = _make_impl("fp8", supports_quant_query_input=False)

    q_descale, k_descale, v_descale = impl._get_descales(
        _StubLayer(), num_seqs=NUM_SEQS
    )

    assert q_descale is None
    assert k_descale is not None
    assert v_descale is not None
