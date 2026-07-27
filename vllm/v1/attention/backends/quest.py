# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QuEST attention backend.

The initial scaffold is intentionally dense-compatible. It inherits the
FlashAttention metadata, KV-cache layout, and forward implementation.

QuEST-specific page metadata, selection, and sparse attention will be added
incrementally after backend registration is validated.
"""

from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
)


class QuestAttentionImpl(FlashAttentionImpl):
    """Dense-compatible implementation scaffold for QuEST."""

    pass


class QuestAttentionBackend(FlashAttentionBackend):
    """QuEST backend using FlashAttention behavior during initial bring-up."""

    @staticmethod
    def get_name() -> str:
        return "QUEST"

    @staticmethod
    def get_impl_cls() -> type[QuestAttentionImpl]:
        return QuestAttentionImpl
