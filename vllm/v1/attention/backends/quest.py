# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QuEST attention backend.

The current implementation establishes QuEST-owned execution routing while
remaining numerically equivalent to dense FlashAttention.

Sparse page selection and sparse KV reads are added in later stages.
"""

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

QUEST_NUM_DENSE_LAYERS = 2


class QuestAttentionImpl(FlashAttentionImpl):
    """QuEST implementation with an initial dense-compatible execution path."""

    def _forward_dense(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute the existing dense FlashAttention implementation."""

        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run QuEST attention.

        At this checkpoint, all branches remain dense:

        - layers 0 and 1 use the permanently dense branch;
        - layers 2 and above use the QuEST 100%-budget branch, which currently
          delegates to dense FlashAttention.

        This separation establishes the control flow that sparse decode will
        replace later.
        """

        layer_index = extract_layer_index(layer.layer_name)

        if layer_index < QUEST_NUM_DENSE_LAYERS:
            return self._forward_dense(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        # QuEST 100%-budget reference path.
        #
        # This deliberately reads the full KV cache through FlashAttention.
        # Sparse page selection will replace only this branch in later stages.
        return self._forward_dense(
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )


class QuestAttentionBackend(FlashAttentionBackend):
    """QuEST backend reusing the FlashAttention KV-cache contract."""

    @staticmethod
    def get_name() -> str:
        return "QUEST"

    @staticmethod
    def get_impl_cls() -> type[QuestAttentionImpl]:
        return QuestAttentionImpl
