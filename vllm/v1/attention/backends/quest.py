# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QuEST attention backend.

The current implementation establishes QuEST-owned execution routing while
remaining numerically equivalent to dense FlashAttention.

Sparse page selection and sparse KV reads are added in later stages.
"""

import os
from typing import Any

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)

QUEST_NUM_DENSE_LAYERS = 2
QUEST_PAGE_BUDGET_PERCENT_ENV = "VLLM_QUEST_PAGE_BUDGET_PERCENT"
QUEST_DEFAULT_PAGE_BUDGET_PERCENT = 100.0


def get_quest_page_budget_percent() -> float:
    """Read and validate the configured QuEST page budget percentage."""

    raw_value = os.getenv(
        QUEST_PAGE_BUDGET_PERCENT_ENV,
        str(QUEST_DEFAULT_PAGE_BUDGET_PERCENT),
    )

    try:
        page_budget_percent = float(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{QUEST_PAGE_BUDGET_PERCENT_ENV} must be numeric, "
            f"but received {raw_value!r}."
        ) from exc

    if not 0.0 < page_budget_percent <= 100.0:
        raise ValueError(
            f"{QUEST_PAGE_BUDGET_PERCENT_ENV} must be greater than 0 "
            f"and at most 100, but received {page_budget_percent}."
        )

    return page_budget_percent


class QuestAttentionImpl(FlashAttentionImpl):
    """QuEST implementation with an initial dense-compatible execution path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.quest_page_budget_percent = get_quest_page_budget_percent()

        if self.quest_page_budget_percent != 100.0:
            raise NotImplementedError(
                "Sparse QuEST attention is not implemented yet. "
                f"{QUEST_PAGE_BUDGET_PERCENT_ENV} must remain 100, "
                f"but received {self.quest_page_budget_percent}."
            )

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

    def _forward_quest_reference(
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
        """Execute the 100%-budget QuEST reference branch.

        This remains dense until sparse page selection is implemented.
        Keeping it separate makes layer routing testable and provides the
        replacement point for the future sparse decode implementation.
        """

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
        return self._forward_quest_reference(
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
