# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""QuEST attention backend.

The current implementation establishes QuEST-owned execution routing while
remaining numerically equivalent to dense FlashAttention.

Sparse page selection and sparse KV reads are added in later stages.
"""

import math
import os
from dataclasses import dataclass, fields
from typing import Any

import torch

from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.ops.quest import QuestPageMetadata
from vllm.v1.attention.ops.quest.vectorized import (
    vectorized_quest_decode_batch1,
)
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.quest.selection_logger import capture_quest_page_selection

QUEST_NUM_DENSE_LAYERS = 2
QUEST_PAGE_BUDGET_PERCENT_ENV = "VLLM_QUEST_PAGE_BUDGET_PERCENT"
QUEST_DEFAULT_PAGE_BUDGET_PERCENT = 100.0
QUEST_SPARSE_START_TOKENS_ENV = "VLLM_QUEST_SPARSE_START_TOKENS"
QUEST_DEFAULT_SPARSE_START_TOKENS = 8192


@dataclass(frozen=True)
class QuestBudgetDecision:
    """Resolved Quest page budget for one decode sequence."""

    sequence_length: int
    valid_pages: int
    selected_pages: int
    sparse_active: bool

    @property
    def effective_percent(self) -> float:
        return 100.0 * self.selected_pages / self.valid_pages


def get_quest_sparse_start_tokens() -> int:
    """Read the sequence-length threshold for sparse Quest decode."""

    raw_value = os.getenv(
        QUEST_SPARSE_START_TOKENS_ENV,
        str(QUEST_DEFAULT_SPARSE_START_TOKENS),
    )

    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(
            f"{QUEST_SPARSE_START_TOKENS_ENV} must be an integer, "
            f"but received {raw_value!r}."
        ) from exc

    if value < 0:
        raise ValueError(
            f"{QUEST_SPARSE_START_TOKENS_ENV} must be non-negative, "
            f"but received {value}."
        )

    return value


def compute_quest_budget(
    *,
    sequence_length: int,
    block_size: int,
    budget_percent: float,
    sparse_start_tokens: int,
) -> QuestBudgetDecision:
    """Resolve the number of valid and selected pages for one request."""

    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    if block_size <= 0:
        raise ValueError("block_size must be positive")

    if not 0.0 < budget_percent <= 100.0:
        raise ValueError(
            "budget_percent must be greater than 0 and at most 100"
        )

    if sparse_start_tokens < 0:
        raise ValueError("sparse_start_tokens must be non-negative")

    valid_pages = math.ceil(sequence_length / block_size)

    sparse_active = (
        sequence_length > sparse_start_tokens
        and budget_percent < 100.0
    )

    if sparse_active:
        selected_pages = max(
            1,
            math.ceil(valid_pages * budget_percent / 100.0),
        )
        selected_pages = min(selected_pages, valid_pages)
    else:
        selected_pages = valid_pages

    return QuestBudgetDecision(
        sequence_length=sequence_length,
        valid_pages=valid_pages,
        selected_pages=selected_pages,
        sparse_active=sparse_active,
    )


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


@dataclass
class QuestAttentionMetadata(FlashAttentionMetadata):
    """FlashAttention metadata extended with exact request phase state."""

    is_prefilling: torch.Tensor | None = None

    @classmethod
    def from_flash_attention_metadata(
        cls,
        metadata: FlashAttentionMetadata,
        is_prefilling: torch.Tensor | None,
    ) -> "QuestAttentionMetadata":
        """Copy FlashAttention metadata and attach the scheduler phase flag."""

        base_values = {
            field.name: getattr(metadata, field.name)
            for field in fields(FlashAttentionMetadata)
        }

        return cls(
            **base_values,
            is_prefilling=is_prefilling,
        )


class QuestAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    """Build FlashAttention metadata plus QuEST's exact phase signal."""

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> QuestAttentionMetadata:
        flash_metadata = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )

        return QuestAttentionMetadata.from_flash_attention_metadata(
            metadata=flash_metadata,
            is_prefilling=common_attn_metadata.is_prefilling,
        )


class QuestAttentionImpl(FlashAttentionImpl):
    """QuEST implementation with an initial dense-compatible execution path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.quest_page_budget_percent = get_quest_page_budget_percent()
        self.quest_sparse_start_tokens = get_quest_sparse_start_tokens()
        self.quest_page_metadata: QuestPageMetadata | None = None

        # QUEST_ROUTE_TRACE
        self.quest_debug = os.getenv("QUEST_DEBUG", "0") == "1"
        self._logged_metadata = False
        self._logged_prefill = False
        self._logged_dense_decode = False
        self._logged_quest_decode = False


    def _get_or_create_page_metadata(
        self,
        kv_cache: torch.Tensor,
    ) -> QuestPageMetadata:
        """Return this layer's QuEST metadata, allocating it if necessary."""

        metadata = self.quest_page_metadata

        if metadata is None:
            metadata = QuestPageMetadata.allocate_from_kv_cache(
                kv_cache=kv_cache,
                expected_num_kv_heads=self.num_kv_heads,
                expected_head_size=self.head_size,
            )
            self.quest_page_metadata = metadata
            return metadata

        expected_shape = (
            metadata.num_blocks,
            2,
            metadata.block_size,
            self.num_kv_heads,
            self.head_size,
        )

        if tuple(kv_cache.shape) != expected_shape:
            raise RuntimeError(
                "The KV-cache shape changed after QuEST metadata "
                f"allocation: expected {expected_shape}, "
                f"received {tuple(kv_cache.shape)}."
            )

        if metadata.key_min.device != kv_cache.device:
            raise RuntimeError(
                "The KV-cache device changed after QuEST metadata "
                f"allocation: expected {metadata.key_min.device}, "
                f"received {kv_cache.device}."
            )

        if metadata.key_min.dtype != kv_cache.dtype:
            raise RuntimeError(
                "The KV-cache dtype changed after QuEST metadata "
                f"allocation: expected {metadata.key_min.dtype}, "
                f"received {kv_cache.dtype}."
            )

        return metadata

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
        """Execute vectorized batch-1 Quest decode.

        Quest runs for every decode token in layers 2 and above. The sparse
        threshold controls only whether the configured reduced page budget
        applies. At or below the threshold, Quest attends every valid page.
        """

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "The vectorized Quest MVP does not support fused "
                "output quantization."
            )

        page_metadata = self.quest_page_metadata
        if page_metadata is None:
            raise RuntimeError(
                "Quest page metadata was not allocated before decode."
            )

        if attn_metadata.max_query_len != 1:
            raise RuntimeError(
                "The vectorized Quest MVP supports only "
                "single-token decode."
            )

        num_tokens = attn_metadata.num_actual_tokens

        if num_tokens != 1 or attn_metadata.seq_lens.numel() != 1:
            raise RuntimeError(
                "The vectorized Quest MVP supports exactly one active "
                "decode request."
            )

        # Batch=1 makes max_seq_len the exact current request length.
        # Using this host-side integer avoids seq_lens[0].item(), which
        # would introduce one CUDA synchronization per Quest layer.
        sequence_length = attn_metadata.max_seq_len

        budget = compute_quest_budget(
            sequence_length=sequence_length,
            block_size=page_metadata.block_size,
            budget_percent=self.quest_page_budget_percent,
            sparse_start_tokens=self.quest_sparse_start_tokens,
        )

        result = vectorized_quest_decode_batch1(
            query=query[:num_tokens],
            kv_cache=kv_cache,
            page_min=page_metadata.key_min,
            page_max=page_metadata.key_max,
            block_table=attn_metadata.block_table[:num_tokens],
            sequence_length=sequence_length,
            page_budget=budget.selected_pages,
            block_size=page_metadata.block_size,
            scale=self.scale,
        )

        capture_quest_page_selection(
            layer_idx=extract_layer_index(layer.layer_name),
            sequence_length=sequence_length,
            physical_page_ids=(
                result.selection.physical_page_ids
            ),
        )

        output_view = output.view(
            output.shape[0],
            self.num_heads,
            self.head_size,
        )
        output_view[:num_tokens].copy_(result.output)

        return output



    @staticmethod
    def _requires_dense_phase(
        attn_metadata: FlashAttentionMetadata,
    ) -> bool:
        """Return whether this batch must use dense attention.

        Missing phase metadata falls back to dense attention. A mixed batch
        containing any prefill request also remains entirely dense during the
        initial single-request QuEST implementation.
        """

        is_prefilling = getattr(attn_metadata, "is_prefilling", None)

        if is_prefilling is None or is_prefilling.numel() == 0:
            return True

        return bool(torch.any(is_prefilling).item())

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

        # Allocate sidecar metadata for QuEST-enabled layers. Allocation happens
        # during prefill so the container is ready before the first decode step.
        if (
            layer_index >= QUEST_NUM_DENSE_LAYERS
            and attn_metadata is not None
            and kv_cache.numel() > 0
        ):
            page_metadata = self._get_or_create_page_metadata(kv_cache)

            if self.quest_debug and not self._logged_metadata:
                print(
                    f"[QUEST] layer={layer_index} "
                    f"metadata_shape={tuple(page_metadata.key_min.shape)}",
                    flush=True,
                )
                self._logged_metadata = True

            # Update the sidecar metadata from the same RoPE-transformed keys
            # and physical slot mapping used by the ordinary KV-cache write.
            page_metadata.update_from_key_slots(
                key=key,
                slot_mapping=attn_metadata.slot_mapping,
            )

        # Prefill remains dense for every layer. Mixed prefill/decode batches
        # also remain dense until broader batching support is implemented.
        if self._requires_dense_phase(attn_metadata):
            if self.quest_debug and not self._logged_prefill:
                print(
                    f"[QUEST] layer={layer_index} "
                    "route=dense phase=prefill_or_mixed",
                    flush=True,
                )
                self._logged_prefill = True

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

        if layer_index < QUEST_NUM_DENSE_LAYERS:
            if self.quest_debug and not self._logged_dense_decode:
                print(
                    f"[QUEST] layer={layer_index} "
                    "route=dense phase=decode",
                    flush=True,
                )
                self._logged_dense_decode = True

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

        if self.quest_debug and not self._logged_quest_decode:
            print(
                f"[QUEST] layer={layer_index} "
                "route=quest_vectorized phase=decode "
                f"budget={self.quest_page_budget_percent}",
                flush=True,
            )
            self._logged_quest_decode = True

        # Vectorized Quest decode path.
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

    @staticmethod
    def get_builder_cls() -> type[QuestAttentionMetadataBuilder]:
        return QuestAttentionMetadataBuilder
