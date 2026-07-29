# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-page key metadata used by QuEST attention."""

from dataclasses import dataclass

import torch

QUEST_PAGE_SIZE = 16


@dataclass
class QuestPageMetadata:
    """Sidecar metadata for the physical KV-cache pages of one layer.

    Each physical page stores an elementwise minimum and maximum key vector
    for every KV head. The metadata does not replace or modify the ordinary
    KV cache.
    """

    key_min: torch.Tensor
    key_max: torch.Tensor
    # 0 means uninitialized/stale; 1..block_size is occupancy.
    num_valid_tokens: torch.Tensor
    block_size: int

    @classmethod
    def allocate_from_kv_cache(
        cls,
        kv_cache: torch.Tensor,
        expected_num_kv_heads: int,
        expected_head_size: int,
    ) -> "QuestPageMetadata":
        """Allocate empty QuEST metadata matching a layer's KV cache."""

        if kv_cache.ndim != 5:
            raise ValueError(
                "QuEST expects KV cache shape "
                "[num_blocks, 2, block_size, num_kv_heads, head_size], "
                f"but received {tuple(kv_cache.shape)}."
            )

        (
            num_blocks,
            num_kv_planes,
            block_size,
            num_kv_heads,
            head_size,
        ) = kv_cache.shape

        if num_kv_planes != 2:
            raise ValueError(
                "QuEST expects separate key and value planes, "
                f"but received dimension 1 size {num_kv_planes}."
            )

        if block_size != QUEST_PAGE_SIZE:
            raise ValueError(
                f"QuEST requires block size {QUEST_PAGE_SIZE}, "
                f"but received {block_size}."
            )

        if num_kv_heads != expected_num_kv_heads:
            raise ValueError(
                "KV-head count mismatch: "
                f"expected {expected_num_kv_heads}, "
                f"received {num_kv_heads}."
            )

        if head_size != expected_head_size:
            raise ValueError(
                "Head-size mismatch: "
                f"expected {expected_head_size}, received {head_size}."
            )

        supported_dtypes = {
            torch.float16,
            torch.bfloat16,
            torch.float32,
        }

        if kv_cache.dtype not in supported_dtypes:
            raise TypeError(
                "QuEST page metadata currently requires a floating-point "
                f"KV cache, but received {kv_cache.dtype}."
            )

        summary_shape = (
            num_blocks,
            num_kv_heads,
            head_size,
        )

        return cls(
            key_min=torch.full(
                summary_shape,
                float("inf"),
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            ),
            key_max=torch.full(
                summary_shape,
                float("-inf"),
                dtype=kv_cache.dtype,
                device=kv_cache.device,
            ),
            num_valid_tokens=torch.zeros(
                num_blocks,
                dtype=torch.uint8,
                device=kv_cache.device,
            ),
            block_size=block_size,
        )

    def update_from_key_slots(
        self,
        key: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Incrementally update metadata for newly written keys.

        Args:
            key:
                Shape [num_tokens, num_kv_heads, head_size].
            slot_mapping:
                Shape [num_actual_tokens]. Each value identifies a physical
                KV-cache token slot.
        """

        if key.ndim != 3:
            raise ValueError(
                "QuEST expects key shape "
                "[num_tokens, num_kv_heads, head_size], "
                f"but received {tuple(key.shape)}."
            )

        expected_tail = (
            self.key_min.shape[1],
            self.key_min.shape[2],
        )

        if tuple(key.shape[1:]) != expected_tail:
            raise ValueError(
                f"Expected key trailing shape {expected_tail}, "
                f"but received {tuple(key.shape[1:])}."
            )

        if slot_mapping.ndim != 1:
            raise ValueError(
                "slot_mapping must be one-dimensional, "
                f"but received {tuple(slot_mapping.shape)}."
            )

        num_actual_tokens = slot_mapping.numel()

        if key.shape[0] < num_actual_tokens:
            raise ValueError(
                "The key tensor contains fewer rows than slot_mapping: "
                f"{key.shape[0]} < {num_actual_tokens}."
            )

        if num_actual_tokens == 0:
            return

        slots = slot_mapping.to(
            device=key.device,
            dtype=torch.long,
        )

        actual_keys = key[:num_actual_tokens]

        # vLLM uses negative slots, normally -1, for padded or inactive
        # tokens that must not be written into the physical KV cache.
        valid_slot_mask = slots >= 0

        if not bool(torch.any(valid_slot_mask).item()):
            return

        slots = slots[valid_slot_mask]
        actual_keys = actual_keys[valid_slot_mask]

        page_indices = torch.div(
            slots,
            self.block_size,
            rounding_mode="floor",
        )
        token_offsets = torch.remainder(
            slots,
            self.block_size,
        )

        if bool(torch.any(page_indices >= self.num_blocks).item()):
            raise ValueError(
                "slot_mapping references a page outside the metadata "
                f"allocation of {self.num_blocks} pages."
            )

        # Multiple tokens may target the same page during prefill. Group by
        # page so each page receives one elementwise min/max update.
        touched_pages = torch.unique(page_indices)

        for page_index_tensor in touched_pages:
            page_index = int(page_index_tensor.item())
            page_mask = page_indices == page_index_tensor
            page_keys = actual_keys[page_mask]
            page_offsets = token_offsets[page_mask]

            incoming_min = page_keys.amin(dim=0)
            incoming_max = page_keys.amax(dim=0)

            current_valid_tokens = int(
                self.num_valid_tokens[page_index].item()
            )
            resets_page = bool(torch.any(page_offsets == 0).item())

            # Offset zero begins a new lifetime for this physical page.
            # Do not merge keys from the page's previous owner.
            if resets_page or current_valid_tokens == 0:
                self.key_min[page_index].copy_(incoming_min)
                self.key_max[page_index].copy_(incoming_max)
                current_valid_tokens = 0
            else:
                self.key_min[page_index].copy_(
                    torch.minimum(
                        self.key_min[page_index],
                        incoming_min,
                    )
                )
                self.key_max[page_index].copy_(
                    torch.maximum(
                        self.key_max[page_index],
                        incoming_max,
                    )
                )

            incoming_valid_tokens = int(page_offsets.max().item()) + 1
            self.num_valid_tokens[page_index] = max(
                current_valid_tokens,
                incoming_valid_tokens,
            )

    def rebuild_from_kv_cache(
        self,
        kv_cache: torch.Tensor,
        valid_tokens: torch.Tensor,
    ) -> None:
        """Recompute all page summaries from the key cache.

        This is a correctness reference implementation. The production
        implementation will update only pages touched by new KV writes.
        """

        expected_shape = (
            self.num_blocks,
            2,
            self.block_size,
            self.key_min.shape[1],
            self.key_min.shape[2],
        )

        if tuple(kv_cache.shape) != expected_shape:
            raise ValueError(
                f"Expected KV-cache shape {expected_shape}, "
                f"but received {tuple(kv_cache.shape)}."
            )

        if valid_tokens.shape != (self.num_blocks,):
            raise ValueError(
                "valid_tokens must have shape "
                f"({self.num_blocks},), but received "
                f"{tuple(valid_tokens.shape)}."
            )

        valid_tokens = valid_tokens.to(
            device=kv_cache.device,
            dtype=torch.int32,
        )

        if bool(torch.any(valid_tokens < 0).item()):
            raise ValueError("valid_tokens cannot contain negative values.")

        if bool(torch.any(valid_tokens > self.block_size).item()):
            raise ValueError(
                f"valid_tokens cannot exceed block size {self.block_size}."
            )

        # Shape:
        # [num_blocks, block_size, num_kv_heads, head_size]
        key_cache = kv_cache[:, 0]

        token_indices = torch.arange(
            self.block_size,
            device=kv_cache.device,
            dtype=torch.int32,
        ).view(1, self.block_size, 1, 1)

        valid_mask = token_indices < valid_tokens.view(-1, 1, 1, 1)

        keys_for_min = torch.where(
            valid_mask,
            key_cache,
            torch.full_like(key_cache, float("inf")),
        )
        keys_for_max = torch.where(
            valid_mask,
            key_cache,
            torch.full_like(key_cache, float("-inf")),
        )

        self.key_min.copy_(keys_for_min.amin(dim=1))
        self.key_max.copy_(keys_for_max.amax(dim=1))
        self.num_valid_tokens.copy_(
            valid_tokens.to(dtype=torch.uint8)
        )

    @property
    def num_blocks(self) -> int:
        """Return the number of represented physical KV blocks."""

        return self.key_min.shape[0]

    @property
    def allocated_bytes(self) -> int:
        """Return the tensor-storage size of this metadata container."""

        tensors = (
            self.key_min,
            self.key_max,
            self.num_valid_tokens,
        )

        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in tensors
        )
