# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.quest.reference import (
    reference_quest_decode,
)
from vllm.v1.attention.ops.quest.vectorized import (
    vectorized_quest_decode_batch1,
)


@pytest.mark.parametrize("page_budget", [1, 2, 3, 100])
def test_vectorized_matches_reference(
    page_budget: int,
) -> None:
    torch.manual_seed(17)

    block_size = 2
    sequence_length = 5

    query = torch.randn(1, 4, 3)
    kv_cache = torch.randn(
        5,
        2,
        block_size,
        2,
        3,
    )

    # Logical pages map to shuffled physical pages.
    block_table = torch.tensor(
        [[3, 0, 4, -1, -1]]
    )
    seq_lens = torch.tensor([sequence_length])

    # The second token slot in the final page is invalid. Large values make
    # incorrect final-page masking immediately visible.
    kv_cache[4, :, 1].fill_(10_000)

    page_min = kv_cache[:, 0].amin(dim=1)
    page_max = kv_cache[:, 0].amax(dim=1)

    reference = reference_quest_decode(
        query=query,
        kv_cache=kv_cache,
        page_min=page_min,
        page_max=page_max,
        block_table=block_table,
        seq_lens=seq_lens,
        page_budget=page_budget,
        block_size=block_size,
    )

    vectorized = vectorized_quest_decode_batch1(
        query=query,
        kv_cache=kv_cache,
        page_min=page_min,
        page_max=page_max,
        block_table=block_table,
        sequence_length=sequence_length,
        page_budget=page_budget,
        block_size=block_size,
    )

    valid_pages = 3

    torch.testing.assert_close(
        vectorized.scores,
        reference.scores[:, :, :valid_pages],
        rtol=1e-5,
        atol=1e-5,
    )

    torch.testing.assert_close(
        vectorized.selection.logical_page_ids,
        reference.selection.logical_page_ids,
    )

    torch.testing.assert_close(
        vectorized.selection.physical_page_ids,
        reference.selection.physical_page_ids,
    )

    torch.testing.assert_close(
        vectorized.output,
        reference.output,
        rtol=1e-5,
        atol=1e-5,
    )


def test_vectorized_always_includes_current_page() -> None:
    torch.manual_seed(23)

    query = torch.randn(1, 4, 2)
    kv_cache = torch.randn(4, 2, 2, 2, 2)
    block_table = torch.tensor([[3, 1, 0, 2]])

    page_min = kv_cache[:, 0].amin(dim=1)
    page_max = kv_cache[:, 0].amax(dim=1)

    result = vectorized_quest_decode_batch1(
        query=query,
        kv_cache=kv_cache,
        page_min=page_min,
        page_max=page_max,
        block_table=block_table,
        sequence_length=7,
        page_budget=2,
        block_size=2,
    )

    # Current logical page is page 3 and must be the final selected page
    # for every query head.
    torch.testing.assert_close(
        result.selection.logical_page_ids[0, :, -1],
        torch.full((4,), 3),
    )

    torch.testing.assert_close(
        result.selection.physical_page_ids[0, :, -1],
        torch.full((4,), 2),
    )


def test_vectorized_hot_path_has_no_tensor_item_calls() -> None:
    import inspect

    source = inspect.getsource(
        vectorized_quest_decode_batch1
    )

    assert ".item(" not in source
