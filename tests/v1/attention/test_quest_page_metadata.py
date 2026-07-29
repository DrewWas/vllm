# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.attention.ops.quest import (
    QUEST_PAGE_SIZE,
    QuestPageMetadata,
)


def make_metadata(
    num_blocks: int = 2,
    num_kv_heads: int = 2,
    head_size: int = 4,
) -> tuple[torch.Tensor, QuestPageMetadata]:
    kv_cache = torch.zeros(
        (
            num_blocks,
            2,
            QUEST_PAGE_SIZE,
            num_kv_heads,
            head_size,
        ),
        dtype=torch.float32,
    )

    metadata = QuestPageMetadata.allocate_from_kv_cache(
        kv_cache=kv_cache,
        expected_num_kv_heads=num_kv_heads,
        expected_head_size=head_size,
    )

    return kv_cache, metadata


def test_rebuild_masks_unused_page_slots() -> None:
    kv_cache, metadata = make_metadata()
    key_cache = kv_cache[:, 0]

    valid_keys = torch.tensor(
        [
            [[1, 2, 3, 4], [10, 20, 30, 40]],
            [[-2, 5, 1, 8], [9, 25, 28, 50]],
            [[3, 0, 7, 6], [12, 18, 35, 42]],
        ],
        dtype=torch.float32,
    )

    key_cache[0, :3].copy_(valid_keys)

    # Invalid slots must not influence the page summaries.
    key_cache[0, 3:].fill_(999)
    key_cache[0, 4::2].fill_(-999)

    metadata.rebuild_from_kv_cache(
        kv_cache=kv_cache,
        valid_tokens=torch.tensor([3, 0]),
    )

    torch.testing.assert_close(
        metadata.key_min[0],
        valid_keys.amin(dim=0),
    )
    torch.testing.assert_close(
        metadata.key_max[0],
        valid_keys.amax(dim=0),
    )

    assert metadata.num_valid_tokens.tolist() == [3, 0]
    assert bool(torch.all(torch.isposinf(metadata.key_min[1])))
    assert bool(torch.all(torch.isneginf(metadata.key_max[1])))


def test_incremental_update_filters_padding_and_appends() -> None:
    _, metadata = make_metadata()

    prefill_keys = torch.tensor(
        [
            [[1, 2, 3, 4], [10, 20, 30, 40]],
            [[999, 999, 999, 999], [999, 999, 999, 999]],
            [[-2, 5, 1, 8], [9, 25, 28, 50]],
            [[-999, -999, -999, -999], [-999, -999, -999, -999]],
        ],
        dtype=torch.float32,
    )

    metadata.update_from_key_slots(
        key=prefill_keys,
        slot_mapping=torch.tensor([0, -1, 1, -1]),
    )

    valid_prefill_keys = prefill_keys[[0, 2]]

    torch.testing.assert_close(
        metadata.key_min[0],
        valid_prefill_keys.amin(dim=0),
    )
    torch.testing.assert_close(
        metadata.key_max[0],
        valid_prefill_keys.amax(dim=0),
    )

    assert metadata.num_valid_tokens.tolist() == [2, 0]

    decode_key = torch.tensor(
        [[[-5, 9, 2, 10], [8, 30, 26, 55]]],
        dtype=torch.float32,
    )

    metadata.update_from_key_slots(
        key=decode_key,
        slot_mapping=torch.tensor([2]),
    )

    all_valid_keys = torch.cat(
        [valid_prefill_keys, decode_key],
        dim=0,
    )

    torch.testing.assert_close(
        metadata.key_min[0],
        all_valid_keys.amin(dim=0),
    )
    torch.testing.assert_close(
        metadata.key_max[0],
        all_valid_keys.amax(dim=0),
    )

    assert metadata.num_valid_tokens.tolist() == [3, 0]


def test_offset_zero_overwrites_reused_physical_page() -> None:
    _, metadata = make_metadata(num_blocks=1)

    old_keys = torch.tensor(
        [
            [[-100, 100, -50, 50], [-80, 80, -40, 40]],
            [[-90, 90, -45, 45], [-70, 70, -35, 35]],
            [[-85, 85, -42, 42], [-60, 60, -30, 30]],
        ],
        dtype=torch.float32,
    )
    metadata.update_from_key_slots(
        key=old_keys,
        slot_mapping=torch.tensor([0, 1, 2]),
    )

    replacement_keys = torch.tensor(
        [
            [[1, 2, 3, 4], [10, 20, 30, 40]],
            [[5, 6, 7, 8], [50, 60, 70, 80]],
        ],
        dtype=torch.float32,
    )
    metadata.update_from_key_slots(
        key=replacement_keys,
        slot_mapping=torch.tensor([0, 1]),
    )

    torch.testing.assert_close(
        metadata.key_min[0],
        replacement_keys.amin(dim=0),
    )
    torch.testing.assert_close(
        metadata.key_max[0],
        replacement_keys.amax(dim=0),
    )
    assert metadata.num_valid_tokens.tolist() == [2]
