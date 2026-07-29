# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.quest import (
    QUEST_PAGE_SIZE,
    QuestPageMetadata,
)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for this smoke test.",
)
def test_quest_page_reuse_cuda() -> None:
    device = torch.device("cuda")

    kv_cache = torch.zeros(
        (1, 2, QUEST_PAGE_SIZE, 2, 4),
        dtype=torch.bfloat16,
        device=device,
    )

    metadata = QuestPageMetadata.allocate_from_kv_cache(
        kv_cache=kv_cache,
        expected_num_kv_heads=2,
        expected_head_size=4,
    )

    # First owner of physical page 0.
    old_keys = torch.tensor(
        [
            [[-100, 100, -50, 50], [-80, 80, -40, 40]],
            [[-90, 90, -45, 45], [-70, 70, -35, 35]],
        ],
        dtype=torch.bfloat16,
        device=device,
    )

    metadata.update_from_key_slots(
        key=old_keys,
        slot_mapping=torch.tensor([0, 1], device=device),
    )

    # New owner starts again at page offset 0.
    replacement_keys = torch.tensor(
        [
            [[1, 2, 3, 4], [10, 20, 30, 40]],
            [[5, 6, 7, 8], [50, 60, 70, 80]],
        ],
        dtype=torch.bfloat16,
        device=device,
    )

    metadata.update_from_key_slots(
        key=replacement_keys,
        slot_mapping=torch.tensor([0, 1], device=device),
    )

    torch.cuda.synchronize()

    # Old extrema must no longer be present.
    torch.testing.assert_close(
        metadata.key_min[0],
        replacement_keys.amin(dim=0),
    )
    torch.testing.assert_close(
        metadata.key_max[0],
        replacement_keys.amax(dim=0),
    )
    assert int(metadata.num_valid_tokens[0].item()) == 2
