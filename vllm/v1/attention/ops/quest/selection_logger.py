"""Asynchronous binary logging of Quest physical-page selections."""

import atexit
import os
import queue
import struct
import threading
from dataclasses import dataclass
from pathlib import Path

import torch


MAGIC = b"QSELBIN1"


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class _WriteJob:
    token_position: int
    layer_ids: tuple[int, ...]
    pages_per_head: int
    cpu_pages: torch.Tensor
    gpu_pages: torch.Tensor
    source_tensors: tuple[torch.Tensor, ...]
    complete_event: torch.cuda.Event


class _QuestSelectionLogger:
    def __init__(self) -> None:
        self.enabled = _env_bool("VLLM_QUEST_LOG_SELECTIONS")

        self.start_tokens = int(
            os.getenv(
                "VLLM_QUEST_SPARSE_START_TOKENS",
                "8192",
            )
        )

        self.last_layer = int(
            os.getenv("VLLM_QUEST_LOG_LAST_LAYER", "31")
        )

        self.path = Path(
            os.getenv(
                "VLLM_QUEST_LOG_PATH",
                "/workspace/quest/logs/quest_pages.qsel",
            )
        )

        self._lock = threading.Lock()
        self._sequence_length: int | None = None
        self._layers: dict[int, torch.Tensor] = {}

        self._copy_stream: torch.cuda.Stream | None = None
        self._queue: queue.Queue[_WriteJob | None] = queue.Queue()
        self._worker: threading.Thread | None = None

    def _start_worker(self) -> None:
        if self._worker is not None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._worker = threading.Thread(
            target=self._writer_loop,
            name="quest-selection-writer",
            daemon=True,
        )
        self._worker.start()

    def capture(
        self,
        *,
        layer_idx: int,
        sequence_length: int,
        physical_page_ids: torch.Tensor,
    ) -> None:
        if not self.enabled:
            return

        if sequence_length <= self.start_tokens:
            return

        with self._lock:
            if (
                self._sequence_length is not None
                and self._sequence_length != sequence_length
            ):
                self._flush_locked()

            self._sequence_length = sequence_length
            self._layers[layer_idx] = physical_page_ids.detach()

            if layer_idx == self.last_layer:
                self._flush_locked()

    def _flush_locked(self) -> None:
        if not self._layers:
            return

        self._start_worker()

        layer_ids = tuple(sorted(self._layers))
        sources = tuple(self._layers[layer] for layer in layer_ids)

        first_shape = sources[0].shape

        if len(first_shape) != 3 or first_shape[0] != 1:
            raise RuntimeError(
                "Quest selections must have shape [1, Hq, K]."
            )

        num_heads = first_shape[1]
        pages_per_head = first_shape[2]

        device = sources[0].device

        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)

        cpu_pages = torch.empty(
            (len(layer_ids), num_heads, pages_per_head),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )

        producer_stream = torch.cuda.current_stream(device=device)
        ready_event = torch.cuda.Event()
        ready_event.record(producer_stream)

        complete_event = torch.cuda.Event()

        with torch.cuda.stream(self._copy_stream):
            self._copy_stream.wait_event(ready_event)

            gpu_pages = torch.cat(
                sources,
                dim=0,
            ).to(dtype=torch.int32)

            cpu_pages.copy_(
                gpu_pages,
                non_blocking=True,
            )

            complete_event.record(self._copy_stream)

        self._queue.put(
            _WriteJob(
                token_position=self._sequence_length - 1,
                layer_ids=layer_ids,
                pages_per_head=pages_per_head,
                cpu_pages=cpu_pages,
                gpu_pages=gpu_pages,
                source_tensors=sources,
                complete_event=complete_event,
            )
        )

        self._sequence_length = None
        self._layers.clear()

    def _writer_loop(self) -> None:
        handle = None
        expected_layers = None
        expected_num_heads = None
        records_since_flush = 0

        try:
            while True:
                job = self._queue.get()

                if job is None:
                    break

                job.complete_event.synchronize()

                num_layers, num_heads, _ = job.cpu_pages.shape

                if handle is None:
                    handle = self.path.open(
                        "wb",
                        buffering=1024 * 1024,
                    )

                    handle.write(MAGIC)
                    handle.write(
                        struct.pack(
                            "<HH",
                            num_layers,
                            num_heads,
                        )
                    )
                    handle.write(
                        struct.pack(
                            f"<{num_layers}H",
                            *job.layer_ids,
                        )
                    )

                    expected_layers = job.layer_ids
                    expected_num_heads = num_heads

                if job.layer_ids != expected_layers:
                    raise RuntimeError(
                        "Quest layer IDs changed during logging."
                    )

                if num_heads != expected_num_heads:
                    raise RuntimeError(
                        "Quest head count changed during logging."
                    )

                handle.write(
                    struct.pack(
                        "<qI",
                        job.token_position,
                        job.pages_per_head,
                    )
                )

                handle.write(
                    job.cpu_pages.numpy().tobytes(order="C")
                )

                records_since_flush += 1

                if records_since_flush >= 64:
                    handle.flush()
                    records_since_flush = 0

        finally:
            if handle is not None:
                handle.flush()
                handle.close()

    def close(self) -> None:
        if not self.enabled:
            return

        with self._lock:
            self._flush_locked()

        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()
            self._worker = None


_LOGGER = _QuestSelectionLogger()
atexit.register(_LOGGER.close)


def capture_quest_page_selection(
    *,
    layer_idx: int,
    sequence_length: int,
    physical_page_ids: torch.Tensor,
) -> None:
    _LOGGER.capture(
        layer_idx=layer_idx,
        sequence_length=sequence_length,
        physical_page_ids=physical_page_ids,
    )
