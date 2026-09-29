"""Pure-Python fallback for the PLE disk row store (``freetoken.kernel.row_store.PleStore``).

The C++ store is Linux-only (``setup.py`` gates it on ``sys.platform == "linux"``), and
on this box the disk backend is the only PLE option that fits: the pinned table is
47.7 GiB and would not coexist with the 63.3 GiB of expert banks. So the platforms
without the extension get this.

Same three operations as the extension, same on-disk geometry, same hashed row ids
(hash reference: ``ple.py`` ``NGramEmbedding.row_ids``), same output layout: for one
token, ``num_ngram_heads`` table rows of ``head_dim`` bytes concatenated. The only
difference is throughput -- reads go through the OS page cache with a thread pool
instead of O_DIRECT + io_uring. At ``num_ngram_heads * head_dim`` = 2560 bytes per
token a 2k prefill moves ~5 MB, which is noise next to the ~65 GiB the MoE expert
prefill already streams.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import torch

# Matches the extension's own tuning (row_store_ext.cpp kReaderThreads).
_READER_THREADS = 16
# Rows within this gap of each other in one extent are read as a single call.
_COALESCE_GAP = 1 << 16


class PyPleStore:
    """Tensor-facing stand-in for the C++ ``PleStore``.

    ``stage`` gathers one hashed row group per new token into ``dst``; ``flush`` is a
    no-op because the reads already completed on the host before the forward is
    enqueued (launch-gating, not the memop flag protocol).
    """

    def __init__(
        self,
        *,
        paths: Sequence[str],
        extent_file: Sequence[int],
        extent_base: Sequence[int],
        rows_per_extent: int,
        row_bytes: int,
        row_stride: int,
        multipliers: Sequence[int],
        head_vocab_sizes: Sequence[int],
        head_offsets: Sequence[int],
        eos_token_id: int,
        **_ignored,
    ) -> None:
        self.paths = list(paths)
        self.extent_file = list(extent_file)
        self.extent_base = list(extent_base)
        self.rows_per_extent = int(rows_per_extent)
        self.row_bytes = int(row_bytes)
        self.row_stride = int(row_stride)
        self.multipliers = [int(x) for x in multipliers]
        self.head_vocab_sizes = [int(x) for x in head_vocab_sizes]
        self.head_offsets = [int(x) for x in head_offsets]
        self.eos_token_id = int(eos_token_id)
        self.num_heads = len(self.head_vocab_sizes)
        self.ngram_size = len(self.multipliers)
        if self.ngram_size < 2:
            raise ValueError(f"n-gram size {self.ngram_size} has no hash windows")
        # one (vocab, offset) pair per n-gram block, matching ple.py's head slicing:
        # ngram_size-1 blocks of heads_per_ngram heads each
        heads_per_ngram = self.num_heads // (self.ngram_size - 1)
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * heads_per_ngram
            blocks.append((start, start + heads_per_ngram))
        self._blocks = blocks
        self._handles: dict[int, object] = {}
        self._handles_lock = threading.Lock()
        self._pool = ThreadPoolExecutor(_READER_THREADS, thread_name_prefix="ple-row")
        self._owns_pool = True

    def __del__(self) -> None:
        pool = getattr(self, "_pool", None)
        if pool is not None and self._owns_pool:
            pool.shutdown(wait=False)

    def io_backend(self) -> str:
        return "python-read"

    def stage(self, run: torch.Tensor, dst: torch.Tensor) -> None:
        """Gather the rows of the ``run.numel() - 2`` tokens that follow the 2 context ids."""
        count = run.numel() - 2
        if count == 0:
            return
        buf = self._gather(self._row_ids(run.tolist(), count))
        dst.view(count, -1).copy_(torch.frombuffer(buf, dtype=torch.uint8).view(count, -1))

    def flush(self, _flag: torch.Tensor | None = None) -> None:
        return None

    # ---------------- hashing ----------------

    def _row_ids(self, ids: list[int], count: int) -> list[int]:
        """Global table row per (token, hash head), in the order ``dst`` expects them.

        ``ids`` is ``[x(t-2), x(t-1), t_0, t_1, ...]``, so output row ``i`` is the token
        ``ids[i + ngram_size - 1]``. ``ple.py`` ``NGramEmbedding.row_ids`` indexes its
        window from the NEW token leftwards -- ``tokens[k] = ids[p - k]`` feeding
        ``layer_multipliers[k]`` -- so the window reverses relative to the run.
        Heads ``(ngram - 2) * heads_per_ngram`` upward, one block per n-gram size.
        """
        multipliers = self.multipliers
        sizes = self.head_vocab_sizes
        offsets = self.head_offsets
        ngram_size = self.ngram_size
        out: list[int] = []
        for i in range(count):
            window = ids[i : i + ngram_size][::-1]
            for block, (start, end) in enumerate(self._blocks):
                mixed = 0
                for position in range(block + 2):
                    mixed ^= window[position] * multipliers[position]
                for head in range(start, end):
                    out.append(mixed % sizes[head] + offsets[head])
        return out

    # ---------------- row reads ----------------

    def _file(self, index: int):
        with self._handles_lock:
            handle = self._handles.get(index)
            if handle is None:
                handle = open(self.paths[index], "rb", buffering=0)
                self._handles[index] = handle
            return handle

    def _locate(self, row: int) -> tuple[int, int]:
        extent = row // self.rows_per_extent
        return self.extent_file[extent], self.extent_base[extent] + (
            row % self.rows_per_extent
        ) * self.row_stride

    def _gather(self, rows: list[int]) -> bytearray:
        """Read the requested rows into one contiguous buffer, in the caller's order."""
        located = [self._locate(row) for row in rows]
        # read in file order so neighbouring rows share a syscall, then scatter back
        order = sorted(range(len(rows)), key=lambda i: located[i])
        buf = bytearray(len(rows) * self.row_bytes)
        view = memoryview(buf)
        jobs: list[tuple[int, int, list[int], list[int]]] = []
        run_start = 0
        while run_start < len(order):
            file_idx, offset = located[order[run_start]]
            run_end = run_start + 1
            while run_end < len(order):
                n_file, n_offset = located[order[run_end]]
                gap = n_offset - offset - (run_end - run_start) * self.row_stride
                if n_file != file_idx or not 0 <= gap <= _COALESCE_GAP:
                    break
                run_end += 1
            picks = order[run_start:run_end]
            jobs.append(
                (
                    file_idx,
                    offset,
                    [located[j][1] - offset for j in picks],  # where in the coalesced read
                    [j for j in picks],                        # where in the output buffer
                )
            )
            run_start = run_end
        list(self._pool.map(lambda job: self._read_run(view, self.row_bytes, *job), jobs))
        return buf

    def _read_run(self, view: memoryview, row_bytes: int, file_idx: int, offset: int,
                  srcs: list[int], dests: list[int]) -> None:
        handle = self._file(file_idx)
        span = srcs[-1] + row_bytes
        # os.pread is POSIX-only; the handle is unbuffered so seek+read stays one syscall
        handle.seek(offset)
        data = handle.read(span)
        if len(data) != span:
            raise OSError(f"{self.paths[file_idx]}: short read at {offset} ({len(data)}/{span})")
        for src, dest in zip(srcs, dests):
            view[dest * row_bytes : (dest + 1) * row_bytes] = data[src : src + row_bytes]


__all__ = ["PyPleStore"]
