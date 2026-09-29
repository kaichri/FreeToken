"""Bounded per-request KV window for the MTP head.

The head keeps its own window rather than taking a layer in the trunk's paged pool, for
three reasons:

* the window is small and hard-bounded: ``window * num_kv_heads * head_dim * 2 (K and V)
  * itemsize`` is 0.25 MiB per request at the shipped geometry, against a paged layer whose
  size is the trunk's page count and which therefore cannot be capped at all;
* no full-prefix prefill, so the head never needs pages handed out by the scheduler's
  allocator, and a draft round adds no page allocation to roll back;
* the head's attention is *exactly* the production one, not an approximation of it. A
  request that sees at most ``indexer_budget + indexer_compress_ratio - 1`` tokens selects
  every block, so sparse and dense attention coincide; the window is 128 against a bound of
  2048 + 4 - 1 = 2051 for this checkpoint.

:class:`~freetoken.models.qwen4_exp.attention.TorchDenseQSAReference` supplies the attention
and already keeps a ``[slot, position]`` KV, so this owns the window bookkeeping, the roll,
and the batch the head's forward sees.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx

from .attention import TorchDenseQSAReference

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig

# The head's KV window is capped here, and the cap is load-bearing rather than a memory
# convenience: the prober's own score.json shows one-step top-1 falling off a cliff as the
# window grows -- code 89.1% at 128 rows, 73.0% at 256, 43.8% at 512, 34.6% at 2048. Measured
# in-tree at 128 with the real engine, so the cap is where the head is worth running.
MTP_KV_WINDOW = 128

# The head owns one attention layer, so its window lives under this layer id in the
# reference's own cache. It is deliberately 0: the reference is a separate object with its
# own cache, so this cannot collide with the trunk's layers.
MTP_LAYER_ID = 0


@dataclass
class MtpWindowReq:
    """The three fields the reference attention reads off a request, in head-window terms.

    ``cached_len`` is the head's own live row count, not the trunk request's prefix length.
    """

    table_idx: int
    cached_len: int
    extend_len: int


@dataclass
class MtpDraftBatch:
    """The batch the head's forward runs against.

    Its rows are head-window rows: one per drafted position, with ``positions`` carrying the
    absolute logical position so the rotary embedding is the model's, not the window's.
    """

    padded_reqs: list[MtpWindowReq]
    positions: torch.Tensor
    # mrope models feed the rotary [3, n] rope rows (rotary.py:179 asserts dim 2), so the head
    # carries the same shape the trunk gives it
    mrope_positions: torch.Tensor | None = None
    phase: str = "decode"
    size_: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.size_ = len(self.padded_reqs)

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_verify(self) -> bool:
        return self.phase == "verify"

    @property
    def size(self) -> int:
        return self.size_

    def get_attn_positions(self) -> torch.Tensor:
        # same precedence as Batch.get_attn_positions (core.py): the 3-axis rope rows when
        # present, else the sequence index
        if self.mrope_positions is not None:
            return self.mrope_positions
        return self.positions


class MtpKV:
    """The head's rolling KV window, one independent window per request slot."""

    def __init__(
        self,
        config: ModelConfig,
        num_slots: int,
        *,
        window: int = MTP_KV_WINDOW,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.window = window
        self.num_slots = num_slots
        self.backend = TorchDenseQSAReference(
            config, num_slots=num_slots, max_len=window, device=device, dtype=dtype
        )
        self._live = [0] * num_slots

    def bytes_per_slot(self) -> int:
        """Device bytes one request's window holds (K and V)."""
        kv, v = self.backend.kv(MTP_LAYER_ID)
        return kv[0].numel() * kv.element_size() + v[0].numel() * v.element_size()

    def live_rows(self, slot: int) -> int:
        return self._live[slot]

    def kv(self, slot: int) -> tuple[torch.Tensor, torch.Tensor]:
        """This request's live window, a view of the reference's cache."""
        k, v = self.backend.kv(MTP_LAYER_ID)
        n = self._live[slot]
        return k[slot, :n], v[slot, :n]

    def reserve(
        self,
        rows: list[tuple[int, int]],
        positions: torch.Tensor,
        mrope_positions: torch.Tensor | None = None,
        *,
        phase: str = "decode",
    ) -> MtpDraftBatch:
        """Claim ``n`` rows for each ``(slot, n)``, rolling the window when it would overflow.

        The rows are claimed before the forward writes them, so the window length is exact
        even if the forward is later rejected and rolled back by the caller.

        ``phase`` is the head's own: a multi-row priming chunk has to reach the attention and
        MoE prefill branches, which a one-row draft does not.
        """
        reqs: list[MtpWindowReq] = []
        for slot, n in rows:
            if not 0 <= slot < self.num_slots:
                raise IndexError(f"MTP window slot {slot} outside 0..{self.num_slots}")
            live = self._live[slot]
            if live + n > self.window:
                self._roll(slot, min(live, self.window - n))
                live = self._live[slot]
            assert live + n <= self.window, "window too small for this draft"
            self._live[slot] = live + n
            reqs.append(MtpWindowReq(table_idx=slot, cached_len=live, extend_len=n))
        return MtpDraftBatch(
            padded_reqs=reqs, positions=positions, mrope_positions=mrope_positions, phase=phase
        )

    def _roll(self, slot: int, keep: int) -> None:
        """Drop the oldest rows so ``keep`` recent ones stay, keeping their rope baked in.

        K is stored post-rotary, so a row carries no position of its own and moving it is safe.
        """
        k, v = self.backend.kv(MTP_LAYER_ID)
        live = self._live[slot]
        if keep < live:
            k[slot, :keep] = k[slot, live - keep : live].clone()
            v[slot, :keep] = v[slot, live - keep : live].clone()
        self._live[slot] = keep

    def rewind(self, slot: int, rows: int) -> None:
        """Give back ``rows`` claimed by :meth:`reserve`, for a rejected draft.

        The rows themselves are not scrubbed: the next draft overwrites them, and the
        reference masks every position past the live length out of the attention.
        """
        self._live[slot] = max(0, self._live[slot] - rows)

    def set_live(self, slot: int, rows: int) -> None:
        """Set a slot's live row count outright, to restore a snapshot.

        A rollback has to return the window to the length the snapshot recorded, not merely
        subtract one: the two differ as soon as a draft is taken after a snapshot that already
        held rows.
        """
        if not 0 <= rows <= self.window:
            raise ValueError(f"window live count {rows} outside 0..{self.window}")
        self._live[slot] = rows

    @contextmanager
    def draft(self, batch: MtpDraftBatch):
        """Run the head against its own window.

        Both the global batch and the attention backend are swapped, not just the batch: the
        head's MoE layers read the global batch for their prefill/decode split, and the head's
        attention must not reach the trunk's paged backend.
        """
        ctx = get_global_ctx()
        saved_batch, saved_backend = ctx._batch, ctx.attn_backend
        ctx._batch = batch
        ctx.attn_backend = self.backend
        try:
            yield batch
        finally:
            ctx._batch = saved_batch
            ctx.attn_backend = saved_backend

    def reset_slot(self, slot: int) -> None:
        """Forget one request's window, so the slot can be handed to the next one.

        The live count is the part that matters: a slot reused with a stale count would hand the
        next request a window of rows that belong to the previous one.
        """
        if not 0 <= slot < self.num_slots:
            raise IndexError(f"MTP window slot {slot} outside 0..{self.num_slots}")
        k, v = self.backend.kv(MTP_LAYER_ID)
        k[slot].zero_()
        v[slot].zero_()
        self._live[slot] = 0

    def reset(self) -> None:
        for slot in range(self.num_slots):
            k, v = self.backend.kv(MTP_LAYER_ID)
            k[slot].zero_()
            v[slot].zero_()
        self._live = [0] * self.num_slots


def window_bytes_per_request(
    config: ModelConfig, *, window: int = MTP_KV_WINDOW, itemsize: int = 2
) -> int:
    """Device bytes one request's window holds, from the config rather than a live pool.

    The engine has to budget for the window before any request exists, so the figure cannot
    come from :meth:`MtpKV.bytes_per_slot` -- that needs an allocated cache. The head's single
    attention layer has the trunk's head geometry, so K and V over ``window`` positions is the
    whole cost.
    """
    per_row = config.num_kv_heads * config.head_dim * 2 * itemsize   # K and V
    return window * per_row


__all__ = [
    "MTP_KV_WINDOW",
    "MTP_LAYER_ID",
    "MtpDraftBatch",
    "MtpKV",
    "MtpWindowReq",
    "window_bytes_per_request",
]
