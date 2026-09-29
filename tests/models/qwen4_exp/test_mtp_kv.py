"""The MTP head's own KV window: bounded, rolling, and reversible on a rejected draft.

The head does not share the trunk's paged pool, so these pin the window bookkeeping that
replaces it: the 128-position cap, the roll, the claim/rewind pair, and that a rolled window
still attends to exactly the same keys it held before.
"""

from __future__ import annotations

import torch

from freetoken.models.qwen4_exp.mtp_kv import MTP_KV_WINDOW, MTP_LAYER_ID, MtpKV

from .common import parsed_config, toy_hf_config


def _kv(num_slots: int = 2, window: int = MTP_KV_WINDOW, device="cpu"):
    config = parsed_config()
    return MtpKV(config, num_slots, window=window, device=torch.device(device), dtype=torch.float32)


def _pos(n: int, start: int = 0, device="cpu") -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.int32, device=device)


def test_window_is_capped_and_counts_claimed_rows():
    kv = _kv(num_slots=2, window=8)
    assert kv.live_rows(0) == 0
    kv.reserve([(0, 1)], _pos(1))
    kv.reserve([(0, 1)], _pos(1, 1))
    assert kv.live_rows(0) == 2
    # the second slot is independent
    assert kv.live_rows(1) == 0
    kv.reserve([(1, 3)], _pos(3))
    assert (kv.live_rows(0), kv.live_rows(1)) == (2, 3)


def test_reserve_rolls_when_it_would_overflow():
    kv = _kv(num_slots=1, window=4)
    for step in range(4):
        kv.reserve([(0, 1)], _pos(1, step))
    assert kv.live_rows(0) == 4
    # the fifth row forces a roll that keeps window - n = 3 recent rows
    kv.reserve([(0, 1)], _pos(1, 4))
    assert kv.live_rows(0) == 4


def test_rewind_gives_the_rows_back():
    kv = _kv(num_slots=1, window=8)
    kv.reserve([(0, 2)], _pos(2))
    kv.reserve([(0, 1)], _pos(1, 2))
    assert kv.live_rows(0) == 3
    kv.rewind(0, 1)
    assert kv.live_rows(0) == 2
    kv.rewind(0, 5)          # never goes negative
    assert kv.live_rows(0) == 0


def test_reserve_rejects_a_slot_it_does_not_own():
    kv = _kv(num_slots=2, window=8)
    try:
        kv.reserve([(2, 1)], _pos(1))
    except IndexError:
        return
    raise AssertionError("a slot outside the window was accepted")


def test_roll_moves_the_recent_rows_to_the_front():
    """K is stored post-rotary, so a roll is a pure move of the newest ``keep`` rows."""
    kv = _kv(num_slots=1, window=4)
    k, v = kv.backend.kv(MTP_LAYER_ID)
    for step in range(4):
        kv.reserve([(0, 1)], _pos(1, step))
    k[0, :4] = torch.arange(4, dtype=torch.float32).view(4, 1, 1).expand(4, *k.shape[2:]).contiguous()
    v[0, :4] = -k[0, :4]
    kv.reserve([(0, 1)], _pos(1, 4))          # forces the roll, keeping 3 rows
    assert kv.live_rows(0) == 4
    assert [k[0, i, 0, 0].item() for i in range(3)] == [1.0, 2.0, 3.0]
    assert v[0, 0, 0, 0].item() == -1.0        # V rode along with K
    # Row 3 is claimed but not yet written, so it still holds the row the roll displaced.
    # That is safe because the reference writes the row before it attends over it, and a
    # rewind leaves the same situation for the next draft; it is not required to be zero.
    assert kv.live_rows(0) == 4


def test_bytes_per_slot_is_the_window_not_a_page():
    """The whole point of the window: its size follows the window, not the trunk's page count."""
    kv = _kv(num_slots=1, window=16)
    small = kv.bytes_per_slot()
    bigger = _kv(num_slots=1, window=32).bytes_per_slot()
    assert bigger == 2 * small
    config = parsed_config()
    per_row = 2 * config.num_kv_heads * config.head_dim * 4        # K and V, fp32
    assert small == 16 * per_row


def test_dense_window_attends_to_everything_it_holds():
    """Sanity on the claim bookkeeping reaching the attention: keys past live stay masked out.

    A key past the live length is an unwritten (zero) row, and a zero key still contributes a
    score, so this asserts the reference's own visibility mask, not MtpKV.
    """
    kv = _kv(num_slots=1, window=8)
    batch = kv.reserve([(0, 2)], _pos(2))
    assert batch.padded_reqs[0].cached_len == 0
    assert batch.padded_reqs[0].extend_len == 2
    assert batch.is_decode and not batch.is_prefill and not batch.is_verify
    assert batch.size == 1
    assert torch.equal(batch.get_attn_positions(), _pos(2))
    nxt = kv.reserve([(0, 1)], _pos(1, 2))
    assert nxt.padded_reqs[0].cached_len == 2


def test_batch_reports_all_three_phases():
    kv = _kv(num_slots=1, window=8)
    batch = kv.reserve([(0, 1)], _pos(1))
    assert (batch.is_decode, batch.is_prefill, batch.is_verify) == (True, False, False)
    batch.phase = "verify"
    assert (batch.is_decode, batch.is_prefill, batch.is_verify) == (False, False, True)
    batch.phase = "prefill"
    assert (batch.is_decode, batch.is_prefill, batch.is_verify) == (False, True, False)


def test_reset_clears_every_slot():
    kv = _kv(num_slots=2, window=8)
    kv.reserve([(0, 1), (1, 1)], _pos(2))
    kv.reset()
    assert kv.live_rows(0) == 0 and kv.live_rows(1) == 0
    k, v = kv.backend.kv(MTP_LAYER_ID)
    assert k.abs().sum().item() == 0.0 and v.abs().sum().item() == 0.0
