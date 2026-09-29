"""A single k=1 self-speculative round for one request, outside the scheduler.

Phase 2 proves the round is *correct* before anything tries to be fast or concurrent, so this
drives one request by hand through the real Engine: real paged KV, real GDN recurrent and conv
state, real PLE state, real expert cache. What it deliberately does not do is multiplex,
overlap, or capture CUDA graphs -- those belong to the scheduler integration this round has to
earn its place in first.

The round, for one request that has already prefilled and is decoding::

    plain target step over x_n       -> y1     (one token, the unchanged production path)
    MTP draft on (y1, R_n)          -> d1     (head's own KV window)
    target verify over [y1, d1]      -> verify (phase="verify", 2-token varlen)
    accept iff d1 is the target's own argmax there
    commit y1 and d1                |  commit y1, then roll the draft back

Why the rollback re-runs the accepted token instead of undoing the draft. The 2-token verify
forward leaves the GDN recurrent and conv state, and both PLE states, holding the state *after*
``d1``; a reject needs the state *after* ``y1*. Those kernels overwrite in place and keep no
previous value, so any undo would have to snapshot 110.29 MiB per request before every step
regardless -- the snapshot is unavoidable either way, so the cheap branch is the one that
reuses it. Re-forwarding the accepted token from that snapshot is exactly the sequence plain AR
would have run, which is what makes the state comparison in the test meaningful rather than
approximate.

Snapshot and restore both run on the engine stream, program-ordered against the forward by that
stream, so neither can race the forward in flight. That is the same discipline
``Scheduler._restore_linear_states`` depends on, and it is the reason this driver does not run
on the scheduler stream.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from freetoken.core import Batch, Req


# How much of the verify's top-k must survive in the re-forward, and how well the two
# distributions must correlate.
#
# Both bars are set by a measurement, not by taste. On this checkpoint bf16 is not run-to-run
# deterministic over 48 layers, so the bars cannot be tight. Measured, same request, same
# state, after a restore:
#
#   verify row 0 vs the re-forward : max|delta| 3.06, correlation 0.957, top-50 overlap 45/50
#   the SAME one-token path, twice   : max|delta| 3.12, correlation 0.961
#
# The cross-path difference is *smaller* than the path's own run-to-run spread, which is the
# definition of "the two computations are the same up to the arithmetic's reassociation". A
# rollback that restored the wrong state would not correlate at all; these bars only need to
# catch that, so they sit below the noise, not inside it.
_MIN_TOP_SET_OVERLAP = 0.5
_MIN_LOGIT_CORRELATION = 0.9

# How many pre-mixer residual rows the prefill capture keeps, and so the bound on that buffer.
# The head's window is 128 rows and the priming needs one residual per primed row, so a
# window-sized tail is the largest a priming pass can consume.
_MTP_PRIME_ROWS = 128


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two logit vectors, centred, so scale cancels."""
    x, y = a.float() - a.float().mean(), b.float() - b.float().mean()
    denom = x.norm() * y.norm()
    return float((x * y).sum() / denom) if float(denom) > 0 else 0.0


def _top_set_overlap(a: torch.Tensor, b: torch.Tensor, k: int = 20) -> float:
    """Fraction of a's top-k token ids that also appear in b's top-k.

    Order-insensitive on purpose: the two paths are the same computation with a different
    reduction order, so near-ties reorder. A wrong state does not reorder, it diverges.
    """
    ta = set(a.topk(k).indices.tolist())
    tb = set(b.topk(k).indices.tolist())
    return len(ta & tb) / max(1, len(ta))


def _confidence(logits: torch.Tensor) -> dict:
    """How decided one distribution is: its top-1 probability and its top-1/top-2 margin.

    Both are wanted because they answer different questions. The margin is a property of the
    top of the distribution and costs one topk; the probability also accounts for the rest of
    the vocabulary, which is what a confidence gate should really look at. Reading either costs
    a device sync, so this is only called under instrumentation.

    The probability is over the *softmax of the row as computed*, i.e. bf16 logits promoted to
    fp32 before the reduction, so it is a property of this forward and not of a re-run.
    """
    row = logits.float()
    top2 = row.topk(2).values
    peak = row.max()
    total = float(torch.exp(row - peak).sum())
    top1 = float(top2[0])
    return {
        "prob": float(torch.exp(top1 - peak)) / total,
        "margin": top1 - float(top2[1]),
        "top1": top1,
    }


@dataclass
class DraftOutcome:
    """What one round decided. ``target_next`` is the token plain AR would emit next."""

    y1: int
    draft: int
    target_next: int
    accepted: bool
    step: int


class SpeculativeRoundDriver:
    """Runs k=1 speculative rounds for exactly one request.

    Explicitly not a serving path: no concurrency, no overlap, no CUDA graphs. It exists so the
    round's state invariants can be checked against plain AR on the real engine.
    """

    def __init__(self, engine, cache_manager, table_manager, req: Req) -> None:
        self.engine = engine
        # the token pool is the table manager's, not the engine's: the scheduler owns the rows
        # (scheduler.py:135) and a driver outside the scheduler has to be handed them
        self.cache_manager = cache_manager
        self.table_manager = table_manager
        self.req = req
        # the pre-mixer residual the trunk produced for the position the last plain step
        # consumed; the head's draft is conditioned on it
        self._last_pre_mixer: torch.Tensor | None = None
        self._snap_slot: int | None = None
        # the request's token-pool and page-table rows as of the last snapshot
        self._snap_token_pool: torch.Tensor | None = None
        self._snap_page_table: torch.Tensor | None = None
        # the head window's live row count as of the last snapshot
        self._snap_window_live: int | None = None
        # cached_len and device_len as of the last snapshot
        self._snap_lengths: tuple[int, int] | None = None
        # rows the prefill primed into the head's window, before any draft
        self._primed_rows = 0
        # a token to draft instead of running the head, for a test that needs one particular
        # accept or reject; None means "ask the head"
        self._forced_draft: int | None = None
        # decision metrics for the acceptance benchmark. Off by default: turning them on costs
        # a topk and a device sync per round, which is fine for a measurement run and not for
        # a decode loop, so the production path never pays for them.
        self._instrument = False
        self._last_metrics: dict = {}
        # The rewind's distributional check is a production guard, and its bars were measured on
        # a five-token prompt. On a two-thousand-token context the same forward is measurably
        # less reproducible, so a long-context run can trip a bar that says nothing about the
        # rollback. A measurement run turns the guard off and records the numbers instead --
        # ``_last_rewind`` is populated either way, so nothing stops being observable.
        self._assert_rewind = True
        self._steps = 0
        self._proposed = 0
        self._accepted = 0
        self._rejects = 0
        # the speculative page(s) the current verify chunk claimed, and the head window row it
        # claimed; both live only between the claim and the commit/rewind
        self._verify_pages: torch.Tensor | None = None
        self._verify_opened: torch.Tensor | None = None
        self._verify_page_row: torch.Tensor | None = None
        self._verify_lo = 0
        self._draft_row_slot: int | None = None
        # the token the verify staged at the second position, to un-write on a reject
        self._draft_slot: int | None = None
        # resolved on the first batch build; mrope needs the [3, T] rope rows
        self._model_is_mrope = False
        # the verify chunk's row 0, kept for the reject path's cross-check
        self._verify_row0: torch.Tensor | None = None
        self._verify_top0 = 0.0
        # the most recent target batch, reused to shape the head's embedding lookup: the head
        # reads the same shared embedding the trunk just used, and embed_input_ids only consults
        # the batch for its multimodal columns, which a text-only draft never has
        self._draft_batch_stub = Batch(reqs=[], phase="decode")
        self._draft_batch_stub.padded_reqs = []

    @property
    def steps(self) -> int:
        return self._steps

    @property
    def proposed(self) -> int:
        return self._proposed

    @property
    def accepted(self) -> int:
        return self._accepted

    @property
    def rejects(self) -> int:
        return self._rejects

    def round(self) -> DraftOutcome:
        """One full speculative round."""
        req = self.req

        # 1. the plain target step: same batch and same path a decode step takes, so y1 is
        #    exactly the token plain AR would have emitted at this position.
        y1 = self._plain_step()
        self._last_pre_mixer = self._take_pre_mixer()

        # 2. snapshot the post-step state. Taken before the verify forward overwrites it. On the
        #    engine stream, so it is ordered against the verify's writes rather than racing them.
        with torch.cuda.stream(self.engine.stream):
            snap, snap_lengths = self._snapshot()

        # 3. the draft, against the head's own window. It claims a window row, which a reject
        #    has to give back -- so the claim exists before the verify, and the snapshot above is
        #    the state a correct rollback returns to.
        draft_pos = req.cached_len          # the position the draft's query sits at
        d1 = self._draft(y1)

        # 4. the verify: [y1, d1] as one 2-token varlen chunk. forward_verify does not sample
        #    and does not complete the request, so the chunk is still uncommitted.
        target_next = self._verify(y1, d1)
        accepted = int(target_next) == int(d1)
        self._proposed += 1
        self._accepted += int(accepted)
        self._rejects += int(not accepted)
        if self._instrument:
            # the position the draft occupied before the commit, so an accept's second token
            # and a reject's re-forward both still report the decision's own coordinates
            self._last_metrics.update({
                "step": self._steps + 1,
                "draft_index": 0,               # k=1: the only draft in the round
                "position": draft_pos,
                "context_len": int(draft_pos),
                "window_live": int(self._window_live()),
                "draft_token": int(d1),
                "target_token": int(target_next),
                "accepted": bool(accepted),
            })

        # 5. commit the accepted prefix. Either way the request ends positioned at the last
        #    committed token, which is what the next round's plain step forwards.
        if accepted:
            self._commit_draft(d1)
        else:
            self._rewind_after_reject(snap, target_next, snap_lengths)

        self._steps += 1
        return DraftOutcome(
            y1=int(y1), draft=int(d1), target_next=int(target_next),
            accepted=accepted, step=self._steps,
        )

    # ------------------------------------------------------------------ target forwards

    def _forward(self, batch: Batch) -> torch.Tensor:
        """The trunk's logits for this batch, keeping the pre-mixer residual for the head.

        The residual comes out of the trunk forward because the head consumes exactly the tensor
        the normal path drops, so re-deriving it would mean a second trunk forward.

        Runs on the engine stream, like every production forward. That is not cosmetic: the
        reject path's snapshot restore has to be program-ordered after the verify's writes to
        the same GDN slot, and only a single stream guarantees that. A driver issuing the
        verify on the ambient stream and the restore on the engine stream would race, and the
        re-forward would read a half-restored state.
        """
        engine = self.engine
        # Same gather the scheduler's _forward does: the (row, column) TUPLE element-wise
        # indexes the pool, which is what yields one token id per position rather than a row.
        batch.input_ids = self.table_manager.token_pool[batch.input_index].to(torch.int32)
        with torch.cuda.stream(engine.stream), engine.ctx.forward_batch(batch), \
                engine.model.forward_host_ctx(batch, False):
            logits, pre_mixer = engine.model.forward_with_residual()
        self._residual = pre_mixer
        return logits

    def _plain_step(self) -> torch.Tensor:
        """One target step over the request's single next token, the unchanged decode path.

        The bookkeeping order is the scheduler's (scheduler.py:1033 then :348): complete_one
        first, which moves the forward's token into the committed prefix and leaves
        ``cached_len`` pointing at the sampled token's own position, and only then append it to
        the host history. Writing the token before complete_one would put it one position early
        and the next forward would read the previous token's slot.
        """
        req = self.req
        batch = self._build_batch("decode")
        logits = self._forward(batch)
        y1 = logits[-1].argmax().to(torch.int32)
        # kept for the cross-path check: this row and the round's verify row 0 are the same
        # distribution taken from the same state by two different paths, so comparing them
        # measures the chunk-vs-step deviation without ever comparing diverged histories
        self._plain_row_last = logits[-1].float()
        self._append_committed(y1)
        return y1

    def _verify(self, y1: torch.Tensor, d1: torch.Tensor) -> torch.Tensor:
        """The target over ``[y1, d1]``, returning the token plain AR would emit next.

        The chunk is staged on the request's host history so ``extend_len`` is 2, which is what
        makes every varlen path (GDN, QSA, MoE, LM head) take its multi-token branch. The
        request is put back afterwards: forward_verify deliberately does not complete it, and
        how much of the chunk is accepted is not known yet.
        """
        req = self.req
        before = (req.cached_len, req.device_len)
        # The chunk is [y1, d1]. The request was left positioned so that the *next* forward
        # would consume y1, i.e. cached_len names y1's own position -- so y1 is the first token
        # of the chunk and d1 the second, at cached_len + 1. extend_len must become 2 to put
        # every varlen path on its multi-token branch; append_host grows only the host view, so
        # device_len is bumped to match.
        assert req.extend_len == 1, (
            f"the verify needs a one-token remainder to extend, got {req.extend_len}"
        )
        self.table_manager.token_pool[req.table_idx, req.cached_len + 1] = d1.to(
            self.engine.device, dtype=torch.int32
        )
        self._draft_slot = req.cached_len + 1
        req.append_host(d1.view(1).cpu())
        req.device_len += 1
        assert req.extend_len == 2, f"verify chunk must be 2 tokens, got {req.extend_len}"
        self.cache_manager.allocate_paged([req])
        # whatever the chunk allocated beyond the request's committed prefix is speculative,
        # and a reject has to give it back. Read after the allocation stamps the page table,
        # which is the only point where the new slots are visible.
        self._verify_pages = self._claim_verify_pages()
        try:
            batch = self._build_batch("verify")
            # every row of the chunk; row 0 is the distribution after y1, which is what d1 is
            # checked against. Row 1 is the next round's free first token, unused here.
            logits = self.engine.forward_verify(batch, self.table_manager.token_pool)
        finally:
            # forward_verify completed nothing, so both lengths go back to what they were. The
            # host history is a view of _ids_buf, so re-taking the view at the restored
            # device_len is what un-appends the drafted token -- _ids_buf keeps it, and the next
            # round's append overwrites that slot. What the chunk wrote into the token pool and
            # the page table is left in place: an accept needs it, and a reject unwinds it in
            # _restore, which is the only place that knows the claims are speculative.
            req.cached_len, req.device_len = before
            req.input_ids = req._ids_buf[: req.device_len]
        # row 0, not the last: the chunk is [y1, d1], so the last row is the distribution after
        # d1, while the draft is checked against the distribution after y1. Kept for the
        # reject-path assertion, which compares the re-forward against this exact row.
        self._verify_row0 = logits[0].float()
        self._verify_top0 = float(self._verify_row0.max())
        if self._instrument:
            self._last_metrics["target"] = _confidence(self._verify_row0)
        return logits[0].argmax().to(torch.int32)

    def _claim_verify_pages(self) -> None:
        """Record what the verify chunk allocated, for a reject to hand back.

        Read after ``allocate_paged`` stamped the page table, which is the only point where the
        new slots are visible, and narrowed to the positions the chunk added: those are exactly
        the speculative ones, and the earlier positions were already charged to the request.

        Nothing is freed here. The chunk may yet be accepted, in which case the drafted token is
        committed and its page belongs to the request; freeing at claim time and re-allocating on
        accept would be a needless round trip through the free list. :meth:`_free_verify_slot` is
        the only place that returns pages.
        """
        req = self.req
        ps = self.cache_manager.page_size
        lo, hi = req.cached_len, req.device_len
        if hi <= lo:
            self._verify_pages = None
            return
        slots = self.engine.page_table[req.table_idx, lo:hi].clone()
        if ps == 1:
            self._verify_pages = slots
            return
        # A page-aligned return cannot be "the slots the chunk added" when the chunk lands
        # mid-page: it shares that page with committed positions, and _free takes whole pages.
        # So record the pages the chunk *opened* -- those the request did not already own -- and
        # the page_table cells it stamped, which do have to be restored either way.
        already_owned = {int(s) // ps for s in
                         self.engine.page_table[req.table_idx, :lo].tolist()}
        opened = sorted({int(s) // ps for s in slots.tolist()} - already_owned)
        self._verify_pages = slots
        self._verify_lo = lo
        self._verify_opened = torch.tensor(
            [page * ps for page in opened], dtype=slots.dtype, device=slots.device
        )
        # and the page table itself, since a freed page must not stay referenced
        self._verify_page_row = self.engine.page_table[req.table_idx, lo:hi].clone()

    def _commit_draft(self, d1: torch.Tensor) -> None:
        """Commit the drafted token after an accept.

        The verify already ran [y1, d1] and left the GDN, PLE and KV state holding d1, so this
        only has to move the request's bookkeeping past d1 -- the same two steps a plain step
        does, in the same order. Nothing is re-run and no page is returned: the state is right.
        """
        req = self.req
        self._append_committed(d1)
        # the verify's pages and the head's window row now belong to the request
        self._verify_pages = None
        self._draft_row_slot = None

    def _restore(
        self, snap: int, lengths: tuple[int, int] | None = None
    ) -> None:
        """Put the request back to the snapshot, on the engine stream.

        Everything a rejected draft claimed has to go back, not just the recurrent state:
        the head's window row and the trunk page the verify chunk allocated. The GDN/PLE state
        is a copy; the window length and the free-page list are bookkeeping, so they are reset
        by hand.

        Split out of :meth:`_rewind_after_reject` so a test can compare the restored state
        against the snapshot before the re-forward overwrites it again -- that comparison is the
        local rollback proof, and it does not depend on any forward's numerics.

        The scalar bookkeeping is read off the snapshot the caller NAMES, not off whatever
        ``self._snap_*`` happens to hold. The driver keeps one snapshot for its life, so
        ``_restore(snap)`` and ``_snapshot()`` are a matched pair today -- but a caller that
        replays a state (a test forcing a decision, say) holds two, and reading the lengths out
        of the wrong one silently hands the request a different position than the state it just
        restored. Measured doing exactly that: the state came back bit-exact while ``cached_len``
        came back one position ahead, so the next forward computed a different chunk and the
        verify named a different token. Hence ``lengths`` is a parameter, and a caller that
        replays passes the pair it snapshotted with.

        The stream context is load-bearing. ``copy_from`` is a plain torch op, so it lands on
        whichever stream is current at the call site; if that is the scheduler's stream while
        the next forward runs on the engine's, the two are unordered and the re-forward can read
        a half-restored state. Doing the copy under the stream the forward uses is what makes
        the rollback correct rather than merely plausible.
        """
        engine = self.engine
        pool = engine.linear_state_pool
        assert pool is not None, "the reject path needs a GDN state pool to snapshot"
        req = self.req
        with torch.cuda.stream(engine.stream):
            pool.copy_from(snap, self.req.linear_slot_idx)
        # the page table cells and the staged token the chunk wrote are speculative, and the
        # snapshot is the only record of what they were -- a freed page must not stay
        # referenced, and the token at that position was not a placeholder.
        width = self.engine.page_table.shape[1]
        self.engine.page_table[self.req.table_idx, :width] = self._snap_page_table
        self.table_manager.token_pool[self.req.table_idx, :width] = self._snap_token_pool
        self._verify_page_row = None
        self._draft_slot = None
        self._free_verify_slot()
        head = engine.model.mtp
        if self._snap_window_live is not None and head is not None and head._kv is not None:
            head.kv.set_live(self._window_slot(), self._snap_window_live)
        cached_len, device_len = lengths if lengths is not None else self._snap_lengths
        if cached_len is not None:
            req.cached_len, req.device_len = cached_len, device_len
            req.input_ids = req._ids_buf[: req.device_len]
        self._draft_row_slot = None

    def _rewind_after_reject(
        self, snap: int, target_next: torch.Tensor, lengths: tuple[int, int] | None = None
    ) -> None:
        """Return to the state plain AR would hold after ``y1``, then commit the re-forward's
        own sample.

        The verify's 2-token chunk left the recurrent, conv and PLE state holding ``d1``, but a
        reject needs them holding the accepted prefix. Those kernels overwrite in place and keep
        no previous value, so the state is rebuilt by restoring the snapshot and re-running the
        one accepted token -- exactly the step plain AR takes. The re-forward is not wasted: its
        output is the token the round commits.
        """
        engine = self.engine
        req = self.req
        self._restore(snap, lengths)
        with torch.cuda.stream(engine.stream):
            batch = self._build_batch("decode")
            logits = self._forward(batch)
        sampled = logits[-1].argmax().to(torch.int32)
        # The chunk's row 0 and this re-forward are the same question asked twice, and the
        # reject path's claim is that the re-forward rebuilds plain AR's state. They agree
        # numerically but not necessarily in argmax: the chunk reduces the same bf16 sums in a
        # different order than a one-token step, and a top-2 gap of ~0.15 is well inside that
        # noise. Measured here: same top-3 set reordered, per-element |delta| up to 0.69
        # against a logit spread of 2.1.
        #
        # So the invariant is distributional, not an argmax equality. What must hold is that
        # the re-forward is the same distribution -- a wrong rollback shows up as a genuinely
        # different distribution, not a reshuffle.
        rf = logits[-1].float()
        agree = _top_set_overlap(self._verify_row0, rf)
        corr = _correlation(self._verify_row0, rf)
        self._last_rewind = {
            "top_overlap": agree,
            "corr": corr,
            "max_abs": float((self._verify_row0 - rf).abs().max()),
            "mean_abs": float((self._verify_row0 - rf).abs().mean()),
            "rel_l2": float(
                (self._verify_row0 - rf).norm()
                / self._verify_row0.norm().clamp_min(1e-6)
            ),
        }
        if self._assert_rewind and (
            agree < _MIN_TOP_SET_OVERLAP or corr < _MIN_LOGIT_CORRELATION
        ):
            raise AssertionError(
                f"the re-forward did not rebuild the verify's distribution: top-20 overlap "
                f"{agree:.0%} (need {_MIN_TOP_SET_OVERLAP:.0%}), correlation {corr:.3f} "
                f"(need {_MIN_LOGIT_CORRELATION}). verify top "
                f"{self._verify_row0.topk(5).indices.tolist()}, re-forward top "
                f"{rf.topk(5).indices.tolist()}"
            )
        # Whichever token the re-forward itself produced is the one this engine would emit
        # next, so it -- not the chunk's row-0 argmax -- is what gets committed.
        # give back the draft's claims, then commit the token the re-forward produced
        self._append_committed(sampled)

    # ------------------------------------------------------------------ draft

    def _append_committed(self, token: torch.Tensor) -> None:
        """Advance the request past the token it just forwarded.

        ``token`` is the sample from the forward that just ran, so it is the token the *next*
        forward will consume: it belongs in the host history and in the token pool at the slot
        the next forward reads, which is ``cached_len`` after ``complete_one``.

        The order is the scheduler's (scheduler.py:348 then :1033): complete_one advances the
        lengths so ``cached_len`` names the new token's slot, then the token is stored there.
        Writing it before the bump would put it one position early, so the next forward would
        read the previous token's slot.
        """
        req = self.req
        req.complete_one()
        self.table_manager.token_pool[req.table_idx, req.cached_len] = token
        req.append_host(token.view(1).cpu())

    def _draft(self, y1: torch.Tensor) -> torch.Tensor:
        """The head's one-token draft.

        The head consumes the embedding of the token it predicts *after*, against the trunk's
        pre-mixer residual for the previous position, and writes one row into its own window.
        No trunk state and no trunk page are touched, which is what makes a reject cheap.

        ``_forced_draft`` short-circuits the head for a test that needs a particular accept or
        reject: the commit and rollback machinery is worth testing on both decisions, and the
        head's own accuracy is not a reliable way to ask for either one.
        """
        engine = self.engine
        model = engine.model
        head = model.mtp
        assert head is not None, "the engine was built without --enable-mtp"
        assert self._last_pre_mixer is not None
        kv = head.ensure_kv(
            self._window_slots(), device=engine.device, dtype=engine.dtype
        )
        if self._forced_draft is not None:
            # still claim the row: a forced draft exercises the same window bookkeeping
            forced = self._forced_draft
            pos = self.req.cached_len
            positions = torch.tensor([pos], dtype=torch.int32, device=engine.device)
            mrope = None
            if self._model_is_mrope:
                mrope = torch.tensor(
                    [[pos], [pos], [pos]], dtype=torch.int32, device=engine.device
                )
            kv.reserve([(self._window_slot(), 1)], positions, mrope)
            self._draft_row_slot = self._window_slot()
            return torch.tensor(forced, dtype=torch.int32)
        # the head's input is the shared embedding, read through the same helper the trunk uses
        from freetoken.models.blocks import embed_input_ids

        embedded = embed_input_ids(
            model.model.embed_tokens, y1.view(1).to(engine.device), self._draft_batch_stub
        )
        pos = self.req.cached_len
        positions = torch.tensor([pos], dtype=torch.int32, device=engine.device)
        # mrope needs the 3-axis rope rows; on a non-mrope model None makes the head use the
        # sequence index, exactly as Batch.get_attn_positions does
        mrope = None
        if self._model_is_mrope:
            mrope = torch.tensor(
                [[pos], [pos], [pos]], dtype=torch.int32, device=engine.device
            )
        window = kv.reserve([(self._window_slot(), 1)], positions, mrope)
        # the pre-mixer residual is the whole [hc_count*hidden] vector, already one row for
        # this single-token step: no view, the decoder re-broadcasts it inside the head
        with kv.draft(window):
            hidden = head.draft_hidden(embedded, self._last_pre_mixer, window)
            # the head collapses to [T, hidden] and owns no vocabulary projection: the shared
            # lm_head is the trunk's, and the argmax has to run over tokens, not over hidden
            # dims. forward_all, not forward: this is one row already, and forward() would want
            # attention metadata to pick a last row out of a batch the head does not have.
            logits = model.lm_head.forward_all(hidden)
        self._draft_row_slot = self._window_slot()
        if self._instrument:
            self._last_metrics["draft"] = _confidence(logits[-1].float())
        return logits[-1].argmax().to(torch.int32).cpu()

    def _window_live(self) -> int:
        """The head window's live row count, or 0 before the head has been used."""
        head = self.engine.model.mtp
        if head is None or head._kv is None:
            return 0
        return head.kv.live_rows(self._window_slot())

    def _window_slot(self) -> int:
        """The head window's slot for this request.

        The window is sized by the page table's row count (``max_running_req + 1``), which is
        exactly the number of request slots, so the table index is a valid window index.
        """
        return self.req.table_idx

    def _window_slots(self) -> int:
        return self.engine.page_table.shape[0]

    def _free_verify_slot(self) -> None:
        """Hand back everything the verify chunk claimed, on an accept-free path.

        The page table cells the chunk stamped are restored from the snapshot (below), and any
        page the chunk *opened* goes back to the free list. A page the chunk merely shared with
        committed positions is left alone: freeing it would return memory the request still owns.
        """
        if self._verify_opened is not None and len(self._verify_opened):
            with self.cache_manager.lazy_free_region():
                self.cache_manager._free(self._verify_opened)
        self._verify_opened = None
        self._verify_pages = None

    def release(self) -> None:
        """Hand the request's slots back, the way the scheduler retires one.

        Self-contained on purpose: ``CacheManager.cache_req`` is the scheduler's path and it
        runs the prefix-cache bookkeeping, which needs a real radix handle. A driver outside the
        scheduler has no such handle, so the pages are returned directly -- page-aligned starts
        of the pages the request's committed prefix occupies, the same granularity ``_free``
        takes. ``table_idx == -1`` marks the request as already gone, so a double release
        cannot hand one slot to two requests.
        """
        req = self.req
        if req.table_idx == -1:
            return
        head = self.engine.model.mtp
        if head is not None and head._kv is not None:
            head.kv.reset_slot(req.table_idx)
        pool = self.engine.linear_state_pool
        if pool is not None and req.linear_slot_idx is not None:
            pool.free([req.linear_slot_idx])
            req.linear_slot_idx = None
        slots = self.engine.page_table[req.table_idx, : req.cached_len]
        if slots.numel():
            page = self.cache_manager.page_size
            starts = torch.unique(torch.div(slots, page, rounding_mode="floor") * page)
            with self.cache_manager.lazy_free_region():
                self.cache_manager._free(starts.to(torch.int32))
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    # ------------------------------------------------------------------ prompt prefill

    def prefill_prompt(self) -> None:
        """Run the request's prompt through the engine's prefill forward.

        The round under test starts from an admitted decoding request; getting there is the
        scheduler's job, so this is the production prefill forward and the production sampling
        args, not a re-implementation.
        """
        from freetoken.core import Batch
        from freetoken.attention.linear import build_fla_metadata
        from freetoken.scheduler.scheduler import (
            _make_input_tuple,
            _make_mrope_positions,
            _make_positions,
        )

        engine = self.engine
        req = self.req
        n = req.device_len
        self._model_is_mrope = bool(engine.config.model_config.model_is_mrope)
        self.table_manager.token_pool[req.table_idx, :n] = req.input_ids.to(
            self.table_manager.token_pool.device
        )
        self.cache_manager.allocate_paged([req])
        batch = Batch(reqs=[req], phase="prefill")
        batch.padded_reqs = [req]
        batch.positions = _make_positions(batch, engine.device)
        if bool(engine.config.model_config.model_is_mrope):
            batch.mrope_positions = _make_mrope_positions(batch, engine.device)
        index = _make_input_tuple(batch, engine.device)
        batch.input_index = index
        batch.out_loc = engine.page_table[index]
        batch.input_ids = self.table_manager.token_pool[index].to(torch.int32)
        if engine.linear_state_pool is not None:
            batch.fla_metadata = build_fla_metadata(batch, engine.device)
        engine.attn_backend.prepare_metadata(batch)
        # the head's window is primed from this forward's residuals, so arm the capture before
        # it runs; it disarms itself and costs nothing when MTP is off
        engine.model.capture_pre_mixer_tail(_MTP_PRIME_ROWS)
        # forward_batch completes the request once, so the prompt is consumed and the request
        # sits exactly where a scheduled prefill would leave it
        out = engine.forward_batch(batch, engine.sampler.prepare(batch))
        # forward_batch already called complete_one for the prefill chunk, which advanced
        # cached_len onto the slot the *sampled* token occupies. The scheduler's drain then
        # writes that token there and appends it to the host history (scheduler.py:348); a
        # driver outside the scheduler has to do the same or the next forward reads whatever the
        # prompt left in the slot. Note complete_one is NOT repeated here: it ran inside the
        # forward, and calling it twice would push cached_len a second slot ahead.
        out.copy_done_event.synchronize()
        first = int(out.next_tokens_cpu[0])
        self.table_manager.token_pool[req.table_idx, req.cached_len] = first
        req.append_host(torch.tensor([first], dtype=req.input_ids.dtype))
        # after the drain, so the window's last row sees the sampled token the first draft
        # continues from
        self._primed_rows = self._prime_head_window(n)

    def _prime_head_window(self, prompt_len: int) -> int:
        """Fill the head's window over the prompt's tail in one forward; returns the row count.

        The first draft's query sits one position past the last primed row and writes its own
        row there, so the window keeps ``window - 1`` primed rows and has exactly one left.
        Priming every row the window could hold would evict the draft's own.

        Row ``t`` is the token at position ``t`` against the residual of ``t-1``, the Phase-0
        prober's ``prime``; hence the ``t >= 1`` floor, and hence ``target_residual`` being the
        tail shifted by one row relative to the tokens.
        """
        from freetoken.models.blocks import embed_input_ids

        engine = self.engine
        head = engine.model.mtp
        assert head is not None, "the engine was built without --enable-mtp"
        kv = head.ensure_kv(self._window_slots(), device=engine.device, dtype=engine.dtype)
        lo = max(1, prompt_len - (kv.window - 1) + 1)
        count = prompt_len - lo + 1
        tail = engine.model.pre_mixer_tail
        assert tail is not None, "the prefill ran without a residual capture"
        assert tail.shape[0] >= count, (
            f"captured {tail.shape[0]} residuals, need {count}"
        )
        # residual of t-1 for t in [lo, prompt_len]: the tail's own last `count` rows
        prev = tail[-count:].to(engine.device, dtype=engine.dtype)
        ids = self.table_manager.token_pool[self.req.table_idx, lo : prompt_len + 1].to(
            engine.device, dtype=torch.int32
        )
        positions = torch.arange(lo, prompt_len + 1, dtype=torch.int32, device=engine.device)
        mrope = None
        if self._model_is_mrope:
            mrope = positions.unsqueeze(0).expand(3, -1).contiguous()
        with torch.cuda.stream(engine.stream):
            embedded = embed_input_ids(
                engine.model.model.embed_tokens, ids, self._draft_batch_stub
            )
            head.prime_window(self._window_slot(), embedded, prev, positions, mrope)
        return count

    # ------------------------------------------------------------------ reference AR

    def plain_step(self) -> torch.Tensor:
        """One plain AR step, with no drafting at all.

        The reference the round is measured against. It runs the same target forward as
        ``_plain_step`` and advances the same bookkeeping, so a speculative run and a plain run
        over the same prompt differ only in what the round does between them.
        """
        return self._plain_step()

    def mtp_kv_state(self) -> dict:
        """The head's window rows for this request, for the state comparison.

        Separate from :meth:`_window_slot` so a test can read the window without reaching into
        the head's private attribute.
        """
        head = self.engine.model.mtp
        assert head is not None and head._kv is not None
        slot = self._window_slot()
        k, v = head.kv.kv(slot)
        return {"k": k.clone(), "v": v.clone(), "live": head.kv.live_rows(slot)}

    def _committed_kv(self) -> torch.Tensor:
        """The trunk KV this request's committed positions point at, gathered by slot.

        Read through the pool rather than by page id, so a slot that has been re-pointed (or
        freed and handed to someone else) shows up as different content instead of silently
        comparing equal. Only the attention layers have paged storage, so the first one is used:
        layer 0 of this model is a GDN layer and has no KV at all.
        """
        req = self.req
        slots = self.engine.page_table[req.table_idx, : req.cached_len].long()
        if slots.numel() == 0:
            return torch.zeros(0, dtype=torch.bfloat16, device=self.engine.device)
        layer = 0
        while layer < self.engine.kv_cache.num_layers:
            try:
                slabs = (self.engine.kv_cache.k_cache(layer), self.engine.kv_cache.v_cache(layer))
                break
            except KeyError:
                layer += 1
        else:
            return torch.zeros(0, dtype=torch.bfloat16, device=self.engine.device)
        parts = []
        for slab in slabs:
            flat = slab.view(-1, *slab.shape[2:])
            parts.append(flat[slots].reshape(slots.numel(), -1))
        return torch.cat(parts, dim=1)

    def rollback_state(self) -> dict:
        """Every state a reject has to put back, as plain tensors.

        The local rollback proof compares this against the snapshot taken before the verify, so
        it covers exactly what the restore is responsible for and nothing that a forward
        recomputes. Token history, the page table and the free-page count are included because a
        rollback that leaves a speculative page allocated leaks it, which a state-only comparison
        would not catch.
        """
        engine = self.engine
        req = self.req
        pool = engine.linear_state_pool
        state: dict = {
            "cached_len": req.cached_len,
            "device_len": req.device_len,
            "hist_len": req.input_ids.numel(),
            "page_table": engine.page_table[req.table_idx, : req.device_len + 2].clone(),
            "free_pages": int(len(self.cache_manager.free_slots)),
            "token_pool": self.table_manager.token_pool[
                req.table_idx, : req.device_len + 2
            ].clone(),
            # the trunk KV the request's committed positions point at: the verify's chunk
            # overwrites the second position's slot, and a rollback has to leave the committed
            # positions' contents exactly as the plain step left them
            "trunk_kv": self._committed_kv(),
        }
        if pool is not None:
            state["gdn_conv"] = pool.conv_states[:, req.linear_slot_idx].clone()
            state["gdn_recurrent"] = pool.recurrent_states[:, req.linear_slot_idx].clone()
            for name, tensor in pool.slot_states.items():
                state[f"ple_{name}"] = tensor[:, req.linear_slot_idx].clone()
        head = engine.model.mtp
        # the window's live count only exists once the head has been used at all, so before the
        # first draft there is nothing to report; 0 is the honest value for "no rows claimed".
        state["mtp_live"] = (
            head.kv.live_rows(self._window_slot()) if head is not None and head._kv is not None
            else 0
        )
        return state

    # ------------------------------------------------------------------ bookkeeping

    def _snapshot(self) -> tuple[int, tuple[int, int]]:
        """Copy the request's live state into the spare slot; return ``(slot, lengths)``.

        The lengths travel with the slot so a caller that replays this state can hand them back
        to :meth:`_restore`; see that method for what goes wrong otherwise.

        The copy is the point: the verify's kernels overwrite conv, recurrent and both PLE
        states in place and keep no previous value, so the only way back is a copy taken before
        they run. The page table and the token pool are recorded alongside it, because a reject
        also has to un-write the cells and the token the chunk staged, and those are plain
        tensors rather than restored state.

        The slot is allocated once and kept for the driver's life -- every step needs it, and
        the pool's free list is LIFO, so re-allocating per step would churn it.
        """
        pool = self.engine.linear_state_pool
        assert pool is not None, "the reject path needs a GDN state pool to snapshot"
        req = self.req
        if self._snap_slot is None:
            self._snap_slot = pool.alloc(1)[0]
        pool.copy_from(self.req.linear_slot_idx, self._snap_slot)
        # the token pool over the request's whole row, so a reject restores what was there
        width = self.engine.page_table.shape[1]
        self._snap_token_pool = self.table_manager.token_pool[
            self.req.table_idx, :width
        ].clone()
        self._snap_page_table = self.engine.page_table[self.req.table_idx, :width].clone()
        head = self.engine.model.mtp
        self._snap_window_live = (
            head.kv.live_rows(self._window_slot()) if head is not None and head._kv is not None
            else 0
        )
        # the lengths and the host-history view, so a restore is a complete state restore and
        # not only a state copy. _rewind_after_reject reaches the same values through _verify's
        # own unwind, so recording them costs nothing there and lets a caller return to a point
        # before the round at all.
        self._snap_lengths = (req.cached_len, req.device_len)
        return self._snap_slot, self._snap_lengths

    def _take_pre_mixer(self) -> torch.Tensor:
        assert self._residual is not None, "no trunk forward has run"
        return self._residual

    def _build_batch(self, phase: str) -> Batch:
        """The one-request batch, built from the scheduler's own metadata helpers.

        No padding (one request, and graphs are off), no multimodal, no SWA bookkeeping: none
        of it is what this driver is testing.
        """
        from freetoken.attention.linear import build_fla_metadata
        from freetoken.scheduler.scheduler import (
            _make_input_tuple,
            _make_mrope_positions,
            _make_positions,
        )

        req = self.req
        engine = self.engine
        device = engine.device
        self._model_is_mrope = bool(engine.config.model_config.model_is_mrope)
        batch = Batch(reqs=[req], phase=phase)
        batch.padded_reqs = [req]
        batch.positions = _make_positions(batch, device)
        # This checkpoint is mrope, so the rotary needs the [3, T] rope rows; get_attn_positions()
        # prefers mrope_positions when set (core.py:Batch), and every backbone here reads that.
        if self._model_is_mrope:
            batch.mrope_positions = _make_mrope_positions(batch, device)
        # _make_input_tuple returns (token_mapping, positions) as a TUPLE, and that tuple is
        # what indexes the 2-D page table: a 2-tensor index is element-wise, so
        # page_table[(rows, cols)] yields the one slot per token. Splitting it would index by
        # row alone and hand back a whole row of slots.
        index = _make_input_tuple(batch, device)
        batch.input_index = index
        batch.out_loc = engine.page_table[index]
        if engine.linear_state_pool is not None:
            # decode needs the per-row GDN state slot staged before build_fla_metadata, which
            # asserts it (linear.py:70); the varlen branch derives the slot itself.
            if phase == "decode":
                batch.linear_table_idx = torch.tensor(
                    [req.linear_slot_idx if req.linear_slot_idx is not None
                     else engine.linear_state_pool.padding_slot],
                    dtype=torch.int32, device="cpu", pin_memory=True,
                ).to(device, non_blocking=True)
            batch.fla_metadata = build_fla_metadata(batch, device)
        engine.attn_backend.prepare_metadata(batch)
        return batch


__all__ = ["DraftOutcome", "SpeculativeRoundDriver"]
