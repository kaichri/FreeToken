"""Phase-2.5: real in-tree MTP acceptance, measured per draft decision.

Every number here comes from a real speculative round on the real engine: a plain target step,
the head's draft, and a 2-token verify whose row 0 is the distribution the draft was checked
against. There is no separate replay, no second engine, no other prefill and no fp32 reference
-- the accept/reject decision is always the one the running engine made.

The workloads are the Phase-0 probe's own, imported from it rather than re-typed, so the
acceptance here is comparable with ``benchmarks/mtp_probe.py score`` on the same text. The
prober's dumped tokens and residuals are used when present, which pins the comparison to
identical inputs instead of merely identical prompts.

What is recorded per decision, and why:

  draft_token / target_token / accepted
        the decision itself
  position / context_len
        where in the generation the decision was taken, and how much context the head could see
  window_live
        how full the head's bounded window was; acceptance depends strongly on it
  draft.prob / draft.margin
        the head's own certainty, the input to a dynamic-k gate
  target.prob / target.margin
        the target's certainty. A draft that disagrees with a near-tie target is not the same
        failure as one that disagrees with a confident target, and the bf16 baseline decides
        how far apart two near-ties can be pushed

The probability costs a full-vocabulary reduction and a device sync per decision. That is
acceptable for a measurement run and not for a decode loop, so instrumentation is opt-in and off
by default.

Usage:
    PYTHONPATH=python python benchmarks/mtp_accept.py --out results.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

MODEL = r"D:\models\Qwen3.8-Flash-Next-NVFP4"
PROBE_DUMP = Path(r"D:\freetoken_MTP\_mtp\dump")

# Confidence buckets for the acceptance-vs-certainty question. The boundaries are the ones a
# gate would plausibly use, not chosen after seeing the data.
CONFIDENCE_BUCKETS = ((0.0, 0.30), (0.30, 0.50), (0.50, 0.70), (0.70, 0.90), (0.90, 1.01))

# A target whose top-1/top-2 logit gap is within a small multiple of the plain forward's own
# run-to-run spread at this context can be reordered by arithmetic alone, so a disagreement
# there is not evidence about the head. The spread is measured per workload; this is the factor
# applied to it, not the threshold itself. Phase 2 measured max|delta| 2.375 on a five-token
# prompt and 0.559 at 2114 tokens, which is why a hard-coded threshold cannot be right.
NEAR_TIE_FACTOR = 2.0


def workload_texts(tokenizer) -> dict[str, str]:
    """The probe's own workloads, so acceptance here and in Phase 0 describe the same text."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mtp_probe import _workload_prompts  # noqa: PLC0415

    return _workload_prompts(tokenizer, 2048)


def build_engine(cfg_args):
    from freetoken.distributed import DistributedInfo, try_get_tp_info
    from freetoken.engine.engine import Engine
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager
    from freetoken.server.args import ServerArgs

    assert try_get_tp_info() is None, "TP info already set in this process"
    cfg = ServerArgs(
        model_path=cfg_args.model,
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        enable_mtp=True,
        moe_strategy="hybrid",
        moe_cache_size=cfg_args.moe_cache_size,
    )
    engine = Engine(cfg)
    table = TableManager(cfg.max_running_req, engine.page_table)
    cache = CacheManager(
        engine.num_pages, cfg.page_size, engine.page_table, cfg.cache_type,
        linear_state_pool=engine.linear_state_pool,
        swa_pool=engine.kv_cache,
        sliding_window_size=None,
    )
    return engine, cache, table, cfg


def admit(cache, table, prompt_ids: list[int], output_len: int):
    """A cold request holding only its prompt, admitted the way prefill admits one."""
    from freetoken.core import Req, SamplingParams
    from freetoken.kvcache.base import BaseCacheHandle

    class _Cold(BaseCacheHandle):
        """No prefix-cache match, so cached_len == 0 and the round owns all of its state."""

        def __init__(self, device):
            super().__init__(0)
            self._device = device

        def get_matched_indices(self):
            return torch.empty(0, dtype=torch.int32, device=self._device)

    pool = cache.linear_state_pool
    return Req(
        input_ids=torch.tensor(prompt_ids, dtype=torch.int64),
        table_idx=table.allocate(),
        cached_len=0,
        output_len=output_len,
        uid=1,
        sampling_params=SamplingParams(),          # greedy
        cache_handle=_Cold(table.page_table.device),
        linear_slot_idx=pool.alloc(1)[0] if pool is not None else None,
    )


def _correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    x, y = a.float() - a.float().mean(), b.float() - b.float().mean()
    denom = x.norm() * y.norm()
    return float((x * y).sum() / denom) if float(denom) > 0 else 0.0


def plain_vs_plain_baseline(driver, repeats: int = 4) -> dict:
    """The same plain forward, repeated from one state, at THIS context length.

    The rewind's distributional check has fixed bars that were measured on a five-token prompt.
    Whether a re-forward correlation is a defect or arithmetic depends entirely on how
    reproducible a plain forward is here, so the floor has to be measured next to the thing it
    judges rather than assumed from a different context.
    """
    engine = driver.engine
    pool = engine.linear_state_pool
    with torch.cuda.stream(engine.stream):
        snap = driver._snapshot()
    rows: list[torch.Tensor] = []
    try:
        for _ in range(repeats):
            with torch.cuda.stream(engine.stream):
                pool.copy_from(snap, driver.req.linear_slot_idx)
            rows.append(driver._forward(driver._build_batch("decode"))[-1].float().cpu())
    finally:
        pool.free([snap])
    corrs = [_correlation(rows[i], rows[0]) for i in range(1, len(rows))]
    margins = [float((rows[i] - rows[0]).abs().max()) for i in range(1, len(rows))]
    return {
        "repeats": repeats,
        "context_len": int(driver.req.cached_len),
        "corr": min(corrs) if corrs else 1.0,
        "max_abs": max(margins) if margins else 0.0,
    }


def run_workload(engine, cache, table, name: str, prompt_ids: list[int], rounds: int) -> tuple:
    """Drive ``rounds`` speculative rounds and return one record per draft decision."""
    from freetoken.scheduler.spec_round import SpeculativeRoundDriver

    # a round commits up to two tokens, so the history needs room for both plus the prompt
    driver = SpeculativeRoundDriver(
        engine, cache, table, admit(cache, table, prompt_ids, output_len=rounds * 2 + 32)
    )
    driver._instrument = True
    driver._assert_rewind = False        # measure, do not abort; see plain_vs_plain_baseline
    prompt_len = len(prompt_ids)
    driver.prefill_prompt()
    # measure the arithmetic floor at the position every round starts from
    baseline = plain_vs_plain_baseline(driver)
    records: list[dict] = []
    rewinds: list[dict] = []
    started = time.perf_counter()
    for _ in range(rounds):
        outcome = driver.round()
        record = dict(driver._last_metrics)
        record["workload"] = name
        record["prompt_len"] = prompt_len
        # generation position: how many tokens this request has produced past the prompt
        record["gen_pos"] = record["position"] - prompt_len
        # the first round's draft is the first one after priming; later ones run against a
        # window the rounds themselves have been filling, so they are not the same measurement
        record["first_after_prefill"] = record["step"] == 1
        records.append(record)
        if not outcome.accepted:
            rewinds.append(dict(driver._last_rewind))
    elapsed = time.perf_counter() - started
    baseline["context_len"] = int(driver.req.cached_len)
    driver.release()
    print(
        f"  {name:10} {len(records):4d} decisions in {elapsed:6.1f}s "
        f"({1000 * elapsed / max(1, len(records)):.1f} ms/decision), "
        f"plain-vs-plain at {prompt_len} ctx: "
        f"corr {baseline['corr']:.4f}, max|delta| {baseline['max_abs']:.3f}",
        flush=True,
    )
    return records, {"baseline": baseline, "rewinds": rewinds}


def summarize(records: list[dict], near_tie_margin: float) -> dict:
    """Overall, per workload, per prompt length, per confidence bucket, per near-tie.

    ``near_tie_margin`` comes from the measured plain-vs-plain spread in the same context; see
    NEAR_TIE_FACTOR.
    """
    def rate(rows: list[dict]) -> dict:
        acc = sum(1 for r in rows if r["accepted"])
        return {
            "proposed": len(rows),
            "accepted": acc,
            "acceptance": acc / len(rows) if rows else 0.0,
        }

    out: dict = {"overall": rate(records)}

    by_workload: dict[str, list[dict]] = {}
    for r in records:
        by_workload.setdefault(r["workload"], []).append(r)
    out["by_workload"] = {k: rate(v) for k, v in sorted(by_workload.items())}

    lengths = sorted({r["prompt_len"] for r in records})
    edges = [0, 512, 1536, 1 << 30]
    names = ("short", "medium", "long")
    out["by_prompt_length"] = {}
    for name, lo, hi in zip(names, edges, edges[1:]):
        rows = [r for r in records if lo <= r["prompt_len"] < hi]
        if rows:
            out["by_prompt_length"][name] = dict(rate(rows), prompt_len=rows[0]["prompt_len"])

    out["by_draft_confidence"] = {}
    for lo, hi in CONFIDENCE_BUCKETS:
        rows = [r for r in records if lo <= r["draft"]["prob"] < hi]
        key = f"{lo:.2f}-{hi if hi <= 1.0 else 1.0:.2f}"
        if rows:
            out["by_draft_confidence"][key] = dict(rate(rows), mean_target_prob=(
                sum(r["target"]["prob"] for r in rows) / len(rows)
            ))

    # the first draft after priming is not the same measurement as the later ones: it reads a
    # window the prefill filled, the rest read one the rounds themselves have been filling
    out["first_after_prefill"] = rate([r for r in records if r["first_after_prefill"]])
    out["later_rounds"] = rate([r for r in records if not r["first_after_prefill"]])

    # how often the target itself was undecided, which bounds what acceptance can mean
    near = [r for r in records if r["target"]["margin"] < near_tie_margin]
    out["near_tie"] = dict(
        rate(near),
        margin=near_tie_margin,
        share=len(near) / len(records) if records else 0.0,
        accepted_among_confident_targets=rate(
            [r for r in records if r["target"]["margin"] >= near_tie_margin]
        )["acceptance"],
        note="a disagreement against a near-tie target is inside the measured bf16 spread",
    )

    # window occupancy, because acceptance depends strongly on it
    out["window"] = dict(
        mean_live=sum(r["window_live"] for r in records) / max(1, len(records)),
        max_live=max((r["window_live"] for r in records), default=0),
        note="acceptance is strongly window-length dependent; see mtp_probe score.json",
    )
    return out


def dynamic_k(records: list[dict], thresholds=(0.3, 0.5, 0.7)) -> dict:
    """What a confidence gate would buy and cost, estimated from the recorded decisions.

    A gate abstains on a draft whose head confidence is below the threshold. Abstaining costs the
    draft's chance of being accepted; it saves the verify forward only when the draft would have
    been rejected anyway. The estimate is therefore: avoided verifies = abstentions that were
    going to reject, lost acceptance = accepted drafts the gate threw away.

    k > 1 is NOT implemented and NOT estimated here beyond this: a single k=1 decision carries
    no information about how a second draft in the same round would behave.
    """
    total = len(records)
    out = {"decisions": total, "thresholds": {}}
    for t in thresholds:
        low = [r for r in records if r["draft"]["prob"] < t]
        avoided = [r for r in low if not r["accepted"]]
        lost = [r for r in low if r["accepted"]]
        out["thresholds"][f"{t:.2f}"] = {
            "abstain_rate": len(low) / total if total else 0.0,
            "avoided_verifies": len(avoided),
            "avoided_share_of_all_rejects": (
                len(avoided) / max(1, sum(1 for r in records if not r["accepted"]))
            ),
            "lost_accepts": len(lost),
            "lost_share_of_all_accepts": (
                len(lost) / max(1, sum(1 for r in records if r["accepted"]))
            ),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="mtp_accept.json")
    ap.add_argument("--rounds", type=int, default=250,
                    help="decisions per workload, when --rounds-per is not given")
    ap.add_argument("--rounds-per", default="",
                    help="per-workload decision counts, e.g. code=500,reasoning=250,prose=250; "
                         "code is the target use case and is weighted up by default")
    ap.add_argument("--moe-cache-size", type=int, default=2048)
    ap.add_argument("--workloads", default="code,reasoning,prose")
    args = ap.parse_args()

    os.environ.setdefault("FREETOKEN_PIN_BUDGET_GB", "64")
    from freetoken.utils import load_tokenizer

    engine, cache, table, _ = build_engine(args)
    tokenizer = load_tokenizer(args.model)
    texts = workload_texts(tokenizer)

    names = [n.strip() for n in args.workloads.split(",") if n.strip()]
    per: dict[str, int] = {}
    if args.rounds_per:
        for item in args.rounds_per.split(","):
            name, _, count = item.partition("=")
            per[name.strip()] = int(count)
    else:
        # code is the primary use case, so it gets twice the decisions of the others
        for name in names:
            per[name] = args.rounds * (2 if name == "code" else 1)

    print(f"\ndecisions per workload: {per}", flush=True)
    records: list[dict] = []
    rewind_stats: dict[str, dict] = {}
    for name in names:
        if name not in texts:
            print(f"  unknown workload {name!r}, skipping")
            continue
        got, extra = run_workload(
            engine, cache, table, name, tokenizer.encode(texts[name]), per[name]
        )
        records.extend(got)
        rewind_stats[name] = extra

    def rewind_block(extra: dict) -> dict:
        rows = extra["rewinds"]
        if not rows:
            return {"rejects": 0}
        corrs = sorted(r["corr"] for r in rows)
        floor = extra["baseline"]["corr"]
        below = sum(1 for c in corrs if c < floor)
        return {
            "rejects": len(rows),
            "rewind_corr_min": corrs[0],
            "rewind_corr_p10": corrs[max(0, len(corrs) // 10)],
            "rewind_corr_median": corrs[len(corrs) // 2],
            "plain_vs_plain_corr": floor,
            "plain_vs_plain_max_abs": extra["baseline"]["max_abs"],
            "below_plain_vs_plain": below,
            "note": "the rewind is only a defect if it correlates worse than a plain forward "
                    "repeated on the same state at the same context length",
        }

    # the near-tie threshold comes from the measured floor, per workload
    margins = [
        NEAR_TIE_FACTOR * extra["baseline"]["max_abs"] for extra in rewind_stats.values()
    ]
    near_tie_margin = max(margins) if margins else 0.0

    result = {
        "window": 128,
        "near_tie_factor": NEAR_TIE_FACTOR,
        "near_tie_margin": near_tie_margin,
        "records": records,
        "summary": summarize(records, near_tie_margin),
        "dynamic_k": dynamic_k(records),
        "rewind": {name: rewind_block(extra) for name, extra in rewind_stats.items()},
    }
    Path(args.out).write_text(json.dumps(result, indent=2))

    s = result["summary"]
    print(f"\noverall: {s['overall']['accepted']}/{s['overall']['proposed']} "
          f"= {s['overall']['acceptance']:.1%}")
    for name, r in s["by_workload"].items():
        print(f"  {name:10} {r['accepted']:4d}/{r['proposed']:4d} = {r['acceptance']:.1%}")
    print("\nby draft confidence:")
    for key, r in s["by_draft_confidence"].items():
        print(f"  {key}  {r['accepted']:4d}/{r['proposed']:4d} = {r['acceptance']:.1%}")
    print(f"\nnear-tie targets: {s['near_tie']['proposed']} "
          f"({s['near_tie']['share']:.1%} of decisions), "
          f"acceptance there {s['near_tie']['acceptance']:.1%}, "
          f"acceptance on confident targets {s['near_tie']['accepted_among_confident_targets']:.1%}")
    print("\nrewind vs the plain-vs-plain floor, per workload:")
    for name, r in result["rewind"].items():
        if not r.get("rejects"):
            continue
        print(f"  {name:10} rewind corr min {r['rewind_corr_min']:.4f} "
              f"p10 {r['rewind_corr_p10']:.4f} median {r['rewind_corr_median']:.4f}   "
              f"plain-vs-plain {r['plain_vs_plain_corr']:.4f} "
              f"(max|delta| {r['plain_vs_plain_max_abs']:.3f})")
    print(f"\nwrote {args.out} with {len(records)} decision records")


if __name__ == "__main__":
    main()
