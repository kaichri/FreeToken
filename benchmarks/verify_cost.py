"""The question a 2-row verify has to answer against two separate 1-row forwards.

Both sides are measured on the SAME request at the SAME state, so the only difference is the
shape of the target forward. For each phase:

  * unique experts per layer, summed and maxed  -- the union, which is the whole point: if the
    second row routes into experts the first row already pulled, the cache pays nothing for it
  * misses and PCIe fetches, from the engine's own per-layer counters
  * H2D bytes, analytically as fetches * the measured per-expert slot size
  * latency, with a device sync around the phase

The union is counted from the raw routed ids the decode path already keeps, reduced once per
phase. No per-layer host sync: counting uniques per layer as the phase runs would serialise
against the very PCIe traffic being measured.

Usage:
    PYTHONPATH=python python benchmarks/verify_cost.py --mode compare
    PYTHONPATH=python python benchmarks/verify_cost.py --mode ab
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
GIB = 1 << 30


def build_engine(*, enable_mtp: bool, moe_cache_size: int, prompt_ids: list[int], stats: bool):
    from freetoken.core import Req, SamplingParams
    from freetoken.distributed import DistributedInfo, try_get_tp_info
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.base import BaseCacheHandle
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.spec_round import SpeculativeRoundDriver
    from freetoken.scheduler.table import TableManager
    from freetoken.server.args import ServerArgs

    assert try_get_tp_info() is None, "TP info already set in this process"
    cfg = ServerArgs(
        model_path=MODEL, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        enable_mtp=enable_mtp, moe_strategy="hybrid",
        # 0 is what asks for the auto path (args.py:862 turns it on for hybrid), and an
        # explicit value must win over auto -- the A/B needs both sides pinned to one number
        moe_cache_size=moe_cache_size, moe_cache_auto=moe_cache_size == 0,
        moe_collect_stats=stats, moe_collect_expert_union=stats,
    )
    engine = Engine(cfg)
    table = TableManager(cfg.max_running_req, engine.page_table)
    cache = CacheManager(
        engine.num_pages, cfg.page_size, engine.page_table, cfg.cache_type,
        linear_state_pool=engine.linear_state_pool,
        swa_pool=engine.kv_cache, sliding_window_size=None,
    )

    class _Cold(BaseCacheHandle):
        def __init__(self, device):
            super().__init__(0)
            self._device = device

        def get_matched_indices(self):
            return torch.empty(0, dtype=torch.int32, device=self._device)

    pool = cache.linear_state_pool
    req = Req(
        input_ids=torch.tensor(prompt_ids, dtype=torch.int64),
        table_idx=table.allocate(), cached_len=0, output_len=64, uid=1,
        sampling_params=SamplingParams(),
        cache_handle=_Cold(table.page_table.device),
        linear_slot_idx=pool.alloc(1)[0] if pool is not None else None,
    )
    driver = SpeculativeRoundDriver(engine, cache, table, req)
    driver.prefill_prompt()
    return engine, cache, driver


def expert_slot_bytes(engine) -> int:
    from freetoken.engine.cache_budget import expert_bytes_per_slot
    return expert_bytes_per_slot(engine.moe_offload_cache.bank_sources)


def _reset_traffic(offload) -> None:
    """Zero the per-phase counters, keeping the cache itself warm.

    The cache is deliberately NOT dropped: the question is what a steady-state round costs, and
    a cold cache measures PCIe once per expert rather than what a decode step actually pays.
    """
    for tensor in (
        offload.stat_missing_layer, offload.stat_fetched_layer, offload.stat_active_layer,
        offload.stat_steps_layer, offload.stat_missing, offload.stat_fetched,
        offload.stat_active, offload.stat_calls,
    ):
        tensor.zero_()
    offload.union_records.clear()


def _traffic(engine, offload, slot_bytes: int, elapsed_ms: float,
             captured: int) -> dict:
    """The expert traffic one phase caused, plus its latency.

    ``captured`` is how many layers recorded routed ids. It travels with the numbers rather
    than being assumed: a phase whose capture came back empty would otherwise read as "this
    forward needed no experts", which is a claim about the model rather than about the
    measurement, and a silent zero is exactly the kind of number that gets believed.
    """
    union = offload.expert_union() if captured else {}
    missing = int(offload.stat_missing.item())
    fetched = int(offload.stat_fetched.item())
    active = int(offload.stat_active.item())
    if not captured and active:
        raise AssertionError(
            f"{elapsed_ms:.1f} ms phase reported {active} routed experts but captured no "
            "routed ids: the union flag is off, so its unique count would be a silent zero"
        )
    return {
        "ms": elapsed_ms,
        "captured_layers": captured,
        "unique": union.get("sum_unique", 0),
        "max_unique": union.get("max_unique", 0),
        "active": active,
        "misses": missing,
        "fetched": fetched,
        "h2d_mib": fetched * slot_bytes / 2**20,
        "miss_rate": (missing / active) if active else 0.0,
    }


def _measure(engine, offload, slot_bytes, fn) -> tuple:
    """Run one forward, timed against a settled device, and collect the traffic it caused."""
    _reset_traffic(offload)
    torch.cuda.synchronize()
    started = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) * 1000
    return out, _traffic(engine, offload, slot_bytes, elapsed, len(offload.union_records))


def _one_round(engine, driver, slot_bytes: int) -> dict:
    """One speculative round, timed phase by phase at one state.

    Each phase's latency is taken with the device settled on both sides, and each phase's
    expert traffic is collected over that phase alone. The two separate 1-row steps the verify
    replaces are what makes the comparison meaningful: same state, same request, one token at a
    time.
    """
    offload = engine.moe_offload_cache
    req = driver.req

    # 1. the plain 1-row target step, which is also what produces y1
    y1, target = _measure(engine, offload, slot_bytes, driver._plain_step)   # noqa: F841
    driver._last_pre_mixer = driver._take_pre_mixer()

    # 2. the head's draft
    d1, draft = _measure(engine, offload, slot_bytes, lambda: driver._draft(y1))

    # 3. the 2-row verify
    with torch.cuda.stream(engine.stream):
        snap, snap_lengths = driver._snapshot()
    verified, verify = _measure(engine, offload, slot_bytes, lambda: driver._verify(y1, d1))
    target_next = int(verified)
    accepted = target_next == int(d1)

    # 4. commit or rewind, outside the measured phases
    round_started = time.perf_counter()
    if accepted:
        driver._commit_draft(d1)
    else:
        driver._rewind_after_reject(snap, torch.tensor(target_next, dtype=torch.int32), snap_lengths)
    torch.cuda.synchronize()
    round_ms = (time.perf_counter() - round_started) * 1000
    return {
        "target_ms": target["ms"],
        "draft_ms": draft["ms"],
        "verify_ms": verify["ms"],
        "round_ms": round_ms + target["ms"] + draft["ms"] + verify["ms"],
        "target_unique": target["unique"],
        "verify_unique": verify["unique"],
        "target_max_unique": target["max_unique"],
        "verify_max_unique": verify["max_unique"],
        "target_misses": target["misses"],
        "verify_misses": verify["misses"],
        "target_h2d_mib": target["h2d_mib"],
        "verify_h2d_mib": verify["h2d_mib"],
        "target_active": target["active"],
        "verify_active": verify["active"],
        "committed": 2 if accepted else 1,
        "accepted": accepted,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("compare", "ab"), default="compare")
    ap.add_argument("--moe-cache-size", type=int, default=0,
                    help="0 means auto-size; both A/B sides must use the same value")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--workload", default="code")
    ap.add_argument("--out", default="verify_cost.json")
    args = ap.parse_args()

    os.environ.setdefault("FREETOKEN_PIN_BUDGET_GB", "64")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mtp_probe import _workload_prompts
    from freetoken.utils import load_tokenizer

    tokenizer = load_tokenizer(MODEL)
    ids = tokenizer.encode(_workload_prompts(tokenizer, 2048)[args.workload])
    print(f"workload={args.workload} prompt={len(ids)} tokens, mode={args.mode}, "
          f"moe_cache_size={args.moe_cache_size or 'auto'}", flush=True)

    engine, cache, driver = build_engine(
        enable_mtp=True, moe_cache_size=args.moe_cache_size, prompt_ids=ids, stats=True
    )
    slot_bytes = expert_slot_bytes(engine)
    print(f"expert slot {slot_bytes / 2**20:.2f} MiB, cache "
          f"{engine.moe_offload_cache.cache_size} slots, "
          f"free {engine._sync_get_memory()[0] / GIB:.3f} GiB", flush=True)

    from freetoken.scheduler.spec_round import SpeculativeRoundDriver  # noqa: F401

    results = {"workload": args.workload, "prompt_tokens": len(ids),
               "moe_cache_size": engine.moe_offload_cache.cache_size,
               "expert_slot_bytes": slot_bytes, "rounds": []}

    for step in range(args.steps):
        driver._assert_rewind = False
        row = _one_round(engine, driver, slot_bytes)
        row["step"] = step
        results["rounds"].append(row)
        print(
            f"  round {step}: plain {row['target_ms']:7.1f} ms | verify {row['verify_ms']:7.1f} ms "
            f"| draft {row['draft_ms']:6.1f} ms | unique tgt {row['target_unique']:4d} -> "
            f"verify {row['verify_unique']:4d} | misses {row['target_misses']:4d} -> "
            f"{row['verify_misses']:4d} | H2D {row['target_h2d_mib']:7.1f} -> "
            f"{row['verify_h2d_mib']:7.1f} MiB | accept {row['accepted']}",
            flush=True,
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    _report(results)
    print(f"\nwrote {args.out}")


def _report(results: dict) -> None:
    rows = results["rounds"]
    n = max(1, len(rows))

    def mean(key):
        return sum(r[key] for r in rows) / n

    t1, v = mean("target_ms"), mean("verify_ms")
    two = 2 * t1
    print("\n=== target forward vs verify k=1, same state")
    print(f"  1-row target forward      {t1:8.2f} ms")
    print(f"  2-row verify              {v:8.2f} ms   ({v / t1:.2f}x a 1-row step)")
    print(f"  two separate 1-row steps  {two:8.2f} ms")
    print(f"  verify vs two steps       {v - two:+8.2f} ms   "
          f"({(two - v) / two * 100:+.1f}% saved by verifying together)")
    print(f"  unique experts  1-row {mean('target_unique'):7.0f}   2-row {mean('verify_unique'):7.0f}"
          f"   (+{mean('verify_unique') - mean('target_unique'):.0f})")
    print(f"  cache misses    1-row {mean('target_misses'):7.0f}   2-row {mean('verify_misses'):7.0f}")
    print(f"  H2D MiB         1-row {mean('target_h2d_mib'):7.1f}   2-row {mean('verify_h2d_mib'):7.1f}")
    print(f"  round latency          {mean('round_ms'):8.2f} ms")
    print(f"  committed tokens per round {mean('committed'):.2f}")
    print(f"  committed tokens per target forward "
          f"{mean('committed') / max(1e-9, mean('target_ms') / 1000):.2f} tok/s-of-work")


if __name__ == "__main__":
    main()
