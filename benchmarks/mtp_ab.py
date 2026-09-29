"""MTP off vs on, at the SAME expert cache, on the same requests.

The comparison the phase needs, and the one that is easy to get wrong: an MTP run sized down
to what the head leaves free, against a plain run with the cache it would otherwise get, is
not an MTP measurement. It measures the cache. So both sides here are pinned to one
--moe-cache-size, and the only difference is --enable-mtp.

Per side and workload: decode tok/s, TTFT, acceptance, target forwards, committed tokens per
target forward, and the latency breakdown of a round. committed_tokens_per_target_forward is
the number that decides whether MTP earns its keep: a high acceptance is worth nothing if the
verify costs more than the token it saves.

Usage:
    PYTHONPATH=python python benchmarks/mtp_ab.py --moe-cache-size 3439 --out ab.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

MODEL = r"D:\models\Qwen3.8-Flash-Next-NVFP4"
GIB = 1 << 30
# A long coding/agent prompt, because that is the target use case and the 2k workloads are
# short enough that the head's window is never the binding constraint
AGENT_PROMPT = (
    "You are working in a Python repository that implements a paged key-value cache. "
    "Below is the current state of several files.\n\n"
    "# cache.py\n"
    "def allocate_paged(self, reqs):\n"
    "    for req in reqs:\n"
    "        while len(self.free_slots) < req.cached_len + req.extend_len:\n"
    "            self._evict_one()\n"
    "        slots = self.free_slots[: req.cached_len + req.extend_len]\n"
    "        self.free_slots = self.free_slots[req.cached_len + req.extend_len:]\n"
    "        self.page_table[req.table_idx, :req.device_len] = slots[:req.device_len]\n"
    "    return reqs\n\n"
    "# linear_state_pool.py\n"
    "class LinearStatePool:\n"
    "    def __init__(self, group, num_slots, dtype, device):\n"
    "        self.conv_states = torch.zeros(num_slots, group.n_layers, *group.conv_shape,\n"
    "                                       dtype=dtype, device=device)\n"
    "        self.recurrent_states = torch.zeros(num_slots, group.n_layers,\n"
    "                                           *group.recurrent_shape, device=device)\n"
    "        self._free = list(range(num_slots - 1, -1, -1))\n\n"
    "    def alloc(self, n):\n"
    "        out = [self._free.pop() for _ in range(n)]\n"
    "        return out if len(out) == 1 else out\n\n"
    "    def free(self, slots):\n"
    "        self._free.extend(slots)\n\n"
    "# scheduler.py\n"
    "def _step(self, batch):\n"
    "    for req in batch.reqs:\n"
    "        if req.extend_len == 1:\n"
    "            self._append(req, self.sampler.sample(self._forward(batch)))\n"
    "        else:\n"
    "            self._chunked_step(batch)\n"
    "    self._maybe_finish(batch)\n\n"
    "# TODO: the eviction policy is the last place the cache can hand a live request a page\n"
    "# another request still holds. The prefix cases are covered by tests against the radix\n"
    "# tree, but the interleaved case where two requests both grow while a third is being\n"
    "# evicted is not covered, and it is the one that broke twice.\n"
    "Question: which of the three functions above is the most likely place for that bug, and "
    "what single test would you write to reproduce it?"
)


def build(*, enable_mtp: bool, moe_cache_size: int, stats: bool = False):
    from freetoken.distributed import DistributedInfo, try_get_tp_info
    from freetoken.engine.engine import Engine
    from freetoken.scheduler.cache import CacheManager
    from freetoken.scheduler.table import TableManager
    from freetoken.server.args import ServerArgs

    assert try_get_tp_info() is None, "TP info already set in this process"
    cfg = ServerArgs(
        model_path=MODEL, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16,
        enable_mtp=enable_mtp, moe_strategy="hybrid",
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
    return engine, cache, table, cfg


def admit(cache, table, ids: list[int], output_len: int):
    from freetoken.core import Req, SamplingParams
    from freetoken.kvcache.base import BaseCacheHandle

    class _Cold(BaseCacheHandle):
        def __init__(self, device):
            super().__init__(0)
            self._device = device

        def get_matched_indices(self):
            return torch.empty(0, dtype=torch.int32, device=self._device)

    pool = cache.linear_state_pool
    return Req(
        input_ids=torch.tensor(ids, dtype=torch.int64), table_idx=table.allocate(),
        cached_len=0, output_len=output_len, uid=1, sampling_params=SamplingParams(),
        cache_handle=_Cold(table.page_table.device),
        linear_slot_idx=pool.alloc(1)[0] if pool is not None else None,
    )


def run_plain(engine, cache, table, ids, tokens: int) -> dict:
    """The MTP-off side: plain autoregressive decode, no drafting, no verify."""
    from freetoken.scheduler.spec_round import SpeculativeRoundDriver

    driver = SpeculativeRoundDriver(
        engine, cache, table, admit(cache, table, ids, tokens * 2 + 16)
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    driver.prefill_prompt()
    torch.cuda.synchronize()
    ttft = time.perf_counter() - started

    step_ms = []
    for _ in range(tokens):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        driver._plain_step()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1000)
    driver.release()
    total_ms = sum(step_ms)
    return {
        "tokens": tokens,
        "ttft_s": ttft,
        "decode_s": total_ms / 1000,
        "tok_s": tokens / (total_ms / 1000) if total_ms else 0.0,
        "target_forward_ms": total_ms / len(step_ms) if step_ms else 0.0,
        "committed_tokens_per_target_forward": 1.0,
        "acceptance": None,
    }


def run_spec(engine, cache, table, ids, tokens: int) -> dict:
    """The MTP-on side: real k=1 rounds, phase by phase."""
    from freetoken.scheduler.spec_round import SpeculativeRoundDriver

    rounds = tokens          # every round commits at least one token
    driver = SpeculativeRoundDriver(
        engine, cache, table, admit(cache, table, ids, rounds * 2 + 16)
    )
    driver._instrument = True
    driver._assert_rewind = False
    torch.cuda.synchronize()
    started = time.perf_counter()
    driver.prefill_prompt()
    torch.cuda.synchronize()
    ttft = time.perf_counter() - started

    plain_ms = draft_ms = verify_ms = 0.0
    committed = accepted = proposed = 0
    while committed < tokens:
        t0 = time.perf_counter()
        y1 = driver._plain_step()
        driver._last_pre_mixer = driver._take_pre_mixer()
        torch.cuda.synchronize()
        plain_ms += (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        d1 = driver._draft(y1)
        torch.cuda.synchronize()
        draft_ms += (time.perf_counter() - t0) * 1000

        with torch.cuda.stream(engine.stream):
            snap, lengths = driver._snapshot()
        t0 = time.perf_counter()
        target_next = int(driver._verify(y1, d1))
        torch.cuda.synchronize()
        verify_ms += (time.perf_counter() - t0) * 1000

        proposed += 1
        if target_next == int(d1):
            driver._commit_draft(d1)
            committed += 2
            accepted += 1
        else:
            driver._rewind_after_reject(
                snap, torch.tensor(target_next, dtype=torch.int32, device=engine.device),
                lengths,
            )
            committed += 1
    driver.release()
    forward_ms = plain_ms + verify_ms          # the target forwards; the head is extra
    wall_ms = plain_ms + draft_ms + verify_ms
    return {
        "tokens": committed,
        "ttft_s": ttft,
        "decode_s": wall_ms / 1000,
        "tok_s": committed / (wall_ms / 1000),
        "target_forward_ms": forward_ms / max(1, proposed),
        "plain_step_ms": plain_ms / max(1, proposed),
        "draft_ms": draft_ms / max(1, proposed),
        "verify_ms": verify_ms / max(1, proposed),
        "proposed": proposed,
        "accepted": accepted,
        "acceptance": accepted / proposed if proposed else 0.0,
        "committed_tokens_per_target_forward": committed / max(1, proposed),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--moe-cache-size", type=int, required=True,
                    help="pinned for BOTH sides; this is what makes the A/B fair")
    ap.add_argument("--tokens", type=int, default=40)
    ap.add_argument("--out", default="mtp_ab.json")
    ap.add_argument("--workload", default=None,
                    help="internal: run this workload in this process and print RESULT")
    ap.add_argument("--mode", choices=("on", "off"), default=None,
                    help="internal: which side to run with --workload")
    ap.add_argument("--merge", action="store_true",
                    help="assemble the report from the per-side JSON files on disk")
    args = ap.parse_args()

    os.environ.setdefault("FREETOKEN_PIN_BUDGET_GB", "64")
    from mtp_probe import _workload_prompts
    from freetoken.utils import load_tokenizer

    tokenizer = load_tokenizer(args.model)
    texts = dict(_workload_prompts(tokenizer, 2048))
    texts["agent"] = AGENT_PROMPT
    results = {}

    # One engine per process: the engine installs the global TP info on construction, so a
    # second engine in the same process cannot be built. Each (workload, side) is therefore
    # its own process, and the results are merged by the driver below.
    # One engine per process: the engine installs the global TP info on construction, so two
    # engines cannot coexist. Rather than the parent spawning children (which then die with
    # it), each (workload, side) is run as its own invocation and writes its own JSON; the
    # report is assembled afterwards by --merge from whatever is on disk. A run therefore
    # survives however it was launched.
    if args.workload is not None:
        name, side = args.workload, args.mode
        ids = tokenizer.encode(texts[name])
        engine, cache, table, _ = build(
            enable_mtp=(side == "on"), moe_cache_size=args.moe_cache_size
        )
        got = int(engine.moe_offload_cache.cache_size)
        free = int(engine._sync_get_memory()[0])
        row = (run_spec if side == "on" else run_plain)(engine, cache, table, ids, args.tokens)
        row.update(moe_cache_size=got, free_vram=free, prompt_tokens=len(ids))
        target = Path(args.out).with_name(
            f"{Path(args.out).stem}.{name}.{side}.json"
        )
        target.write_text(json.dumps(row, indent=2))
        print(f"\n[side {side}] {name}: {row['tok_s']:.2f} tok/s, TTFT {row['ttft_s']:.1f} s, "
              f"cache {got}, free {free / GIB:.2f} GiB -> {target.name}", flush=True)
        return

    results = {}
    if args.merge:
        stem = Path(args.out).stem
        for name in ("code", "reasoning", "prose", "agent"):
            row = {}
            for side in ("off", "on"):
                part = Path(args.out).with_name(f"{stem}.{name}.{side}.json")
                if part.exists():
                    row[f"mtp_{side}"] = json.loads(part.read_text())
            if row:
                results[name] = row
        Path(args.out).write_text(json.dumps(results, indent=2))
        _report(results, args)
        print(f"\nmerged into {args.out}")
        return

    print("no --workload and no --merge: run one side per invocation, then --merge", flush=True)


def _report(results: dict, args) -> None:
    print(f"\n=== A/B, identical trunk cache ({args.moe_cache_size} slots both sides)")
    print(f"{'workload':10} {'off tok/s':>10} {'on tok/s':>9} {'ratio':>8} "
          f"{'accept':>7} {'tok/fwd':>8} {'verify ms':>9} {'draft ms':>9} {'TTFT off':>9} "
          f"{'free off':>9} {'free on':>8}")
    for name, r in results.items():
        off, on = r.get("mtp_off"), r.get("mtp_on")
        if not off or not on:
            print(f"{name:10} (incomplete: "
                  f"{'off' if not off else ''}{' on' if not on else ''})")
            continue
        print(f"{name:10} {off['tok_s']:10.2f} {on['tok_s']:9.2f} "
              f"{on['tok_s'] / off['tok_s']:7.3f}x {on['acceptance']:6.1%} "
              f"{on['committed_tokens_per_target_forward']:8.2f} "
              f"{on['verify_ms']:9.1f} {on['draft_ms']:9.1f} {off['ttft_s']:8.1f}s "
              f"{off['free_vram'] / GIB:8.2f}G {on['free_vram'] / GIB:7.2f}G")


if __name__ == "__main__":
    main()
