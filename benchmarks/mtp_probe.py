"""Phase 0 feasibility probe for Qwen3.8-Flash-Next MTP (self-speculative decoding).

Two stages, so the 125B target model is loaded at most once:

``dump``   runs the real target engine in-process, greedy, and records for every
           position the token and the pre-final-mixer hyper-connection residual
           ``R [T, hc_count*hidden] = [T, 10240]``. That residual is exactly what
           the MTP head consumes, and it is not exposed by the model today, so it
           is captured with an instance-level hook on ``hyper_connection_mixer.mix``.

``score``  builds the checkpoint's 31 MTP tensors on top of the *real* Qwen4Exp
           layer ops (Qwen4ExpAttention / Qwen4ExpMoE / GatedResidual) and a dense
           QSA oracle, then measures

             * which fusion of ``pre_fc_norm_embedding`` / ``fc_embedding`` and
               ``pre_fc_norm_hidden`` / ``fc_hidden`` with the pre-mixer residual
               actually predicts the next token,
             * how much prefix the head's own KV needs (the priming sweep),
             * draft acceptance for k = 1..4,
             * the single-token MTP forward latency.

Reusing the production ops matters: a hand-written re-derivation would measure my
transcription of the architecture instead of the architecture.

Run:
    PYTHONPATH=python python benchmarks/mtp_probe.py dump  --model /path/to/ckpt --out /tmp/mtp
    PYTHONPATH=python python benchmarks/mtp_probe.py score --dump /tmp/mtp
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

# --------------------------------------------------------------------------------------
# workload prompts (grown to ~2k tokens deterministically)
# --------------------------------------------------------------------------------------

_CODE_SEED = '''\
You are reviewing a Python module that implements a paged KV cache for an LLM serving engine.
Here is the file:

```python
"""Paged KV cache: fixed-size pages, an LRU free list, a per-request page table."""

import torch

PAGE_SIZE = 16


class PagedKVCache:
    def __init__(self, num_pages: int, num_layers: int, num_kv_heads: int, head_dim: int, dtype):
        self.page_size = PAGE_SIZE
        self.num_pages = num_pages
        self.free_pages = list(range(num_pages))
        self.kv = [
            torch.zeros(num_pages, PAGE_SIZE, num_kv_heads, head_dim, dtype=dtype)
            for _ in range(num_layers)
        ]
        self.page_table = {}

    def allocate(self, req_id: str, num_new_tokens: int):
        pages = []
        remaining = num_new_tokens
        while remaining > 0:
            if not self.free_pages:
                raise RuntimeError("out of pages")
            page = self.free_pages.pop()
            pages.append(page)
            remaining -= PAGE_SIZE
        self.page_table[req_id] = pages
        return pages

    def write(self, req_id: str, layer: int, slots: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        pages = self.page_table[req_id]
        for i, slot in enumerate(slots.tolist()):
            page, offset = pages[slot // PAGE_SIZE], slot % PAGE_SIZE
            self.kv[layer][page, offset] = keys[i]
            self.kv[layer][page, offset] = values[i]

    def free(self, req_id: str):
        for page in self.page_table.pop(req_id, []):
            self.free_pages.append(page)
```

Find and fix every bug in it. For each one, say which line is wrong, what breaks, and the corrected code.
Work through them in order of severity, and finish with the complete corrected file.
'''

_REASONING_SEED = '''\
Solve the following problems in order. Show your reasoning, then state each final answer on its own
line in the form `Answer N: <value>`.

1. A train travels 60 km in the first hour and 40 km in each of the next two hours. What is its
   average speed over the whole trip, and why is it not 50 km/h?

2. Three boxes are labelled "apples", "oranges" and "mixed". Every label is wrong. You may draw one
   fruit from one box. Which box do you draw from, and what does the result tell you?

3. Let f(x) = x^2 - 4x + 3. Find the largest integer n such that f(f(n)) < 0. Verify your answer by
   evaluating f twice at the boundary values.

4. A 3x3 grid is filled with the numbers 1..9 so that every row, column and both diagonals sum to 15.
   Give one solution and prove it is unique up to rotation and reflection.

5. Six people sit around a round table. Each knows only their neighbours' incomes. Each says "my
   neighbour earns more than me". What are the possible income orderings around the table? Explain
   why the all-equal case is excluded.

Be precise with each answer and do not skip the reasoning.
'''

_PROSE_SEED = '''\
Write a reflective, essay-style piece of about 700 words on the following topic:

"Somewhere between the tool and the craft there is a moment where the work stops being work."

Draw on everyday experience rather than technical examples. Take a clear position, develop it across
several paragraphs, include at least one concrete scene, and end with a conclusion that does not
simply restate the opening. Aim for a thoughtful, unhurried register. Do not use bullet points, and
do not use headings.
'''

_FILLER = [
    "For reference, the following notes are part of the background you should read before answering:",
    "Note that all identifiers in the excerpt use snake_case and every public function carries a "
    "docstring in the imperative mood.",
    "The engine runs on a single accelerator with 32 GB of device memory and a large host buffer, so "
    "capacity planning is the central concern in this design.",
    "Any change you propose should keep the existing call sites working and should not require a "
    "flagship-style rebuild of the surrounding infrastructure.",
    "Where you have to choose, prefer the change that is easier to reason about over the change that "
    "is marginally faster, and say so explicitly when you do.",
    "Assume the reader is competent but has not read the surrounding code base.",
]


def _grow(tokenizer, seed: str, target_tokens: int) -> str:
    """Append filler paragraphs until the prompt is at least ``target_tokens`` tokens."""
    extra: list[str] = []
    prompt = seed
    while len(tokenizer.encode(prompt)) < target_tokens:
        extra.append(_FILLER[len(extra) % len(_FILLER)])
        prompt = seed + "\n" + "\n".join(extra) + "\n"
    return prompt


def _workload_prompts(tokenizer, target_tokens: int) -> dict[str, str]:
    return {
        "code": _grow(tokenizer, _CODE_SEED, target_tokens),
        "reasoning": _grow(tokenizer, _REASONING_SEED, target_tokens),
        "prose": _grow(tokenizer, _PROSE_SEED, target_tokens),
    }


# --------------------------------------------------------------------------------------
# stage 1: dump the target's pre-mixer residuals
# --------------------------------------------------------------------------------------


def stage_dump(args: argparse.Namespace) -> int:
    from freetoken.core import SamplingParams
    from freetoken.llm import LLM

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        args.model,
        dtype=torch.bfloat16,
        max_running_req=args.max_running_requests,
        moe_strategy=args.moe_strategy,
        moe_cache_size=args.moe_cache_size,
        moe_cpu_layers=args.moe_cpu_layers,
        moe_cpu_threads=args.moe_cpu_threads,
        ple_backend=args.ple_backend,
        # graphs replay without re-entering python, so the capture hook would not fire
        cuda_graph_max_bs=0,
        num_token_override=args.max_seq_len,
    )

    # Instance-level capture: Qwen4ExpModel.forward calls hyper_connection_mixer.mix(hidden)
    # exactly once per forward, on the [T, hc_count*hidden] pre-mixer residual.
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    mixer = llm.engine.model.model.hyper_connection_mixer
    original_mix = mixer.mix

    def capturing_mix(R: torch.Tensor):
        batch = llm.engine.ctx.batch
        captured.append(
            (
                batch.input_ids.detach().to("cpu", copy=True),
                batch.positions.detach().to("cpu", copy=True),
                R.detach().to("cpu", copy=True),
            )
        )
        return original_mix(R)

    mixer.mix = capturing_mix
    try:
        prompts = _workload_prompts(llm.tokenizer, args.prompt_tokens)
        for name, prompt in prompts.items():
            captured.clear()
            t0 = time.perf_counter()
            result = llm.generate(
                [prompt],
                SamplingParams(
                    temperature=0.0, top_k=-1, top_p=1.0,
                    max_tokens=args.decode, ignore_eos=True,
                ),
            )
            elapsed = time.perf_counter() - t0

            tokens, residuals = _assemble(captured, result[0]["token_ids"])
            path = out_dir / f"{name}.pt"
            torch.save({"tokens": tokens, "residuals": residuals, "workload": name}, path)
            print(
                f"[dump] {name:10s} prompt={args.prompt_tokens:5d} tok "
                f"decoded={len(result[0]['token_ids']):4d} "
                f"positions_with_residual={residuals.shape[0]:4d} "
                f"residuals={tuple(residuals.shape)} {elapsed:6.1f}s -> {path}",
                flush=True,
            )
    finally:
        mixer.mix = original_mix
    return 0


def _assemble(
    captured: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    output_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleave the per-forward token/position/residual rows into one per-position view.

    Position p's residual is the target's state after consuming ``tokens[p]``, so the
    head's input at position p is ``(embed(tokens[p]), residuals[p - 1])``.
    """
    by_position: dict[int, tuple[int, torch.Tensor]] = {}
    for ids, positions, residual in captured:
        for row, pos in enumerate(positions.tolist()):
            by_position[pos] = (int(ids[row]), residual[row])
    if not by_position:
        raise RuntimeError("no forwards captured")
    last = max(by_position)
    tokens = [by_position[p][0] for p in range(last + 1)] + list(output_ids)
    residuals = torch.stack([by_position[p][1] for p in range(last + 1)])
    return torch.tensor(tokens, dtype=torch.long), residuals


# --------------------------------------------------------------------------------------
# stage 2: the MTP head
# --------------------------------------------------------------------------------------


@dataclass
class ProbeReq:
    """The three fields ``TorchDenseQSAReference`` reads off a request."""
    table_idx: int
    cached_len: int
    extend_len: int


@dataclass
class ProbeBatch:
    reqs: list[ProbeReq]
    positions: torch.Tensor
    mrope: bool = True
    phase: str = "prefill"
    padded_reqs: list[ProbeReq] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.padded_reqs:
            self.padded_reqs = self.reqs

    @property
    def is_prefill(self) -> bool:
        return self.phase == "prefill"

    @property
    def is_decode(self) -> bool:
        return self.phase == "decode"

    def get_attn_positions(self) -> torch.Tensor:
        # text-only mrope: all three axes carry the same position, which reduces
        # MRotaryEmbedding to plain rope
        return self.positions.unsqueeze(0).expand(3, -1) if self.mrope else self.positions


def _rms(x: torch.Tensor, w: torch.Tensor, eps: float, groups: int, plus_one: bool) -> torch.Tensor:
    """``groups`` slices of the last dim, each on its own fp32 RMS statistic (hc.py)."""
    xf = x.float().unflatten(-1, (groups, -1))
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    scale = 1.0 + w.float() if plus_one else w.float()
    return (xf.flatten(-2) * scale).to(x.dtype)


@dataclass(frozen=True)
class FusionVariant:
    """One hypothesis for the four fusion tensors.

    Shapes already force most of it: ``fc_embedding`` [H,H] and ``fc_hidden`` [H,H]
    cannot consume a concatenated 2H vector, so the fusion is additive, and
    ``pre_fc_norm_hidden`` [hc*H] can only see the full pre-mixer residual. What is
    left open is the norm's group structure and whether its weight is zero-centered.
    """

    name: str
    hidden_groups: int  # 4 = per-branch RMS statistic, 1 = one statistic over all 10240
    plus_one: bool  # checkpoint stores scale-1
    normalize_hidden: bool = True


VARIANTS = (
    FusionVariant("grouped_plus1", hidden_groups=4, plus_one=True),
    FusionVariant("plain_plus1", hidden_groups=1, plus_one=True),
    FusionVariant("grouped_raw", hidden_groups=4, plus_one=False),
    FusionVariant("plain_raw", hidden_groups=1, plus_one=False),
    FusionVariant("grouped_plus1_nonorm", hidden_groups=4, plus_one=True, normalize_hidden=False),
)


class MTPHead:
    """The checkpoint's ``mtp.*`` block, assembled from the production Qwen4Exp ops.

    Dataflow (the Qwen3.8 shape set fixes all of it):

        e = embed_tokens(x)                      [T, H]        shared with the trunk
        e = fc_embedding(pre_fc_norm_embedding(e))
        h = pre_fc_norm_hidden(R_prev)           [T, hc*H]     R_prev = pre-mixer residual
        h = fc_hidden(h.view(T, hc, H))          [T, hc, H]    one weight shared per branch
        R = (e[:, None, :] + h).view(T, hc*H)
        R = decoder_layer(R)                     one full_attention layer, no PLE, no GDN
        logits = lm_head(mixer.mix(R))
    """

    def __init__(self, config, weights: dict[str, torch.Tensor], variant: FusionVariant) -> None:
        from freetoken.models.qwen4_exp.attention import Qwen4ExpAttention
        from freetoken.models.qwen4_exp.hc import GatedResidual
        from freetoken.models.qwen4_exp.moe import Qwen4ExpMoE
        from freetoken.utils import torch_dtype

        self.variant = variant
        self.hc = config.qwen4_args.hc_count
        self.hidden = config.qwen4_args.hidden_size
        self.eps = config.rms_norm_eps
        dtype = torch.bfloat16
        # same construction context as Engine.__init__: meta allocates the params and
        # get_rope() materialises its cos/sin cache on the rope device
        with torch.device("meta"), torch_dtype(dtype):
            self.attn = Qwen4ExpAttention(config, 0, prefix="mtp.layers.0.self_attn")
            self.attn_hc = GatedResidual(config, prefix="mtp.layers.0.attn_hyper_connection")
            self.mlp_hc = GatedResidual(config, prefix="mtp.layers.0.mlp_hyper_connection")
            self.mlp = Qwen4ExpMoE(config, prefix="mtp.layers.0.mlp")
            self.mixer = GatedResidual(config, use_combine=False, prefix="mtp.hyper_connection_mixer")
        for module, prefix in (
            (self.attn, "mtp.layers.0.self_attn"),
            (self.attn_hc, "mtp.layers.0.attn_hyper_connection"),
            (self.mlp_hc, "mtp.layers.0.mlp_hyper_connection"),
            (self.mlp, "mtp.layers.0.mlp"),
            (self.mixer, "mtp.hyper_connection_mixer"),
        ):
            subtree = {k: v for k, v in weights.items() if k.startswith(prefix + ".")}
            missing = set(module.state_dict(prefix=prefix)) - set(subtree)
            if missing:
                raise RuntimeError(f"{prefix}: no checkpoint tensor for {sorted(missing)}")
            module.load_state_dict(subtree, prefix=prefix)

        self.w_pre_embedding = weights["mtp.pre_fc_norm_embedding.weight"]
        self.w_pre_hidden = weights["mtp.pre_fc_norm_hidden.weight"]
        self.fc_embedding = weights["mtp.fc_embedding.weight"]
        self.fc_hidden = weights["mtp.fc_hidden.weight"]
        self.embed = weights["embed_tokens"]
        self.lm_head = weights["lm_head"]

    def fuse(self, tokens: torch.Tensor, prev_residual: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[0]
        e = F.embedding(tokens, self.embed)
        e = F.linear(_rms(e, self.w_pre_embedding, self.eps, 1, True), self.fc_embedding)
        if self.variant.normalize_hidden:
            h = _rms(prev_residual, self.w_pre_hidden, self.eps, self.variant.hidden_groups,
                     self.variant.plus_one)
        else:
            h = prev_residual
        h = F.linear(h.view(t, self.hc, self.hidden), self.fc_hidden)
        return (e.unsqueeze(1) + h).reshape(t, self.hc * self.hidden)

    def layer(self, residual: torch.Tensor, batch: ProbeBatch) -> torch.Tensor:
        x, inject = self.attn_hc.mix(residual)
        residual = self.attn_hc.combine(residual, self.attn.forward(x, batch), inject)
        x, inject = self.mlp_hc.mix(residual)
        return self.mlp_hc.combine(residual, self.mlp.forward(x), inject)

    def logits(self, residual: torch.Tensor) -> torch.Tensor:
        """The shared trunk ``lm_head`` on the head's collapsed hidden."""
        return F.linear(self.mixer.mix(residual)[0], self.lm_head)

    def forward(self, tokens: torch.Tensor, prev_residual: torch.Tensor, batch: ProbeBatch):
        """Returns ``(logits, pre_mixer_residual)``; the residual feeds the next draft step."""
        residual = self.fuse(tokens, prev_residual)
        residual = self.layer(residual, batch)
        return self.logits(residual), residual


# --------------------------------------------------------------------------------------
# stage 2: acceptance measurement
# --------------------------------------------------------------------------------------


class MTPProber:
    """Primes the head's KV over a window, then runs draft chains from sampled positions."""

    def __init__(self, head: MTPHead, backend, device: torch.device) -> None:
        from freetoken.core import get_global_ctx

        self.head = head
        self.device = device
        self.backend = backend
        self.ctx = get_global_ctx()
        self.primed: torch.Tensor | None = None  # head pre-mixer residuals, one per position
        self.primed_top: torch.Tensor | None = None  # [n, 5] top-5 token ids per position
        self.window: tuple[int, int] = (0, 0)

    def _active(self, batch: ProbeBatch):
        # MoELayer reads the active batch off the global context to pick its kernel
        return self.ctx.forward_batch(batch)

    def _batch(self, start: int, count: int) -> ProbeBatch:
        positions = torch.arange(start, start + count, device=self.device, dtype=torch.int32)
        req = ProbeReq(table_idx=0, cached_len=start, extend_len=count)
        return ProbeBatch(reqs=[req], positions=positions)

    def _step(self, token: int, prev_residual: torch.Tensor, position: int) -> torch.Tensor:
        tokens = torch.tensor([token], device=self.device, dtype=torch.long)
        with self._active(self._batch(position, 1)):
            logits, _ = self.head.forward(tokens, prev_residual.unsqueeze(0), self._batch(position, 1))
        return logits[0]

    @torch.inference_mode()
    def prime(self, tokens: torch.Tensor, residuals: torch.Tensor, lo: int, hi: int,
              chunk: int = 256) -> None:
        """Fill the head's KV over ``[lo, hi)``.

        Records each position's pre-mixer residual (the next draft step's hidden input)
        and its top-5 prediction, so the one-step agreement needs no extra forwards.
        """
        assert lo >= 1, "the head consumes the residual of the previous position, so lo >= 1"
        width = self.head.hc * self.head.hidden
        out = torch.empty(hi - lo, width, dtype=torch.bfloat16)
        top = torch.empty(hi - lo, 5, dtype=torch.int32)
        for start in range(lo, hi, chunk):
            end = min(start + chunk, hi)
            ids = tokens[start:end].to(self.device)
            prev = residuals[start - 1 : end - 1].to(self.device)
            batch = self._batch(start, end - start)
            with self._active(batch):
                residual = self.head.fuse(ids, prev)
                residual = self.head.layer(residual, batch)
                logits = self.head.logits(residual)
            out[start - lo : end - lo] = residual.to("cpu")
            top[start - lo : end - lo] = torch.topk(logits, 5, dim=-1).indices.to("cpu", torch.int32)
        self.primed = out
        self.primed_top = top
        self.window = (lo, hi)

    def _snapshot(self, lo: int, hi: int) -> list[torch.Tensor]:
        k, v = self.backend._layer_cache(0)
        return [k[:, lo:hi].clone(), v[:, lo:hi].clone()]

    def _restore(self, saved: list[torch.Tensor], lo: int) -> None:
        k, v = self.backend._layer_cache(0)
        k[:, lo : lo + saved[0].shape[1]] = saved[0]
        v[:, lo : lo + saved[1].shape[1]] = saved[1]

    @torch.inference_mode()
    def draft_chain(self, tokens: torch.Tensor, position: int, k: int) -> list[int]:
        """Draft up to k tokens autoregressively from ``position``.

        Step 0 reuses the primed forward's top-1, so only k-1 head forwards run.
        """
        assert self.primed is not None and self.primed_top is not None
        lo, _ = self.window
        drafts: list[int] = [int(self.primed_top[position - lo, 0])]
        prev = self.primed[position - lo].to(self.device)
        for j in range(1, k):
            # the chain rewrites the head's KV at position+j; keep the primed rows
            saved = self._snapshot(position + j, position + j + 1)
            batch = self._batch(position + j, 1)
            with self._active(batch):
                logits, prev = self.head.forward(
                    torch.tensor([drafts[j - 1]], device=self.device, dtype=torch.long),
                    prev.unsqueeze(0), batch,
                )
            self._restore(saved, position + j)
            drafts.append(int(torch.argmax(logits[0]).item()))
        return drafts

    @torch.inference_mode()
    def latency_ms(self, tokens: torch.Tensor, residuals: torch.Tensor, lo: int, repeats: int = 50) -> float:
        """Single-position head forward, the cost one draft step adds per decode round."""
        prev = residuals[lo - 1 : lo].to(self.device)
        for _ in range(5):
            self._step(int(tokens[lo]), prev[0], lo)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(repeats):
            self._step(int(tokens[lo]), prev[0], lo)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeats


def _load_mtp_weights(model_path: str, device: torch.device) -> dict[str, torch.Tensor]:
    from freetoken.models.loader import ShardReader

    reader = ShardReader(model_path, "cpu")
    try:
        wanted = {f"mtp.{leaf}" for leaf in _MTP_LEAVES}
        raw = {name: reader.get_tensor(name) for name in wanted}
        raw["embed_tokens"] = reader.get_tensor("model.language_model.embed_tokens.weight")
        raw["lm_head"] = reader.get_tensor("lm_head.weight")
    finally:
        reader.close()
    weights = _fuse_linear_names(raw)
    return {k: v.to(device=device, dtype=torch.bfloat16) for k, v in weights.items()}


# The checkpoint ships split projections; the loader concatenates them. Fusing by hand here
# keeps the probe independent of the loader change this measurement is meant to justify.
_FUSED_AWAY = (
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
    "mtp.layers.0.mlp.shared_expert.up_proj.weight",
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.attn_hyper_connection.block_inject_weight.weight",
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "mtp.layers.0.mlp_hyper_connection.block_inject_weight.weight",
)


def _fuse_linear_names(raw: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {k: v for k, v in raw.items() if k not in _FUSED_AWAY}

    def cat(prefix: str, fused: str, leaves: list[str]) -> None:
        out[f"{prefix}.{fused}.weight"] = torch.cat([raw[f"{prefix}.{leaf}.weight"] for leaf in leaves], 0)

    layer = "mtp.layers.0"
    cat(f"{layer}.self_attn", "qkv_proj", ["q_proj", "k_proj", "v_proj"])
    cat(f"{layer}.mlp.shared_expert", "gate_up_proj", ["gate_proj", "up_proj"])
    for role in ("attn_hyper_connection", "mlp_hyper_connection"):
        down = raw[f"{layer}.{role}.input_mix_weight_down.weight"]
        inject = raw[f"{layer}.{role}.block_inject_weight.weight"]
        pad = (-(down.shape[0] + inject.shape[0])) % 16
        out[f"{layer}.{role}.input_mix_weight_down_block_inject.weight"] = torch.cat(
            [down, inject, down.new_zeros(pad, down.shape[1])], 0
        )
    return out


_MTP_LEAVES = [
    "fc_embedding.weight", "fc_hidden.weight",
    "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight",
    "hyper_connection_mixer.hc_norm.weight",
    "hyper_connection_mixer.input_mix_weight_down.weight",
    "hyper_connection_mixer.input_mix_weight_up.weight",
    "layers.0.attn_hyper_connection.hc_norm.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_down.weight",
    "layers.0.attn_hyper_connection.input_mix_weight_up.weight",
    "layers.0.attn_hyper_connection.block_inject_weight.weight",
    "layers.0.mlp_hyper_connection.hc_norm.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_down.weight",
    "layers.0.mlp_hyper_connection.input_mix_weight_up.weight",
    "layers.0.mlp_hyper_connection.block_inject_weight.weight",
    "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight",
    "layers.0.self_attn.indexer.index_qk_proj.weight",
    "layers.0.self_attn.indexer.q_layernorm.weight",
    "layers.0.self_attn.indexer.k_layernorm.weight",
    "layers.0.mlp.gate.weight", "layers.0.mlp.shared_expert_gate.weight",
    "layers.0.mlp.shared_expert.gate_proj.weight",
    "layers.0.mlp.shared_expert.up_proj.weight",
    "layers.0.mlp.shared_expert.down_proj.weight",
    "layers.0.mlp.experts.gate_up_proj", "layers.0.mlp.experts.down_proj",
]


def stage_score(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from freetoken.core import Context, set_global_ctx
    from freetoken.distributed import DistributedInfo, set_tp_info, try_get_tp_info
    from freetoken.engine import EngineConfig
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    set_rope_device(device)
    torch.cuda.set_device(device)

    config = EngineConfig(
        model_path=args.model, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16
    ).model_config
    dump_dir = Path(args.dump)
    runs = {}
    for path in sorted(dump_dir.glob("*.pt")):
        blob = torch.load(path, weights_only=False)
        runs[blob["workload"]] = (blob["tokens"], blob["residuals"])
    if not runs:
        sys.exit(f"no dumps in {dump_dir}")

    max_len = max(int(residuals.shape[0]) + 2 for _, residuals in runs.values())
    print(f"[score] loaded {len(runs)} runs, head KV window {max_len}", flush=True)

    # the head's experts are BF16 while the trunk's are NVFP4, so the head is built
    # resident rather than through the offload bank
    head_config = replace(config, moe_strategy="fused")
    ctx = Context(page_size=1)
    ctx.page_table = torch.zeros(1, max_len, dtype=torch.int32, device=device)
    set_global_ctx(ctx)

    weights = _load_mtp_weights(args.model, device)
    print(
        f"[score] {len(weights)} mtp tensors resident "
        f"({sum(v.numel() * 2 for v in weights.values()) / 2**30:.2f} GiB)",
        flush=True,
    )

    results: list[dict] = []
    for variant in VARIANTS:
        if args.variants and variant.name not in args.variants:
            continue
        head = MTPHead(head_config, weights, variant)
        backend = TorchDenseQSAReference(
            head_config, num_slots=1, max_len=max_len, device=device, dtype=torch.bfloat16
        )
        ctx.attn_backend = backend
        prober = MTPProber(head, backend, device)
        for workload, (tokens, residuals) in runs.items():
            row = _measure(prober, tokens, residuals, variant.name, workload, args)
            results.append(row)
            print(_fmt(row), flush=True)
        del head, backend, prober
        torch.cuda.empty_cache()

    out = Path(args.json) if args.json else dump_dir / "score.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"[score] wrote {out}")
    return 0


def _measure(prober: MTPProber, tokens: torch.Tensor, residuals: torch.Tensor,
             variant: str, workload: str, args: argparse.Namespace) -> dict:
    n = int(residuals.shape[0])
    # the head consumes (token at p, residual at p-1) and predicts tokens[p+1]
    last = n - 2
    wanted = [last if p == 0 else p for p in args.primes]
    primes = [p for p in wanted if 0 < p <= last]
    row: dict = {"variant": variant, "workload": workload, "positions": last + 1, "prime": {}}
    for prime in primes:
        lo = max(1, last - prime + 1)
        prober.prime(tokens, residuals, lo, last + 1)
        assert prober.primed is not None and prober.primed_top is not None
        truth = torch.tensor(tokens[lo + 1 : last + 2], dtype=torch.int64)
        top = prober.primed_top.long()

        # one-step agreement over the whole primed window, straight from the priming pass
        agree = int((top[:, 0] == truth).sum())
        in_top5 = int((top == truth.unsqueeze(1)).any(dim=1).sum())
        counted = truth.numel()

        # k-step acceptance, sampled so the chains stay cheap
        span = last - lo
        stride = max(1, span // max(1, args.chains))
        starts = list(range(lo, last, stride))
        accepted: dict[int, list[int]] = {k: [] for k in args.k}
        for p in starts:
            if p + max(args.k) > last:
                break
            drafts = prober.draft_chain(tokens, p, max(args.k))
            for k in args.k:
                n_ok = 0
                for j in range(k):
                    if drafts[j] == int(tokens[p + 1 + j]):
                        n_ok += 1
                    else:
                        break
                accepted[k].append(n_ok)

        row["prime"][str(prime)] = {
            "window": [lo, last + 1],
            "one_step_top1": agree / counted,
            "one_step_top5": in_top5 / counted,
            "chains": len(starts),
            "accepted_per_chain": {str(k): sum(v) / max(1, len(v)) for k, v in accepted.items()},
            "accept_at_least_one": {
                str(k): sum(1 for v in accepted[k] if v >= 1) / max(1, len(accepted[k]))
                for k in args.k
            },
        }
    row["mtp_forward_ms"] = prober.latency_ms(
        tokens, residuals, max(1, last - 32), repeats=args.latency_repeats
    )
    return row


def _fmt(row: dict) -> str:
    lines = [f"--- {row['workload']:10s} variant={row['variant']:22s} "
             f"mtp_forward={row['mtp_forward_ms']:.3f} ms"]
    for prime, m in row["prime"].items():
        acc = "  ".join(
            f"k={k}:{v:.2f}" for k, v in sorted(m["accepted_per_chain"].items(), key=lambda kv: int(kv[0]))
        )
        lines.append(
            f"      prime={prime:>5s} top1={m['one_step_top1']*100:5.1f}% "
            f"top5={m['one_step_top5']*100:5.1f}%  avg_accepted {acc}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="stage", required=True)

    dump = sub.add_parser("dump")
    dump.add_argument("--model", required=True)
    dump.add_argument("--out", required=True)
    dump.add_argument("--prompt-tokens", type=int, default=2000)
    dump.add_argument("--decode", type=int, default=200)
    dump.add_argument("--max-seq-len", type=int, default=16384)
    dump.add_argument("--max-running-requests", type=int, default=1)
    dump.add_argument("--moe-cpu-layers", default=None)
    dump.add_argument("--moe-cpu-threads", type=int, default=0)
    dump.add_argument("--moe-strategy", default="offload")
    dump.add_argument("--moe-cache-size", type=int, default=5120)
    dump.add_argument("--ple-backend", default="disk")
    dump.set_defaults(func=stage_dump)

    score = sub.add_parser("score")
    score.add_argument("--model", required=True)
    score.add_argument("--dump", required=True)
    score.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4])
    score.add_argument("--primes", type=int, nargs="+", default=[256, 512, 2048, 0],
                       help="head KV priming lengths; 0 means the full available prefix")
    score.add_argument("--variants", nargs="+", default=None,
                       help="subset of variant names (default: all)")
    score.add_argument("--chains", type=int, default=24, help="draft chains per prime length")
    score.add_argument("--latency-repeats", type=int, default=50)
    score.add_argument("--json", default=None)
    score.set_defaults(func=stage_score)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # MTP_PROBE_STACK_AFTER=<seconds> dumps every thread's stack and exits, so a stuck
    # run says where it is stuck instead of only that it is
    after = os.environ.get("MTP_PROBE_STACK_AFTER")
    if after:
        import faulthandler

        faulthandler.dump_traceback_later(int(after), exit=True)
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
