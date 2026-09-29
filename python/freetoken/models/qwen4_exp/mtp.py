"""Qwen3.8-Flash-Next MTP head: the checkpoint's ``mtp.*`` block, for self-speculative decoding.

Qwen3.8 ships a one-layer MTP head (``text_config.mtp``: ``num_hidden_layers=1``,
``layer_types=["full_attention"]``) with all 31 tensors in BF16 -- the release's
``quantization_config.ignore`` lists ``mtp.*``, so every module below resolves to
``QuantKind.NONE`` through the existing ignore globs and no weight is requantized.

Semantics, confirmed numerically in ``benchmarks/mtp_probe.py`` (variants that differ
in one of these choices were measured against the checkpoint's own weights; the
zero-centered-norm reading scores 0% next-token agreement, i.e. chance, against 89%
for the one implemented here)::

    R_target = [T, hc_count*hidden]      pre-final-mixer backbone residual, NOT the
                                        [T, hidden] tensor that feeds lm_head
    e = fc_embedding(pre_fc_norm_embedding(embed_tokens(x)))       [T, hidden]
    h = pre_fc_norm_hidden(R_target).view(T, hc_count, hidden)    [T, hc_count, hidden]
    h = fc_hidden(h)                        one [hidden, hidden] weight, shared by all
                                             hc_count branches
    R = (e[:, None, :] + h).view(T, hc_count*hidden)   additive, not a concatenation
    R = layers.0(R)                         one full_attention decoder layer, no PLE, no GDN
    logits = lm_head(hyper_connection_mixer.mix(R)[0])

``embed_tokens`` and ``lm_head`` are the trunk's: the checkpoint has no dedicated ones
(``mtp_use_dedicated_embeddings: false``), so this module owns neither and the caller
supplies the embedding and applies the head. That keeps the sharing structural instead
of duplicating ~1.3 GiB of weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.layers import BaseOP, GemmaPlusOneRMSNorm, LinearReplicated, OPList

from .attention import Qwen4ExpAttention
from .hc import GatedResidual, GroupedPlusOneRMSNorm
from .moe import Qwen4ExpMoE

if TYPE_CHECKING:
    from freetoken.core import Batch
    from freetoken.models.config import ModelConfig

MTP_KEY_PREFIX = "mtp."


class Qwen4ExpMTPDecoderLayer(BaseOP):
    """``mtp.layers.0``: one full_attention decoder layer, without PLE and without GDN.

    The same mix/combine flow as :class:`~freetoken.models.qwen4_exp.model.Qwen4ExpDecoderLayer`,
    spelled out here so the MTP path does not depend on the trunk's layer-type table
    (``config.is_linear_layer``) or on the trunk's PLE placement.
    """

    def __init__(self, config: ModelConfig, *, prefix: str) -> None:
        self.self_attn = Qwen4ExpAttention(config, 0, prefix=f"{prefix}.self_attn")
        self.mlp = Qwen4ExpMoE(config, prefix=f"{prefix}.mlp")
        self.attn_hyper_connection = GatedResidual(config, prefix=f"{prefix}.attn_hyper_connection")
        self.mlp_hyper_connection = GatedResidual(config, prefix=f"{prefix}.mlp_hyper_connection")

    def forward(self, residual: torch.Tensor, batch: Batch) -> torch.Tensor:
        block_input, inject = self.attn_hyper_connection.mix(residual)
        block_output = self.self_attn.forward(block_input, batch)
        residual = self.attn_hyper_connection.combine(residual, block_output, inject)
        block_input, inject = self.mlp_hyper_connection.mix(residual)
        return self.mlp_hyper_connection.combine(residual, self.mlp.forward(block_input), inject)


class Qwen4ExpMTP(BaseOP):
    """The MTP draft head. ``forward`` returns the pre-final-mixer residual it was built from.

    That residual, not the collapsed ``[T, hidden]``, is what a chained draft step
    consumes, so the head hands it back rather than collapsing internally.
    """

    def __init__(self, config: ModelConfig, *, prefix: str = MTP_KEY_PREFIX.rstrip(".")) -> None:
        args = config.qwen4_args
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        # zero-centered checkpoint weights, (1+w) applied in fp32 at runtime: same
        # convention as hc_norm / q_norm / k_norm
        self.pre_fc_norm_embedding = GemmaPlusOneRMSNorm(args.hidden_size, eps=config.rms_norm_eps)
        # one RMS statistic per hyper-connection branch, matching hc_norm's grouping
        self.pre_fc_norm_hidden = GroupedPlusOneRMSNorm(
            args.ple_state_width, config.rms_norm_eps, self.hc_count
        )
        self.fc_embedding = LinearReplicated(
            args.hidden_size, args.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_embedding",
        )
        self.fc_hidden = LinearReplicated(
            args.hidden_size, args.hidden_size, has_bias=False,
            quant_config=config.quant, prefix=f"{prefix}.fc_hidden",
        )
        # OPList so the state-dict path is ``mtp.layers.0.*``, the checkpoint's own name
        self.layers = OPList([Qwen4ExpMTPDecoderLayer(config, prefix=f"{prefix}.layers.0")])
        self.hyper_connection_mixer = GatedResidual(
            config, use_combine=False, prefix=f"{prefix}.hyper_connection_mixer"
        )

    def fuse(self, embedded: torch.Tensor, target_residual: torch.Tensor) -> torch.Tensor:
        """The additive merge of the trunk's token embedding and the pre-mixer residual.

        ``embedded`` is ``embed_tokens(x)`` from the trunk; ``target_residual`` is the
        backbone's ``[T, hc_count*hidden]`` pre-final-mixer stream.
        """
        t = embedded.shape[0]
        e = self.fc_embedding.forward(self.pre_fc_norm_embedding.forward(embedded))
        h = self.pre_fc_norm_hidden.forward(target_residual).view(t, self.hc_count, self.hidden_size)
        return (e.unsqueeze(1) + self.fc_hidden.forward(h)).view(t, self.hc_count * self.hidden_size)

    def forward(self, embedded: torch.Tensor, target_residual: torch.Tensor, batch: Batch):
        """Returns ``(hidden_for_lm_head, pre_mixer_residual)``."""
        residual = self.fuse(embedded, target_residual)
        residual = self.layers.op_list[0].forward(residual, batch)
        return self.hyper_connection_mixer.mix(residual)[0], residual

    def logits(self, residual: torch.Tensor) -> torch.Tensor:
        """Collapse the pre-mixer residual to ``[T, hidden]`` for the shared lm_head."""
        return self.hyper_connection_mixer.mix(residual)[0]


def mtp_state_bytes(head: Qwen4ExpMTP) -> int:
    """Device bytes the built head holds, summed from its own buffers.

    Read off the module rather than from a hand-written formula: the stacked expert
    tensors dominate and their size tracks the checkpoint, not a constant here.
    """
    return sum(t.numel() * t.element_size() for t in head.state_dict().values())


__all__ = ["MTP_KEY_PREFIX", "Qwen4ExpMTP", "Qwen4ExpMTPDecoderLayer", "mtp_state_bytes"]
