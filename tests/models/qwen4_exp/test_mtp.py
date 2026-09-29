"""Qwen3.8-Flash-Next MTP head: optionality, state-dict coverage and the Phase-0 semantics.

The norm and fusion choices here are the ones benchmarks/mtp_probe.py measured against
the real checkpoint: the two ``pre_fc_norm_*`` weights are zero-centered (1+w applied
at runtime) and the hidden norm is per-hyper-connection-branch. The variants that
differ scored 0% next-token agreement, i.e. chance, against 89% for these.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.hc import GatedResidual, GroupedPlusOneRMSNorm
from freetoken.models.qwen4_exp.mtp import MTP_KEY_PREFIX, Qwen4ExpMTP

from .common import requires_cuda, toy_hf_config

# the module buffers, i.e. the 31 checkpoint tensors after the reader's fusions
MTP_SHAPES = {
    "mtp.pre_fc_norm_embedding.weight": (128,),
    "mtp.pre_fc_norm_hidden.weight": (512,),          # hc_count * hidden
    "mtp.fc_embedding.weight": (128, 128),
    "mtp.fc_hidden.weight": (128, 128),               # one weight, shared by all 4 branches
    "mtp.hyper_connection_mixer.hc_norm.weight": (512,),
    "mtp.hyper_connection_mixer.input_mix_weight_down.weight": (16, 512),
    "mtp.hyper_connection_mixer.input_mix_weight_up.weight": (512, 16),
    "mtp.layers.0.self_attn.qkv_proj.weight": (640, 128),   # q|k|v fused
    "mtp.layers.0.self_attn.o_proj.weight": (128, 256),
    "mtp.layers.0.self_attn.q_norm.weight": (64,),
    "mtp.layers.0.self_attn.k_norm.weight": (64,),
    "mtp.layers.0.self_attn.indexer.index_qk_proj.weight": (192, 128),
    "mtp.layers.0.self_attn.indexer.q_layernorm.weight": (64,),
    "mtp.layers.0.self_attn.indexer.k_layernorm.weight": (64,),
    "mtp.layers.0.attn_hyper_connection.hc_norm.weight": (512,),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_down_block_inject.weight": (32, 512),
    "mtp.layers.0.attn_hyper_connection.input_mix_weight_up.weight": (512, 16),
    "mtp.layers.0.mlp_hyper_connection.hc_norm.weight": (512,),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_down_block_inject.weight": (32, 512),
    "mtp.layers.0.mlp_hyper_connection.input_mix_weight_up.weight": (512, 16),
    "mtp.layers.0.mlp.gate.weight": (8, 128),
    "mtp.layers.0.mlp.shared_expert_gate.weight": (1, 128),
    "mtp.layers.0.mlp.shared_expert.gate_up_proj.weight": (128, 128),  # gate|up fused
    "mtp.layers.0.mlp.shared_expert.down_proj.weight": (128, 64),
    "mtp.layers.0.mlp.experts.gate_up_proj": (8, 128, 128),   # stacked, one tensor
    "mtp.layers.0.mlp.experts.down_proj": (8, 128, 64),
}


def _config(*, enable_mtp: bool):
    config = parse_config(toy_hf_config())
    assert config.qwen4_args.mtp_num_hidden_layers == 1
    assert config.qwen4_args.mtp_layer_types == ("full_attention",)
    return replace(config, mtp_enabled=enable_mtp, moe_strategy="fused")


def _head(*, enable_mtp: bool = True) -> tuple[Qwen4ExpMTP, object]:
    from freetoken.utils import torch_dtype

    config = _config(enable_mtp=enable_mtp)
    with torch_dtype(torch.bfloat16):
        return Qwen4ExpMTP(config), config


def _initialised_head() -> tuple[Qwen4ExpMTP, object]:
    """The module allocates with torch.empty, so the numeric checks need real values."""
    head, config = _head()
    gen = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for tensor in head.state_dict().values():
            if tensor.dtype.is_floating_point and tensor.dim() <= 2:
                # norm weights near 1 (zero-centered) rather than near 0, so (1+w) is exercised
                scale = 0.05 if tensor.dim() == 1 else 0.05
                tensor.copy_(torch.randn(tensor.shape, generator=gen).to(tensor.dtype) * scale)
    return head, config


def test_head_is_absent_unless_enabled():
    config = _config(enable_mtp=False)
    assert config.mtp_enabled is False
    with torch_dtype_bf16():
        model = _trunk(config)
    assert model.mtp is None
    assert not any(k.startswith(MTP_KEY_PREFIX) for k in model.state_dict())


def torch_dtype_bf16():
    from freetoken.utils import torch_dtype

    return torch_dtype(torch.bfloat16)


def _trunk(config):
    from freetoken.models.qwen4_exp.model import Qwen4ExpForCausalLM

    return Qwen4ExpForCausalLM(config)


def test_state_dict_covers_exactly_the_checkpoint_buffers():
    head, _ = _head()
    got = {f"mtp.{k}": v.shape for k, v in head.state_dict().items()}
    assert got == {k: torch.Size(v) for k, v in MTP_SHAPES.items()}
    assert all(v.dtype is torch.bfloat16 for v in head.state_dict().values())


def test_head_shares_the_trunk_embedding_and_lm_head():
    """The checkpoint has no mtp.embed_tokens / mtp.lm_head; sharing must be structural."""
    head, _ = _head()
    assert not hasattr(head, "embed_tokens")
    assert not hasattr(head, "lm_head")
    assert not any("embed_tokens" in k or "lm_head" in k for k in head.state_dict())


def test_layer_is_one_full_attention_layer_without_ple_or_gdn():
    head, _ = _head()
    layer = head.layers.op_list[0]
    assert hasattr(layer, "self_attn")
    assert not hasattr(layer, "linear_attn")
    assert not hasattr(layer, "ple")
    assert isinstance(layer.attn_hyper_connection, GatedResidual)
    assert layer.attn_hyper_connection.use_combine is True
    assert head.hyper_connection_mixer.use_combine is False


def test_expert_layout_matches_the_fused_bf16_kernel():
    """The stacked tensors are the per-expert kernel banks with a leading expert axis."""
    head, config = _head()
    experts = head.layers.op_list[0].mlp.experts
    layout = experts.quant_method.layout()
    assert set(layout) == {"gate_up", "down"}
    i, hidden, e = config.moe_intermediate_size, config.hidden_size, config.num_experts
    assert tuple(layout["gate_up"].shape) == (2 * i, hidden)
    assert tuple(layout["down"].shape) == (hidden, i)
    assert tuple(experts.gate_up_proj.shape) == (e, *layout["gate_up"].shape)
    assert tuple(experts.down_proj.shape) == (e, *layout["down"].shape)
    assert experts.gate_up_proj.is_contiguous() and experts.down_proj.is_contiguous()


def test_experts_stay_unquantized_bf16_under_the_checkpoints_ignore_list():
    """``quantization_config.ignore`` lists ``mtp.*``, so no MTP weight is requantized."""
    head, _ = _head()
    for tensor in head.state_dict().values():
        assert tensor.dtype is torch.bfloat16
    experts = head.layers.op_list[0].mlp.experts
    assert experts.gate_up_proj.dtype is torch.bfloat16
    assert experts.down_proj.dtype is torch.bfloat16


def test_pre_fc_norms_are_zero_centered_and_branch_grouped():
    head, config = _initialised_head()
    hc = config.qwen4_args.hc_count
    assert isinstance(head.pre_fc_norm_hidden, GroupedPlusOneRMSNorm)
    assert head.pre_fc_norm_hidden.num_groups == hc
    assert head.pre_fc_norm_hidden.weight.shape == (hc * config.hidden_size,)
    assert head.pre_fc_norm_embedding.weight.shape == (config.hidden_size,)

    x = torch.randn(1, 8, hc * config.hidden_size, dtype=torch.float32)
    got = head.pre_fc_norm_hidden.forward(x)
    # per-branch fp32 RMS, scaled by (1+w)
    branches = x.view(1, 8, hc, -1)
    want = branches * torch.rsqrt(branches.pow(2).mean(-1, keepdim=True) + 1e-6)
    want = (want * (1.0 + head.pre_fc_norm_hidden.weight.float()).view(1, 1, hc, -1)).flatten(-2)
    assert torch.allclose(got.float(), want, atol=1e-3)
    # the raw weight (no +1) is a different function, which is why the reading is load-bearing
    raw = (branches * torch.rsqrt(branches.pow(2).mean(-1, keepdim=True) + 1e-6)
           * head.pre_fc_norm_hidden.weight.float().view(1, 1, hc, -1)).flatten(-2)
    assert not torch.allclose(want, raw, atol=1e-2)


@requires_cuda
def test_pre_fc_norm_embedding_is_plus_one_scaled():
    """The kernel path: GemmaPlusOneRMSNorm applies (1+w) in fp32."""
    head, config = _initialised_head()
    head = _to_cuda(head)
    x = torch.randn(1, 4, config.hidden_size, device="cuda", dtype=torch.bfloat16)
    got = head.pre_fc_norm_embedding.forward(x)
    want = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + config.rms_norm_eps) * (
        1.0 + head.pre_fc_norm_embedding.weight.float()
    )
    assert torch.allclose(got.float(), want, atol=2e-2)


@requires_cuda
def test_fuse_is_additive_over_branches_not_a_concatenation():
    head, config = _initialised_head()
    head = _to_cuda(head)
    t, hc, h = 3, config.qwen4_args.hc_count, config.hidden_size
    embedded = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    target = torch.randn(t, hc * h, device="cuda", dtype=torch.bfloat16)
    got = head.fuse(embedded, target)
    e = head.fc_embedding.forward(head.pre_fc_norm_embedding.forward(embedded))
    n = head.pre_fc_norm_hidden.forward(target).view(t, hc, h)
    want = (e.unsqueeze(1) + head.fc_hidden.forward(n)).view(t, hc * h)
    assert got.shape == (t, hc * h)
    assert torch.equal(got, want)
    # a concatenation would need 2*hidden per branch; the checkpoint's [hidden, hidden]
    # fc_hidden cannot consume that, so the merge must be additive
    assert head.fc_hidden.full_input_size == h


def _to_cuda(module, dtype=None):
    """BaseOP has no .to(); walk the attributes the state_dict is built from."""
    for name, value in list(module.__dict__.items()):
        if isinstance(value, torch.Tensor):
            setattr(module, name, value.to("cuda", dtype) if dtype else value.to("cuda"))
        elif hasattr(value, "__dict__"):
            _to_cuda(value, dtype)
    return module


def test_normal_path_is_unchanged_by_the_flag():
    """Same trunk keys and shapes either way; only the mtp.* subtree differs."""
    with torch_dtype_bf16():
        without = _trunk(_config(enable_mtp=False)).state_dict()
        with_head = _trunk(_config(enable_mtp=True)).state_dict()
    trunk_without = {k: v.shape for k, v in without.items()}
    trunk_with = {k: v.shape for k, v in with_head.items() if not k.startswith(MTP_KEY_PREFIX)}
    assert trunk_without == trunk_with
    assert set(with_head) - set(without) == set(MTP_SHAPES)


MODEL_ENV = "FREETOKEN_TEST_MODEL"


@requires_cuda
@pytest.mark.needs_weights
def test_matches_the_phase_zero_probe_on_the_real_checkpoint():
    """The in-tree head and benchmarks/mtp_probe.py's standalone re-derivation must agree.

    The probe is a separate implementation of the Phase-0 semantics (its own fusion, its
    own grouped norm), so agreement pins the in-tree wiring -- key names, fusions, norm
    classes, residual flow -- to the measurement the Go/No-Go was taken from.
    """
    import os
    import sys
    from pathlib import Path

    model_path = os.environ.get(MODEL_ENV)
    if not model_path or not Path(model_path).is_dir():
        pytest.skip(f"set {MODEL_ENV} to a local checkpoint dir")
    repo = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo / "benchmarks"))

    import mtp_probe as probe
    from freetoken.core import Context, set_global_ctx
    from freetoken.distributed import DistributedInfo
    from freetoken.engine import EngineConfig
    from freetoken.layers.rotary import set_rope_device
    from freetoken.models.qwen4_exp.attention import TorchDenseQSAReference
    from freetoken.utils import torch_dtype

    set_rope_device(torch.device("cuda", 0))
    config = replace(
        EngineConfig(model_path=model_path, tp_info=DistributedInfo(0, 1), dtype=torch.bfloat16).model_config,
        moe_strategy="fused", decode_target="gpu", mtp_enabled=True,
    )
    torch.manual_seed(0)
    # _load_mtp_weights already returns the fused, module-keyed dict
    fused = probe._load_mtp_weights(model_path, torch.device("cuda", 0))

    # both heads, same tensors
    ref = probe.MTPHead(config, fused, probe.VARIANTS[0])
    with torch_dtype(torch.bfloat16):
        head = Qwen4ExpMTP(config)
    for name, tensor in list(head.state_dict().items()):
        src = fused[f"mtp.{name}"]
        assert src.shape == tensor.shape, name
        setattr_tensor(head, name, src)
    _to_cuda(head)

    max_len = 64
    ctx = Context(page_size=1)
    ctx.page_table = torch.zeros(1, max_len, dtype=torch.int32, device="cuda")
    set_global_ctx(ctx)
    backend = TorchDenseQSAReference(config, num_slots=1, max_len=max_len,
                                     device=torch.device("cuda"), dtype=torch.bfloat16)
    ctx.attn_backend = backend
    batch = probe.ProbeBatch(reqs=[probe.ProbeReq(table_idx=0, cached_len=0, extend_len=max_len)],
                            positions=torch.arange(max_len, device="cuda", dtype=torch.int32))
    with torch.inference_mode(), ctx.forward_batch(batch):
        ids = torch.randint(0, 1000, (max_len,), device="cuda")
        residual = torch.randn(max_len, hc_width(config), device="cuda", dtype=torch.bfloat16)
        embedded = torch.nn.functional.embedding(ids, ref.embed)

        # The fusion is the whole of the Phase-0 semantics: two norms, two GEMMs and an add.
        # Not bit-equal because the in-tree grouped norm runs the vendored Triton kernel and
        # the probe runs the torch path, but the tolerance is ~100x tighter than any of the
        # rejected variants (whose agreement was 0% against this one's 89%).
        ref_fused = ref.fuse(ids, residual)
        mine_fused = head.fuse(embedded, residual)
        _assert_close(mine_fused, ref_fused, "fuse")

        # Everything past the fusion is the SAME production code in both objects, so a tight
        # comparison there would only measure bf16 noise through the MoE. These two run the
        # whole path and catch a gross fault (wrong layer slot, unwritten KV, broken mixer)
        # without pretending to a precision the dtype does not carry.
        ref_residual = ref.layer(ref_fused, batch)
        mine_residual = head.layers.op_list[0].forward(mine_fused, batch)
        _assert_close(head.logits(mine_residual), ref.mixer.mix(ref_residual)[0], "mixer", rel=0.25)

        # The probe's logits() is mixer + shared lm_head; the in-tree one is the collapse
        # only (Qwen4ExpForCausalLM.mtp_logits applies the head), so add it here.
        ref_logits = ref.logits(ref_residual)
        mine_logits = torch.nn.functional.linear(head.logits(mine_residual), ref.lm_head)
    _assert_close(mine_logits, ref_logits, "logits", rel=0.5)
    agree = (mine_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean()
    assert agree >= 0.9, f"top-1 agreement {agree:.3f}"


def _assert_close(got, want, what: str, rel: float = 0.05):
    scale = want.float().std().clamp_min(1e-6)
    worst = (got.float() - want.float()).abs().max() / scale
    assert worst <= rel, f"{what}: worst deviation {worst:.4f} of the signal std (limit {rel})"


def hc_width(config) -> int:
    return config.qwen4_args.hc_count * config.hidden_size


def setattr_tensor(head, name, tensor):
    target = head
    parts = name.split(".")
    for part in parts[:-1]:
        target = getattr(target, part) if not part.isdigit() else target.op_list[int(part)]
    setattr(target, parts[-1], tensor)
