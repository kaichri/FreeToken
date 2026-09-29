"""Pure-Python PLE disk row store: hashed row ids against the GPU oracle, byte fidelity, extents.

Runs on every platform (no ``freetoken.kernel._row_store`` dependency, unlike
test_ple_disk.py) because the store is the only PLE disk backend off Linux.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.qwen4_exp.config import parse_config
from freetoken.models.qwen4_exp.ple import NGramEmbedding
from freetoken.models.qwen4_exp.ple_disk_py import PyPleStore

from .common import EOS, hash_constants, toy_hf_config
from .test_ple import _meta

_HEADS = None


def _args():
    return parse_config(toy_hf_config()).qwen4_args


def _table(tmp_path, *, n_extents=1):
    """A synthetic fp8 table split over ``n_extents`` files; returns (store, table, args)."""
    args = _args()
    multipliers, sizes, offsets = hash_constants(args)
    total_rows = int(offsets[-1] + sizes[-1])
    gen = torch.Generator().manual_seed(5)
    table = torch.randint(0, 256, (total_rows, args.ngram_head_dim), dtype=torch.uint8, generator=gen)
    per_extent = total_rows // n_extents
    assert per_extent * n_extents == total_rows, "toy table must split evenly"
    paths, files, bases = [], [], []
    for e in range(n_extents):
        path = tmp_path / f"ple-{e}.bin"
        chunk = table[e * per_extent : (e + 1) * per_extent]
        path.write_bytes(chunk.numpy().tobytes())
        paths.append(str(path))
        files.append(e)
        # extent_base is the byte offset INSIDE its own file (source_from_safetensors
        # stores the file's header size there), and each extent gets its own file here
        bases.append(0)
    store = PyPleStore(
        paths=paths,
        extent_file=files,
        extent_base=bases,
        rows_per_extent=per_extent,
        row_bytes=args.ngram_head_dim,
        row_stride=args.ngram_head_dim,
        multipliers=multipliers.tolist(),
        head_vocab_sizes=sizes.tolist(),
        head_offsets=offsets.tolist(),
        eos_token_id=EOS,
    )
    return store, table, args


def _run(n_tokens=64, seed=3):
    """``[c0, c1, t0 ... t(n-1)]``; no token may equal EOS or the hash would ignore the window."""
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(EOS + 1, 512, (n_tokens + 2,), generator=gen, dtype=torch.int64)


def _embedding(args):
    emb = NGramEmbedding(args)
    multipliers, sizes, offsets = hash_constants(args)
    emb.layer_multipliers.copy_(multipliers)
    emb.ngram_heads_vocab_sizes.copy_(sizes)
    emb.ngram_heads_offsets.copy_(offsets)
    return emb


def test_row_ids_match_the_gpu_oracle_hash(tmp_path):
    """The sliding window the store hashes must be the one ple.py hashes."""
    store, _table_t, args = _table(tmp_path)
    ids = _run()
    emb = _embedding(args)
    # the whole run is the sequence; the two leading ids are the context, as host_fill_batch builds it
    oracle = emb.row_ids(_meta([ids.tolist()], [[EOS, EOS]]))
    # oracle position p hashes ids[p-2 : p+1]; the store hashes ids[i : i+3] for token i
    want = oracle[2:].reshape(-1)
    assert torch.equal(torch.tensor(store._row_ids(ids.tolist(), ids.numel() - 2), dtype=torch.int64), want)


@pytest.mark.parametrize("n_extents", [1, 3])
def test_staged_bytes_are_the_gathered_rows(tmp_path, n_extents):
    store, table, args = _table(tmp_path, n_extents=n_extents)
    ids = _run()
    n = ids.numel() - 2
    dst = torch.zeros(n * args.num_ngram_heads * args.ngram_head_dim, dtype=torch.uint8)
    store.stage(ids, dst)
    want = table[torch.tensor(store._row_ids(ids.tolist(), n), dtype=torch.int64)]
    assert torch.equal(dst.view(torch.uint16), want.reshape(-1).view(torch.uint16))


def test_empty_run_is_a_noop(tmp_path):
    store, _t, _a = _table(tmp_path)
    dst = torch.zeros(8, dtype=torch.uint8)
    dst.fill_(7)
    store.stage(torch.tensor([1, 2], dtype=torch.int64), dst)
    assert dst.tolist() == [7] * 8


def test_short_file_raises(tmp_path):
    store, _t, args = _table(tmp_path)
    store.paths[0] = store.paths[0]  # keep the handle, truncate below it
    with open(store.paths[0], "r+b") as fh:
        fh.truncate(4)
    ids = _run(n_tokens=8)
    dst = torch.zeros((ids.numel() - 2) * args.num_ngram_heads * args.ngram_head_dim, dtype=torch.uint8)
    with pytest.raises(OSError, match="short read"):
        store.stage(ids, dst)


def test_make_row_store_reports_no_flag_sync_when_extension_is_missing(monkeypatch):
    """Launch-gating, not the memop flag protocol: the Python store fills on the host already."""
    import freetoken.models.qwen4_exp.ple_disk as ple_disk

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def blocked(name, *args, **kwargs):
        if name.endswith("row_store") or name == "freetoken.kernel.row_store":
            raise ImportError("no extension")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", blocked)
    source = ple_disk.PleRowSource(
        paths=["x"], extent_file=[0], extent_base=[0],
        rows_per_extent=8, row_bytes=4, row_stride=4, scale=1.0,
    )
    store, flag_sync = ple_disk.make_row_store(
        source, {"num_ngram_heads": 2, "layer_multipliers": [1, 2, 3],
                 "per_head_vocab_sizes": [4, 4], "per_head_offsets": [0, 4], "eos_token_id": EOS},
    )
    assert isinstance(store, PyPleStore)
    assert flag_sync is False
