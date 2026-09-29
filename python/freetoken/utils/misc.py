from __future__ import annotations

import os


def drop_file_cache(path: str, offset: int = 0, nbytes: int = 0) -> None:
    """Best-effort POSIX_FADV_DONTNEED on ``path[offset : offset + nbytes]`` (whole file when nbytes=0).

    Host RAM cannot hold the expert banks and the checkpoint page cache at once, so
    loaded ranges are dropped to keep the banks resident. A no-op where the platform
    has no posix_fadvise (Windows) and on any OSError -- it is an optimisation.
    """
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return
    try:
        fd = os.open(path, os.O_RDONLY)
        try:
            fadvise(fd, offset, nbytes, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except OSError:
        pass


def call_if_main(name: str = "__main__", discard: bool | None = None):
    """Decorator to ensure a function will call when the script is run as main."""
    if name != "__main__":
        discard = False if discard is None else discard
        if discard:
            return lambda _: None
        else:
            return lambda f: f
    else:
        discard = True if discard is None else discard
        if discard:
            return lambda f: (f() or True) and None
        else:
            return lambda f: (f() and None) or f


def div_even(a: int, b: int, allow_replicate: bool = False) -> int:
    """Divides two integers. If allow_replicate=True, allows b > a when b % a == 0, returning 1."""
    if allow_replicate and b > a:
        assert b % a == 0, f"{b = } must be divisible by {a = } for KV head replication"
        return 1
    assert a % b == 0, f"{a = } must be divisible by {b = }"
    return a // b


def div_ceil(a: int, b: int) -> int:
    """Divides two integers, rounding up"""
    return (a + b - 1) // b


def align_ceil(a: int, b: int) -> int:
    """Aligns a to the next multiple of b"""
    return div_ceil(a, b) * b


def align_down(a: int, b: int) -> int:
    """Aligns a to the previous multiple of b"""
    return (a // b) * b


def mem_GB(size: int) -> str:
    """Format a byte count as a human-readable GiB string for logs."""
    return f"{size / (1024**3):.2f} GiB"


class Unset:
    pass


UNSET = Unset()
