"""Package-wide runtime hygiene: TP info set once, the global ctx never leaks across tests."""

import pytest


@pytest.fixture(autouse=True)
def _runtime():
    import freetoken.core as core
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.rotary import get_rope

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    # get_rope is memoised without the device in its key, so a module built on CPU hands a
    # CPU cos_sin cache to the next test that asks for the same geometry.
    get_rope.cache_clear()
    core._GLOBAL_CTX = None
    yield
    core._GLOBAL_CTX = None
    get_rope.cache_clear()
