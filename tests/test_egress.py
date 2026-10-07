import numpy as np
import pytest

from device.egress import EgressBlocked, EgressGuard, Tainted


def test_blocks_secret_on_sync():
    g = EgressGuard(["Okafor"])
    with pytest.raises(EgressBlocked):
        g.send("sync", "client Okafor")


def test_blocks_raw_index_on_pir():
    g = EgressGuard([])
    onehot = np.zeros(710, dtype=np.int64)
    onehot[3] = 1
    with pytest.raises(EgressBlocked):
        g.send("pir", onehot)


def test_blocks_tainted_and_unknown_channel():
    g = EgressGuard([])
    with pytest.raises(EgressBlocked):
        g.send("pir", Tainted(np.zeros(4), "facts"))
    with pytest.raises(EgressBlocked):
        g.send("http", b"anything")


def test_allows_lwe_batch():
    g = EgressGuard(["Okafor"])
    q = np.random.default_rng(0).integers(0, 1 << 32, size=(56, 710), dtype=np.uint32)
    assert g.send("pir", q)
