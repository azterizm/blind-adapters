import numpy as np
import pytest

from device.egress import EgressGuard
from pir.adapters import AdapterCloud, AdapterFetcher, IntegrityError, build


def _lib(N, c, sizes=(3001, 2500), seed=0):
    rng = np.random.default_rng(seed)
    recs = [(f"adapter-{i}", rng.bytes(s)) for i, s in enumerate(sizes)]
    D, manifest, lay = build(recs, N, c)
    cloud = AdapterCloud(D, lay)
    fetcher = AdapterFetcher(cloud.seed, cloud.hint(), lay, manifest, rng_seed=seed)
    return recs, cloud, fetcher, manifest, lay


@pytest.mark.parametrize("N", [2, 5, 16])
@pytest.mark.parametrize("c", [1, 3, 8])
def test_fetch_byte_exact(N, c):
    recs, cloud, fetcher, manifest, lay = _lib(N, c)
    for i, (_, raw) in enumerate(recs):
        assert fetcher.fetch(i, cloud) == raw
    assert fetcher.fetch(N - 1, cloud)          # a filler decodes and verifies too
    assert all(v.shape == (c, lay["m"]) for v in cloud.views)


def test_sha_mismatch_rejected():
    recs, cloud, fetcher, manifest, lay = _lib(4, 3)
    manifest[1]["sha256"] = "0" * 64
    with pytest.raises(IntegrityError):
        fetcher.fetch(1, cloud)


def test_tampered_record_rejected():
    # a cloud that flips one byte of adapter 1 and serves a hint consistent with it
    recs, cloud, fetcher, manifest, lay = _lib(4, 3)
    cloud.server.D[1 * 3 + 2, 5] ^= 1
    fetcher = AdapterFetcher(cloud.seed, cloud.hint(), lay, manifest, rng_seed=1)
    assert fetcher.fetch(0, cloud) == recs[0][1]
    with pytest.raises(IntegrityError):
        fetcher.fetch(1, cloud)


def test_stale_hint_rejected():
    # a cloud that changes the DB after publishing the hint breaks every fetch
    recs, cloud, fetcher, manifest, lay = _lib(4, 3)
    cloud.server.D[1 * 3 + 2, 5] ^= 1
    for i in (0, 1):
        with pytest.raises(IntegrityError):
            fetcher.fetch(i, cloud)


def test_fixed_schedule_and_guard():
    recs, cloud, fetcher, manifest, lay = _lib(16, 4)
    g = EgressGuard(["secret"])
    for i in (0, 1, 15):
        fetcher.fetch(i, cloud, guard=g)
    assert [r["allowed"] for r in g.log] == [True] * 3
    assert {v.shape for v in cloud.views} == {(4, 64)}


def test_view_independent_of_adapter_index():
    recs, cloud, fetcher, manifest, lay = _lib(16, 4)
    a = np.concatenate([fetcher.pir.query([0 * 4 + j for j in range(4)])[0] for _ in range(40)]).astype(float)
    b = np.concatenate([fetcher.pir.query([15 * 4 + j for j in range(4)])[0] for _ in range(40)]).astype(float)
    q = float(1 << 32)
    assert abs(a.mean() - b.mean()) / (q / 2) < 0.01
    # the wanted rows do not stand out from the rest of the query
    assert abs(a[:, :4].mean() - a[:, 4:].mean()) / (q / 2) < 0.05
    assert abs(b[:, -4:].mean() - b[:, :-4].mean()) / (q / 2) < 0.05
