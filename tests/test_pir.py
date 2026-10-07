import numpy as np
import pytest

from pir.simplepir import (PIRClient, PIRServer, cells_for, pack, plaintext_bits,
                           server_view_independent_of_index, unpack)


@pytest.mark.parametrize("b", [4, 8, 10, 13])
def test_pack_roundtrip(b):
    data = np.random.default_rng(b).bytes(1001)
    assert unpack(pack(data, b), b, len(data)) == data


def test_plaintext_bits_follow_bound():
    # the SimplePIR paper's p = 991 at 2^13 rows rounds down to 2^9
    assert plaintext_bits(1 << 13) == 9
    assert plaintext_bits(1 << 20) == 8
    assert plaintext_bits(710) == 10


@pytest.mark.parametrize("m,nbytes", [(3, 100), (710, 2048), (300, 4097)])
def test_batched_rows_exact(m, nbytes):
    rng = np.random.default_rng(m)
    b = plaintext_bits(m)
    L = cells_for(nbytes, b)
    recs = [rng.bytes(nbytes) for _ in range(m)]
    srv = PIRServer(np.stack([pack(r, b, L) for r in recs]), b, seed=5)
    cli = PIRClient(5, srv.hint(), m, b, rng_seed=6)
    want = sorted({0, m - 1, m // 2})
    Qu, S = cli.query(want)
    rows = cli.decode(srv.answer(Qu), S)
    assert [unpack(rows[j], b, nbytes) for j in range(len(want))] == [recs[r] for r in want]


def test_view_independent_of_index():
    rng = np.random.default_rng(0)
    b = plaintext_bits(256)
    D = rng.integers(0, 1 << b, size=(256, 64)).astype(np.uint16)
    pv = server_view_independent_of_index(D, b)
    assert pv["rel_gap"] < 0.01
    assert pv["top_nibble_tv"] < 0.03
    for k in ("targeted_col_i0", "targeted_col_iN"):
        assert abs(pv[k] - pv["uniform_ref"]) / pv["uniform_ref"] < 0.3
