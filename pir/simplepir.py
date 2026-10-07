# from blind-counsel 8b1bf8f pir/simplepir.py
"""SimplePIR (Henzinger et al., USENIX Security 2023) over Z_q, numpy only.

Changes from blind-counsel: the paper's LWE parameters (n = 1024, q = 2^32, rounded
Gaussian noise with sigma 6.4) replace n = 512 with uniform noise; the plaintext
modulus p = 2^b is chosen per row count by the correctness bound; records are packed b
bits per cell; the LWE matrix A is expanded from a public seed on both sides, as in the
paper; a record may span c rows (chunked layout); and any number of queries are
answered as one matrix product (batched).

Protocol (m rows of L cells, k wanted rows):
  DB     D in Z_p^{m x L}, centred to [-p/2, p/2) when multiplied
  A      in Z_q^{m x n}, expanded from a public seed
  hint   H = A^T D in Z_q^{n x L}                   downloaded once (offline)
  query  Qu[j] = A s_j + e_j + Delta * u_{r_j}      k vectors of m entries -> server
  answer Ans = Qu D in Z_q^{k x L}                  one product            -> client
  decode Ans[j] - s_j^T H = e_j^T D + Delta * D[r_j], rounded.

Correctness holds with probability 1 - 2^-40 per cell when
  sigma * sqrt(m) * (p/2) * sqrt(2 ln(2 / 2^-40)) < q / (2p).
The server's view of each query is an LWE sample whose distribution does not depend on
the wanted row; `server_view_independent_of_index` checks this empirically.
"""
from __future__ import annotations

import math

import numpy as np

Q = 1 << 32           # ciphertext modulus; uint32 arithmetic wraps mod Q
N_LWE = 1024          # LWE secret dimension
SIGMA = 6.4           # rounded Gaussian noise
FAIL_LOG2 = -40       # per-cell decryption failure probability
MAX_BITS = 16         # cells are stored as uint8 (b <= 8) or uint16
_MASK = np.int64(0xFFFFFFFF)
_ELEMS = 1 << 24      # float64 elements per block (128 MB)


def plaintext_bits(m: int) -> int:
    """Largest b with p = 2^b meeting the correctness bound for m rows."""
    t = math.sqrt(2 * math.log(2 / 2.0 ** FAIL_LOG2))
    best = 0
    for b in range(1, MAX_BITS + 1):
        p = 1 << b
        if SIGMA * math.sqrt(m) * (p / 2) * t < Q / (2 * p):
            best = b
    if best == 0:
        raise ValueError(f"no plaintext modulus is correct for {m} rows")
    return best


def cell_dtype(b: int):
    return np.uint8 if b <= 8 else np.uint16


def cells_for(nbytes: int, b: int) -> int:
    return -(-8 * nbytes // b)


def pack(data: bytes, b: int, L: int | None = None) -> np.ndarray:
    """Bytes -> cells of b bits (big-endian bit order), zero-padded to L cells."""
    raw = np.frombuffer(data, dtype=np.uint8)
    n = cells_for(len(raw), b) if L is None else L
    if b == 8:
        out = np.zeros(n, dtype=np.uint8)
        out[: len(raw)] = raw
        return out
    bits = np.zeros(n * b, dtype=np.uint8)
    bits[: 8 * len(raw)] = np.unpackbits(raw)
    w = (1 << np.arange(b - 1, -1, -1)).astype(np.uint32)
    return (bits.reshape(n, b).astype(np.uint32) @ w).astype(cell_dtype(b))


def unpack(cells: np.ndarray, b: int, nbytes: int) -> bytes:
    if b == 8:
        return cells.astype(np.uint8)[:nbytes].tobytes()
    shifts = np.arange(b - 1, -1, -1, dtype=np.uint32)
    bits = ((cells.astype(np.uint32)[:, None] >> shifts) & 1).astype(np.uint8).ravel()
    return np.packbits(bits[: 8 * nbytes]).tobytes()


# ------------------------------------------------------------------ arithmetic mod q
def _limbs(X: np.ndarray):
    X = X.astype(np.uint32)
    return (X & 0xFFFF).astype(np.float64), (X >> 16).astype(np.float64)


def _wrap(x: np.ndarray) -> np.ndarray:
    """float64 holding an exact integer -> int64 reduced mod q."""
    return x.astype(np.int64) & _MASK


def mul_full(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(X @ Y) mod q for uint32 matrices with full 32-bit entries and a contraction
    dimension of at most 2^20. Three float64 products of 16-bit limbs, each exact."""
    x0, x1 = _limbs(X)
    y0, y1 = _limbs(Y)
    lo = _wrap(x0 @ y0)
    mid = (_wrap(x0 @ y1) + _wrap(x1 @ y0)) & _MASK
    return ((lo + (mid << 16)) & _MASK).astype(np.uint32)


def mul_db(X: np.ndarray, D: np.ndarray, centre: int, row0: int = 0) -> np.ndarray:
    """(X @ (D - centre)) mod q for X uint32 [k, m'] and cells D [m', L].

    Blocked over rows and columns so the float64 copy of D stays small. Each block sum
    is below 2^16 * 2^15 * rows <= 2^53, so float64 is exact."""
    k, m = X.shape
    L = D.shape[1]
    cols = min(L, max(1, _ELEMS // 1024))
    rows = max(1, min(m, 1 << 14, _ELEMS // cols))
    x0, x1 = _limbs(X)
    lo = np.zeros((k, L), dtype=np.int64)
    hi = np.zeros((k, L), dtype=np.int64)
    for c0 in range(0, L, cols):
        c1 = min(L, c0 + cols)
        for r0 in range(0, m, rows):
            r1 = min(m, r0 + rows)
            Db = D[r0:r1, c0:c1].astype(np.float64) - centre
            lo[:, c0:c1] = (lo[:, c0:c1] + _wrap(x0[:, r0:r1] @ Db)) & _MASK
            hi[:, c0:c1] = (hi[:, c0:c1] + _wrap(x1[:, r0:r1] @ Db)) & _MASK
    return ((lo + (hi << 16)) & _MASK).astype(np.uint32)


def expand_A(seed: int, r0: int, r1: int) -> np.ndarray:
    """Rows [r0, r1) of the public LWE matrix A, generated in fixed blocks of 1024 rows
    so any range can be produced independently by client and server."""
    B = 1024
    out = []
    for blk in range(r0 // B, -(-r1 // B)):
        rng = np.random.default_rng([seed, blk])
        a = rng.integers(0, Q, size=(B, N_LWE), dtype=np.uint32)
        lo, hi = max(r0, blk * B), min(r1, (blk + 1) * B)
        out.append(a[lo - blk * B: hi - blk * B])
    return np.concatenate(out) if out else np.zeros((0, N_LWE), np.uint32)


def _row_blocks(m: int, step: int = 1 << 14):
    for r0 in range(0, m, step):
        yield r0, min(m, r0 + step)


# ------------------------------------------------------------------ roles
class PIRServer:
    """The untrusted cloud. Holds D (cells) and the public seed; sees only queries."""

    def __init__(self, D: np.ndarray, b: int, seed: int = 0):
        assert D.dtype == cell_dtype(b) and D.ndim == 2
        self.D, self.b, self.seed = D, b, seed
        self.m, self.L = D.shape
        self.p = 1 << b
        self.views: list[np.ndarray] = []           # what the cloud saw, for the audit

    def hint(self) -> np.ndarray:
        H = np.zeros((N_LWE, self.L), dtype=np.uint32)
        for r0, r1 in _row_blocks(self.m):
            At = np.ascontiguousarray(expand_A(self.seed, r0, r1).T)
            H += mul_db(At, self.D[r0:r1], self.p // 2)   # uint32 add wraps mod q
        return H

    def answer(self, Qu: np.ndarray) -> np.ndarray:
        """Qu: uint32 [k, m] (k batched queries). Returns uint32 [k, L]."""
        Qu = np.atleast_2d(Qu)
        assert Qu.dtype == np.uint32 and Qu.shape[1] == self.m
        self.views.append(Qu.copy())
        return mul_db(Qu, self.D, self.p // 2)


class PIRClient:
    """The device. Knows the seed, the hint and the shape; keeps the secrets."""

    def __init__(self, seed: int, H: np.ndarray, m: int, b: int, rng_seed: int | None = None):
        self.seed, self.H, self.m, self.b = seed, H, m, b
        self.p = 1 << b
        self.delta = Q // self.p
        self.rng = np.random.default_rng(rng_seed)

    def query(self, rows: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """One batched query for the given rows. Returns (Qu [k, m] uint32, secrets)."""
        k = len(rows)
        S = self.rng.integers(0, Q, size=(N_LWE, k), dtype=np.uint32)
        AS = np.zeros((self.m, k), dtype=np.uint32)
        for r0, r1 in _row_blocks(self.m):
            AS[r0:r1] = mul_full(expand_A(self.seed, r0, r1), S)
        e = np.rint(self.rng.normal(0, SIGMA, size=(self.m, k))).astype(np.int64)
        Qu = (AS.astype(np.int64) + e) & _MASK
        Qu[rows, np.arange(k)] = (Qu[rows, np.arange(k)] + self.delta) & _MASK
        return np.ascontiguousarray(Qu.T.astype(np.uint32)), S

    def decode(self, Ans: np.ndarray, S: np.ndarray) -> np.ndarray:
        """Returns the wanted rows as cells, uint [k, L]."""
        resid = (Ans.astype(np.int64) - mul_full(S.T, self.H).astype(np.int64)) & _MASK
        v = np.floor(resid / self.delta + 0.5).astype(np.int64)
        return ((v + self.p // 2) % self.p).astype(cell_dtype(self.b))


def server_view_independent_of_index(D: np.ndarray, b: int, trials: int = 64) -> dict:
    """Empirical privacy check. The query the server sees for row 0 and for row m-1
    come from the same distribution: both look uniform on Z_q, and the wanted
    coordinate does not stand out."""
    srv = PIRServer(D, b, seed=7)
    cli = PIRClient(7, srv.hint(), srv.m, b, rng_seed=99)
    q0 = cli.query([0] * trials)[0].astype(np.float64)
    q1 = cli.query([srv.m - 1] * trials)[0].astype(np.float64)
    top0 = np.bincount((q0.ravel() / (Q / 16)).astype(int), minlength=16) / q0.size
    top1 = np.bincount((q1.ravel() / (Q / 16)).astype(int), minlength=16) / q1.size
    return dict(mean_i0=q0.mean(), mean_iN=q1.mean(), uniform_ref=Q / 2,
                targeted_col_i0=q0[:, 0].mean(), targeted_col_iN=q1[:, -1].mean(),
                rel_gap=abs(q0.mean() - q1.mean()) / (Q / 2),
                top_nibble_tv=0.5 * float(np.abs(top0 - top1).sum()))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    m, nbytes = 400, 512
    b = plaintext_bits(m)
    L = cells_for(nbytes, b)
    recs = [rng.bytes(nbytes) for _ in range(m)]
    D = np.stack([pack(r, b, L) for r in recs])
    srv = PIRServer(D, b)
    cli = PIRClient(0, srv.hint(), m, b, rng_seed=1)
    want = [0, 1, 7, 123, m - 1]
    Qu, S = cli.query(want)
    got = cli.decode(srv.answer(Qu), S)
    ok = sum(unpack(got[j], b, nbytes) == recs[r] for j, r in enumerate(want))
    print(f"m={m} rows, p=2^{b}: {ok}/{len(want)} rows recovered exactly in one batched query")
    pv = server_view_independent_of_index(D, b)
    print(f"privacy: mean(query|i=0)={pv['mean_i0']:.3e} mean(query|i=N-1)={pv['mean_iN']:.3e} "
          f"(uniform {pv['uniform_ref']:.3e}, rel gap {pv['rel_gap']:.2e}, top-nibble TV {pv['top_nibble_tv']:.3f})")
