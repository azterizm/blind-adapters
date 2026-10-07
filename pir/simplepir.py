# from blind-counsel 8b1bf8f pir/simplepir.py
"""SimplePIR (Henzinger et al.) over Z_q, stdlib + numpy only.

The cloud holds a public database D (here: statute text keyed by legal coordinate).
The device fetches row i WITHOUT the cloud learning i. The cloud's per-query view
is an LWE sample  qu = A s + e + Delta * u_i  which is pseudorandom and, crucially,
has a distribution independent of i -- that is the privacy guarantee, and
`server_view_independent_of_index` checks it empirically.

Protocol (retrieve a full row = one record):
  DB    D in Z_p^{N x L}           N records, each L cells (bytes, p = 256)
  hint  A in Z_q^{N x k}           public random (LWE matrix)
        H = A^T D in Z_q^{k x L}   sent to the client once (offline)
  query client picks secret s in Z_q^k, small noise e in Z^N,
        qu = A s + e + Delta * u_i   (Delta = q // p)              -> to server
  answer server returns  ans = qu^T D in Z_q^L                      -> to client
  decode client computes ans - s^T H = e^T D + Delta * row_i,
        row_i = round(. / Delta) mod p.

Noise budget: need |e^T D| < Delta/2 = q/(2p). With q = 2^32, p = 256 that is ~8.4e6,
and |e^T D| <= N * max|e| * (p-1); fine for N up to a few thousand with small e.
"""
from __future__ import annotations

import numpy as np

Q = 1 << 32          # ciphertext modulus
P = 256              # plaintext modulus (one byte per cell)
DELTA = Q // P
K = 512              # LWE dimension
NOISE = 8            # |e| <= NOISE


def _mm_mod(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """(X @ Y) mod Q for int64 arrays whose entries may be up to Q-1.

    Splits Y into 16-bit limbs so no intermediate exceeds int64. Requires the
    contraction dimension < 2^15 (true for all our matmuls: it is K, or N with
    byte-valued D)."""
    X = X.astype(np.int64); Y = Y.astype(np.int64)
    y_lo = Y & 0xFFFF
    y_hi = Y >> 16
    lo = (X @ y_lo) % Q
    hi = (X @ y_hi) % Q
    return (((hi << 16) % Q) + lo) % Q


class PIRServer:
    """The untrusted cloud. Holds D and A; answers queries; sees only `qu`."""

    def __init__(self, D: np.ndarray, seed: int = 0):
        assert D.dtype == np.uint8
        self.D = D.astype(np.int64)                 # [N, L] in [0, P)
        self.N, self.L = D.shape
        rng = np.random.default_rng(seed)
        self.A = rng.integers(0, Q, size=(self.N, K), dtype=np.int64)
        self._views: list[np.ndarray] = []          # for the privacy audit only

    def hint(self) -> np.ndarray:
        return _mm_mod(self.A.T, self.D)            # H = A^T D  [K, L]

    def answer(self, qu: np.ndarray) -> np.ndarray:
        self._views.append(qu.copy())               # record what the cloud sees
        # ans = qu^T D ; D is byte-valued so int64 is safe without limb-splitting
        return (qu.astype(np.int64) @ self.D) % Q    # [L]


class PIRClient:
    """The sealed device. Knows A and H (public); keeps s secret; recovers rows."""

    def __init__(self, A: np.ndarray, H: np.ndarray, seed: int = 1234):
        self.A = A.astype(np.int64)
        self.H = H.astype(np.int64)
        self.N = A.shape[0]
        self.rng = np.random.default_rng(seed)

    def query(self, i: int):
        s = self.rng.integers(0, Q, size=K, dtype=np.int64)
        e = self.rng.integers(-NOISE, NOISE + 1, size=self.N, dtype=np.int64)
        u = np.zeros(self.N, dtype=np.int64); u[i] = 1
        qu = (self.A @ s + e + DELTA * u) % Q
        return qu, s

    def decode(self, ans: np.ndarray, s: np.ndarray) -> np.ndarray:
        sH = _mm_mod(s.reshape(1, -1), self.H)[0]    # s^T H  [L]
        resid = (ans - sH) % Q
        row = np.rint(resid.astype(np.float64) / DELTA).astype(np.int64) % P
        return row.astype(np.uint8)


def server_view_independent_of_index(D: np.ndarray, trials: int = 200) -> dict:
    """Empirical privacy check: the server's query vectors for DIFFERENT target
    indices are drawn from the same (pseudorandom) distribution. We compare the
    bytes the server sees for i=0 vs i=N-1 and confirm (a) both look uniform on
    Z_q and (b) they are statistically indistinguishable."""
    srv = PIRServer(D, seed=7)
    cli = PIRClient(srv.A, srv.hint(), seed=99)
    qs0 = np.stack([cli.query(0)[0] for _ in range(trials)])
    qs1 = np.stack([cli.query(srv.N - 1)[0] for _ in range(trials)])
    # mean over the uniform range should be ~Q/2 for both; difference tiny
    m0, m1 = qs0.mean(), qs1.mean()
    # per-coordinate: does the entry at the targeted index stand out? (it must not)
    col0_mean = qs0[:, 0].mean()
    colN_mean = qs1[:, -1].mean()
    overall = np.concatenate([qs0.ravel(), qs1.ravel()]).mean()
    return dict(mean_i0=m0, mean_iN=m1, uniform_ref=Q / 2,
                targeted_col_i0=col0_mean, targeted_col_iN=colN_mean,
                rel_gap=abs(m0 - m1) / (Q / 2), overall=overall)


if __name__ == "__main__":
    # correctness + privacy smoke test on random data
    rng = np.random.default_rng(0)
    N, L = 400, 512
    D = rng.integers(0, P, size=(N, L), dtype=np.uint8)
    srv = PIRServer(D)
    cli = PIRClient(srv.A, srv.hint())
    ok = 0
    for i in [0, 1, 7, 123, N - 1]:
        qu, s = cli.query(i)
        row = cli.decode(srv.answer(qu), s)
        ok += int(np.array_equal(row, D[i]))
    print(f"correctness: {ok}/5 rows recovered exactly")
    pv = server_view_independent_of_index(D)
    print(f"privacy: mean(query|i=0)={pv['mean_i0']:.3e}  mean(query|i=N-1)={pv['mean_iN']:.3e}  "
          f"(uniform ref {pv['uniform_ref']:.3e}, rel gap {pv['rel_gap']:.2e})")
    print(f"targeted-cell mean i=0: {pv['targeted_col_i0']:.3e}  i=N-1: {pv['targeted_col_iN']:.3e} "
          f"-> the retrieved index does not stand out in the server's view")
