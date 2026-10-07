"""Adapter library served by chunked SimplePIR.

Records: every adapter is serialized (adapters/convert.py), and every record is padded
to the largest size S. Filler records of random bytes scale the library to N. The
public manifest (index, name, size, sha256) is downloaded in full by every device.

Layout: a record is split into c chunks of ceil(S / c) bytes; row r = index * c + j.
So the DB has m = N * c rows of L = cells_for(S / c) cells. A fetch is always exactly
c queries (rows index*c .. index*c + c - 1), sent and answered as one batched product.
Cells are bytes (b = min(correctness bound, 8)) so the cloud stores one byte per cell.

Costs per fetch (n = 1024, 4-byte ring elements):
  hint      4 n L        bytes, downloaded once per library version
  upload    4 c m        bytes
  download  4 c L        bytes  (about 4 S at b = 8)
  server    c m L        multiply-adds, one pass over the DB per batch

Run:  python -m pir.adapters --sweep     cost table against download-all and text PIR
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os

import numpy as np

from pir.simplepir import (N_LWE, PIRClient, PIRServer, cells_for, pack, plaintext_bits,
                           unpack)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEXT_PIR = dict(rows=710, row_bytes=2048, queries=56)     # blind-counsel text DB


class IntegrityError(Exception):
    pass


def bits_for(m: int) -> int:
    return min(plaintext_bits(m), 8)


def layout(N: int, S: int, c: int) -> dict:
    m = N * c
    b = bits_for(m)
    chunk = -(-S // c)
    L = cells_for(chunk, b)
    return dict(N=N, S=S, c=c, m=m, b=b, chunk=chunk, L=L,
                hint=4 * N_LWE * L, upload=4 * c * m, download=4 * c * L, server_madds=c * m * L)


def filler(index: int, S: int, seed: int = 0) -> bytes:
    return np.random.default_rng([seed, index]).bytes(S)


def build(records: list[tuple[str, bytes]], N: int, c: int) -> tuple[np.ndarray, list[dict], dict]:
    """records: the real adapters (name, bytes). Returns (D, manifest, layout)."""
    S = max(len(r) for _, r in records)
    lay = layout(N, S, c)
    D = np.zeros((lay["m"], lay["L"]), dtype=np.uint8)
    manifest = []
    for i in range(N):
        name, raw = records[i] if i < len(records) else (f"filler-{i:04d}", filler(i, S))
        manifest.append(dict(index=i, name=name, size=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
        padded = raw.ljust(S, b"\x00")
        for j in range(c):
            D[i * c + j] = pack(padded[j * lay["chunk"]:(j + 1) * lay["chunk"]], lay["b"], lay["L"])
    return D, manifest, lay


class AdapterCloud:
    """Untrusted: holds the library, answers batched queries, records its view."""

    def __init__(self, D: np.ndarray, lay: dict, seed: int = 11):
        self.lay = lay
        self.server = PIRServer(D, lay["b"], seed=seed)
        self.seed = seed

    def hint(self) -> np.ndarray:
        return self.server.hint()

    def answer(self, Qu: np.ndarray) -> np.ndarray:
        return self.server.answer(Qu)

    @property
    def views(self) -> list[np.ndarray]:
        return self.server.views


class AdapterFetcher:
    """Device side. Fetches one adapter per matter with exactly c queries."""

    def __init__(self, seed: int, H: np.ndarray, lay: dict, manifest: list[dict], rng_seed=None):
        self.lay, self.manifest = lay, manifest
        self.pir = PIRClient(seed, H, lay["m"], lay["b"], rng_seed=rng_seed)

    def fetch(self, index: int, cloud, guard=None) -> bytes:
        c = self.lay["c"]
        Qu, S = self.pir.query([index * c + j for j in range(c)])
        if guard is not None:
            guard.send("pir", Qu, note="adapter library rows")
        rows = self.pir.decode(cloud.answer(Qu), S)
        raw = b"".join(unpack(rows[j], self.lay["b"], self.lay["chunk"]) for j in range(c))
        entry = self.manifest[index]
        raw = raw[: entry["size"]]
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise IntegrityError(f"adapter {entry['name']}: sha256 does not match the manifest")
        return raw


# ------------------------------------------------------------------ sweep
def _mb(x: float) -> str:
    return f"{x / 1e6:,.1f}" if x < 1e11 else f"{x / 1e9:,.0f} GB"


def sweep(S: int, Ns=(2, 16, 128, 1024), cs=(16, 64, 256, 1024), measured: dict | None = None) -> list[dict]:
    measured = measured or {}
    t = TEXT_PIR
    tb = plaintext_bits(t["rows"])
    tL = cells_for(t["row_bytes"], tb)
    print(f"Adapter record S = {S / 1e6:.1f} MB. Sizes in MB unless marked. n = {N_LWE}, q = 2^32.")
    print(f"Text PIR (blind-counsel DB): hint {_mb(4 * N_LWE * tL)}, upload {_mb(4 * t['queries'] * t['rows'])}, "
          f"download {_mb(4 * t['queries'] * tL)} per matter ({t['queries']} queries, p = 2^{tb}).")
    hdr = f"{'N':>5} {'c':>5} {'rows':>8} {'p':>5} {'hint':>10} {'upload':>9} {'download':>9} {'server madds':>13} " \
          f"{'srv np s':>9} {'srv go s':>9} {'cli s':>7} {'all-dl':>10}"
    print(hdr)
    rows = []
    for N in Ns:
        for c in cs:
            lay = layout(N, S, c)
            mk = f"{N},{c}"
            mm = measured.get(mk, {})
            rows.append(dict(lay, measured=mm, download_all=N * S))
            f = lambda k: f"{mm[k]:.2f}" if k in mm else "-"  # noqa: E731
            print(f"{N:>5} {c:>5} {lay['m']:>8} {'2^' + str(lay['b']):>5} {_mb(lay['hint']):>10} "
                  f"{_mb(lay['upload']):>9} {_mb(lay['download']):>9} {lay['server_madds']:>13.2e} "
                  f"{f('server_numpy_s'):>9} {f('server_go_s'):>9} {f('client_s'):>7} {_mb(N * S):>10}")
    # crossover: smallest N where one fetch plus the hint moves fewer bytes than download-all
    print("\nCrossover against download-all (hint + upload + download of one fetch < N x S):")
    for c in cs:
        for N in (2 ** k for k in range(1, 16)):
            lay = layout(N, S, c)
            if lay["hint"] + lay["upload"] + lay["download"] < N * S:
                print(f"  c = {c:>5}: N >= {N} (download-all {_mb(N * S)} MB)")
                break
    return rows


def client_bench(S: int, Ns=(2, 16, 128, 1024), cs=(16, 64, 256, 1024), max_hint=3.1e9, max_upload=1.5e9) -> dict:
    """Laptop time for one fetch on the device: build c queries and decode c answers,
    with a random hint of the right shape (timing does not depend on its content)."""
    import time
    out = {}
    for N in Ns:
        for c in cs:
            lay = layout(N, S, c)
            if lay["hint"] > max_hint or lay["upload"] > max_upload:
                continue
            rng = np.random.default_rng(0)
            H = rng.integers(0, 1 << 32, size=(N_LWE, lay["L"]), dtype=np.uint32)
            cli = PIRClient(1, H, lay["m"], lay["b"], rng_seed=2)
            t = time.perf_counter()
            Qu, Sec = cli.query([j for j in range(c)])
            q_s = time.perf_counter() - t
            ans = rng.integers(0, 1 << 32, size=(c, lay["L"]), dtype=np.uint32)
            t = time.perf_counter()
            cli.decode(ans, Sec)
            d_s = time.perf_counter() - t
            out[f"{N},{c}"] = dict(client_s=q_s + d_s, client_query_s=q_s, client_decode_s=d_s)
            print(f"client N={N} c={c}: query {q_s:.2f}s decode {d_s:.2f}s", flush=True)
            del H, Qu, ans
    return out


TASKS = ["uk_unfair_dismissal", "uk_redundancy_payment"]


def write_library(variant: str) -> dict:
    """Publish the real adapters as delivery records plus the public manifest."""
    import shutil
    d = os.path.join(ROOT, "artifacts", "library")
    os.makedirs(d, exist_ok=True)
    recs = []
    for i, t in enumerate(TASKS):
        src = os.path.join(ROOT, "artifacts", "adapters", t, f"adapter.{variant}")
        shutil.copyfile(src, os.path.join(d, f"{t}.{variant}"))
        raw = open(src, "rb").read()
        recs.append(dict(index=i, name=t, file=f"{t}.{variant}", size=len(raw),
                         sha256=hashlib.sha256(raw).hexdigest()))
    man = dict(variant=variant, records=recs)
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    return man


def load_library(N: int, c: int, seed: int = 11):
    """The real library padded with fillers to N: (cloud, public manifest, layout)."""
    d = os.path.join(ROOT, "artifacts", "library")
    man = json.load(open(os.path.join(d, "manifest.json")))
    recs = [(e["name"], open(os.path.join(d, e["file"]), "rb").read()) for e in man["records"]]
    D, manifest, lay = build(recs, N, c)
    return AdapterCloud(D, lay, seed=seed), manifest, lay


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--library", choices=["fp16", "int8"], help="publish the delivery records")
    ap.add_argument("--client-bench", action="store_true", help="time the device side, store in results/pir_sweep.json")
    ap.add_argument("--size", type=int, default=0, help="record bytes S (default: the largest delivery record)")
    a = ap.parse_args()
    if a.library:
        print(json.dumps(write_library(a.library), indent=1))
    if a.client_bench:
        S = a.size or max(e["size"] for e in json.load(open(os.path.join(ROOT, "artifacts", "library", "manifest.json")))["records"])
        res = os.path.join(ROOT, "results", "pir_sweep.json")
        old = json.load(open(res)) if os.path.exists(res) else dict(S=S, measured={})
        for k, v in client_bench(S).items():
            old["measured"].setdefault(k, {}).update(v)
        old["meta"] = dict(old.get("meta", {}), client_machine="laptop (Apple M4, 16 GB)")
        os.makedirs(os.path.dirname(res), exist_ok=True)
        json.dump(old, open(res, "w"), indent=1)
    if a.sweep:
        S = a.size
        if not S:
            m = os.path.join(ROOT, "artifacts", "library", "manifest.json")
            S = max(e["size"] for e in json.load(open(m))["records"]) if os.path.exists(m) else 43_646_976
        res = os.path.join(ROOT, "results", "pir_sweep.json")
        sweep(S, measured=json.load(open(res))["measured"] if os.path.exists(res) else None)
