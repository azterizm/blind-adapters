"""M4 on Modal: the cloud role for the adapter library.

  bench_numpy  -- this repo's numpy server (pir/simplepir.py, unoptimized) answering one
                  batch of c queries over a random library with the exact layout of
                  (N, c, S). Labelled unoptimized in the report.
  bench_go     -- the reference SimplePIR Go code (ahenzinger/simplepir e9020b0) on a
                  random DB of the same dimensions: one query over the whole DB, single
                  thread; c queries are reported as c times that.
  serve        -- the real library (2 trained adapters + fillers) for the laptop-to-Modal
                  network test: hint download, query upload, answer download, wall time.

    modal run pir/modal_app.py::sweep              # numpy + Go, writes results/pir_sweep.json
    modal run pir/modal_app.py::network --n 16     # laptop <-> Modal fetch of the real adapters
"""

import json
import os
import time

import modal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOREF = "e9020b03bf2872c75b8954e749e32408b5db87ed"
NS = (2, 16, 128, 1024)
CS = (16, 64, 256, 1024)

app = modal.App("blind-adapters-pir")
vol = modal.Volume.from_name("blind-adapters-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "golang", "build-essential")
         .pip_install("numpy==2.4.2")
         .run_commands(f"git clone https://github.com/ahenzinger/simplepir /opt/simplepir && "
                       f"cd /opt/simplepir && git checkout {GOREF}")
         .add_local_file(os.path.join(ROOT, "pir", "goref", "main.go"), "/opt/goref/main.go", copy=True)
         .run_commands("cd /opt/goref && printf 'module goref\\n\\ngo 1.18\\n\\nrequire github.com/ahenzinger/simplepir v0.0.0\\n\\n"
                       "replace github.com/ahenzinger/simplepir => /opt/simplepir\\n' > go.mod && go build -o /opt/bench .")
         .add_local_python_source("pir"))


def _cpu_info() -> dict:
    return dict(cpus=os.cpu_count(), model=next((l.split(":")[1].strip() for l in open("/proc/cpuinfo")
                                                 if l.startswith("model name")), "?"))


@app.function(image=image, cpu=32, memory=128 * 1024, timeout=4 * 3600)
def bench_numpy(N: int, S: int, cs: list[int]) -> dict:
    import numpy as np

    from pir.adapters import layout
    from pir.simplepir import mul_db
    rng = np.random.default_rng(N)
    lays = {c: layout(N, S, c) for c in cs}
    need = max(l["m"] * l["L"] for l in lays.values())
    t = time.time()
    flat = rng.integers(0, 256, size=need, dtype=np.uint8)
    gen = time.time() - t
    out = {}
    for c, lay in lays.items():
        D = flat[: lay["m"] * lay["L"]].reshape(lay["m"], lay["L"])
        Qu = rng.integers(0, 1 << 32, size=(c, lay["m"]), dtype=np.uint32)
        t = time.time()
        mul_db(Qu, D, 128)
        out[f"{N},{c}"] = dict(server_numpy_s=time.time() - t)
        print(f"numpy N={N} c={c} m={lay['m']} L={lay['L']}: {out[f'{N},{c}']['server_numpy_s']:.2f}s", flush=True)
    return dict(results=out, db_gen_s=gen, cpu=_cpu_info())


@app.function(image=image, cpu=2, memory=96 * 1024, timeout=4 * 3600)
def bench_go(N: int, S: int, c: int) -> dict:
    import subprocess

    from pir.adapters import layout
    lay = layout(N, S, c)
    r = subprocess.run(["/opt/bench", str(lay["L"]), str(lay["m"]), "3"], capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    if line is None:
        return dict(key=f"{N},{c}", error=(r.stdout + r.stderr)[-500:])
    kv = dict(x.split("=") for x in line.split()[1:])
    one = float(kv["seconds"])
    return dict(key=f"{N},{c}", server_go_one_query_s=one, server_go_s=c * one, go_p=int(kv["p"]),
                cpu=_cpu_info())


@app.local_entrypoint()
def sweep(size: int = 0, go_max_n: int = 128):
    """size: record bytes S (default: the largest delivery record in the manifest)."""
    if not size:
        size = max(e["size"] for e in json.load(open(os.path.join(ROOT, "artifacts", "library", "manifest.json")))["records"])
    measured, meta = {}, {}
    go_jobs = [(N, size, c) for N in NS for c in CS if N <= go_max_n]
    go_calls = [bench_go.spawn(*j) for j in go_jobs]
    for N, r in zip(NS, bench_numpy.starmap([(N, size, list(CS)) for N in NS])):
        for k, v in r["results"].items():
            measured.setdefault(k, {}).update(v)
        meta[f"numpy_N{N}"] = dict(db_gen_s=r["db_gen_s"], cpu=r["cpu"])
    for call in go_calls:
        r = call.get()
        if "error" in r:
            print("go error", r)
            continue
        measured.setdefault(r["key"], {}).update(
            server_go_s=r["server_go_s"], server_go_one_query_s=r["server_go_one_query_s"], go_p=r["go_p"])
        meta["go_cpu"] = r["cpu"]
    path = os.path.join(ROOT, "results", "pir_sweep.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    for k, v in old.get("measured", {}).items():          # keep laptop client timings
        measured.setdefault(k, {}).update({x: y for x, y in v.items() if x.startswith("client")})
    json.dump(dict(S=size, measured=measured, meta=meta), open(path, "w"), indent=1)
    print(json.dumps(measured, indent=1))


# ------------------------------------------------------------------ network test
@app.cls(image=image, cpu=16, memory=32 * 1024, volumes={"/vol": vol}, timeout=3600, scaledown_window=600)
class Library:
    n: int = modal.parameter(default=16)
    c: int = modal.parameter(default=256)

    @modal.enter()
    def load(self):
        import numpy as np

        from pir.adapters import AdapterCloud, build
        vol.reload()
        man = json.load(open("/vol/library/manifest.json"))
        recs = [(e["name"], open(f"/vol/library/{e['file']}", "rb").read()) for e in man["records"]]
        D, self.manifest, self.lay = build(recs, self.n, self.c)
        self.cloud = AdapterCloud(D, self.lay)
        t = time.time()
        self.H = self.cloud.hint()
        self.hint_s = time.time() - t
        np.save("/tmp/hint.npy", self.H)

    @modal.method()
    def public(self) -> dict:
        return dict(manifest=self.manifest, layout=self.lay, seed=self.cloud.seed, hint_compute_s=self.hint_s)

    @modal.method()
    def hint(self) -> bytes:
        return open("/tmp/hint.npy", "rb").read()

    @modal.method()
    def answer(self, Qu_bytes: bytes) -> tuple[bytes, float]:
        import io

        import numpy as np
        Qu = np.load(io.BytesIO(Qu_bytes))
        t = time.time()
        ans = self.cloud.answer(Qu)
        s = time.time() - t
        buf = io.BytesIO()
        np.save(buf, ans)
        return buf.getvalue(), s


@app.local_entrypoint()
def network(n: int = 16, c: int = 256, fetches: int = 2):
    """Laptop <-> Modal fetch of each real adapter, byte-exact and sha256-checked."""
    import io

    import numpy as np

    from pir.adapters import AdapterFetcher, IntegrityError

    man = json.load(open(os.path.join(ROOT, "artifacts", "library", "manifest.json")))
    with vol.batch_upload(force=True) as b:
        b.put_file(os.path.join(ROOT, "artifacts", "library", "manifest.json"), "/library/manifest.json")
        for e in man["records"]:
            b.put_file(os.path.join(ROOT, "artifacts", "library", e["file"]), f"/library/{e['file']}")
    lib = Library(n=n, c=c)
    t = time.time()
    pub = lib.public.remote()
    setup_s = time.time() - t
    t = time.time()
    hb = lib.hint.remote()
    hint_s = time.time() - t
    H = np.load(io.BytesIO(hb))
    f = AdapterFetcher(pub["seed"], H, pub["layout"], pub["manifest"])

    class Remote:
        def __init__(self):
            self.up = self.down = 0
            self.server_s = 0.0

        def answer(self, Qu):
            buf = io.BytesIO()
            np.save(buf, Qu)
            self.up += len(buf.getvalue())
            raw, s = lib.answer.remote(buf.getvalue())
            self.down += len(raw)
            self.server_s += s
            return np.load(io.BytesIO(raw))

    runs = []
    for k in range(fetches):
        for e in man["records"]:
            r = Remote()
            t = time.time()
            try:
                raw = f.fetch(e["index"], r)
                ok = raw == open(os.path.join(ROOT, "artifacts", "library", e["file"]), "rb").read()
            except IntegrityError:
                ok = False
            runs.append(dict(name=e["name"], byte_exact_and_sha=ok, wall_s=time.time() - t,
                             upload=r.up, download=r.down, server_s=r.server_s))
            print(runs[-1], flush=True)
    res = dict(N=n, c=c, layout=pub["layout"], hint_bytes=len(hb), hint_download_s=hint_s,
               hint_compute_s=pub["hint_compute_s"], cold_start_and_setup_s=setup_s, fetches=runs)
    json.dump(res, open(os.path.join(ROOT, "results", f"pir_network_N{n}_c{c}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "fetches"}, indent=1))
