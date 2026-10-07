# from blind-counsel 8b1bf8f cloud/precompute.py
"""Cloud-side Lane B: precompute the PUBLIC statute database the device fetches
from via PIR.

The cloud only ever touches public law (legislation.gov.uk, OGL-3.0) from the
legal-rag-router corpus. For each section coordinate it renders the section's full
text (the subtree, in document order), records a sha256 for byte-exact referencing,
and packs everything into a fixed-cell byte matrix suitable for SimplePIR. The
cloud never sees any case data.

Output (all public), written to artifacts/cloud/:
  pir_db.npy        uint8 [N, L]  the PIR matrix (statute text, one chunk per row)
  catalogue.json    coordinate -> {row, n_rows, title, sha256, nbytes}
  meta.json         {N, L, instrument_ids, built}
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np

from config import LRR
ART = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "cloud"))
CELL = 2048                      # bytes per PIR row
DEMO_INSTRUMENTS = ["uk_ukpga_1996_18"]   # Employment Rights Act 1996


def _instrument_path(instrument_id: str) -> str:
    # uk_ukpga_1996_18 -> data/uk/ukpga/1996/uk_ukpga_1996_18.jsonl
    parts = instrument_id.split("_")
    series, year = parts[1], parts[2]
    return os.path.join(LRR, "data", "uk", series, year, f"{instrument_id}.jsonl")


def render_sections(instrument_id: str) -> dict[str, str]:
    """Return {section_coordinate: rendered_text} for every top-level section."""
    path = _instrument_path(instrument_id)
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    provs = [r for r in recs if r.get("record_type") == "provision"]
    provs.sort(key=lambda r: r.get("order", 0))

    # group by section: coordinate .../sNN  (depth-1 provision). key by its prefix.
    def section_key(coord: str) -> str | None:
        # uk/ukpga/1996/18/s98/2/a -> uk/ukpga/1996/18/s98
        tail = coord.split("/")[4:]          # after uk/ukpga/1996/18
        if not tail:
            return None
        return "/".join(coord.split("/")[:5])  # keep through the first provision seg

    sections: dict[str, list] = {}
    for r in provs:
        k = section_key(r["coordinate"])
        if k:
            sections.setdefault(k, []).append(r)

    out = {}
    for k, rs in sections.items():
        rs.sort(key=lambda r: r.get("order", 0))
        lines = []
        base_depth = k.count("/")
        # text_after belongs AFTER a provision's children, so hold it on a stack
        # and emit it when the walk leaves that provision's subtree.
        stack: list[tuple[str, str, str]] = []

        def close_until(coord: str) -> None:
            while stack and not coord.startswith(stack[-1][0] + "/"):
                _, ind, aft = stack.pop()
                if aft:
                    lines.append(f"{ind}{aft}")

        for r in rs:
            coord = r["coordinate"]
            close_until(coord)
            depth = coord.count("/") - base_depth
            label = (r.get("number_label") or "").strip()
            text = (r.get("text") or "").strip()
            after = (r.get("text_after") or "").strip()
            indent = "  " * depth
            lab = f"({label}) " if (label and depth > 0) else ""
            if text:
                lines.append(f"{indent}{lab}{text}")
            stack.append((coord, indent, after))
        close_until("")
        rendered = "\n".join(lines).strip()
        if rendered:
            out[k] = rendered
    return out


def build(instruments=DEMO_INSTRUMENTS):
    os.makedirs(ART, exist_ok=True)
    catalogue: dict[str, dict] = {}
    rows: list[bytes] = []
    for iid in instruments:
        secs = render_sections(iid)
        for coord, text in sorted(secs.items()):
            raw = text.encode("utf-8")
            sha = hashlib.sha256(raw).hexdigest()
            n_rows = max(1, (len(raw) + CELL - 1) // CELL)
            catalogue[coord] = dict(row=len(rows), n_rows=n_rows, nbytes=len(raw),
                                    sha256=sha, title=coord)
            for c in range(n_rows):
                chunk = raw[c * CELL:(c + 1) * CELL].ljust(CELL, b"\x00")
                rows.append(chunk)
    N = len(rows)
    D = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(N, CELL)
    np.save(os.path.join(ART, "pir_db.npy"), D)
    json.dump(catalogue, open(os.path.join(ART, "catalogue.json"), "w"), indent=0)
    json.dump(dict(N=N, L=CELL, instruments=instruments, n_sections=len(catalogue)),
              open(os.path.join(ART, "meta.json"), "w"), indent=2)
    print(f"built PIR DB: {N} rows x {CELL} bytes  ({D.nbytes/1e6:.1f} MB), "
          f"{len(catalogue)} sections from {instruments}")
    # show a sample
    sample = next(iter(catalogue.items()))
    print(f"sample section: {sample[0]}  ({sample[1]['nbytes']} bytes, sha {sample[1]['sha256'][:12]})")
    return D, catalogue


if __name__ == "__main__":
    build()
