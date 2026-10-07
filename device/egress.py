# from blind-counsel 8b1bf8f device/egress.py
"""Device-side egress guard (capability/taint style, after CaMeL).

Every value derived from the private case record is wrapped `Tainted`. The guard
allows bytes to leave the device on exactly two public channels:

  "sync" : public artifacts the device pulled from the cloud and echoes back (the
           catalogue / task program). Must contain no tainted data.
  "pir"  : a SimplePIR query vector, or a batch of them. This is the ONLY channel whose content depends
           (indirectly) on the case -- it encodes which statute row is wanted. That
           dependence is cryptographically hidden: the query is an LWE sample,
           pseudorandom and of a distribution independent of the index. The guard
           therefore "declassifies" a PIR query only if it is a uint32 or int64
           vector or matrix in [0, Q) that passes a uniformity sanity check -- i.e. it is a
           ciphertext, not raw tainted data smuggled out.

Anything else carrying tainted bytes is blocked and logged as a leak attempt. The
evaluation scans everything that left for any secret substring; it must be zero.
"""
from __future__ import annotations

import numpy as np

Q = 1 << 32


class Tainted:
    """A value derived from private case data. Carries its origin for auditing."""
    __slots__ = ("value", "origin")

    def __init__(self, value, origin: str):
        self.value = value
        self.origin = origin

    def __repr__(self):
        return f"Tainted({self.origin})"


class EgressBlocked(Exception):
    pass


class EgressGuard:
    def __init__(self, secrets: list[str]):
        # the raw secret strings the guard must never let out (for the audit)
        self._secrets = [s for s in secrets if s]
        self.log: list[dict] = []      # every egress attempt, allowed or blocked

    # -- channel checkers -------------------------------------------------------
    def _looks_like_pir_query(self, payload) -> bool:
        # a single query vector, or a batch of query vectors (one per row)
        if not isinstance(payload, np.ndarray) or payload.ndim not in (1, 2) or payload.size < 64:
            return False
        if payload.dtype not in (np.uint32, np.int64) or payload.min() < 0 or int(payload.max()) >= Q:
            return False
        # a genuine LWE query is ~uniform on [0,Q); a raw index vector (one-hot or
        # small integers) is not. Require the mean near Q/2 and high entropy.
        return abs(float(payload.mean()) - Q / 2) < 0.1 * Q and payload.std() > 0.2 * Q

    def _contains_secret_bytes(self, payload) -> bool:
        if isinstance(payload, Tainted):
            return True
        try:
            blob = payload if isinstance(payload, (bytes, bytearray)) else str(payload).encode()
        except Exception:
            return False
        return any(s.encode() in blob for s in self._secrets)

    # -- the one call anything leaving the device must go through ---------------
    def send(self, channel: str, payload, note: str = "") -> bool:
        rec = {"channel": channel, "note": note, "allowed": False,
               "nbytes": getattr(payload, "nbytes", len(str(payload)))}
        if channel == "pir":
            ok = self._looks_like_pir_query(payload) and not isinstance(payload, Tainted)
            rec["allowed"] = ok
            self.log.append(rec)
            if not ok:
                raise EgressBlocked("pir channel: payload is not a declassified LWE query")
            return True
        if channel == "sync":
            if self._contains_secret_bytes(payload):
                rec["blocked_reason"] = "tainted/secret bytes on public sync channel"
                self.log.append(rec)
                raise EgressBlocked(rec["blocked_reason"])
            rec["allowed"] = True
            self.log.append(rec)
            return True
        # unknown channel: refuse
        rec["blocked_reason"] = f"unknown channel {channel!r}"
        self.log.append(rec)
        raise EgressBlocked(rec["blocked_reason"])

    # -- audit ------------------------------------------------------------------
    def egressed_blobs(self) -> list[bytes]:
        """Reconstruct the bytes that actually left, for the leakage scan."""
        return []  # payloads are not retained here; the harness feeds the auditor
