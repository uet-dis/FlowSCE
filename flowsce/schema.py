"""Feature schema, raw-space validator V_k and projection Π."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DATASETS, _np
from .data import Prep

# ══════════════════════════════ Schema & raw-space validator ══════════════════════════════
class Domain:
    """Feature schema + validator V_k + projection Π, all evaluated in RAW coordinates.
    Units/coupled relations are NOT guessed: `relations` is an explicit, initially empty list, and the
    report states which constraint classes were (not) checked."""
    def __init__(self, prep: Prep, df_train, ds):
        c = DATASETS[ds]; raw = prep.raw_frame(df_train); self.ds = ds
        self.names = prep.cols; Fd = len(self.names)
        oh = np.array([any(n.startswith(k + "=") for k in c["cat"]) for n in self.names])
        self.mutable = np.array([not (n in c["immutable"] or o) for n, o in zip(self.names, oh)])
        self.is_log = np.array([n in prep.log for n in self.names])
        self.mu = prep.mu.values.astype(np.float64); self.sd = prep.sd.values.astype(np.float64)
        self.lo = raw.min().values; self.hi = raw.max().values
        self.integer = np.array([np.allclose(raw[n].values, np.round(raw[n].values)) for n in self.names])
        self.nonneg = self.lo >= 0
        self.relations = []          # e.g. dict(kind='rate', rate=..., bytes=..., dur=..., factor=8) — only if documented
        self.mask = torch.tensor(self.mutable, dtype=torch.float32)
    def schema(self):
        from .units import unit_of
        return [dict(name=n, mutable=bool(m), integer=bool(i), nonneg=bool(z), lo=float(l), hi=float(h),
                     transform="log1p+standardize" if g else "standardize", unit=unit_of(self.ds, n)[0], unit_note=unit_of(self.ds, n)[1])
                for n, m, i, z, l, h, g in zip(self.names, self.mutable, self.integer, self.nonneg, self.lo, self.hi, self.is_log)]
    def to_raw(self, z):
        x = np.asarray(z, np.float64) * self.sd + self.mu
        x[..., self.is_log] = np.expm1(x[..., self.is_log]); return x
    def from_raw(self, x):
        y = np.array(x, np.float64, copy=True); y[..., self.is_log] = np.log1p(np.maximum(y[..., self.is_log], 0))
        return (y - self.mu) / self.sd
    def project(self, a, delta):
        """Π: clip to train range, round integer fields, freeze immutable; unchanged coords stay exactly 0."""
        a = _np(a); d = _np(delta).astype(np.float64); ch = (d != 0) & self.mutable
        if not ch.any(): return np.zeros_like(d, dtype=np.float32)
        raw = self.to_raw(a + d); r = raw.copy()
        r[ch] = np.clip(r[ch], self.lo[ch], self.hi[ch])
        ints = ch & self.integer; r[ints] = np.round(r[ints])
        out = np.zeros_like(d); out[ch] = (self.from_raw(r) - a.astype(np.float64))[ch]
        return out.astype(np.float32)
    def check(self, a, delta, tol=1e-4):
        """Returns (ok, reasons). Only CHANGED coordinates are tested: the validator certifies the intervention."""
        a = _np(a); d = _np(delta).astype(np.float64); reasons = []
        if not np.all(np.isfinite(d)): return False, ["nonfinite"]
        if np.any((d != 0) & ~self.mutable): reasons.append("immutable_changed")
        ch = (d != 0) & self.mutable
        if ch.any():
            r = self.to_raw(a + d)[ch]; lo, hi = self.lo[ch], self.hi[ch]; span = np.maximum(hi - lo, 1.0)
            if np.any(r < lo - tol * span) or np.any(r > hi + tol * span): reasons.append("out_of_range")
            ints = self.integer[ch]
            ri = r[ints]
            if np.any(np.abs(ri - np.round(ri)) > np.maximum(1e-3, 1e-5 * np.abs(ri))): reasons.append("non_integer")   # float32 round-trip tolerance
        for rel in self.relations: reasons += rel_check(rel, self, a, d)
        return len(reasons) == 0, reasons

def rel_check(rel, dom, a, d):
    return []   # hook for documented coupled relations (none declared for these parquet releases)
