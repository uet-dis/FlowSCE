"""Counterfactual assessment (nominal vs reconstructed), ledger rows and rate/quality summaries."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE

# ══════════════════════════════ Counterfactual metrics & ledger ══════════════════════════════
class BenignPool:
    """q nearest TRAINING-benign neighbours (L1, processed space, deterministic tie-break by row id)."""
    def __init__(self, Xtr, ytr, rows, q=5):
        b = ytr == 0; order = np.argsort(rows[b], kind="stable")
        self.X = torch.tensor(Xtr[b][order], device=DEVICE); self.q = q
    def dist(self, x):
        d = torch.cdist(torch.as_tensor(x, device=DEVICE, dtype=torch.float32)[None], self.X, p=1)[0]
        if len(d) < self.q: return float("nan")
        return float(torch.topk(d, self.q, largest=False).values.mean())
    def nearest(self, x):
        d = torch.cdist(torch.as_tensor(x, device=DEVICE, dtype=torch.float32)[None], self.X, p=1)[0]
        return self.X[int(torch.argmin(d))].cpu().numpy()

POOL: BenignPool | None = None   # training-benign kNN pool; set by the run scripts after loading a dataset

def assess(orc, dom, sp, d, mode, cfg):
    """Nominal (variant's own predictor) vs reconstructed (shared oracle) assessment."""
    if d is None:
        return dict(n=0, v=0, returned=0, status="failure", q_nominal=np.nan, q_reconstructed=np.nan, domain_ok=None,
                    domain_reasons="", support_exact=np.nan, support_eps=np.nan, l1=np.nan, l2=np.nan, mahal=np.nan,
                    knn_benign=np.nan, margin_logit=np.nan, margin_prob=np.nan, disc_stale=np.nan, delta=None)
    a = orc.a.cpu().numpy(); ok, why = dom.check(a, d)
    qn = orc.q0(d, mode); qr = orc.q0(d, "A"); qs = orc.q0(d, "B")
    n = int(qn < 0 and ok); v = int(qr < 0 and ok)
    pr = 1 / (1 + math.exp(-(qr + orc.thr)))
    status = "valid" if v else ("spurious" if n else "invalid")
    return dict(n=n, v=v, returned=1, status=status, q_nominal=qn, q_reconstructed=qr, domain_ok=ok,
                domain_reasons=";".join(why), support_exact=int((d != 0).sum()), support_eps=int((np.abs(d) > cfg.eps_support).sum()),
                l1=float(np.abs(d).sum()), l2=float(np.linalg.norm(d)), mahal=sp.mahal(d),
                knn_benign=POOL.dist(a + d) if POOL is not None else np.nan,
                margin_logit=-qr, margin_prob=cfg.tau - pr, disc_stale=abs(qr - qs), delta=d.tolist())

def rates(df):
    """CSR / SVR / PRF_all with explicit denominators; zero denominators → None (never 0)."""
    Q = len(df); nn_ = int(df.n.sum()); nv = int(df.v.sum()); spur = int(((df.n == 1) & (df.v == 0)).sum())
    return dict(attempted=Q, returned=int(df.returned.sum()), nominal=nn_, verified=nv, spurious=spur,
                CSR=nv / Q if Q else None, SVR=spur / nn_ if nn_ else None, PRF_all=spur / Q if Q else None)

def quality(df):
    s = df[df.v == 1]
    out = dict(n_verified=len(s))
    for c in ["support_exact", "support_eps", "l1", "l2", "mahal", "knn_benign", "margin_logit", "margin_prob", "n_fwd", "n_bwd", "updates"]:
        if c in s and len(s): out[c + "_median"] = float(s[c].median()); out[c + "_mean"] = float(s[c].mean())
    return out

def summarize(df, keys):
    rows = []
    for k, sub in df.groupby(keys):
        r = dict(zip(keys, k if isinstance(k, tuple) else (k,))); r.update(rates(sub)); r.update(quality(sub)); rows.append(r)
    return pd.DataFrame(rows)
