"""Optimisation coordinates ξ (scale, consistent Mahalanobis metric) and the domain box."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE
from .schema import Domain

# ══════════════════════════════ FlowSCE: search & restoration ══════════════════════════════
class Space:
    """ξ-coordinates: δ = ξ ⊙ s ⊙ m; Q_ξ = S Σ_a^{-1} S on mutable coordinates (consistent metric)."""
    def __init__(self, Xtr, dom: Domain, ridge=1e-3, s_pct=95):
        sd = Xtr.std(0); self.s = torch.tensor(np.minimum(np.maximum(sd, 1e-6), np.percentile(sd, s_pct)), dtype=torch.float32, device=DEVICE)
        mi = dom.mutable; C = np.cov(Xtr[:, mi].T) + ridge * np.eye(int(mi.sum()))
        P = np.zeros((len(mi), len(mi))); P[np.ix_(mi, mi)] = np.linalg.inv(C)        # precision in processed coords
        self.Pa = torch.tensor(P, dtype=torch.float32, device=DEVICE)
        S = np.diag(self.s.cpu().numpy()); self.Qxi = torch.tensor(S @ P @ S, dtype=torch.float32, device=DEVICE)
        self.m = dom.mask.to(DEVICE); self.F = len(mi)
    def mahal(self, d):
        d = torch.as_tensor(d, device=DEVICE, dtype=torch.float32); return float(torch.sqrt(torch.clamp(d @ self.Pa @ d, min=0)))

def xi_box(dom: Domain, sp: Space, a):
    """Bounds on ξ such that a+δ stays inside the train box (processed coordinates); staying at the original
    value is always admissible, so a test flow already outside the box is never forced to move."""
    lo = np.where(np.isfinite(dom.lo), dom.lo, 0.0); hi = np.where(np.isfinite(dom.hi), dom.hi, 0.0)
    lo_p = np.where(np.isfinite(dom.lo), dom.from_raw(lo), -np.inf); hi_p = np.where(np.isfinite(dom.hi), dom.from_raw(hi), np.inf)
    s = sp.s.detach().cpu().numpy().astype(np.float64); a = np.asarray(a, np.float64)
    XLO = np.minimum(lo_p - a, 0) / s; XHI = np.maximum(hi_p - a, 0) / s
    return torch.tensor(XLO, dtype=torch.float32, device=DEVICE), torch.tensor(XHI, dtype=torch.float32, device=DEVICE)

def subset_space(sp: Space, dom: Domain, keep_idx):
    """Copies with a reduced mutable set (runtime scaling over #editable features)."""
    sp2, dom2 = copy.copy(sp), copy.copy(dom)
    mut = np.zeros_like(dom.mutable); mut[keep_idx] = True
    dom2.mutable = mut; dom2.mask = torch.tensor(mut, dtype=torch.float32); sp2.m = dom2.mask.to(DEVICE)
    return sp2, dom2
