"""Policy-consistent query oracle f̄_{π,k} (modes A / B / C) with forward/backward counters."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .graph import window_subgraph

# ══════════════════════════════ Policy-consistent oracle ══════════════════════════════
class QueryOracle:
    """f̄_{π,k}: modifies only flow k, reconstructs host summaries by the exact affine update (Eq. exact),
    evaluates the frozen detector and returns q_k (Eq. score). Modes:
      'A' reconstructed forward + total derivative      'B' stale summaries (diagnostic negative control)
      'C' reconstructed values, detached context derivative
    Counts forward / backward evaluations."""
    def __init__(self, model, g, k, tau=0.5):
        self.m, self.k = model, int(k); self.g, kk = window_subgraph(g, k); self.kk = kk
        u, v = self.g["src"][kk], self.g["dst"][kk]
        self.idx = torch.stack([u, v]); self.wv = torch.stack([1 / self.g["deg"][u], 1 / self.g["deg"][v]])
        wfull = torch.zeros(self.g["N"], device=u.device, dtype=self.wv.dtype).index_add(0, self.idx, self.wv)
        self.omega = float(wfull.norm()); self.deg_u = float(self.g["deg"][u]); self.deg_v = float(self.g["deg"][v])
        self.thr = math.log(tau / (1 - tau)); self.e = torch.tensor([kk], device=u.device)
        self.nf = 0; self.nb = 0; self.uses_context = model.kind != "edge"
    def xv(self, delta, mode):
        if mode == "B" or not self.uses_context: return self.g["xv"]
        d = delta if mode == "A" else delta.detach()
        return self.g["xv"].index_add(0, self.idx, self.wv[:, None].to(d.dtype) * d[None, :])
    def q(self, delta, mode="A"):
        g = self.g; xe = g["xe"].index_add(0, self.e, delta[None, :].to(g["xe"].dtype))
        lg = self.m(xe, self.xv(delta, mode), g["src"], g["dst"], self.e)[0]; self.nf += 1
        return lg[1] - lg[0] - self.thr
    @torch.no_grad()
    def q0(self, delta, mode="A"): return float(self.q(torch.as_tensor(delta, device=self.e.device, dtype=torch.float32), mode))
    @property
    def a(self): return self.g["xe"][self.kk].detach()
