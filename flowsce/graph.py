"""Graph-construction policy: snapshot / windowed host means and contract checks."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DATASETS, DEVICE

# ══════════════════════════════ Graph-construction policy ══════════════════════════════
def host_means(xe, src, dst, N, deg):
    """X_V = D^{-1} B X_E (self-loop counts twice)."""
    s = torch.zeros(N, xe.shape[1], dtype=xe.dtype, device=xe.device).index_add(0, src, xe).index_add(0, dst, xe)
    return s / deg.clamp(min=1).unsqueeze(1).to(xe.dtype)

def build_graph(dfp, X, ds, policy="full", device=DEVICE):
    """policy='full': one snapshot for the partition. policy=('win', W): one snapshot per W consecutive
    flows (time order). policy=('time', dt): one snapshot per wall-clock interval of dt seconds (tumbling windows
    aligned to the epoch, as a streaming detector would emit them). Windows are disjoint graphs.
    Parallel flows are kept; labels are not used."""
    c = DATASETS[ds]; E = len(dfp)
    order = np.argsort(dfp["_t"].values, kind="stable"); win = np.zeros(E, int)
    if policy != "full" and policy[0] == "win": win[order] = np.arange(E) // policy[1]
    elif policy != "full":
        b = np.floor(dfp["_t"].values.astype(np.float64) * c["tunit"] / policy[1]).astype(np.int64)
        win = pd.factorize(b, sort=True)[0]
    w = pd.Series(win).astype(str).values
    codes, uniq = pd.factorize(np.concatenate([w + "|" + dfp[c["src"]].values, w + "|" + dfp[c["dst"]].values]))
    src = torch.tensor(codes[:E], device=device); dst = torch.tensor(codes[E:], device=device); N = len(uniq)
    deg = torch.zeros(N, device=device).index_add_(0, src, torch.ones(E, device=device)).index_add_(0, dst, torch.ones(E, device=device))
    xe = torch.tensor(X, device=device)
    xv = host_means(xe.double(), src, dst, N, deg).float()   # accumulate in float64 → deterministic up to one rounding
    return dict(xe=xe, xv=xv, src=src, dst=dst, N=N, deg=deg,
                y=torch.tensor(dfp["_y"].values, device=device), row_id=dfp["row_id"].values,
                block=dfp["_block"].values if "_block" in dfp else None, win=torch.tensor(win, device=device),
                policy=policy if policy == "full" else f"{policy[0]}{policy[1]:g}")

def check_contract(g, rtol=1e-6):
    """Eq. (original-contract): supplied X_V must equal A_π(X_E). The reference is rebuilt in float64;
    the tolerance is relative to float32 storage (GPU index_add uses atomics, so float32 sums are not bitwise
    reproducible — this is rounding, not a policy mismatch)."""
    ref = host_means(g["xe"].double(), g["src"], g["dst"], g["N"], g["deg"])
    diff = (g["xv"].double() - ref).abs()
    rel = (diff / ref.abs().clamp(min=1.0)).max().item()
    return dict(max_abs_residual=diff.max().item(), max_rel_residual=rel, rtol=rtol, ok=rel <= rtol)

def window_subgraph(g, k):
    """Exact restriction of a windowed policy to the query's window (windows are disjoint graphs)."""
    if g["policy"] == "full": return g, int(k)
    eid = (g["win"] == g["win"][k]).nonzero().squeeze(1)
    nodes, inv = torch.unique(torch.cat([g["src"][eid], g["dst"][eid]]), return_inverse=True)
    E = len(eid)
    sub = dict(xe=g["xe"][eid], xv=g["xv"][nodes], src=inv[:E], dst=inv[E:], N=len(nodes), deg=g["deg"][nodes],
               y=g["y"][eid], policy=g["policy"], win=torch.zeros(E, dtype=torch.long, device=eid.device))
    return sub, int((eid == k).nonzero())

def degree_stats(g):
    e = torch.cat([g["src"], g["dst"]]); d = g["deg"][e].float().cpu().numpy()
    w = torch.sqrt(1 / g["deg"][g["src"]] ** 2 + 1 / g["deg"][g["dst"]] ** 2).cpu().numpy()
    same = (g["src"] == g["dst"]).cpu().numpy(); w[same] = (2 / g["deg"][g["src"]][same]).cpu().numpy()
    return dict(nodes=g["N"], flows=len(g["src"]), deg_median=float(np.median(d)), deg_p10=float(np.percentile(d, 10)),
                omega_median=float(np.median(w)), omega_p90=float(np.percentile(w, 90)), self_loops=int(same.sum()))
