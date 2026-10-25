"""RQ1 checks on real graphs: reconstruction vs rebuild, total gradient vs FD / chain rule, A–C alignment."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE
from .graph import host_means
from .oracle import QueryOracle

# ══════════════════════════════ RQ1 checks on real graphs ══════════════════════════════
def check_reconstruction(g, ks, rng, scale=1.0):
    """Fast rank-one update vs independent rebuild from X'_E (float64)."""
    gd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in g.items()}
    gd["xv"] = host_means(gd["xe"], gd["src"], gd["dst"], gd["N"], gd["deg"])   # original state in float64
    worst = 0.0
    for k in ks:
        d = torch.tensor(rng.normal(0, scale, gd["xe"].shape[1]), dtype=torch.float64, device=DEVICE)
        xe = gd["xe"].clone(); xe[k] += d
        u, v = gd["src"][k], gd["dst"][k]
        fast = gd["xv"].index_add(0, torch.stack([u, v]), torch.stack([1 / gd["deg"][u], 1 / gd["deg"][v]])[:, None] * d[None])
        slow = host_means(xe, gd["src"], gd["dst"], gd["N"], gd["deg"])
        worst = max(worst, (fast - slow).abs().max().item())
    return worst

def check_gradients(model, g, ks, sp, rng, h=1e-6):
    """Total gradient (autograd through reconstruction) vs central finite differences and vs explicit chain rule
    ∇a q + Σ_v w_v ∇x_v q (Eq. gradient), in float64 with dropout off. Evaluated at a random nonzero δ0."""
    m64 = copy.deepcopy(model).double().eval()
    g64 = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in g.items()}
    out = []
    for k in ks:
        orc = QueryOracle(m64, g64, k); mut = sp.m.double()
        d0 = torch.tensor(rng.normal(0, .1, sp.F), dtype=torch.float64, device=DEVICE) * mut
        x = d0.clone().requires_grad_(True); gA = torch.autograd.grad(orc.q(x, "A"), x)[0] * mut
        fd = torch.zeros_like(gA)
        for j in torch.nonzero(mut).squeeze(1).tolist():
            e = torch.zeros_like(d0); e[j] = h
            with torch.no_grad(): fd[j] = (orc.q(d0 + e, "A") - orc.q(d0 - e, "A")) / (2 * h)
        # explicit chain rule with independent leaf tensors
        gs = orc.g; kk = orc.kk
        xe = gs["xe"].clone(); xe[kk] += d0; xe = xe.detach().requires_grad_(True)
        xv = orc.xv(d0, "A").detach().requires_grad_(True)
        lg = m64(xe, xv, gs["src"], gs["dst"], orc.e)[0]; qq = lg[1] - lg[0] - orc.thr
        ga, gv = torch.autograd.grad(qq, [xe, xv])
        chain = (ga[kk] + (orc.wv[:, None] * gv[orc.idx]).sum(0)) * mut
        out.append(dict(query=int(k), fd_max_abs=float((gA - fd).abs().max()), chain_max_abs=float((gA - chain).abs().max()),
                        grad_norm=float(gA.norm()), ctx_share=float(((orc.wv[:, None] * gv[orc.idx]).sum(0) * mut).norm() / (gA.norm() + 1e-30))))
    return pd.DataFrame(out)

def grad_alignment(model, g, k, sp):
    """cos(∇_A, ∇_C) at δ=0: how much the detached context derivative changes the search direction."""
    orc = QueryOracle(model, g, k); z = torch.zeros(sp.F, device=DEVICE, requires_grad=True)
    gA = torch.autograd.grad(orc.q(z * sp.m, "A"), z, allow_unused=True)[0]
    z = torch.zeros(sp.F, device=DEVICE, requires_grad=True)
    gC = torch.autograd.grad(orc.q(z * sp.m, "C"), z, allow_unused=True)[0]
    gA = torch.zeros(sp.F, device=DEVICE) if gA is None else gA; gC = torch.zeros(sp.F, device=DEVICE) if gC is None else gC
    return dict(cos_AC=float(F.cosine_similarity(gA, gC, 0, eps=1e-30)), rel_diff_AC=float((gA - gC).norm() / (gA.norm() + 1e-30)))


# ══════════════════════════════ Naive self-loop update rule (diagnostic) ══════════════════════════════
def legacy_update(xv, u, v, w_u, w_v, delta):
    """Naive host-summary update (an earlier prototype): x[src] = x0[src] + δ/cnt_src, then x[dst] = x0[dst] + δ/cnt_dst.
    The second assignment overwrites the first when src = dst, so a self-loop receives δ/cnt instead of 2δ/cnt."""
    out = xv.clone(); out[u] = xv[u] + w_u * delta; out[v] = xv[v] + w_v * delta
    return out

def legacy_update_check(model, g, ks, deltas):
    """For flows ks with perturbations deltas (rows), compare the legacy update with an independent rebuild:
    max |X_V(legacy) - X_V(rebuild)| over hosts, and the score difference q(legacy) - q(exact) of the edited flow."""
    gd = {k: (v.double() if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in g.items()}
    xv0 = host_means(gd["xe"], gd["src"], gd["dst"], gd["N"], gd["deg"]); thr = 0.0; rows = []
    for k, d in zip(ks, deltas):
        d = torch.as_tensor(d, dtype=torch.float64, device=DEVICE); u, v = gd["src"][k], gd["dst"][k]
        xe = gd["xe"].clone(); xe[k] += d
        rebuild = host_means(xe, gd["src"], gd["dst"], gd["N"], gd["deg"])
        leg = legacy_update(xv0, u, v, 1 / gd["deg"][u], 1 / gd["deg"][v], d)
        with torch.no_grad():
            e = torch.tensor([int(k)], device=DEVICE)
            dt = next(model.parameters()).dtype
            lq = lambda X: model(xe.to(dt), X.to(dt), g["src"], g["dst"], e)[0]
            q_ex, q_lg = lq(rebuild), lq(leg); q_ex = float(q_ex[1] - q_ex[0]); q_lg = float(q_lg[1] - q_lg[0])
        rows.append(dict(flow=int(k), self_loop=bool(u == v), xv_max_abs=float((leg - rebuild).abs().max()),
                         q_exact=q_ex, q_legacy=q_lg, dq=q_lg - q_ex, label_flip=bool((q_ex < thr) != (q_lg < thr))))
    return pd.DataFrame(rows)
