"""Counterfactual baselines through the same oracle: CANDICE-style, DiCE-style, LBS-adapted, minimizers."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from . import metrics as _M
from .config import DEVICE, ExplainCfg
from .oracle import QueryOracle
from .space import xi_box
from .metrics import assess
from .explain import restore

# ══════════════════════════════ Baselines (all through QueryOracle) ══════════════════════════════
def _final_row(orc, dom, sp, d, cfg, method, extra=None):
    row = dict(method=method, n_fwd_method=orc.nf); row.update(assess(orc, dom, sp, d, "A", cfg)); row["n_fwd"], row["n_bwd"] = orc.nf, orc.nb
    if extra: row.update(extra)
    return row

def candice_style(model, g, k, sp, dom, cfg: ExplainCfg, seed, n_cf=5, lam1=0.5, lam2=0.5, lr=0.05, iters=500, init_noise=0.05,
                  method="CANDICE-style"):
    """CANDICE objective (Li et al. 2025, Eq. 7): BCE validity + λ1·Mahalanobis + λ2·L1, mutable mask,
    train min–max ranges (projection Π), Adam, early stop at first valid; d=5 counterfactuals and
    change-frequency importance (Eq. 8–9). Adaptation: queried through the reconstructed oracle; the shared
    per-query forward budget is split across the 5 runs. Returned CF: valid one with minimum support."""
    orc = QueryOracle(model, g, k, cfg.tau); a = orc.a.cpu().numpy(); gen = torch.Generator(device="cpu").manual_seed(seed)
    cands = []; per = max(1, cfg.max_fwd // n_cf)
    if cfg.box_proj: XLO, XHI = xi_box(dom, sp, a)
    for r in range(n_cf):
        start = orc.nf
        xi = (torch.randn(sp.F, generator=gen) * init_noise).to(DEVICE) * sp.m; xi.requires_grad_(True)
        opt = torch.optim.Adam([xi], lr=lr)
        for t in range(iters + 1):
            d = xi * sp.s * sp.m; q = orc.q(d, "A")
            if q.item() < 0:
                dp = dom.project(a, d.detach())
                if dom.check(a, dp)[0] and orc.q0(dp) < 0: cands.append(dp); break
            if t == iters or orc.nf - start >= per: break
            p = torch.sigmoid(q + orc.thr)                       # attack probability
            loss = -torch.log(1 - p + 1e-12) + lam1 * torch.sqrt(xi @ sp.Qxi @ xi + 1e-8) + lam2 * xi.abs().sum()
            opt.zero_grad(); loss.backward(); orc.nb += 1; opt.step()
            with torch.no_grad():
                xi.mul_(sp.m)
                if cfg.box_proj: xi.copy_(torch.maximum(torch.minimum(xi, XHI), XLO))
    fi = (np.mean([(c != 0) for c in cands], 0) if cands else np.zeros(sp.F)).tolist()
    best = min(cands, key=lambda c: ((np.abs(c) > cfg.eps_support).sum(), np.abs(c).sum())) if cands else None
    return _final_row(orc, dom, sp, best, cfg, method, dict(n_valid_runs=len(cands), feature_importance=fi, all_cfs=[c.tolist() for c in cands]))

def dice_style(model, g, k, sp, dom, cfg: ExplainCfg, seed, K=4, lam_prox=0.5, lam_div=1.0, lr=0.05, iters=500,
               init_noise=0.1, posthoc=True):
    """Gradient DiCE (Mothilal et al. 2020): K CFs jointly, hinge validity + proximity (L1 in ξ units)
    − λ_div·det(K_dpp), K_ij = 1/(1+L1(c_i,c_j)); then DiCE post-hoc sparsity enhancement
    (revert features in ascending |δ| order, partial revert by bisection when a full revert breaks validity)."""
    orc = QueryOracle(model, g, k, cfg.tau); a = orc.a.cpu().numpy(); gen = torch.Generator(device="cpu").manual_seed(seed)
    XI = ((torch.randn(K, sp.F, generator=gen) * init_noise).to(DEVICE) * sp.m).requires_grad_(True)
    opt = torch.optim.Adam([XI], lr=lr); valid = {}
    if cfg.box_proj: XLO, XHI = xi_box(dom, sp, a)
    for t in range(iters + 1):
        D = XI * sp.s * sp.m; qs = torch.stack([orc.q(D[i], "A") for i in range(K)])
        for i in range(K):
            if i not in valid and qs[i].item() < 0:
                dp = dom.project(a, D[i].detach())
                if dom.check(a, dp)[0] and orc.q0(dp) < 0: valid[i] = dp
        if len(valid) == K or t == iters or orc.nf >= cfg.max_fwd: break
        dist = torch.cdist(XI, XI, p=1); Kd = 1 / (1 + dist) + 1e-4 * torch.eye(K, device=DEVICE)
        loss = F.relu(qs + 0.5).mean() + lam_prox * XI.abs().sum(1).mean() - lam_div * torch.det(Kd)
        opt.zero_grad(); loss.backward(); orc.nb += 1; opt.step()
        with torch.no_grad():
            XI.mul_(sp.m)
            if cfg.box_proj: XI.copy_(torch.maximum(torch.minimum(XI, XHI[None]), XLO[None]))
    cfs = list(valid.values()); pre = [c.copy() for c in cfs]
    if posthoc: cfs = [dice_posthoc(orc, dom, c, cfg) for c in cfs]
    div = float(np.mean([np.abs(x - y).sum() for i, x in enumerate(cfs) for y in cfs[i + 1:]])) if len(cfs) > 1 else np.nan
    best = min(cfs, key=lambda c: ((c != 0).sum(), np.abs(c).sum())) if cfs else None
    return _final_row(orc, dom, sp, best, cfg, "DiCE-style" + ("" if posthoc else "-noposthoc"),
                      dict(n_valid_cfs=len(cfs), diversity_l1=div, support_before_posthoc=float(np.mean([(c != 0).sum() for c in pre])) if pre else np.nan))

def dice_posthoc(orc, dom, d, cfg, bisect_steps=5):
    a = orc.a.cpu().numpy(); d = np.array(d, np.float32)
    for j in np.argsort(np.abs(d)):
        if d[j] == 0 or not dom.mutable[j] or orc.nf >= cfg.max_fwd: continue
        tr = d.copy(); tr[j] = 0
        if dom.check(a, tr)[0] and orc.q0(tr) < 0: d = tr; continue
        lo, hi = 0.0, 1.0                                    # fraction of δ_j kept
        for _ in range(bisect_steps):
            mid = (lo + hi) / 2; tr = d.copy(); tr[j] = d[j] * mid; tr = dom.project(a, tr)
            if dom.check(a, tr)[0] and orc.q0(tr) < 0: hi = mid
            else: lo = mid
        if hi < 1.0:
            tr = d.copy(); tr[j] = d[j] * hi; d = dom.project(a, tr)
    return d

def lbs_adapted(model, g, k, sp, dom, cfg: ExplainCfg, seed_delta, seed, T_del=None, T_swap=20, lam=1.0):
    """Local Bounded Search (García et al., Machine Learning 115:224, 2026), Algorithm 2, adapted to ONE flow's
    numerical attributes. Adaptation (recorded, not a reproduction of the graph-edit benchmark):
      • edit universe E(V)  → mutable coordinates of flow k;  solution S = set of changed coordinates;
      • structural candidate S' → coordinates allowed to differ from the original flow;
      • attribute strategies A (cheapest → richest, first accept) assign values to the coordinates in S':
          α1 keep incumbent values (new coordinates take the benign-neighbour difference),
          α2 incumbent values ×1.5, α3 ×2 (compensation for removed edits),
          α4 nearest training-benign neighbour values on S', α5 midpoint of α1 and α4;
      • every candidate is projected by Π, checked by V_k and evaluated by the reconstructed oracle;
      • a shared forward budget (cfg.max_fwd) replaces the per-instance oracle-call cap.
    Strategies 1a (single removal), 1b (shrinking-overshooting, Eq. 2), 2 (replacement), 3 (compensating addition)
    follow the published priority order and bound logic."""
    orc = QueryOracle(model, g, k, cfg.tau); a = orc.a.cpu().numpy(); rng = np.random.default_rng(seed)
    if seed_delta is None: return _final_row(orc, dom, sp, None, cfg, "LBS-adapted", dict(lbs_iters=0, lbs_moves=""))
    seed_d = np.array(seed_delta, np.float32)
    if not (dom.check(a, seed_d)[0] and orc.q0(seed_d) < 0):
        return _final_row(orc, dom, sp, None, cfg, "LBS-adapted", dict(lbs_iters=0, lbs_moves="invalid_seed"))
    nn_d = (_M.POOL.nearest(a) - a).astype(np.float32) if _M.POOL is not None else seed_d.copy()
    U = np.nonzero(dom.mutable)[0]
    Xhat = seed_d.copy(); Shat = set(np.nonzero(Xhat)[0].tolist()); b = len(Shat); moves = []
    def values(Sp, j_strategy):
        base = np.zeros_like(Xhat)
        for j in Sp: base[j] = Xhat[j] if Xhat[j] != 0 else (seed_d[j] if seed_d[j] != 0 else nn_d[j])
        if j_strategy == 0: return base
        if j_strategy == 1: return base * 1.5
        if j_strategy == 2: return base * 2.0
        nnv = np.zeros_like(Xhat); idx = list(Sp); nnv[idx] = nn_d[idx]
        return nnv if j_strategy == 3 else 0.5 * (base + nnv)
    def EVAL(Sp):
        if len(Sp) == 0: return False, None
        for js in range(5):
            if orc.nf >= cfg.max_fwd: return False, None
            d = dom.project(a, values(Sp, js))
            if (d != 0).sum() == 0: continue
            if dom.check(a, d)[0] and orc.q0(d) < 0: return True, d
        return False, None
    it = 0
    while orc.nf < cfg.max_fwd:
        it += 1; acc = False; S = sorted(Shat)
        for _ in range(T_del or max(1, len(S))):                                   # 1a
            if not S: break
            e = rng.choice(S); ok, d = EVAL(set(S) - {e})
            if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); b = len(Shat); acc = True; moves.append("1a"); break
        if not acc and b >= 2:                                                      # 1b
            K = max(1, b // 2); ks = np.arange(1, K + 1); p = np.exp(-lam * (K - ks)); p /= p.sum()
            kk = int(rng.choice(ks, p=p)); R = set(rng.choice(S, size=min(kk, len(S)), replace=False).tolist())
            ok, d = EVAL(set(S) - R)
            if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); acc = True; moves.append("1b")
        if not acc:                                                                 # 2
            out = [j for j in U if j not in Shat]
            for _ in range(T_swap):
                if not out or not S: break
                e, e2 = rng.choice(S), rng.choice(out); ok, d = EVAL((set(S) - {e}) | {e2})
                if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); acc = True; moves.append("2"); break
        if not acc and len(Shat) + 1 < b:                                           # 3
            for e in [j for j in U if j not in Shat]:
                ok, d = EVAL(Shat | {e})
                if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); b = len(Shat); acc = True; moves.append("3"); break
        if not acc: break
    return _final_row(orc, dom, sp, Xhat, cfg, "LBS-adapted", dict(lbs_iters=it, lbs_moves="".join(m[-1] for m in moves),
                                                                   lbs_budget_stop=orc.nf >= cfg.max_fwd))

def drive(orc, gen, ctx):
    """Runs a coroutine of flowsce.fast with the sequential oracle: every request (delta, mode) is answered by q0;
    ctx.nf is advanced exactly as run_coroutines does in the batched path."""
    try:
        req = next(gen)
        while True:
            q = orc.q0(req[0], req[1]); ctx.nf += 1; req = gen.send(q)
    except StopIteration as s: return s.value

def minimizer_from_seed(model, g, k, sp, dom, cfg, seed_delta, method, seed=0, budget=None, dml=None):
    """Minimizer comparison from the SAME saved valid candidate. RHC and the dice-ml post-hoc algorithm are written once,
    as coroutines in flowsce.fast, and driven here by the sequential oracle (the batched path drives them with BatchOracle)."""
    if method in ("RHC", "DiCE post-hoc") or budget is not None:
        from . import fast as FF
        orc = QueryOracle(model, g, k, cfg.tau); a = orc.a.cpu().numpy(); nn_d = (_M.POOL.nearest(a) - a).astype(np.float32)
        ctx = FF.Ctx(int(k), a, orc.thr)
        row = drive(orc, FF.minimizer_co(ctx, dom, sp, cfg, seed_delta, method, seed, nn_d, budget, dml), ctx)
        row["n_fwd"] = orc.nf; return row
    if method == "LBS-adapted": return lbs_adapted(model, g, k, sp, dom, cfg, seed_delta, seed)
    orc = QueryOracle(model, g, k, cfg.tau)
    if seed_delta is None: return _final_row(orc, dom, sp, None, cfg, method)
    d = np.array(seed_delta, np.float32)
    if method == "none (A0)": pass
    elif method == "FlowSCE single (A)": d, _ = restore(orc, dom, d, cfg, "A", repeated=False)
    elif method == "FlowSCE repeated (D)": d, _ = restore(orc, dom, d, cfg, "A", repeated=True)
    elif method == "DiCE post-hoc": d = dice_posthoc(orc, dom, d, cfg)
    else: raise ValueError(method)
    return _final_row(orc, dom, sp, d, cfg, method)
