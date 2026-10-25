"""FlowSCE: Algorithm 1 generation, restoration (single / repeated), the A0/A/B/C/D variants, and FlowSCE (restoration,
margin-floored repair, margin polishing)."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE, ExplainCfg
from .schema import Domain
from .oracle import QueryOracle
from .space import Space, xi_box
from .metrics import assess

def generate(orc: QueryOracle, sp: Space, dom: Domain, cfg: ExplainCfg, mode="A", mask=None, xi0=None):
    """Algorithm 1, generation loop. Returns dict(delta|None, status, updates)."""
    m = sp.m if mask is None else mask
    xi = (torch.zeros(sp.F, device=DEVICE) if xi0 is None else xi0.clone()).requires_grad_(True)
    opt = torch.optim.Adam([xi], lr=cfg.lr); a = orc.a.cpu().numpy()
    if cfg.box_proj: XLO, XHI = xi_box(dom, sp, a)
    for t in range(cfg.T + 1):
        d = xi * sp.s * m; q = orc.q(d, mode)
        if q.item() < -cfg.min_margin:
            dp = dom.project(a, d.detach())
            if dom.check(a, dp)[0] and orc.q0(dp, mode) < -cfg.min_margin:
                return dict(delta=dp, status="valid", updates=t)
        if t == cfg.T or orc.nf >= cfg.max_fwd:
            return dict(delta=None, status="budget_stop", updates=t)
        loss = F.relu(q + cfg.search_margin) + cfg.lam1 * torch.sqrt(xi @ sp.Qxi @ xi + 1e-8) + cfg.lam2 * xi.abs().sum()
        opt.zero_grad(); loss.backward(); orc.nb += 1; opt.step()
        with torch.no_grad():
            xi.mul_(m)
            if cfg.box_proj: xi.copy_(torch.maximum(torch.minimum(xi, XHI), XLO))

def restore(orc: QueryOracle, dom: Domain, delta, cfg: ExplainCfg, mode="A", repeated=False):
    """Algorithm 1 restoration (single pass = A) or repeated passes until no-change (= D)."""
    a = orc.a.cpu().numpy(); d = np.array(delta, np.float32); trials, passes = 0, 0
    while True:
        passes += 1; changed = False
        order = [j for j in np.argsort(np.abs(d)) if dom.mutable[j] and abs(d[j]) > cfg.eps_support]
        for j in order:
            if trials >= cfg.max_restore_trials or orc.nf >= cfg.max_fwd:
                return d, dict(trials=trials, passes=passes, stop="budget_stop")
            tr = d.copy(); tr[j] = 0.0; trials += 1
            if dom.check(a, tr)[0] and orc.q0(tr, mode) < -cfg.min_margin: d = tr; changed = True
        if not repeated: return d, dict(trials=trials, passes=passes, stop="single_pass")
        if not changed: return d, dict(trials=trials, passes=passes, stop="no_change")

# Readable names of the variants; ledgers keep the short code in `variant` and add `variant_name`.
VARIANT_NAMES = {"A0": "flowsce_noprune", "A": "flowsce", "B": "stale_context", "C": "detached_grad", "D": "flowsce_repeated"}

def explain_query(model, g, k, sp, dom, cfg: ExplainCfg, variant: str, seed_delta=None):
    """Runs one variant on one query and returns a ledger row (see §8).
    A0: generation only. A: + single restoration. B: stale context. C: detached gradient.
    D: repeated restoration starting from A0's saved candidate (seed_delta)."""
    orc = QueryOracle(model, g, k, cfg.tau); t0 = time.perf_counter()
    q_orig = orc.q0(np.zeros(sp.F, np.float32), "A")
    if q_orig < 0: return dict(status="not_alert", q_orig=q_orig)
    mode = {"A0": "A", "A": "A", "B": "B", "C": "C", "D": "A"}[variant]
    if variant == "D":
        gen = dict(delta=None if seed_delta is None else np.array(seed_delta, np.float32),
                   status="valid" if seed_delta is not None else "no_seed", updates=0)
        if gen["delta"] is not None and not (orc.q0(gen["delta"], "A") < 0): gen["status"] = "invalid_seed"; gen["delta"] = None
    else:
        gen = generate(orc, sp, dom, cfg, mode)
    t_gen = time.perf_counter() - t0; rinfo = dict(trials=0, passes=0, stop="none"); d = gen["delta"]
    if d is not None and variant in ("A", "B", "C", "D"):
        d, rinfo = restore(orc, dom, d, cfg, mode, repeated=(variant == "D"))
    t_rest = time.perf_counter() - t0 - t_gen
    row = dict(variant=variant, variant_name=VARIANT_NAMES[variant], gen_status=gen["status"], updates=gen["updates"], restore_trials=rinfo["trials"],
               restore_passes=rinfo["passes"], restore_stop=rinfo["stop"], q_orig=q_orig,
               omega=orc.omega, deg_u=orc.deg_u, deg_v=orc.deg_v, t_generate=t_gen, t_restore=t_rest)
    row.update(assess(orc, dom, sp, d, mode, cfg))
    row["t_assess"] = time.perf_counter() - t0 - t_gen - t_rest
    row["n_fwd"], row["n_bwd"] = orc.nf, orc.nb
    return row


# ══════════════════════════════ FlowSCE: restoration, margin-floored repair, margin polishing ══════════════════════════════
def support_eps(d, eps):
    return int((np.abs(d) > eps).sum())

def repair(orc: QueryOracle, sp: Space, dom: Domain, delta, cfg: ExplainCfg, mode="A", T_rep=25, min_margin=0.0):
    """Margin-floored repair (Section 4, Proposition repair): from a valid candidate, visit the changed coordinates in
    ascending |δ_j|; try δ_j ← 0, and if that breaks validity re-run the projected search of Algorithm 1 for at most T_rep
    updates from the restored point, masked to the remaining support. A trial is accepted only if it passes V_k, its
    reconstructed score satisfies q < −min_margin, and its ε-support is strictly smaller. Passes repeat until none accepts."""
    a = orc.a.cpu().numpy(); d = np.array(delta, np.float32); eps = cfg.eps_support
    rcfg = ExplainCfg(**{**vars(cfg), "T": T_rep, "min_margin": 0.0}); trials = passes = repairs = 0; stop = "no_change"
    thr0 = orc.thr; orc.thr = thr0 - min_margin          # q' = q + min_margin:  q' < 0  ⇔  q < −min_margin
    try:
        while True:
            passes += 1; changed = False
            for j in [j for j in np.argsort(np.abs(d)) if dom.mutable[j] and abs(d[j]) > eps]:
                if abs(d[j]) <= eps: continue
                if orc.nf >= cfg.max_fwd or trials >= cfg.max_restore_trials: stop = "budget_stop"; break
                tr = d.copy(); tr[j] = 0.0; trials += 1
                if dom.check(a, tr)[0] and orc.q0(tr, mode) < 0: d = tr; changed = True; continue
                keep = (np.abs(tr) > eps) & dom.mutable
                if not keep.any(): continue
                mask = torch.tensor(keep, dtype=torch.float32, device=DEVICE)
                xi0 = torch.tensor(tr, dtype=torch.float32, device=DEVICE) / sp.s * mask
                res = generate(orc, sp, dom, rcfg, mode, mask=mask, xi0=xi0)
                if res["delta"] is not None and support_eps(res["delta"], eps) < support_eps(d, eps):
                    d = np.asarray(res["delta"], np.float32); changed = True; repairs += 1
            if stop == "budget_stop" or not changed: break
    finally:
        orc.thr = thr0
    return d, dict(trials=trials, passes=passes, repairs=repairs, stop=stop)

def polish_margin(orc: QueryOracle, sp: Space, dom: Domain, delta, cfg: ExplainCfg, m_target, T=100, mode="A"):
    """Margin polishing: re-run the projected search restricted to the candidate's own support, from the candidate, until
    q < −m_target. The support can only shrink; if the target is not reached the input candidate is returned unchanged."""
    d = np.array(delta, np.float32); keep = (np.abs(d) > cfg.eps_support) & dom.mutable
    if not keep.any() or orc.q0(d, mode) < -m_target: return d, dict(polished=False, stop="already")
    mask = torch.tensor(keep, dtype=torch.float32, device=DEVICE); xi0 = torch.tensor(d, dtype=torch.float32, device=DEVICE) / sp.s * mask
    thr0 = orc.thr; orc.thr = thr0 - m_target
    try: res = generate(orc, sp, dom, ExplainCfg(**{**vars(cfg), "T": T, "min_margin": 0.0}), mode, mask=mask, xi0=xi0)
    finally: orc.thr = thr0
    if res["delta"] is None: return d, dict(polished=False, stop=res["status"])
    return np.asarray(res["delta"], np.float32), dict(polished=True, stop="target", updates=res["updates"])

M_TARGET = float(np.log(3))      # polishing target: reconstructed attack probability ≤ 0.25 (fixed a priori, not tuned)

def sparse_probe(orc, sp, dom, d, cfg, k=5, T=25, min_margin=0.0, mode="A", init="zero", free=False, select="first"):
    """Single-feature probes: for the k mutable coordinates with the largest |total gradient| of q at δ = 0 and the k
    largest |δ_j| of the incumbent, run the projected search masked to that one coordinate (from δ = 0). A probe is
    accepted if it is valid on the reconstructed graph, passes V_k, satisfies q < −min_margin, and has smaller ε-support."""
    a = orc.a.cpu().numpy(); eps = cfg.eps_support; best = np.array(d, np.float32)
    if support_eps(best, eps) <= 1 or (k is not None and k <= 0): return best, 0
    z = torch.zeros(sp.F, device=DEVICE, requires_grad=True)
    gq = torch.autograd.grad(orc.q(z * sp.m, mode), z)[0].abs().cpu().numpy(); orc.nb += 1
    mut = np.nonzero(dom.mutable)[0]
    from . import metrics as _M
    nn_d = (_M.POOL.nearest(a) - a).astype(np.float32) if _M.POOL is not None and init in ("benign", "both") else None
    if k is None: cand = [int(j) for j in mut[np.argsort(-gq[mut])]]                   # every mutable feature, largest gradient first
    else: cand = list(dict.fromkeys([int(j) for j in mut[np.argsort(-gq[mut])][:k]] + [int(j) for j in np.argsort(-np.abs(best)) if dom.mutable[j]][:k]))
    pcfg = ExplainCfg(**{**vars(cfg), "T": T, "min_margin": 0.0, **({"lam1": 0.0, "lam2": 0.0} if free else {})})
    thr0 = orc.thr; orc.thr = thr0 - min_margin; hits = 0
    try:
        for j in cand:
            if orc.nf >= cfg.max_fwd: break
            mask = torch.zeros(sp.F, device=DEVICE); mask[j] = 1.0
            starts = []
            if init in ("incumbent", "both") and abs(d[j]) > eps: starts.append(float(d[j]))        # incumbent value of coordinate j
            if init in ("benign", "both") and nn_d is not None and abs(nn_d[j]) > eps: starts.append(float(nn_d[j]))  # nearest benign value
            if not starts: starts = [0.0]
            for v0 in starts:
                xi0 = torch.zeros(sp.F, device=DEVICE); xi0[j] = v0 / float(sp.s[j])
                res = generate(orc, sp, dom, pcfg, mode, mask=mask, xi0=xi0)
                if res["delta"] is not None: break
            if res["delta"] is not None:
                cd = np.asarray(res["delta"], np.float32); sc, sb = support_eps(cd, eps), support_eps(best, eps)
                if sc < sb or (select == "best_l1" and sc == sb and np.abs(cd).sum() < np.abs(best).sum()): best = cd; hits += 1
                if select == "first" and support_eps(best, eps) <= 1: break
    finally:
        orc.thr = thr0
    return best, hits

def shrink_l1(orc, dom, d, cfg, min_margin, steps=6, mode="A"):
    """Moves every remaining change toward the original value by bisection, keeping validity and q < −min_margin; the
    support never grows and the L1 deviation never increases."""
    a = orc.a.cpu().numpy(); d = np.array(d, np.float32)
    for j in [j for j in np.argsort(-np.abs(d)) if abs(d[j]) > cfg.eps_support]:
        lo, hi = 0.0, 1.0                                         # fraction of δ_j kept
        for _ in range(steps):
            if orc.nf >= cfg.max_fwd: return d
            mid = (lo + hi) / 2; tr = d.copy(); tr[j] = d[j] * mid; tr = dom.project(a, tr)
            if dom.check(a, tr)[0] and orc.q0(tr, mode) < -min_margin: hi = mid
            else: lo = mid
        if hi < 1.0:
            tr = d.copy(); tr[j] = d[j] * hi; tr = dom.project(a, tr)
            if dom.check(a, tr)[0] and orc.q0(tr, mode) < -min_margin and support_eps(tr, cfg.eps_support) <= support_eps(d, cfg.eps_support): d = tr
    return d

def plus_from_seed(orc, sp, dom, cfg, seed_delta, T_rep=25, T_polish=100, m_target=M_TARGET, start="A", probe_k=0, probe_T=25,
                   shrink=False, probe_init="zero", probe_free=False, probe_select="first", probe_floor="A", floor_mode="A"):
    """FlowSCE refinement of a valid candidate: restoration (single pass = variant A, or repeated = D), single-feature
    probes, repair with A's margin as floor, polishing to q < −m_target, and optional L1 shrinking at the reached margin.
    Returns (delta, info) with the support and margin of A for the audit."""
    d = np.array(seed_delta, np.float32)
    dA, rinfo = restore(orc, dom, d, cfg, "A", repeated=False)
    mA = -orc.q0(dA, "A"); sA = support_eps(dA, cfg.eps_support)
    # margin floor: A's margin, or (floor_mode="target") the smaller of A's margin and the polishing target
    floor = max(0.0, (min(mA, m_target) if floor_mode == "target" else mA) - 1e-6)
    d0 = restore(orc, dom, dA, cfg, "A", repeated=True)[0] if start == "D" else dA
    if start == "D" and -orc.q0(d0, "A") < floor: d0 = dA            # keep the margin floor of A
    def finish(d_in, fl):
        dR, pinfo = repair(orc, sp, dom, d_in, cfg, "A", T_rep=T_rep, min_margin=fl)
        dP, qinfo = polish_margin(orc, sp, dom, dR, cfg, m_target, T=T_polish)
        if shrink: dP = shrink_l1(orc, dom, dP, cfg, min_margin=max(floor, m_target if -orc.q0(dP, "A") > m_target else floor))
        return dP, pinfo, qinfo
    if probe_floor == "zero":
        # single-feature probes without the margin floor, polished afterwards; kept only if the final margin respects A's floor
        dz, hits = sparse_probe(orc, sp, dom, d0, cfg, k=probe_k, T=probe_T, min_margin=0.0, init=probe_init, free=probe_free, select=probe_select)
        if hits:
            dP, pinfo, qinfo = finish(dz, 0.0)
            if -orc.q0(dP, "A") >= floor and support_eps(dP, cfg.eps_support) <= sA:
                return dP, dict(support_A=sA, margin_A=mA, repairs=pinfo["repairs"], repair_stop=pinfo["stop"], polished=qinfo["polished"],
                                polish_stop=qinfo["stop"], restore_trials=rinfo["trials"] + pinfo["trials"], probe_hits=hits, probe_path="free")
        d0, hits = sparse_probe(orc, sp, dom, d0, cfg, k=probe_k, T=probe_T, min_margin=floor, init=probe_init, free=probe_free, select=probe_select)
    else:
        d0, hits = sparse_probe(orc, sp, dom, d0, cfg, k=probe_k, T=probe_T, min_margin=floor, init=probe_init, free=probe_free, select=probe_select)
    dP, pinfo, qinfo = finish(d0, floor)
    return dP, dict(support_A=sA, margin_A=mA, repairs=pinfo["repairs"], repair_stop=pinfo["stop"], polished=qinfo["polished"],
                    polish_stop=qinfo["stop"], restore_trials=rinfo["trials"] + pinfo["trials"], probe_hits=hits, probe_path="floor")

def explain_plus(model, g, k, sp, dom, cfg: ExplainCfg, seed_delta=None, T_rep=25, T_polish=100, m_target=M_TARGET, **opts):
    """FlowSCE on query k: Algorithm 1 generation (or a given seed, e.g. the saved A0 candidate or another generator's
    output), restoration, margin-floored repair, margin polishing — all through the same reconstructed oracle and budget."""
    orc = QueryOracle(model, g, k, cfg.tau); t0 = time.perf_counter()
    q_orig = orc.q0(np.zeros(sp.F, np.float32), "A"); info = {}
    if seed_delta is None: gen = generate(orc, sp, dom, cfg, "A"); d = gen["delta"]
    else: d = np.array(seed_delta, np.float32) if orc.q0(np.array(seed_delta, np.float32), "A") < 0 else None
    t_gen = time.perf_counter() - t0
    if d is not None: d, info = plus_from_seed(orc, sp, dom, cfg, d, T_rep, T_polish, m_target, **opts)
    row = dict(method="FlowSCE", variant="Plus", q_orig=q_orig, omega=orc.omega, t_generate=t_gen,
               t_restore=time.perf_counter() - t0 - t_gen, T_rep=T_rep, T_polish=T_polish, n_fwd_method=orc.nf, **info)
    row.update(assess(orc, dom, sp, d, "A", cfg)); row["n_fwd"], row["n_bwd"] = orc.nf, orc.nb
    return row


def plus_kwargs(pc):
    """Keyword arguments of explain_plus / plus_from_seed from an entry of protocol/plus_config.json."""
    return dict(T_rep=pc["T_rep"], T_polish=pc["T_polish"], **pc.get("opts", {}))
