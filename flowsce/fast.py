"""
flowsce.fast — fast, numerically equivalent execution of the explainers in the flowsce reference modules.

Two exact accelerations (same mathematics, different evaluation order):
 1. Multiplicity compression of the multigraph: parallel flows between the same (src, dst) hosts send identical
    GAT messages with identical attention scores, so message passing runs on the U unique host pairs with
    integer weights c_uv (U = 98 / 164 / 1297 instead of ~12 000 edges for DoH / UNSW / CIC).
 2. Query batching: B (query, δ) pairs are evaluated in one forward. Only X_V and the query row change, so the
    detector is run on a (B, N, F) stack of reconstructed host matrices.
Gradient methods (CANDICE-style, DiCE-style) are vectorised with a per-row Adam identical to torch.optim.Adam;
forward-only procedures (restoration, DiCE post-hoc, LBS-adapted, assessment) are written as coroutines that
yield oracle requests, and a scheduler batches the pending requests of all queries.
Call counts (n_fwd, n_bwd) follow the sequential implementation exactly.
"""
from __future__ import annotations
import copy, math, time, numpy as np, torch, torch.nn.functional as F
import flowsce as L          # package namespace (reference implementation); POOL is read at call time

DEV = L.DEVICE

# ══════════════════════ 1. Compressed, batched detector evaluation ══════════════════════
class CompressedGraph:
    def __init__(self, g):
        assert g["policy"] == "full", "fast path implemented for the per-partition snapshot policy"
        pair, cnt = torch.unique(torch.stack([g["src"], g["dst"]]), dim=1, return_counts=True)
        self.src, self.dst, self.cnt = pair[0], pair[1], cnt.to(g["xe"].dtype)
        self.g, self.N = g, g["N"]
        self.e_src, self.e_dst, self.deg = g["src"], g["dst"], g["deg"]

def _gat_batched(layer, h, cg: CompressedGraph):
    """h: (B, N, Fin) → (B, N, H*d). Equivalent to GATLayer on the full multigraph (eval mode)."""
    B, N, _ = h.shape; H, d = layer.H, layer.d; src, dst, c = cg.src, cg.dst, cg.cnt.to(h.dtype)
    Wh = layer.lin(h).view(B, N, H, d)
    e = F.leaky_relu((Wh * layer.a_s).sum(-1)[:, src] + (Wh * layer.a_d).sum(-1)[:, dst], 0.2)      # (B,U,H)
    idx = dst.view(1, -1, 1).expand(B, -1, H)
    m = torch.full((B, N, H), -torch.finfo(h.dtype).max, dtype=h.dtype, device=h.device)
    m = m.scatter_reduce(1, idx, e, "amax", include_self=True)
    ex = torch.exp(e - m[:, dst]) * c.view(1, -1, 1)                                                  # multiplicity
    den = torch.zeros(B, N, H, dtype=h.dtype, device=h.device).index_add(1, dst, ex)
    al = ex / den[:, dst].clamp(min=1e-16)
    out = torch.zeros(B, N, H, d, dtype=h.dtype, device=h.device).index_add(1, dst, al.unsqueeze(-1) * Wh[:, src])
    return out.reshape(B, N, H * d) + layer.b

class BatchOracle:
    """q for a batch of (query k, δ, mode); identical semantics to flowsce.QueryOracle."""
    def __init__(self, model, g, tau=0.5):
        self.m, self.cg, self.g = model.eval(), CompressedGraph(g), g
        self.thr = math.log(tau / (1 - tau)); self.ctx = model.kind != "edge"
        self.forwards = 0; self.rows = 0
    def q(self, ks, deltas, modes):
        """ks: LongTensor (B,), deltas: (B,F) tensor (may require grad), modes: list of 'A'/'B'/'C'."""
        g, m = self.g, self.m; B = len(ks); ks = torch.as_tensor(np.array(ks), device=DEV)
        self.forwards += 1; self.rows += B
        a = g["xe"][ks] + deltas
        parts = [F.relu(m.ep(a))]
        if self.ctx:
            u, v = g["src"][ks], g["dst"][ks]
            mode_w = torch.tensor([0.0 if md == "B" else 1.0 for md in modes], device=DEV, dtype=deltas.dtype)
            det = torch.tensor([md == "C" for md in modes], device=DEV)
            d_ctx = torch.where(det[:, None], deltas.detach(), deltas) * mode_w[:, None]
            X = g["xv"].unsqueeze(0).expand(B, -1, -1).reshape(B * g["N"], -1)
            off = torch.arange(B, device=DEV) * g["N"]
            X = X.index_add(0, off + u, d_ctx / g["deg"][u][:, None]).index_add(0, off + v, d_ctx / g["deg"][v][:, None])
            h = F.relu(m.ne(X.view(B, g["N"], -1)))
            if hasattr(m, "sages"):                                  # GraphSAGE-mean over incoming flows (multiplicity-weighted)
                c = self.cg.cnt.to(h.dtype); incnt = torch.zeros(g["N"], dtype=h.dtype, device=DEV).index_add(0, self.cg.dst, c).clamp(min=1)
                for lin in m.sages:
                    agg = torch.zeros_like(h).index_add(1, self.cg.dst, h[:, self.cg.src] * c.view(1, -1, 1)) / incnt.view(1, -1, 1)
                    h = F.relu(lin(torch.cat([h, agg], -1)))
            else:
                for layer in m.gats: h = F.elu(_gat_batched(layer, h, self.cg))
            bi = torch.arange(B, device=DEV)
            cs, cd = (h[bi, u], h[bi, v]) if getattr(m, "full", False) else (F.relu(m.ps(h[bi, u])), F.relu(m.pd(h[bi, v])))
            if m.kind.endswith("_gate"):
                gt = torch.sigmoid(m.gate(parts[0])); cs, cd = gt[:, :1] * cs, gt[:, 1:] * cd
            parts += [cs, cd]
        lg = m.mlp(torch.cat(parts, -1))
        return lg[:, 1] - lg[:, 0] - self.thr

    @torch.no_grad()
    def q_nograd(self, ks, deltas, modes, chunk=1024):
        out = []
        for i in range(0, len(ks), chunk):
            d = torch.as_tensor(np.asarray(deltas[i:i + chunk]), device=DEV, dtype=torch.float32)
            out.append(self.q(ks[i:i + chunk], d, modes[i:i + chunk]))
        return torch.cat(out).cpu().numpy() if out else np.zeros(0)

# ══════════════════════ 2. Coroutine scheduler for forward-only procedures ══════════════════════
class Ctx:
    """Per-(query, method) oracle bookkeeping, mirrors QueryOracle.nf / nb."""
    __slots__ = ("k", "a", "nf", "nb", "thr")
    def __init__(self, k, a, thr): self.k, self.a, self.nf, self.nb, self.thr = k, a, 0, 0, thr

def run_coroutines(bo: BatchOracle, tasks, chunk=1024):
    """tasks: list of (ctx, generator). A generator yields (delta ndarray, mode) and receives q (float);
    its return value is the result. All pending requests are evaluated together each round."""
    results = [None] * len(tasks); pending = {}
    for i, (ctx, gen) in enumerate(tasks):
        try: pending[i] = next(gen)
        except StopIteration as s: results[i] = s.value
    while pending:
        ids = list(pending); ks = [tasks[i][0].k for i in ids]
        qs = bo.q_nograd(ks, [pending[i][0] for i in ids], [pending[i][1] for i in ids], chunk)
        for i, qv in zip(ids, qs):
            tasks[i][0].nf += 1
            try: pending[i] = tasks[i][1].send(float(qv))
            except StopIteration as s: results[i] = s.value; del pending[i]
    return results

# --- coroutine versions (line-by-line mirrors of the flowsce reference modules) ---
def assess_co(ctx, dom, sp, d, mode, cfg):
    if d is None: return _empty_assess()
    a = ctx.a; ok, why = dom.check(a, d)
    qn = yield (d, mode); qr = yield (d, "A"); qs = yield (d, "B")
    n = int(qn < 0 and ok); v = int(qr < 0 and ok); pr = 1 / (1 + math.exp(-(qr + ctx.thr)))
    return dict(n=n, v=v, returned=1, status="valid" if v else ("spurious" if n else "invalid"), q_nominal=qn,
                q_reconstructed=qr, domain_ok=ok, domain_reasons=";".join(why), support_exact=int((d != 0).sum()),
                support_eps=int((np.abs(d) > cfg.eps_support).sum()), l1=float(np.abs(d).sum()), l2=float(np.linalg.norm(d)),
                mahal=sp.mahal(d), knn_benign=L.POOL.dist(a + d) if L.POOL is not None else np.nan,
                margin_logit=-qr, margin_prob=cfg.tau - pr, disc_stale=abs(qr - qs), delta=d.tolist())

def _empty_assess():
    return dict(n=0, v=0, returned=0, status="failure", q_nominal=np.nan, q_reconstructed=np.nan, domain_ok=None,
                domain_reasons="", support_exact=np.nan, support_eps=np.nan, l1=np.nan, l2=np.nan, mahal=np.nan,
                knn_benign=np.nan, margin_logit=np.nan, margin_prob=np.nan, disc_stale=np.nan, delta=None)

def final_row_co(ctx, dom, sp, d, cfg, method, extra=None):
    row = dict(method=method, n_fwd_method=ctx.nf)
    if d is None: row.update(_empty_assess())
    else: row.update((yield from assess_co(ctx, dom, sp, d, "A", cfg)))
    row["n_fwd"], row["n_bwd"] = ctx.nf, ctx.nb
    if extra: row.update(extra)
    return row

def restore_co(ctx, dom, delta, cfg, mode="A", repeated=False):
    a = ctx.a; d = np.array(delta, np.float32); trials, passes = 0, 0
    while True:
        passes += 1; changed = False
        order = [j for j in np.argsort(np.abs(d)) if dom.mutable[j] and abs(d[j]) > cfg.eps_support]
        for j in order:
            if trials >= cfg.max_restore_trials or ctx.nf >= cfg.max_fwd:
                return d, dict(trials=trials, passes=passes, stop="budget_stop")
            tr = d.copy(); tr[j] = 0.0; trials += 1
            if dom.check(a, tr)[0] and (yield (tr, mode)) < -cfg.min_margin: d = tr; changed = True
        if not repeated: return d, dict(trials=trials, passes=passes, stop="single_pass")
        if not changed: return d, dict(trials=trials, passes=passes, stop="no_change")

def dice_posthoc_co(ctx, dom, d, cfg, bisect_steps=5):
    a = ctx.a; d = np.array(d, np.float32)
    for j in np.argsort(np.abs(d)):
        if d[j] == 0 or not dom.mutable[j] or ctx.nf >= cfg.max_fwd: continue
        tr = d.copy(); tr[j] = 0
        if dom.check(a, tr)[0] and (yield (tr, "A")) < 0: d = tr; continue
        lo, hi = 0.0, 1.0
        for _ in range(bisect_steps):
            mid = (lo + hi) / 2; tr = d.copy(); tr[j] = d[j] * mid; tr = dom.project(a, tr)
            if dom.check(a, tr)[0] and (yield (tr, "A")) < 0: hi = mid
            else: lo = mid
        if hi < 1.0:
            tr = d.copy(); tr[j] = d[j] * hi; d = dom.project(a, tr)
    return d

def lbs_co(ctx, dom, sp, cfg, seed_delta, seed, nn_d, T_del=None, T_swap=20, lam=1.0):
    """Mirror of flowsce.lbs_adapted (same RNG call sequence)."""
    a = ctx.a; rng = np.random.default_rng(seed)
    if seed_delta is None:
        return (yield from final_row_co(ctx, dom, sp, None, cfg, "LBS-adapted", dict(lbs_iters=0, lbs_moves="")))
    seed_d = np.array(seed_delta, np.float32)
    if not (dom.check(a, seed_d)[0] and (yield (seed_d, "A")) < 0):
        return (yield from final_row_co(ctx, dom, sp, None, cfg, "LBS-adapted", dict(lbs_iters=0, lbs_moves="invalid_seed")))
    U = np.nonzero(dom.mutable)[0]
    Xhat = seed_d.copy(); Shat = set(np.nonzero(Xhat)[0].tolist()); b = len(Shat); moves = []
    def values(Sp, js):
        base = np.zeros_like(Xhat)
        for j in Sp: base[j] = Xhat[j] if Xhat[j] != 0 else (seed_d[j] if seed_d[j] != 0 else nn_d[j])
        if js == 0: return base
        if js == 1: return base * 1.5
        if js == 2: return base * 2.0
        nnv = np.zeros_like(Xhat); idx = list(Sp); nnv[idx] = nn_d[idx]
        return nnv if js == 3 else 0.5 * (base + nnv)
    def EVAL(Sp):
        if len(Sp) == 0: return False, None
        for js in range(5):
            if ctx.nf >= cfg.max_fwd: return False, None
            d = dom.project(a, values(Sp, js))
            if (d != 0).sum() == 0: continue
            if dom.check(a, d)[0] and (yield (d, "A")) < 0: return True, d
        return False, None
    it = 0
    while ctx.nf < cfg.max_fwd:
        it += 1; acc = False; S = sorted(Shat)
        for _ in range(T_del or max(1, len(S))):
            if not S: break
            e = rng.choice(S); ok, d = yield from EVAL(set(S) - {e})
            if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); b = len(Shat); acc = True; moves.append("1a"); break
        if not acc and b >= 2:
            K = max(1, b // 2); ks = np.arange(1, K + 1); p = np.exp(-lam * (K - ks)); p /= p.sum()
            kk = int(rng.choice(ks, p=p)); R = set(rng.choice(S, size=min(kk, len(S)), replace=False).tolist())
            ok, d = yield from EVAL(set(S) - R)
            if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); acc = True; moves.append("1b")
        if not acc:
            out = [j for j in U if j not in Shat]
            for _ in range(T_swap):
                if not out or not S: break
                e, e2 = rng.choice(S), rng.choice(out); ok, d = yield from EVAL((set(S) - {e}) | {e2})
                if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); acc = True; moves.append("2"); break
        if not acc and len(Shat) + 1 < b:
            for e in [j for j in U if j not in Shat]:
                ok, d = yield from EVAL(Shat | {e})
                if ok: Xhat = d; Shat = set(np.nonzero(d)[0].tolist()); b = len(Shat); acc = True; moves.append("3"); break
        if not acc: break
    return (yield from final_row_co(ctx, dom, sp, Xhat, cfg, "LBS-adapted",
                                    dict(lbs_iters=it, lbs_moves="".join(m[-1] for m in moves), lbs_budget_stop=ctx.nf >= cfg.max_fwd)))

def minimizer_co(ctx, dom, sp, cfg, seed_delta, method, seed, nn_d, budget=None, dml=None):
    """budget: cap on the oracle calls of the minimizer itself (None: the shared per-query budget cfg.max_fwd).
    dml: parameters of the dice-ml post-hoc sparsity algorithm (dice_posthoc_params)."""
    if budget is not None:
        cfg = copy.copy(cfg); cfg.max_fwd = int(budget); cfg.max_restore_trials = int(budget)
    if method == "LBS-adapted": return (yield from lbs_co(ctx, dom, sp, cfg, seed_delta, seed, nn_d))
    if method == "RHC": return (yield from rhc_co(ctx, dom, sp, cfg, seed_delta, seed, nn_d))
    if seed_delta is None: return (yield from final_row_co(ctx, dom, sp, None, cfg, method))
    d = np.array(seed_delta, np.float32); info = {}
    if method == "FlowSCE single (A)": d, info = yield from restore_co(ctx, dom, d, cfg, "A", False)
    elif method == "FlowSCE repeated (D)": d, info = yield from restore_co(ctx, dom, d, cfg, "A", True)
    elif method == "DiCE post-hoc": d, info = yield from dice_posthoc_dml_co(ctx, dom, d, cfg, dml)
    elif method != "none (A0)": raise ValueError(method)
    return (yield from final_row_co(ctx, dom, sp, d, cfg, method, {"minimizer_stop": info.get("stop", "")}))

def minimizers_batch(model, g, ks, seeds_delta, init_seeds, sp, dom, cfg, methods, chunk=1024, budgets=None, dml=None):
    """All (query × method) minimizer runs of one checkpoint, batched. Returns list of rows (query-major).
    budgets: optional {method: per-query list (or scalar) of oracle-call caps}."""
    bo = BatchOracle(model, g, cfg.tau); tasks = []
    for qi, (k, sd, s0) in enumerate(zip(ks, seeds_delta, init_seeds)):
        a = g["xe"][int(k)].cpu().numpy(); nn_d = (L.POOL.nearest(a) - a).astype(np.float32)
        for meth in methods:
            b = None if not budgets or meth not in budgets else (budgets[meth][qi] if np.ndim(budgets[meth]) else budgets[meth])
            ctx = Ctx(int(k), a, bo.thr); tasks.append((ctx, minimizer_co(ctx, dom, sp, cfg, sd, meth, s0, nn_d, b, dml)))
    return run_coroutines(bo, tasks, chunk)


def rhc_co(ctx, dom, sp, cfg, seed_delta, seed, nn_d):
    """Random hill-climbing minimizer (the RHC baseline of García et al., 2026, Section 5.1.3): the same edit operations
    as LBS-adapted — deletion of one changed coordinate, replacement of one by an unchanged one, and compensating
    addition (two removed, one added) — and the same five value-assignment strategies, but drawn uniformly at random
    instead of in LBS's priority order, without the shrinking/overshooting move. A proposal is accepted if it is valid
    and does not increase the epsilon-support. Runs until its oracle budget (cfg.max_fwd, set to LBS's calls) is spent."""
    a = ctx.a; rng = np.random.default_rng(seed)
    if seed_delta is None: return (yield from final_row_co(ctx, dom, sp, None, cfg, "RHC", dict(rhc_iters=0)))
    seed_d = np.array(seed_delta, np.float32)
    if not (dom.check(a, seed_d)[0] and (yield (seed_d, "A")) < 0):
        return (yield from final_row_co(ctx, dom, sp, None, cfg, "RHC", dict(rhc_iters=0, rhc_moves="invalid_seed")))
    U = np.nonzero(dom.mutable)[0]; Xhat = seed_d.copy(); it = acc_n = 0
    def values(Sp, js):
        base = np.zeros_like(Xhat)
        for j in Sp: base[j] = Xhat[j] if Xhat[j] != 0 else (seed_d[j] if seed_d[j] != 0 else nn_d[j])
        if js == 0: return base
        if js == 1: return base * 1.5
        if js == 2: return base * 2.0
        nnv = np.zeros_like(Xhat); idx = list(Sp); nnv[idx] = nn_d[idx]
        return nnv if js == 3 else 0.5 * (base + nnv)
    supp = lambda d: int((np.abs(d) > cfg.eps_support).sum())
    while ctx.nf < cfg.max_fwd:
        it += 1; S = sorted(set(np.nonzero(Xhat)[0].tolist())); out = [j for j in U if j not in S]
        ops = ["del"] + (["rep"] if out and S else []) + (["add"] if out and len(S) >= 2 else [])
        op = ops[rng.integers(len(ops))]
        if op == "del": Sp = set(S) - {rng.choice(S)}
        elif op == "rep": Sp = (set(S) - {rng.choice(S)}) | {rng.choice(out)}
        else: Sp = (set(S) - set(rng.choice(S, 2, replace=False).tolist())) | {rng.choice(out)}
        if not Sp: continue
        d = dom.project(a, values(Sp, int(rng.integers(5))))
        if (d != 0).sum() == 0 or supp(d) > supp(Xhat) or not dom.check(a, d)[0]: continue
        if (yield (d, "A")) < 0: Xhat = d; acc_n += 1
    return (yield from final_row_co(ctx, dom, sp, Xhat, cfg, "RHC", dict(rhc_iters=it, rhc_accepted=acc_n)))


def decimal_precision(col):
    """dice-ml's rule (PublicData.get_decimal_precisions): the largest number of decimals among the modes of the column,
    read in positional notation (dice-ml 0.12 itself fails on modes printed in scientific notation, e.g. 1e-05)."""
    if np.allclose(col.values, np.round(col.values)): return 0
    p_ = 0
    for mv in col.mode().values:
        sv = np.format_float_positional(float(mv), trim="-"); p_ = max(p_, len(sv.split(".")[1]) if "." in sv else 0)
    return p_

def dice_posthoc_params(prep, df_train, quantile=0.1):
    """Inputs of dice-ml's post-hoc sparsity enhancement (ExplainerBase.do_posthoc_sparsity_enhancement), computed on the
    training data in raw units: per feature the `quantile` of |unique value - median of unique values|, capped by the MAD
    (a zero MAD is replaced by 1, as in get_valid_mads), the decimal precision of the modes, and the visiting order
    (decreasing quantile)."""
    R = prep.raw_frame(df_train); qs, prec = {}, {}
    for j, c in enumerate(R.columns):
        u = np.unique(R[c].values); qv = float(np.quantile(np.abs(u - np.median(u)), quantile))
        mad = float(np.median(np.abs(R[c].values - np.median(R[c].values)))); mad = mad if mad > 0 else 1.0
        qs[j] = min(qv, mad)
        prec[j] = decimal_precision(R[c])
    order = [j for j, _ in sorted(qs.items(), key=lambda kv: kv[1], reverse=True)]
    return dict(quantile=qs, prec=prec, order=order, limit_steps=10000)

def dice_posthoc_dml_co(ctx, dom, d, cfg, P):
    """dice-ml post-hoc sparsity, 'linear' algorithm, applied to one candidate in raw units: features in decreasing
    quantile order; a feature whose |query - CF| is at most its quantile is moved toward the query value in steps of
    10^-precision until the prediction would turn back to attack (that step is undone), the value is reached, or the
    step limit is hit. Validity is decided by the reconstructed oracle and the schema validator."""
    a = ctx.a; raw_a = dom.to_raw(a); raw = dom.to_raw(a + np.asarray(d, np.float64)); steps = 0
    for j in P["order"]:
        if not dom.mutable[j]: continue
        diff = raw_a[j] - raw[j]
        if not (abs(diff) <= P["quantile"][j]): continue
        change = 10.0 ** -P["prec"][j]; old_diff = diff; cnt = 0
        while abs(diff) > 10e-4 and np.sign(diff * old_diff) > 0 and cnt < P["limit_steps"]:
            if ctx.nf >= cfg.max_fwd: return _from_raw(dom, a, raw), dict(stop="budget_stop", steps=steps)
            old = raw[j]; raw[j] = raw[j] + np.sign(diff) * change
            tr = _from_raw(dom, a, raw); ok = dom.check(a, tr)[0] and (yield (tr, "A")) < 0; steps += 1; cnt += 1
            if not ok: raw[j] = old; break
            old_diff = diff; diff = raw_a[j] - raw[j]
    return _from_raw(dom, a, raw), dict(stop="done", steps=steps)

def _from_raw(dom, a, raw):
    d = (dom.from_raw(raw) - a).astype(np.float32); d[np.isclose(raw, dom.to_raw(a), rtol=0, atol=1e-9)] = 0.0
    return d

# ══════════════════════ 3. Vectorised gradient baselines ══════════════════════
def row_box(dom, sp, A_rows):
    """Stack xi_box bounds for many rows (A_rows: (R,F) original flows)."""
    lo, hi = zip(*[L.xi_box(dom, sp, a) for a in A_rows]); return torch.stack(lo), torch.stack(hi)

class RowAdam:
    """Adam with independent state per row (identical update to torch.optim.Adam on each row separately)."""
    def __init__(self, P, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr = lr if torch.is_tensor(lr) else torch.full((P.shape[0],), lr, device=P.device)
        self.b1, self.b2, self.eps = b1, b2, eps
        self.m = torch.zeros_like(P); self.v = torch.zeros_like(P); self.t = torch.zeros(P.shape[0], device=P.device)
    @torch.no_grad()
    def step(self, P, G, rows):
        """rows: bool mask of rows that take a step."""
        if not rows.any(): return
        r = rows; self.t[r] += 1; t = self.t[r][:, None]
        self.m[r] = self.b1 * self.m[r] + (1 - self.b1) * G[r]
        self.v[r] = self.b2 * self.v[r] + (1 - self.b2) * G[r] * G[r]
        bc1, bc2 = 1 - self.b1 ** t, 1 - self.b2 ** t
        denom = self.v[r].sqrt() / bc2.sqrt() + self.eps
        P[r] -= (self.lr[r][:, None] / bc1) * self.m[r] / denom

def _confirm(bo, rows_k, cand, dom, ctxs, owner, idx_rows):
    """Π + validator (CPU) then one batched no-grad forward for candidates that pass the validator.
    Returns (confirmed {row: delta}, evaluated set of rows). Sequential code: `check and q0(dp) < 0`."""
    to_eval = []
    for r in idx_rows:
        a = ctxs[owner[r]].a; dp = dom.project(a, cand[r])
        if dom.check(a, dp)[0]: to_eval.append((r, dp))
    ok, ev = {}, set()
    if to_eval:
        qs = bo.q_nograd([rows_k[r] for r, _ in to_eval], [dp for _, dp in to_eval], ["A"] * len(to_eval))
        for (r, dp), qv in zip(to_eval, qs):
            ev.add(r)
            if qv < 0: ok[r] = dp
    return ok, ev

def candice_batch(model, g, ks, init_seeds, sp, dom, cfg, n_cf=5, lam1=0.5, lam2=0.5, lr=0.05, iters=500, init_noise=0.05,
                  grad_chunk=512, method="CANDICE-style"):
    """Vectorised mirror of flowsce.candice_style over all queries × n_cf runs (runs are independent)."""
    bo = BatchOracle(model, g, cfg.tau); Q = len(ks); R = Q * n_cf; Fd = sp.F; per = max(1, cfg.max_fwd // n_cf)
    ctxs = [Ctx(int(k), g["xe"][int(k)].cpu().numpy(), bo.thr) for k in ks]
    owner = np.repeat(np.arange(Q), n_cf); rows_k = np.repeat(np.asarray(ks, int), n_cf)
    XI = torch.zeros(R, Fd, device=DEV)
    for qi, s0 in enumerate(init_seeds):                       # identical RNG stream to the sequential code
        gen = torch.Generator(device="cpu").manual_seed(int(s0))
        for r in range(n_cf): XI[qi * n_cf + r] = (torch.randn(Fd, generator=gen) * init_noise).to(DEV) * sp.m
    opt = RowAdam(XI, lr); active = np.ones(R, bool)
    if cfg.box_proj: XLO, XHI = row_box(dom, sp, [ctxs[o].a for o in owner])
    run_f = np.zeros(R, int); run_b = np.zeros(R, int); t_it = np.zeros(R, int); cands = {}
    while active.any():
        An = np.where(active)[0]; G = torch.zeros_like(XI); qv = np.empty(len(An))
        for i in range(0, len(An), grad_chunk):
            sl = torch.as_tensor(An[i:i + grad_chunk], device=DEV); xi = XI[sl].clone().requires_grad_(True)
            q = bo.q(rows_k[An[i:i + grad_chunk]], xi * sp.s * sp.m, ["A"] * len(sl))
            p = torch.sigmoid(q + bo.thr)
            loss = -torch.log(1 - p + 1e-12) + lam1 * torch.sqrt(((xi @ sp.Qxi) * xi).sum(1) + 1e-8) + lam2 * xi.abs().sum(1)
            G[sl] = torch.autograd.grad(loss.sum(), xi)[0]; qv[i:i + len(sl)] = q.detach().cpu().numpy()
        run_f[An] += 1
        neg = An[qv < 0]
        ok, ev = _confirm(bo, rows_k, (XI * sp.s * sp.m).cpu().numpy(), dom, ctxs, owner, neg)
        for r in ev: run_f[r] += 1
        cont = []
        for r in An:
            if r in ok: cands[r] = ok[r]; active[r] = False
            elif t_it[r] == iters or run_f[r] >= per: active[r] = False
            else: cont.append(r)
        if cont:
            mask = torch.zeros(R, dtype=torch.bool, device=DEV); mask[torch.as_tensor(cont, device=DEV)] = True
            opt.step(XI, G, mask)
            with torch.no_grad():
                XI.mul_(sp.m)
                if cfg.box_proj: XI.copy_(torch.where(mask[:, None], torch.maximum(torch.minimum(XI, XHI), XLO), XI))
            run_b[cont] += 1; t_it[cont] += 1
    tasks = []
    for qi, ctx in enumerate(ctxs):
        sl = slice(qi * n_cf, (qi + 1) * n_cf); ctx.nf, ctx.nb = int(run_f[sl].sum()), int(run_b[sl].sum())
        cs = [cands[r] for r in range(qi * n_cf, (qi + 1) * n_cf) if r in cands]
        fi = (np.mean([(c != 0) for c in cs], 0) if cs else np.zeros(sp.F)).tolist()
        best = min(cs, key=lambda c: ((np.abs(c) > cfg.eps_support).sum(), np.abs(c).sum())) if cs else None
        tasks.append((ctx, final_row_co(ctx, dom, sp, best, cfg, method, dict(n_valid_runs=len(cs), feature_importance=fi,
                                                                                 all_cfs=[c.tolist() for c in cs]))))
    return run_coroutines(bo, tasks)

def dice_batch(model, g, ks, init_seeds, sp, dom, cfg, K=4, lam_prox=0.5, lam_div=1.0, lr=0.05, iters=500, init_noise=0.1,
               posthoc=True, grad_chunk_q=128):
    """Vectorised mirror of flowsce.dice_style (K CFs per query optimised jointly; queries independent)."""
    bo = BatchOracle(model, g, cfg.tau); Q = len(ks); Fd = sp.F
    ctxs = [Ctx(int(k), g["xe"][int(k)].cpu().numpy(), bo.thr) for k in ks]
    owner = np.repeat(np.arange(Q), K); rows_k = np.repeat(np.asarray(ks, int), K)
    XI = torch.zeros(Q * K, Fd, device=DEV)
    for qi, s0 in enumerate(init_seeds):
        gen = torch.Generator(device="cpu").manual_seed(int(s0))
        XI[qi * K:(qi + 1) * K] = (torch.randn(K, Fd, generator=gen) * init_noise).to(DEV) * sp.m
    opt = RowAdam(XI, lr); active = np.ones(Q, bool); t_it = np.zeros(Q, int); valid = [dict() for _ in range(Q)]
    if cfg.box_proj: XLO, XHI = row_box(dom, sp, [ctxs[o].a for o in owner])
    nf = np.zeros(Q, int); nb = np.zeros(Q, int); eye = torch.eye(K, device=DEV)
    while active.any():
        Aq = np.where(active)[0]; G = torch.zeros_like(XI); qv = np.empty((len(Aq), K))
        for i in range(0, len(Aq), grad_chunk_q):
            qq = Aq[i:i + grad_chunk_q]; rows = (qq[:, None] * K + np.arange(K)).ravel()
            rt = torch.as_tensor(rows, device=DEV); xi = XI[rt].clone().requires_grad_(True)
            q = bo.q(rows_k[rows], xi * sp.s * sp.m, ["A"] * len(rows)); X3 = xi.view(len(qq), K, Fd)
            Kd = 1 / (1 + torch.cdist(X3, X3, p=1)) + 1e-4 * eye
            loss = F.relu(q.view(len(qq), K) + 0.5).mean(1) + lam_prox * X3.abs().sum(2).mean(1) - lam_div * torch.det(Kd)
            G[rt] = torch.autograd.grad(loss.sum(), xi)[0]; qv[i:i + len(qq)] = q.detach().cpu().numpy().reshape(len(qq), K)
        nf[Aq] += K
        neg = [qi * K + j for n_, qi in enumerate(Aq) for j in range(K) if j not in valid[qi] and qv[n_, j] < 0]
        ok, ev = _confirm(bo, rows_k, (XI * sp.s * sp.m).cpu().numpy(), dom, ctxs, owner, neg)
        for r in sorted(ev): nf[owner[r]] += 1
        for r in sorted(ok): valid[owner[r]][r % K] = ok[r]
        cont = []
        for qi in Aq:
            if len(valid[qi]) == K or t_it[qi] == iters or nf[qi] >= cfg.max_fwd: active[qi] = False
            else: cont.append(qi)
        if cont:
            rows = (np.asarray(cont)[:, None] * K + np.arange(K)).ravel()
            mask = torch.zeros(Q * K, dtype=torch.bool, device=DEV); mask[torch.as_tensor(rows, device=DEV)] = True
            opt.step(XI, G, mask)
            with torch.no_grad():
                XI.mul_(sp.m)
                if cfg.box_proj: XI.copy_(torch.where(mask[:, None], torch.maximum(torch.minimum(XI, XHI), XLO), XI))
            nb[cont] += 1; t_it[cont] += 1
    tasks = []
    for qi, ctx in enumerate(ctxs):
        ctx.nf, ctx.nb = int(nf[qi]), int(nb[qi])
        tasks.append((ctx, _dice_finish_co(ctx, dom, sp, cfg, list(valid[qi].values()), posthoc)))
    return run_coroutines(bo, tasks)

def _dice_finish_co(ctx, dom, sp, cfg, cfs, posthoc):
    pre = [c.copy() for c in cfs]
    if posthoc:
        out = []
        for c in cfs: out.append((yield from dice_posthoc_co(ctx, dom, c, cfg)))
        cfs = out
    div = float(np.mean([np.abs(x - y).sum() for i, x in enumerate(cfs) for y in cfs[i + 1:]])) if len(cfs) > 1 else np.nan
    best = min(cfs, key=lambda c: ((c != 0).sum(), np.abs(c).sum())) if cfs else None
    return (yield from final_row_co(ctx, dom, sp, best, cfg, "DiCE-style" + ("" if posthoc else "-noposthoc"),
            dict(n_valid_cfs=len(cfs), diversity_l1=div,
                 support_before_posthoc=float(np.mean([(c != 0).sum() for c in pre])) if pre else np.nan)))

# ══════════════════════ 4. Batched FlowSCE generation (mirror of flowsce.generate) + diagnostics ══════════════════════
def generate_batch(model, g, ks, sp, dom, cfg, mode="A", grad_chunk=512, box=None, return_raw=False):
    """Algorithm 1 generation for all queries at once (ξ0 = 0). Besides the result, records WHY a search fails:
    neg_hits = iterations where the raw candidate already had q<0, proj_reject = of those, rejected by Π/V_k,
    proj_flip = projected candidate passed V_k but was no longer benign; q_min = best raw score reached."""
    bo = BatchOracle(model, g, cfg.tau); Q = len(ks); ks = np.asarray(ks, int)
    ctxs = [Ctx(int(k), g["xe"][int(k)].cpu().numpy(), bo.thr) for k in ks]
    XI = torch.zeros(Q, sp.F, device=DEV); opt = RowAdam(XI, cfg.lr); active = np.ones(Q, bool)
    nf = np.zeros(Q, int); nb = np.zeros(Q, int); t_it = np.zeros(Q, int); out = [None] * Q
    status = np.array(["budget_stop"] * Q, dtype=object)
    box = cfg.box_proj if box is None else box
    if box: XLO, XHI = row_box(dom, sp, [c.a for c in ctxs])
    neg_hits = np.zeros(Q, int); proj_reject = np.zeros(Q, int); proj_flip = np.zeros(Q, int); q_min = np.full(Q, np.inf)
    while active.any():
        An = np.where(active)[0]; G = torch.zeros_like(XI); qv = np.empty(len(An))
        for i in range(0, len(An), grad_chunk):
            sl = torch.as_tensor(An[i:i + grad_chunk], device=DEV); xi = XI[sl].clone().requires_grad_(True)
            q = bo.q(ks[An[i:i + grad_chunk]], xi * sp.s * sp.m, [mode] * len(sl))
            loss = F.relu(q + cfg.search_margin) + cfg.lam1 * torch.sqrt(((xi @ sp.Qxi) * xi).sum(1) + 1e-8) + cfg.lam2 * xi.abs().sum(1)
            G[sl] = torch.autograd.grad(loss.sum(), xi)[0]; qv[i:i + len(sl)] = q.detach().cpu().numpy()
        nf[An] += 1; q_min[An] = np.minimum(q_min[An], qv)
        D = (XI * sp.s * sp.m).cpu().numpy(); to_eval = []
        for r, qq in zip(An, qv):
            if qq < -cfg.min_margin:
                neg_hits[r] += 1; dp = dom.project(ctxs[r].a, D[r])
                if dom.check(ctxs[r].a, dp)[0]: to_eval.append((r, dp))
                else: proj_reject[r] += 1
        if to_eval:
            q2 = bo.q_nograd(ks[[r for r, _ in to_eval]], [dp for _, dp in to_eval], [mode] * len(to_eval))
            for (r, dp), qq in zip(to_eval, q2):
                nf[r] += 1
                if qq < -cfg.min_margin: out[r] = dp; status[r] = "valid"; active[r] = False
                else: proj_flip[r] += 1
        cont = [r for r in An if active[r] and not (t_it[r] == cfg.T or nf[r] >= cfg.max_fwd)]
        for r in An:
            if active[r] and r not in cont: active[r] = False
        if cont:
            mask = torch.zeros(Q, dtype=torch.bool, device=DEV); mask[torch.as_tensor(cont, device=DEV)] = True
            opt.step(XI, G, mask)
            with torch.no_grad():
                XI.mul_(sp.m)
                if box: XI.copy_(torch.where(mask[:, None], torch.maximum(torch.minimum(XI, XHI), XLO), XI))
            nb[cont] += 1; t_it[cont] += 1
    if return_raw: return out, status, nf, nb, t_it
    for r, c in enumerate(ctxs): c.nf, c.nb = int(nf[r]), int(nb[r])
    rows = run_coroutines(bo, [(c, final_row_co(c, dom, sp, out[r], cfg, "FlowSCE-gen (A0)")) for r, c in enumerate(ctxs)])
    for r, row in enumerate(rows):
        row.update(gen_status=status[r], updates=int(t_it[r]), neg_hits=int(neg_hits[r]), proj_reject=int(proj_reject[r]),
                   proj_flip=int(proj_flip[r]), q_min_raw=float(q_min[r]), q_orig=None)
    return rows


# ══════════════════════ 5. Batched ablation (A0/A/B/C/D) — mirror of flowsce.explain_query ══════════════════════
def _e7_co(ctx, dom, sp, cfg, variant, d, gen_status, updates, q_orig, geo, t_gen):
    mode = {"A0": "A", "A": "A", "B": "B", "C": "C", "D": "A"}[variant]
    if variant == "D":
        if d is None: gen_status = "no_seed"
        else:
            gen_status = "valid"
            if not ((yield (d, "A")) < 0): gen_status = "invalid_seed"; d = None
    rinfo = dict(trials=0, passes=0, stop="none")
    if d is not None and variant in ("A", "B", "C", "D"):
        d, rinfo = yield from restore_co(ctx, dom, d, cfg, mode, repeated=(variant == "D"))
    row = dict(variant=variant, gen_status=gen_status, updates=updates, restore_trials=rinfo["trials"], restore_passes=rinfo["passes"],
               restore_stop=rinfo["stop"], q_orig=q_orig, **geo, t_generate=t_gen, t_restore=np.nan)
    row.update(_empty_assess() if d is None else (yield from assess_co(ctx, dom, sp, d, mode, cfg)))
    row["t_assess"] = np.nan; row["n_fwd"], row["n_bwd"] = ctx.nf, ctx.nb
    return row

def e7_batch(model, g, ks, sp, dom, cfg, chunk=1024):
    """All five variants for all queries of one checkpoint. Call counts follow explain_query exactly
    (q_orig forward + generation + restoration + assessment). Timing columns are batch averages."""
    ks = np.asarray(ks, int); Q = len(ks); bo = BatchOracle(model, g, cfg.tau)
    q0 = bo.q_nograd(ks, np.zeros((Q, sp.F), np.float32), ["A"] * Q)
    geo = []
    for k in ks:
        u, v = g["src"][k], g["dst"][k]; w = torch.zeros(g["N"], device=DEV)
        w = w.index_add(0, torch.stack([u, v]), torch.stack([1 / g["deg"][u], 1 / g["deg"][v]]))
        geo.append(dict(omega=float(w.norm()), deg_u=float(g["deg"][u]), deg_v=float(g["deg"][v])))
    z = torch.zeros(Q, sp.F, device=DEV, requires_grad=True)                   # grad alignment A vs C at δ = 0
    gA = torch.autograd.grad(bo.q(ks, z * sp.m, ["A"] * Q).sum(), z)[0]
    z = torch.zeros(Q, sp.F, device=DEV, requires_grad=True)
    gC = torch.autograd.grad(bo.q(ks, z * sp.m, ["C"] * Q).sum(), z, allow_unused=True)[0]
    gC = torch.zeros_like(gA) if gC is None else gC
    align = [dict(cos_AC=float(F.cosine_similarity(gA[i], gC[i], 0, eps=1e-30)),
                  rel_diff_AC=float((gA[i] - gC[i]).norm() / (gA[i].norm() + 1e-30))) for i in range(Q)]
    search = {}
    for mode in "ABC":
        t = time.time(); search[mode] = generate_batch(model, g, ks, sp, dom, cfg, mode=mode, return_raw=True) + ((time.time() - t) / Q,)
    tasks, meta = [], []
    for i, k in enumerate(ks):
        a = g["xe"][int(k)].cpu().numpy()
        for var in ["A0", "A", "B", "C", "D"]:
            md = {"A0": "A", "A": "A", "B": "B", "C": "C", "D": "A"}[var]
            out, status, nf, nb, t_it, tg = search[md]
            ctx = Ctx(int(k), a, bo.thr)
            if var == "D": ctx.nf, ctx.nb = 1, 0; d0, gs, up = search["A"][0][i], "valid", 0
            else: ctx.nf, ctx.nb = 1 + int(nf[i]), int(nb[i]); d0, gs, up = out[i], status[i], int(t_it[i])
            tasks.append((ctx, _e7_co(ctx, dom, sp, cfg, var, None if d0 is None else d0.copy(), gs, up, float(q0[i]), geo[i], tg)))
            meta.append(align[i])
    rows = run_coroutines(bo, tasks, chunk)
    for r, al in zip(rows, meta): r.update(al)
    return rows

def explain_A_batch(model, g, ks, sp, dom, cfg):
    """Variant A only (search + single restoration + assessment), batched — used for §5 tuning."""
    ks = np.asarray(ks, int); bo = BatchOracle(model, g, cfg.tau)
    out, status, nf, nb, t_it = generate_batch(model, g, ks, sp, dom, cfg, mode="A", return_raw=True)
    tasks = []
    for i, k in enumerate(ks):
        ctx = Ctx(int(k), g["xe"][int(k)].cpu().numpy(), bo.thr); ctx.nf, ctx.nb = 1 + int(nf[i]), int(nb[i])
        tasks.append((ctx, _e7_co(ctx, dom, sp, cfg, "A", out[i], status[i], int(t_it[i]), np.nan, {}, np.nan)))
    return run_coroutines(bo, tasks)
