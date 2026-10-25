#!/usr/bin/env python3
"""
Runtime scaling (RQ5), one-factor-at-a-time sensitivity and retrievable case studies. Requires scripts/run_main.py.

    python scripts/run_analysis.py            # each part is skipped when its output exists (--force recomputes)
    python scripts/run_analysis.py --parts sensitivity cases   # runtime separately, on an otherwise idle GPU

Outputs: results/runtime_scaling.csv, results/sensitivity.csv, results/case_studies.md
  runtime      sequential variant A, seed-11 checkpoint, 20 alerts per cell; test graphs built from 25/50/100% of the test time blocks
               × the first 25/50/100% of the mutable features; one warm-up query excluded; GPU-synchronised timing.
  sensitivity  variant A on validation alerts of the seed-11 checkpoint; T, λ1, λ2, κ, the minimum acceptance margin m and the
               support tolerance ε varied one at a time.
  case studies spurious validity of B, restoration that keeps the label but lowers the margin, a failure, and a FlowSCE explanation
               next to the search output it starts from (from the query ledgers).
"""
import argparse, glob, json, os, resource, sys, time
from dataclasses import asdict
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--force", action="store_true")
ap.add_argument("--parts", nargs="+", default=["runtime", "sensitivity", "cases"], choices=["runtime", "sensitivity", "cases"]); args = ap.parse_args(); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L                      # noqa: E402

OUT = args.out; CFG = json.load(open(f"{OUT}/protocol/config.json")); DATASETS, MAIN = CFG["datasets"], CFG["main"]
done = lambda p: os.path.exists(p) and not args.force
EXP = {ds: L.ExplainCfg(**v) for ds, v in json.load(open(f"{OUT}/protocol/explain_config.json")).items()}
S = {ds: L.prepare_dataset(ds, args.data_dir, OUT, write=False) for ds in DATASETS}
SP = {ds: L.Space(S[ds]["X"]["train"], S[ds]["dom"]) for ds in DATASETS}
POOLS = {ds: L.BenignPool(S[ds]["X"]["train"], S[ds]["D"]["train"]["_y"].values, S[ds]["D"]["train"]["row_id"].values) for ds in DATASETS}
detector = lambda ds, seed: L.load_detector(f"{OUT}/ckpt/{ds}_{MAIN}_{seed}.pt", S[ds]["X"]["train"].shape[1])

# ---- runtime scaling ----
rt_path = f"{OUT}/runtime_scaling.csv"
if "runtime" in args.parts and not done(rt_path):
    rows = []
    for ds in DATASETS:
        L.POOL = POOLS[ds]; st, sp = S[ds], SP[ds]; m = detector(ds, 11); Dt, Xt = st["D"]["test"], st["X"]["test"]
        blocks = sorted(Dt["_block"].unique(), key=lambda b: int(b.split("-")[1])); mut_idx = np.where(st["dom"].mutable)[0]
        for frac in [.25, .5, 1.]:
            keep = np.isin(Dt["_block"].values, blocks[::int(round(1 / frac))])          # evenly spaced time blocks
            g = L.build_graph(Dt[keep], Xt[keep], ds); al = np.where(L.predict(m, g) >= .5)[0]
            if len(al) == 0: continue
            ks = np.random.default_rng(0).choice(al, min(20, len(al)), replace=False)
            for mf in [.25, .5, 1.]:
                sp2, dom2 = L.subset_space(sp, st["dom"], mut_idx[: max(1, int(round(mf * len(mut_idx))))])
                L.explain_query(m, g, ks[0], sp2, dom2, EXP[ds], "A")                    # warm-up, not timed
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                for k in ks:
                    L.sync(); t0 = time.perf_counter(); r = L.explain_query(m, g, k, sp2, dom2, EXP[ds], "A"); L.sync()
                    rows.append(dict(dataset=ds, frac_flows=frac, N=g["N"], M=len(g["src"]), F_mut=int(dom2.mutable.sum()),
                                     latency_s=time.perf_counter() - t0, t_generate=r["t_generate"], t_restore=r["t_restore"], t_assess=r["t_assess"],
                                     n_fwd=r["n_fwd"], n_bwd=r["n_bwd"], updates=r["updates"], v=r["v"],
                                     gpu_peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None,
                                     cpu_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
        print("runtime", ds, "done", flush=True)
    pd.DataFrame(rows).to_csv(rt_path, index=False)

# ---- sensitivity (validation alerts only) ----
se_path = f"{OUT}/sensitivity.csv"
if "sensitivity" in args.parts and not done(se_path):
    rows = []
    for ds in DATASETS:
        L.POOL = POOLS[ds]; st, sp = S[ds], SP[ds]; m = detector(ds, 11); gv = st["G"]["val"]; base = EXP[ds]
        al = np.where(L.predict(m, gv) >= .5)[0]; qv = np.random.default_rng(1).choice(al, min(CFG["n_val_queries"], len(al)), replace=False)
        factors = dict(T=[50, 150, 500], lam1=[0., base.lam1, 10 * base.lam1], lam2=[0., base.lam2, 10 * base.lam2], search_margin=[0., base.search_margin, 2 * base.search_margin, 4 * base.search_margin],
                       min_margin=[0., .5, float(np.log(3))])           # minimum acceptance margin m in search and restoration
        for f, vals in factors.items():
            for val in vals:
                cfg = L.ExplainCfg(**{**asdict(base), f: val})
                R = pd.DataFrame([L.explain_query(m, gv, k, sp, st["dom"], cfg, "A") for k in qv])
                rows.append(dict(dataset=ds, factor=f, value=val, **L.rates(R), **L.quality(R)))
        R = pd.DataFrame([L.explain_query(m, gv, k, sp, st["dom"], base, "A") for k in qv]); s = R[R.v == 1]
        for eps in [1e-8, 1e-4, 1e-2]:
            rows.append(dict(dataset=ds, factor="eps_support", value=eps, **L.rates(R),
                             support_eps_median=float(np.median([(np.abs(np.array(d)) > eps).sum() for d in s.delta])) if len(s) else None))
        print("sensitivity", ds, "done", flush=True)
    pd.DataFrame(rows).to_csv(se_path, index=False)

# ---- case studies from the query ledger ----
if "cases" not in args.parts: sys.exit(0)
E7 = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
from flowsce.units import unit_of
def fmt(x):
    if isinstance(x, (float, np.floating)):
        if abs(x) < 1e-6: return "0"
        return f"{x:.0f}" if abs(x - round(x)) < 1e-6 and abs(x) < 1e12 else f"{x:.4g}"
    return str(x)
def md_table(df):
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)] +
                     ["| " + " | ".join(fmt(x) for x in r) + " |" for r in df.itertuples(index=False)])
def raw_table(ds, a, delta, feats=None):
    dom = S[ds]["dom"]; d = np.zeros_like(a) if delta is None else np.array(delta)
    ra, rc = dom.to_raw(a), dom.to_raw(a + d); ch = np.nonzero(np.abs(d) > 1e-8)[0] if feats is None else feats
    return pd.DataFrame(dict(feature=[dom.names[j] for j in ch], unit=[unit_of(ds, dom.names[j])[0] for j in ch],
                             raw_original=ra[ch], raw_counterfactual=rc[ch] if delta is not None else ["–"] * len(ch),
                             processed_original=a[ch], processed_counterfactual=(a + d)[ch] if delta is not None else ["–"] * len(ch),
                             transform=["log1p+std" if dom.is_log[j] else "std" for j in ch]))
md = ["# Case studies (generated from the query ledgers)\n",
      "Model-input contrasts in raw feature units (rounded for display), not validated network remediation. Coupled statistics are not checked.\n"]
TW = pd.read_parquet(f"{OUT}/coupling_sweep_time_queries.parquet") if os.path.exists(f"{OUT}/coupling_sweep_time_queries.parquet") else None
for ds in DATASETS:
    E = E7[E7.dataset == ds]; md.append(f"\n## {ds}\n"); xe = lambda k: S[ds]["G"]["test"]["xe"][int(k)].cpu().numpy()
    spur = E[(E.variant == "B") & (E.n == 1) & (E.v == 0)]
    md.append("### (i) Spurious validity (B)\n" + ("None observed on the full snapshot: every nominal B success was confirmed by reconstruction.\n" if spur.empty else ""))
    if not spur.empty:
        r = spur.iloc[0]; md += [f"query {r.query_id} (row {r.row_id}, seed {r.seed}): nominal (stale) score q={r.q_nominal:.4f}, reconstructed score q={r.q_reconstructed:.4f}\n",
                                 md_table(raw_table(ds, xe(r.query_id), r.delta)) + "\n"]
    a0 = E[E.variant == "A0"].set_index(["seed", "query_id"]); a1 = E[E.variant == "A"].set_index(["seed", "query_id"])
    j = a0.join(a1, lsuffix="_A0", rsuffix="_A").query("v_A0 == 1 and v_A == 1")
    if not j.empty:
        j["margin_drop"] = j.margin_logit_A0 - j.margin_logit_A; (seed, k), r = next(iter(j.sort_values("margin_drop", ascending=False).iterrows()))
        md += [f"### (ii) Restoration keeps the label but lowers the margin\nquery {k} (seed {seed}): support {r.support_eps_A0:.0f} -> {r.support_eps_A:.0f}, "
               f"logit margin {r.margin_logit_A0:.3f} -> {r.margin_logit_A:.3f}; reconstructed score q={r.q_reconstructed_A:.4f} (nominal q={r.q_nominal_A:.4f})\n",
               md_table(raw_table(ds, xe(k), r.delta_A)) + "\n"]
    pl = sorted(glob.glob(f"{OUT}/ledger/PLUS_{ds}_*.parquet"))
    if pl:
        P = pd.concat([pd.read_parquet(x) for x in pl]).set_index(["seed", "query_id"]); jj = P.join(a0[["support_eps", "l1", "margin_logit"]], rsuffix="_A0")
        jj = jj[(jj.v == 1) & (jj.support_eps_A0 > 0)]
        if len(jj):                                           # the alert whose FlowSCE support equals the median, largest A0 support
            med = jj.support_eps.median(); c = jj[jj.support_eps == jj.support_eps.iloc[(jj.support_eps - med).abs().argmin()]]
            (seed, k), r = next(iter(c.sort_values(["support_eps_A0", "l1"], ascending=[False, True]).iterrows()))
            md += [f"### (iii) FlowSCE explanation\nquery {k} (seed {seed}, {r.alert_type}): search output (A0) support {r.support_eps_A0:.0f}, "
                   f"L1 {r.l1_A0:.2f}, logit margin {r.margin_logit_A0:.3f} -> FlowSCE support {r.support_eps:.0f}, L1 {r.l1:.2f}, logit margin {r.margin_logit:.3f}; "
                   f"reconstructed score q={r.q_reconstructed:.4f}; {r.n_fwd_method:.0f} oracle calls\n", md_table(raw_table(ds, xe(k), r.delta)) + "\n"]
    f = E[(E.variant == "A") & (E.v == 0)]
    md.append("### (iv) Failure\n")
    if len(f):
        r = f.iloc[0]; md.append(f"full snapshot, query {r.query_id} ({r.alert_type}), stop={r.gen_status}, original q={r.q_orig:.3f}\n")
    elif TW is not None and len(TW[(TW.dataset == ds) & (TW.variant == "A") & (TW.v == 0)]):
        t = TW[(TW.dataset == ds) & (TW.variant == "A") & (TW.v == 0)].sort_values(["W", "seed", "query_id"]).iloc[0]
        dom = S[ds]["dom"]; a = S[ds]["X"]["test"][int(t.query_id)]; mut = np.nonzero(dom.mutable)[0]
        md += [f"No failure among the registered alerts of the full snapshot. Example from the time-window sweep (Δt = {t.W:g} s, detector seed {t.seed}, "
               f"retrained for that policy): query {t.query_id} ({'true' if t.y_true == 1 else 'false'} alert), stop={t.gen_status}, original score q={t.q_orig:.3f}, "
               f"coupling ω={t.omega:.3f}; no candidate passed the reconstructed oracle and the validator within the budget "
               f"({t.updates:.0f} updates, {t.n_fwd:.0f} forward evaluations). Mutable features of the alert:\n",
               md_table(raw_table(ds, a, None, mut)) + "\n"]
    else:
        rt = pd.read_csv(f"{OUT}/runtime_scaling.csv") if os.path.exists(f"{OUT}/runtime_scaling.csv") else None
        x = rt[(rt.dataset == ds) & (rt.v == 0)] if rt is not None else []
        md.append("No failure among the registered alerts or in the time-window sweep." + (
            f" Failures occur only when the mutable set is reduced (runtime study, Table T8): {len(x)} of "
            f"{len(rt[(rt.dataset == ds) & (rt.F_mut < rt[rt.dataset == ds].F_mut.max())])} timed searches with a reduced mutable set "
            f"exhaust the {int(x.updates.max())}-update budget." if len(x) else "") + "\n")
open(f"{OUT}/case_studies.md", "w").write("".join(md)); print("case studies written")
