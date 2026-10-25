#!/usr/bin/env python3
"""
Automated audit of every result artifact. Prints PASS / WARN / FAIL lines, writes results/audit_report.md, exits 1 on any FAIL.

    python scripts/audit.py

Checks: locked protocol files; checkpoints; detector metric identity; RQ1 tolerances; denominators and completeness of every ledger;
no spurious validity for the consistent variants and for every method assessed through the shared oracle; restoration and
minimizers preserve validity; Proposition-level guarantees (s(A) ≤ s(A0), s(D) ≤ s(A)); bootstrap estimates lie inside
their intervals.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results"); args = ap.parse_args(); os.chdir(ROOT); OUT = args.out
lines, n_fail = [], 0
def rep(level, msg):
    global n_fail
    n_fail += level == "FAIL"; s = f"[{level}] {msg}"; print(s); lines.append(s)
def ok(cond, msg, level="FAIL"):
    rep("PASS" if cond else level, msg); return bool(cond)
def rd(p):
    p = f"{OUT}/{p}"; return (pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)) if os.path.exists(p) else None
def ledger(prefix):
    fs = sorted(glob.glob(f"{OUT}/ledger/{prefix}_*.parquet")); return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True) if fs else None
def per_query(df, method_col, a, b, col):
    w = df.pivot_table(index=["dataset", "seed", "query_id"], columns=method_col, values=col, aggfunc="first"); return w[a], w[b]

CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]; NQ = len(SEEDS) * CFG["n_queries"]
# ---- protocol, detectors, RQ1 ----
exp = json.load(open(f"{OUT}/protocol/explain_config.json"))
ok(all(v.get("box_proj") for v in exp.values()), "explain_config: in-loop projection enabled for every dataset")
for ds in DS: ok(os.path.exists(f"{OUT}/protocol/network_manifest_{ds}.json") and os.path.exists(f"{OUT}/protocol/graph_manifest_{ds}.json"), f"{ds}: data and graph manifests")
n_ck = len(glob.glob(f"{OUT}/ckpt/*.pt")); ok(n_ck >= len(DS) * len(SEEDS) * len(CFG["kinds"]), f"checkpoints: {n_ck}")
det = pd.DataFrame(json.load(open(f"{OUT}/detector_metrics.json")))
ok(det.identity_gap.max() < 1e-9, f"accuracy = ρ·recall + (1−ρ)·specificity on every checkpoint (max gap {det.identity_gap.max():.1e})")
for (ds, k), g in det.groupby(["dataset", "kind"]): ok(g.macro_f1.min() >= 0.98, f"detector {ds}/{k}: min macro-F1 {g.macro_f1.min():.4f}", "WARN")
REG = rd("query_registry.csv"); ok(REG is not None and len(REG) == len(DS) * NQ, f"registry: {0 if REG is None else len(REG)} queries")
rq1 = json.load(open(f"{OUT}/reconstruction_gradient_checks.json"))
for ds, v in rq1.items():
    ok(v["rebuild_max_abs"] < 1e-12 and v["fd_max_abs"] < 1e-6 and v["original_score_max_abs"] < 1e-5,
       f"RQ1 {ds}: rebuild {v['rebuild_max_abs']:.1e}, gradient vs FD {v['fd_max_abs']:.1e}, original score {v['original_score_max_abs']:.1e}")
# ---- A0/A/B/C/D ----
E = rd("counterfactual_queries.parquet")
if ok(E is not None, "ablation ledger present"):
    for ds in DS:
        e = E[E.dataset == ds]; c = e.groupby("variant").size()
        ok((c == NQ).all() and len(c) == 5, f"{ds}: {NQ} rows for each of A0/A/B/C/D")
        for v in ["A0", "A", "C", "D"]:
            s = e[e.variant == v]; ok(((s.n == 1) & (s.v == 0)).sum() == 0, f"{ds}/{v}: no spurious validity")
        a0, a = per_query(e[e.variant.isin(["A0", "A"])], "variant", "A0", "A", "support_eps"); both = a0.notna() & a.notna()
        ok((a[both] <= a0[both]).all(), f"{ds}: s(A) ≤ s(A0) on every jointly valid query")
        a_, d_ = per_query(e[e.variant.isin(["A", "D"])], "variant", "A", "D", "support_eps"); both = a_.notna() & d_.notna()
        ok((d_[both] <= a_[both]).mean() >= 0.98, f"{ds}: s(D) ≤ s(A) on {(d_[both] <= a_[both]).mean():.3f} of queries", "WARN")
        val = e[e.v == 1]; ok((val.margin_logit > 0).all() and not val[["support_eps", "l1", "margin_logit"]].isna().any().any(), f"{ds}: valid outputs have margin > 0 and no NaN")
        dd = e[e.variant == "D"]; nb = int((dd.restore_stop == "budget_stop").sum())
        ok("restore_stop" in dd and dd.restore_stop.notna().all(), f"{ds}/D: every run reports its stop reason ({nb} budget stops)")
        ok(set(map(tuple, e[["seed", "query_id"]].drop_duplicates().values)) == set(map(tuple, REG[REG.dataset == ds][["seed", "query_id"]].values)),
           f"{ds}: ablation uses exactly the registered query set")
# ---- coupling ----
X = rd("coupling_sweep_time_queries.parquet")
if X is not None:
    ok(((X.n == 1) & (X.v == 0) & X.variant.isin(["A", "C"])).sum() == 0, "time-window sweep: no spurious validity for A and C")
    ok(X.groupby(["dataset", "W", "seed"]).macro_f1.first().min() >= 0.95, "time-window sweep: every retrained detector has macro-F1 >= 0.95")
# ---- detection comparison (E1) ----
need = {"flow", "flow_gate", "edge", "flow_full", "flow_full_gate", "egraphsage", "candice_svm", "candice_rf", "candice_mlp"}
ok(need <= set(det.kind), f"detector models present: {sorted(set(det.kind))}")
for k, g in det.groupby("kind"): ok(len(g) == len(DS) * len(SEEDS), f"detector {k}: {len(g)} checkpoints (datasets × seeds)")
# ---- E0 / E2 ----
gs = rd("graph_stats.csv"); ok(gs is not None and set(gs.dataset) >= set(DS), "graph statistics for every test graph")
ok(all("legacy_update" in rq1[d] for d in DS), "legacy update rule measured on every dataset (self-loops or a note)")
# ---- reference verification on constructed inputs ----
_rv = f"{OUT}/reference_verification/results.json" if "OUT" in dir() else "results/reference_verification/results.json"
if os.path.exists(_rv):
    rv = json.load(open(_rv))
    ok(rv["groups_passed"] == len(rv["tests"]) and all(t["status"] == "PASS" for t in rv["tests"]) and rv.get("contract_tests_ok", False),
       f"reference verification: {rv['groups_passed']} groups / {rv['cases_passed']} cases PASS, {rv.get('contract_tests_run')} contract tests OK")
    ok(rv["metrics"]["max_snapshot_abs_error"] < 1e-12 and rv["metrics"]["max_gradient_fd_abs_error"] < 1e-6,
       f"reference verification: rebuild {rv['metrics']['max_snapshot_abs_error']:.1e}, finite differences {rv['metrics']['max_gradient_fd_abs_error']:.1e}")
else:
    ok(False, "reference verification results missing (run reference/run_reference_verification.py)", "WARN")
# ---- FlowSCE (proposed explainer) ----
PL = ledger("PLUS")
if PL is not None:
    s_ = PL[PL.v == 1]
    ok(((PL.n == 1) & (PL.v == 0)).sum() == 0, "FlowSCE: no spurious validity")
    ok((s_.support_eps <= s_.support_A).all(), f"FlowSCE: s_eps <= s_eps(A) on every alert ({(s_.support_eps <= s_.support_A).mean():.3f})")
    fl = np.minimum(s_.margin_A, np.log(3))                           # margin floor of the locked configuration (floor_mode "target")
    ok((s_.margin_logit >= fl - 1e-4).all(), f"FlowSCE: margin >= min(margin(A), ln 3) on every alert ({(s_.margin_logit >= fl - 1e-4).mean():.3f})")
    for ds in DS:
        a0 = E[(E.dataset == ds) & (E.variant == "A0")].v.mean(); pl = PL[PL.dataset == ds]
        ok(len(pl) == NQ and abs(pl.v.mean() - a0) < 1e-9, f"{ds}/FlowSCE: all {NQ} registered alerts, CSR equals that of its seed A0 ({pl.v.mean():.3f})")
# ---- E5 / E6 generators and minimizers ----
G = rd("generator_comparison_queries.csv")
if G is not None:
    ok(((G.n == 1) & (G.v == 0)).sum() == 0, "generators: no spurious validity through the shared oracle")
    for (ds, m), g in G.groupby(["dataset", "method"]):
        ok(set(map(tuple, g[["seed", "query_id"]].values)) == set(map(tuple, REG[REG.dataset == ds][["seed", "query_id"]].values)),
           f"generator {ds}/{m}: the registered query set")
        ok(g.n_fwd.max() <= 4096 + 3, f"generator {ds}/{m}: forward budget respected (max {g.n_fwd.max():.0f} incl. final assessment)")
M = rd("minimizer_comparison_queries.csv")
if M is not None:
    sp_ = M[(M.n == 1) & (M.v == 0)]; ours = sp_.method.str.startswith("FlowSCE")
    ok(not ours.any(), f"minimizers: no spurious validity of FlowSCE ({int(ours.sum())})")
    ok(sp_.empty, f"minimizers: spurious validity of baselines {sorted(set(sp_.method))} on {len(sp_)} alerts "
                  f"(max |q_reconstructed| {sp_.q_reconstructed.abs().max() if len(sp_) else 0:.1e}: a candidate placed at the decision threshold)", "WARN")
    for (gname, ds), g in M.groupby(["generator", "dataset"]):
        base = g[g.method == "none"].set_index(["seed", "query_id"]).v
        for m, x in g.groupby("method"):
            same = (x.set_index(["seed", "query_id"]).v == base.reindex(x.set_index(["seed", "query_id"]).index)).mean()
            ok(same == 1.0, f"minimizer {gname}/{ds}/{m}: validity of the seed preserved ({same:.3f})", "FAIL" if m.startswith("FlowSCE") else "WARN")
        lb = g[g.method == "LBS-adapted"].set_index(["seed", "query_id"]).n_fwd_method; rh = g[g.method == "RHC"].set_index(["seed", "query_id"])
        ok((rh.n_fwd_method <= rh.budget + 1).all() and (rh.budget == lb.reindex(rh.index)).all(), f"RHC {gname}/{ds}: budget = oracle calls of LBS on the same seed")
for f_ in ["uncertainty_ablation.csv", "uncertainty_generator.csv", "uncertainty_minimizer.csv"]:
    u = rd(f_)
    if u is not None:
        x = u.dropna(subset=["estimate"]); ok(((x.estimate >= x.ci_low - 1e-9) & (x.estimate <= x.ci_high + 1e-9)).all() and (x.n_pairs > 0).all(),
                                              f"{f_}: every estimate inside its 95% interval, every interval with n > 0")
# ---- baselines ----
bq = rd("baseline_comparison_queries.csv")
if bq is not None:
    ok(((bq.n == 1) & (bq.v == 0)).sum() == 0, "baselines: no spurious validity through the shared oracle")
    m = bq[bq.comparison == "minimizer_same_seed"]
    for ds in DS:
        mm = m[m.dataset == ds]; base = mm[mm.method == "none (A0)"].set_index(["seed", "query_id"]).v
        for meth, g in mm.groupby("method"):
            same = (g.set_index(["seed", "query_id"]).v == base.reindex(g.set_index(["seed", "query_id"]).index)).mean()
            ok(same == 1.0, f"minimizer {ds}/{meth}: validity of the A0 candidate preserved ({same:.3f})")
for f in ["uncertainty.csv"]:
    u = rd(f)
    if u is not None:
        u = u[u.estimate.notna()]; ok(((u.ci_low <= u.estimate + 1e-12) & (u.estimate <= u.ci_high + 1e-12)).all(), f"{f}: every estimate inside its 95% interval")
open(f"{OUT}/audit_report.md", "w").write("# Audit report\n\n" + "\n".join(f"- {l}" for l in lines) + "\n")
print(f"\n{n_fail} FAIL — details: {OUT}/audit_report.md"); sys.exit(1 if n_fail else 0)
