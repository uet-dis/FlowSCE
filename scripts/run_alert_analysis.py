#!/usr/bin/env python3
"""
Per-alert analyses of stored results (no new explanations):

    python scripts/run_alert_analysis.py

  coupling   spurious validity of the stale-context variant B among its nominal successes in all wall-clock window policies
             (subsets and complete DoHBrw-2020 capture), by the coupling omega of the alert and by B's stale logit margin
  windows    per window policy: CSR of B after the reconstructed check (stale search + final check), the A-C CSR difference with
             a Bonferroni-adjusted paired bootstrap interval over the 15 window policies (alerts resampled within checkpoints), and the empirical ratio
             |q_rec - q_stale| / (omega * ||delta||_2) of B's candidates (a data estimate of L_V in Proposition 2)
  floor      share of FlowSCE explanations whose margin floor min(mu_A, ln 3) lies below ln 3
  alerts     whether the FlowSCE explanation separates false from true alerts: AUROC of the L1 deviation, the restoration
             margin and the search support, next to the AUROC of the detector score itself (datasets with >= 20 false alerts)
Output: results/alert_analysis.csv
"""
import glob, os, sys
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); OUT = "results"
X = pd.concat([pd.read_parquet(f"{OUT}/coupling_sweep_time_queries.parquet"), pd.read_parquet(f"{OUT}/coupling_fullcapture_doh_queries.parquet")],
              ignore_index=True)
B = X[(X.variant == "B") & (X.n == 1)].assign(spur=lambda d: (d.v == 0).astype(int), stale_margin=lambda d: -d.q_nominal)
rows = []
for lab, col, edges in [("coupling", "omega", [0, 0.01, 0.1, 0.5, np.inf]), ("stale_margin", "stale_margin", [-np.inf, 0.05, 0.25, 1.0, np.inf])]:
    b = pd.cut(B[col], edges, right=True)
    for iv, g in B.groupby(b, observed=True):
        rows.append(dict(analysis=lab, lo=iv.left, hi=iv.right, nominal=len(g), spurious=int(g.spur.sum()), rate=float(g.spur.mean())))
for (ds, iv), g in B.groupby(["dataset", pd.cut(B.omega, [0, 0.01, 0.1, 0.5, np.inf])], observed=True):   # stratified by dataset
    rows.append(dict(analysis="coupling_by_dataset", dataset=ds, lo=iv.left, hi=iv.right, nominal=len(g), spurious=int(g.spur.sum()), rate=float(g.spur.mean())))
rng = np.random.default_rng(0); NW = X.groupby(["dataset", "W"]).ngroups
for (ds, W), g in X.groupby(["dataset", "W"]):
    a_, c_ = (g[g.variant == v].set_index(["seed", "query_id"]) for v in "AC")
    j = a_.index.intersection(c_.index); d = (a_.loc[j, "v"] - c_.loc[j, "v"]).rename("d").reset_index()
    groups = {s_: sub.d.values for s_, sub in d.groupby("seed")}           # alerts resampled within each checkpoint (no block id stored)
    boots = [np.mean([v[rng.integers(0, len(v), len(v))].mean() for v in groups.values()]) for _ in range(2000)]
    lo, hi = np.percentile(boots, [100 * 0.025 / NW, 100 * (1 - 0.025 / NW)])
    b_ = g[(g.variant == "B") & (g.n == 1)]; ratio = (b_.disc_stale / (b_.omega * b_.l2)).replace([np.inf, -np.inf], np.nan).dropna()
    rows.append(dict(analysis="window", dataset=ds, W=W, csr_B_checked=float((g[g.variant == "B"].v).mean()), csr_A=float(a_.v.mean()),
                     dAC=float(d.d.mean()), bonf_lo=float(lo), bonf_hi=float(hi), LV_ratio_median=float(ratio.median()) if len(ratio) else np.nan))
PL = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet"))], ignore_index=True)
for ds, g in PL[PL.v == 1].groupby("dataset"):
    rows.append(dict(analysis="floor", dataset=ds, share_floor_below_ln3=float((g.margin_A < np.log(3)).mean())))
    y = (g.alert_type == "false_alert").astype(int)
    if y.sum() < 20: continue
    for c in ["l1", "margin_A", "support_A", "q_orig"]:
        sc = -g[c] if c == "q_orig" else g[c]                                       # lower detector score -> false alert
        a = float(roc_auc_score(y, sc)); rows.append(dict(analysis="false_vs_true", dataset=ds, score=c, n_false=int(y.sum()), auroc=max(a, 1 - a),
                                                          direction="higher for false alerts" if a >= 0.5 else "lower for false alerts"))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/alert_analysis.csv", index=False); pd.set_option("display.width", 200); print(R.round(3).to_string())
