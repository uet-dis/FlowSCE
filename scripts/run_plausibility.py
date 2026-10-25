#!/usr/bin/env python3
"""
Plausibility analysis of the stored counterfactuals and a case table (no new explanations are computed).

    python scripts/run_plausibility.py

  reference   median distance to the five nearest benign training flows of (a) the registered alerts themselves and
              (b) 500 benign test flows, as a scale for the kNN column of the quality tables
  percentile  for every changed feature of a valid counterfactual: the percentile of its raw counterfactual value in the
              training distribution of that feature, and whether it lies inside the central 98% (1st-99th percentile)
  features    the three features most often changed by FlowSCE (share of its valid explanations)
  cases       per dataset, one true alert and one false alert explained by FlowSCE (the alert whose support equals the
              median, largest search support first), with raw values and units
Outputs: results/plausibility.csv, results/case_table.csv
"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L                                                                # noqa: E402
from flowsce.units import unit_of                                                  # noqa: E402
OUT = "results"; CFG = json.load(open(f"{OUT}/protocol/config.json"))
REG = pd.read_csv(f"{OUT}/query_registry.csv"); E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
PL = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet"))], ignore_index=True)
MN = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/MIN2_*.parquet"))], ignore_index=True)
GB = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/GENB_*.parquet"))], ignore_index=True)
rows, cases = [], []
for ds in CFG["datasets"]:
    st = L.prepare_dataset(ds, "data", OUT, write=False); dom = st["dom"]; Xtr, Xte = st["X"]["train"], st["X"]["test"]
    pool = L.BenignPool(Xtr, st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    xe = st["G"]["test"]["xe"].cpu().numpy(); yte = st["D"]["test"]["_y"].values
    knn = lambda x: float(pool.dist(x))
    al = REG[REG.dataset == ds].query_id.unique(); ben = np.random.default_rng(0).choice(np.where(yte == 0)[0], 500, replace=False)
    rows.append(dict(dataset=ds, kind="reference", method="registered alerts (original)", knn_median=float(np.median([knn(xe[k]) for k in al]))))
    rows.append(dict(dataset=ds, kind="reference", method="benign test flows", knn_median=float(np.median([knn(xe[k]) for k in ben]))))
    raw_tr = dom.to_raw(Xtr)                                                        # training distribution in raw units
    srcs = {"FlowSCE": PL[PL.dataset == ds], "LBS-adapted": MN[(MN.dataset == ds) & (MN.generator == "FlowSCE (A0)") & (MN.method == "LBS-adapted")],
            "CANDICE (tuned λ)": GB[(GB.dataset == ds) & (GB.method == "CANDICE (tuned λ)")]}
    for meth, T in srcs.items():
        pc, inside = [], []
        for r in T[T.v == 1].itertuples():
            d = np.array(r.delta); a = xe[int(r.query_id)]; ch = np.nonzero(np.abs(d) > 1e-8)[0]
            if not len(ch): continue
            rc = dom.to_raw((a + d)[None])[0]
            for j in ch:
                p = float((raw_tr[:, j] <= rc[j]).mean()); pc.append(p); inside.append(0.01 <= p <= 0.99)
        rows.append(dict(dataset=ds, kind="percentile", method=meth, changed_values=len(pc), share_inside_p1_p99=float(np.mean(inside)) if pc else np.nan,
                         share_at_extremes=float(np.mean([(p <= 0.001) or (p >= 0.999) for p in pc])) if pc else np.nan))
    cnt = {}                                                                         # which features FlowSCE changes
    pv = PL[(PL.dataset == ds) & (PL.v == 1)]
    for d in pv.delta:
        for j in np.nonzero(np.abs(np.array(d)) > 1e-8)[0]: cnt[dom.names[j]] = cnt.get(dom.names[j], 0) + 1
    for fn, c in sorted(cnt.items(), key=lambda t: -t[1])[:3]:
        rows.append(dict(dataset=ds, kind="feature_share", method=fn, share_of_explanations=c / len(pv), distinct_features=len(cnt)))
    a0 = E[(E.dataset == ds) & (E.variant == "A0")].set_index(["seed", "query_id"])
    P = PL[(PL.dataset == ds) & (PL.v == 1)].set_index(["seed", "query_id"]).join(a0[["support_eps"]], rsuffix="_A0")
    for at in ["true_alert", "false_alert"]:
        x = P[P.alert_type == at]
        if x.empty: continue
        med = x.support_eps.median(); c = x[x.support_eps == x.support_eps.iloc[(x.support_eps - med).abs().argmin()]]
        (seed, k), r = next(iter(c.sort_values(["support_eps_A0", "l1"], ascending=[False, True]).iterrows()))
        d = np.array(r.delta); a = xe[int(k)]; ch = np.nonzero(np.abs(d) > 1e-8)[0]
        ra, rc = dom.to_raw(a[None])[0], dom.to_raw((a + d)[None])[0]
        lab = st["D"]["test"].iloc[int(k)]
        atk = str(lab.get("attack_type", lab.get("_y")))
        for j in ch:
            cases.append(dict(dataset=ds, alert=at.replace("_alert", ""), seed=int(seed), query_id=int(k), attack_type=atk, feature=dom.names[j],
                              unit=unit_of(ds, dom.names[j])[0], raw_original=float(ra[j]), raw_counterfactual=float(rc[j]),
                              support_A0=float(r.support_eps_A0), support=float(r.support_eps), margin=float(r.margin_logit), calls=float(r.n_fwd_method)))
    print(ds, "done", flush=True)
pd.DataFrame(rows).to_csv(f"{OUT}/plausibility.csv", index=False); pd.DataFrame(cases).to_csv(f"{OUT}/case_table.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(cases).to_string())
