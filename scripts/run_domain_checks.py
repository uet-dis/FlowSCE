#!/usr/bin/env python3
"""
E10 — byte-duration-rate relations (diagnostic only; the validator does not enforce them). Table A5.

    python scripts/run_domain_checks.py

Step 1, on the raw subsets: does the documented relation hold on every flow (relative tolerance 1%, zero-duration flows
excluded)?  DoHBrw-2020: FlowSentRate = FlowBytesSent / Duration. UNSW-NB15: Sload = 8 * sbytes / dur (bits/s).
CIC-IDS2018 (NetFlow v3): SRC_TO_DST_AVG_THROUGHPUT = 8 * IN_BYTES / duration, with the duration
FLOW_DURATION_MILLISECONDS in seconds (the duration is not a model feature, so this relation cannot be checked on
counterfactuals even where it holds).
Step 2, only where the relation holds on every flow and all its fields are features: the share of valid counterfactuals of
FlowSCE, FlowSCE (A), CANDICE (paper configuration), DiCE (dice-ml) and A0 + LBS-adapted that violate it, next to the share among the
original alerts. Output: results/domain_checks.csv.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L
OUT = "results"; TOL = 0.01
REL = {"doh": dict(rate="FlowSentRate", num="FlowBytesSent", dur="Duration", factor=1.0, dur_scale=1.0),
       "unsw": dict(rate="Sload", num="sbytes", dur="dur", factor=8.0, dur_scale=1.0),
       "cic": dict(rate="SRC_TO_DST_AVG_THROUGHPUT", num="IN_BYTES", dur=None, factor=8.0, dur_scale=1e-3)}

def holds(rate, num, dur, factor):
    ok = dur > 0; pred = factor * num[ok] / dur[ok]
    return ok, np.abs(rate[ok] - pred) <= TOL * np.maximum(np.abs(pred), 1e-9)

rows = []
E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
GB = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/GENB_*.parquet"))])
DC = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/DICE_*.parquet"))])
MN = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/MIN2_*.parquet"))])
PL = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet"))])
for ds, r in REL.items():
    df = L.load(ds, "data")
    dur = df["FLOW_DURATION_MILLISECONDS"].values.astype(float) * r["dur_scale"] if r["dur"] is None else df[r["dur"]].values.astype(float)
    ok, h = holds(df[r["rate"]].values.astype(float), df[r["num"]].values.astype(float), dur, r["factor"])
    share = float(h.mean()); rows.append(dict(dataset=ds, step="raw data", method="—", relation=f"{r['rate']} = {r['factor']:g}*{r['num']}/duration",
                                              n=int(ok.sum()), share_holding=share, share_violating=1 - share))
    st = L.prepare_dataset(ds, "data", OUT, write=False); dom = st["dom"]; names = list(dom.names)
    fields = [r["rate"], r["num"]] + ([r["dur"]] if r["dur"] else [])
    if share < 1.0 or not all(f in names for f in fields):
        why = "relation does not hold on every flow of the raw data" if share < 1.0 else "a field of the relation is not a model feature"
        rows.append(dict(dataset=ds, step="counterfactuals", method="all", relation="skipped", n=0, note=why)); continue
    jr, jn, jd = names.index(r["rate"]), names.index(r["num"]), names.index(r["dur"])
    def viol(a, d):
        x = dom.to_raw(np.asarray(a, float) + np.asarray(d, float)); return not holds(np.array([x[jr]]), np.array([x[jn]]), np.array([x[jd]]), r["factor"])[1].all()
    X = st["X"]["test"]
    srcs = {"original alerts": E[(E.dataset == ds) & (E.variant == "A")].assign(delta=lambda t: [np.zeros(X.shape[1])] * len(t), v=1),
            "FlowSCE": PL[PL.dataset == ds], "FlowSCE (A)": E[(E.dataset == ds) & (E.variant == "A")], "CANDICE": GB[(GB.dataset == ds) & (GB.method == "CANDICE")],
            "DiCE": DC[DC.dataset == ds], "A0 + LBS-adapted": MN[(MN.dataset == ds) & (MN.generator == "FlowSCE (A0)") & (MN.method == "LBS-adapted")]}
    for meth, T in srcs.items():
        T = T[T.v == 1]; vs = [viol(X[int(k)], d) for k, d in zip(T.query_id, T.delta)]
        rows.append(dict(dataset=ds, step="counterfactuals", method=meth, relation=f"{r['rate']} = {r['num']}/{r['dur']}", n=len(vs),
                         share_violating=float(np.mean(vs)) if vs else np.nan))
pd.DataFrame(rows).to_csv(f"{OUT}/domain_checks.csv", index=False); print(pd.DataFrame(rows).to_string())
