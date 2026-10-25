#!/usr/bin/env python3
"""
Equivalence of the sequential reference (flowsce.minimizer_from_seed, QueryOracle) and the batched path
(flowsce.fast.minimizers_batch, BatchOracle) for the minimizers of the comparison study, on registered alerts:
validity must agree; forward counts may differ only through floating-point noise near the decision boundary.

    python tests/test_minimizers_equivalence.py --n 5
"""
import argparse, json, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L
from flowsce import fast as FF

ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=5); ap.add_argument("--seed", type=int, default=11)
args = ap.parse_args()
E = pd.read_parquet("results/counterfactual_queries.parquet"); REG = pd.read_csv("results/query_registry.csv")
METHODS = ["FlowSCE single (A)", "FlowSCE repeated (D)", "DiCE post-hoc", "LBS-adapted", "RHC"]
bad, rows = 0, []
for ds in ["cic", "doh", "unsw"]:
    st = L.prepare_dataset(ds, "data", "results", write=False); sp = L.Space(st["X"]["train"], st["dom"])
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    cfg = L.ExplainCfg(**json.load(open("results/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"results/ckpt/{ds}_flow_{args.seed}.pt", st["X"]["train"].shape[1]); g = st["G"]["test"]
    Q = REG[(REG.dataset == ds) & (REG.seed == args.seed)].head(args.n); ks = Q.query_id.values.astype(int); init = Q.init_seed.values.astype(int)
    a0 = E[(E.dataset == ds) & (E.seed == args.seed) & (E.variant == "A0")].set_index("query_id")
    sd = [np.array(a0.loc[k, "delta"], np.float32) if a0.loc[k, "v"] == 1 else None for k in ks]
    dml = FF.dice_posthoc_params(st["prep"], st["D"]["train"]); budgets = {"RHC": 60}
    fast = FF.minimizers_batch(m, g, ks, sd, init, sp, st["dom"], cfg, METHODS, budgets=budgets, dml=dml)
    for i, (k, s, s0) in enumerate(zip(ks, sd, init)):
        for j, meth in enumerate(METHODS):
            seq = L.minimizer_from_seed(m, g, int(k), sp, st["dom"], cfg, s, meth, int(s0), budget=budgets.get(meth), dml=dml)
            fb = fast[i * len(METHODS) + j]
            same = seq["v"] == fb["v"]; bad += (not same)
            rows.append(dict(dataset=ds, query=int(k), method=meth, v_seq=seq["v"], v_fast=fb["v"], nf_seq=seq["n_fwd_method"],
                             nf_fast=fb["n_fwd_method"], s_seq=seq["support_eps"], s_fast=fb["support_eps"]))
R = pd.DataFrame(rows); print(R.to_string())
print(f"validity disagreements: {bad} / {len(R)}; n_fwd equal on {(R.nf_seq == R.nf_fast).mean():.3f}; support equal on {(R.s_seq == R.s_fast).mean():.3f}")
sys.exit(1 if bad else 0)
