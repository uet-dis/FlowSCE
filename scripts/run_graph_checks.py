#!/usr/bin/env python3
"""
E0 + E2 (supplement): structure of the test graphs and a naive self-loop update rule.

    python scripts/run_graph_checks.py

E0 -> results/graph_stats.csv: hosts, flows, self-loops, parallel-flow pairs, endpoint-degree distribution per test graph.
E2 -> block "legacy_update" in results/reconstruction_gradient_checks.json: on every self-loop of each test graph, the legacy
update (x[src] = x0[src] + δ/cnt; x[dst] = x0[dst] + δ/cnt, so a self-loop gets δ/cnt instead of 2δ/cnt) is compared with an
independent rebuild, for (i) a random perturbation of the mutable features (N(0,1) in processed units, seed 0) and (ii) the
saved A0 counterfactual of every registered alert that is a self-loop. Detector: flow-centric checkpoint of every seed.
"""
import json, os, sys
import numpy as np, pandas as pd, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L
OUT = "results"; CFG = json.load(open(f"{OUT}/protocol/config.json")); SEEDS = CFG["seeds"]; MAIN = CFG["main"]
REG = pd.read_csv(f"{OUT}/query_registry.csv"); E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")

stats, LEG = [], {}
for ds in CFG["datasets"]:
    st = L.prepare_dataset(ds, "data", OUT, write=False); g = st["G"]["test"]; sp = L.Space(st["X"]["train"], st["dom"])
    src, dst = g["src"].cpu().numpy(), g["dst"].cpu().numpy(); deg = g["deg"].cpu().numpy()
    pairs = pd.Series(list(zip(src, dst))).value_counts(); par = pairs[pairs > 1]
    loops = np.where(src == dst)[0]; dq = np.percentile(deg, [0, 10, 50, 90, 100])
    stats.append(dict(dataset=ds, hosts=int(g["N"]), flows=len(src), self_loops=len(loops), parallel_pairs=len(par),
                      flows_in_parallel_pairs=int(par.sum()), deg_min=dq[0], deg_p10=dq[1], deg_median=dq[2], deg_p90=dq[3], deg_max=dq[4]))
    rng = np.random.default_rng(0); res = {}
    for seed in SEEDS:
        m = L.load_detector(f"{OUT}/ckpt/{ds}_{MAIN}_{seed}.pt", st["X"]["train"].shape[1]).double().eval()
        R = []
        if len(loops):
            dr = rng.normal(0, 1, (len(loops), sp.F)) * sp.m.cpu().numpy()
            R.append(L.legacy_update_check(m, g, loops, dr).assign(delta_kind="random"))
        a0 = E[(E.dataset == ds) & (E.seed == seed) & (E.variant == "A0") & (E.v == 1)]
        a0 = a0[np.isin(a0.query_id.values, loops)]
        if len(a0): R.append(L.legacy_update_check(m, g, a0.query_id.values, np.stack(a0.delta.values)).assign(delta_kind="A0_counterfactual"))
        if R: res[seed] = pd.concat(R)
    allr = pd.concat(res.values()) if res else pd.DataFrame(columns=["delta_kind", "xv_max_abs", "dq", "label_flip", "self_loop"])
    blk = dict(self_loops=len(loops), note="no self-loop in this test graph" if not len(loops) else "")
    for kind, x in allr.groupby("delta_kind"):
        blk[kind] = dict(n=len(x), xv_max_abs=float(x.xv_max_abs.max()), dq_median_abs=float(x.dq.abs().median()),
                         dq_max_abs=float(x.dq.abs().max()), label_flips=int(x.label_flip.sum()))
    LEG[ds] = blk; print(ds, stats[-1], blk)
pd.DataFrame(stats).to_csv(f"{OUT}/graph_stats.csv", index=False)
p = f"{OUT}/reconstruction_gradient_checks.json"; R1 = json.load(open(p)); 
for ds, b in LEG.items(): R1[ds]["legacy_update"] = b
json.dump(R1, open(p, "w"), indent=1); print("wrote results/graph_stats.csv and legacy_update blocks")
