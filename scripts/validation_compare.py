#!/usr/bin/env python3
"""
Validation-only comparison of FlowSCE configurations with the minimizer baselines (development of the explainer; never
touches test alerts). Same 50 validation alerts per checkpoint as every tuning step; seeds = FlowSCE search (A0) outputs.

    python scripts/validation_compare.py --configs '{"T_rep":25,"T_polish":100}' --procs 4 --tag v1

Writes results/validation/compare_<tag>.csv (one row per alert and method) and prints mean support, L1, kNN, margin and
oracle calls per dataset and method.
"""
import argparse, json, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--procs", type=int, default=4); ap.add_argument("--tag", default="v")
ap.add_argument("--configs", nargs="+", default=['{"T_rep":25,"T_polish":100}'], help="JSON keyword sets for flowsce.explain_plus")
ap.add_argument("--baselines", action="store_true", help="also run LBS-adapted and RHC")
ap.add_argument("--datasets", nargs="+", default=None)
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out

def job(ds, seed):
    import torch, flowsce as L; from flowsce import fast as FF
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, "data", OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"]); dom = st["dom"]
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1]); g = st["G"]["val"]
    al = np.where(L.predict(m, g) >= .5)[0]; qv = np.random.default_rng(seed).choice(al, min(CFG["n_val_queries"], len(al)), replace=False)
    seeds = []
    for k in qv:
        orc = L.QueryOracle(m, g, int(k), cfg.tau); gen = L.generate(orc, sp, dom, cfg, "A"); seeds.append(gen["delta"])
    rows = []
    keep = lambda r, meth: dict(dataset=ds, seed=seed, query_id=int(r.get("query_id", -1)), method=meth, v=r["v"], support_eps=r["support_eps"],
                                l1=r["l1"], knn=r["knn_benign"], margin=r["margin_logit"], calls=r.get("n_fwd_method", np.nan))
    if args.baselines:
        R = FF.minimizers_batch(m, g, qv, seeds, [0] * len(qv), sp, dom, cfg, ["FlowSCE single (A)", "LBS-adapted"])
        lbs = [r for r in R if r["method"] == "LBS-adapted"]; budget = [int(r["n_fwd_method"]) for r in lbs]
        R2 = FF.minimizers_batch(m, g, qv, seeds, [0] * len(qv), sp, dom, cfg, ["RHC"], budgets={"RHC": budget})
        for k, r in zip(np.repeat(qv, 2), R): r["query_id"] = k; rows.append(keep(r, r["method"]))
        for k, r in zip(qv, R2): r["query_id"] = k; rows.append(keep(r, "RHC"))
    for c in args.configs:
        kw = json.loads(c)
        for k, sd in zip(qv, seeds):
            r = L.explain_plus(m, g, int(k), sp, dom, cfg, seed_delta=sd, **kw); r["query_id"] = k; rows.append(keep(r, "FlowSCE " + c))
    return rows

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS = args.datasets or CFG["datasets"]; out = []
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, d, s): (d, s) for d in DS for s in CFG["seeds"]}
        for f in as_completed(futs):
            try: out += f.result()
            except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
    R = pd.DataFrame(out); os.makedirs(f"{OUT}/validation", exist_ok=True); R.to_csv(f"{OUT}/validation/compare_{args.tag}.csv", index=False)
    s = R[R.v == 1].groupby(["dataset", "method"]).agg(n=("v", "size"), supp=("support_eps", "mean"), supp_med=("support_eps", "median"),
                                                       l1=("l1", "mean"), knn=("knn", "mean"), margin=("margin", "mean"), calls=("calls", "mean")).reset_index()
    pd.set_option("display.width", 250); print(s.round(3).to_string(index=False))
