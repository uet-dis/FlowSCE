#!/usr/bin/env python3
"""
Percentile-constrained domain: the schema's training range of every feature is narrowed to the 1st-99th training
percentile (raw units), so that projection, validator and search box exclude the extreme values that single-feature
counterfactuals tend to reach. Everything else is unchanged (locked explainer and refinement settings, no tuning).

    python scripts/run_constrained.py --procs 3

On every registered alert: FlowSCE (search + refinement) and the same search followed by LBS-adapted, both under the
constrained domain. Output: ledger/CONS_<ds>_<seed>.parquet; summary constrained.csv.
"""
import argparse, copy, glob, json, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--procs", type=int, default=3)
ap.add_argument("--stage", default="run", choices=["run", "summary"])
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out

def constrained(dom, Xtr, lo_q=1.0, hi_q=99.0):
    d = copy.copy(dom); raw = dom.to_raw(Xtr)
    d.lo = np.percentile(raw, lo_q, axis=0); d.hi = np.percentile(raw, hi_q, axis=0)
    return d

def job(ds, seed):
    import torch, flowsce as L
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, "data", OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"]); g = st["G"]["test"]
    dom = constrained(st["dom"], st["X"]["train"])
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    P = json.load(open(f"{OUT}/protocol/plus_config.json"))[ds]
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1])
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    rows = []
    for q in Q.itertuples():
        k = int(q.query_id); meta = dict(dataset=ds, seed=seed, query_id=k, alert_type=q.alert_type, block=q.block)
        r = L.explain_plus(m, g, k, sp, dom, cfg, **L.plus_kwargs(P)); r.update(meta, method="FlowSCE (constrained)"); rows.append(r)
        orc = L.QueryOracle(m, g, k, cfg.tau); gen = L.generate(orc, sp, dom, cfg, "A")
        r2 = L.lbs_adapted(m, g, k, sp, dom, cfg, gen["delta"], seed=int(q.init_seed)); r2.update(meta, method="A0 + LBS-adapted (constrained)"); rows.append(r2)
    path = f"{OUT}/ledger/CONS_{ds}_{seed}.parquet"; pd.DataFrame(rows).to_parquet(path + ".tmp.parquet"); os.replace(path + ".tmp.parquet", path)
    return len(rows)

if __name__ == "__main__":
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    if args.stage == "run":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [(d, s) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/CONS_{d}_{s}.parquet")]
        T0 = time.time()
        with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(job, *j): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                try: f.result(); print(f"[{i}/{len(jobs)}] {futs[f]} (elapsed {time.time() - T0:.0f}s)", flush=True)
                except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
    R = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/CONS_*.parquet"))], ignore_index=True)
    rows = []
    for (mth, d), g in R.groupby(["method", "dataset"]):
        s = g[g.v == 1]
        rows.append(dict(method=mth, dataset=d, n=len(g), CSR=g.v.mean(), spurious=int(((g.n == 1) & (g.v == 0)).sum()),
                         support_eps_mean=s.support_eps.mean(), single_feature=float((s.support_eps <= 1).mean()), l1_median=s.l1.median(),
                         knn_benign_median=s.knn_benign.median(), margin_logit_median=s.margin_logit.median(), oracle_calls_median=s.n_fwd_method.median()))
    import flowsce as L
    for d, sub in R.groupby("dataset"):                                     # paired cluster bootstrap, FlowSCE - LBS
        for met in ["support_eps", "l1", "knn_benign", "margin_logit", "n_fwd_method"]:
            rows.append(dict(method="FlowSCE - LBS (paired)", dataset=d, metric=met,
                             **L.paired_cluster_bootstrap(sub, "method", "FlowSCE (constrained)", "A0 + LBS-adapted (constrained)", met, success_only=True)))
    S = pd.DataFrame(rows); S.to_csv(f"{OUT}/constrained.csv", index=False); pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))
