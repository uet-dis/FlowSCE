#!/usr/bin/env python3
"""
Ablations of the FlowSCE refinement on the registered alerts (the locked configuration is not changed; no tuning):

    python scripts/run_plus_ablation.py --procs 4

  components  the refinement without one step at a time: no single-feature probes, no repair (T_rep = 0, i.e. restoration
              under the margin floor), no polishing (T_pol = 0), no L1 shrinking
  budget      the complete refinement with the forward budget capped at 200 / 500 / 1000 evaluations, comparable with the
              oracle calls of LBS-adapted on the same seeds (Table A2)
  probe_only  no generator: the single-feature probes run directly from the alert (every mutable feature, largest gradient
              first, started from the nearest-benign value), the best valid single-feature candidate (smallest L1) is kept, then
              polished to ln 3 and shrunk as in FlowSCE; a failure if no single feature suffices
All variants start from the saved FlowSCE search output (A0) of each registered alert.
Output: ledger/PLUSABL_<variant>_<ds>_<seed>.parquet; summary plus_ablation.csv (refine_seconds_median: wall-clock time of the
refinement per alert, measured in the parallel runs, so comparable across variants but not with the sequential reference).
"""
import argparse, glob, json, os, sys, time, traceback
from dataclasses import asdict
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--stage", default="run", choices=["run", "summary"])
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out
VARIANTS = {"no_probe": dict(opts=dict(probe_k=0)), "no_repair": dict(T_rep=0), "no_polish": dict(T_polish=0),
            "no_shrink": dict(opts=dict(shrink=False)), "cap200": dict(max_fwd=200), "cap500": dict(max_fwd=500), "cap1000": dict(max_fwd=1000),
            "probe_only": dict(probe_only=True)}

def probe_only(L, m, g, k, sp, dom, cfg, P):
    """Single-feature probes from the alert itself (no generator), then polishing and shrinking as in FlowSCE."""
    import time
    orc = L.QueryOracle(m, g, k, cfg.tau); t0 = time.perf_counter(); q0 = orc.q0(np.zeros(sp.F, np.float32), "A")
    dummy = np.where(dom.mutable, 1.0, 0.0).astype(np.float32)                 # placeholder incumbent: any valid single feature beats it
    best, hits = L.sparse_probe(orc, sp, dom, dummy, cfg, k=None, T=P["opts"]["probe_T"], min_margin=0.0, init="benign", free=True, select="best_l1")
    d = None
    if hits:
        d, _ = L.polish_margin(orc, sp, dom, best, cfg, L.M_TARGET, T=P["T_polish"])
        fl = L.M_TARGET if -orc.q0(d, "A") > L.M_TARGET else max(0.0, -orc.q0(d, "A") - 1e-6)
        d = L.shrink_l1(orc, dom, d, cfg, min_margin=fl)
    row = dict(method="probe-only", variant="probe_only", q_orig=q0, t_generate=0.0, t_restore=time.perf_counter() - t0, n_fwd_method=orc.nf, probe_hits=hits)
    row.update(L.assess(orc, dom, sp, d, "A", cfg)); row["n_fwd"], row["n_bwd"] = orc.nf, orc.nb
    return row

def job(ds, seed, var):
    import torch, flowsce as L
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, "data", OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"]); g, dom = st["G"]["test"], st["dom"]
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); base = json.load(open(f"{OUT}/protocol/explain_config.json"))[ds]
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1])
    P = json.load(open(f"{OUT}/protocol/plus_config.json"))[ds]; V = VARIANTS[var]
    kw = L.plus_kwargs(P); kw.update({k: v for k, v in V.items() if k in ("T_rep", "T_polish")}); kw.update(V.get("opts", {}))
    cfg = L.ExplainCfg(**{**base, **({"max_fwd": V["max_fwd"]} if "max_fwd" in V else {})})
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet"); a0 = E[(E.dataset == ds) & (E.seed == seed) & (E.variant == "A0")].set_index("query_id")
    rows = []
    for q in Q.itertuples():
        k = int(q.query_id); sd = np.array(a0.loc[k, "delta"], np.float32) if a0.loc[k, "v"] == 1 else None
        r = probe_only(L, m, g, k, sp, dom, cfg, P) if V.get("probe_only") else L.explain_plus(m, g, k, sp, dom, cfg, seed_delta=sd, **kw)
        r.update(ablation=var, dataset=ds, seed=seed, query_id=k, alert_type=q.alert_type, block=q.block); rows.append(r)
    path = f"{OUT}/ledger/PLUSABL_{var}_{ds}_{seed}.parquet"; pd.DataFrame(rows).to_parquet(path + ".tmp.parquet"); os.replace(path + ".tmp.parquet", path)
    return len(rows)

if __name__ == "__main__":
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    if args.stage == "run":
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed
        jobs = [(d, s, v) for v in VARIANTS for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/PLUSABL_{v}_{d}_{s}.parquet")]
        T0 = time.time()
        with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
            futs = {ex.submit(job, *j): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                try: f.result(); print(f"[{i}/{len(jobs)}] {futs[f]} (elapsed {time.time() - T0:.0f}s)", flush=True)
                except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
    R = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUSABL_*.parquet"))]
                  + [pd.read_parquet(p).assign(ablation="full") for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet"))], ignore_index=True)
    rows = []
    for (v, d), g in R.groupby(["ablation", "dataset"]):
        s = g[g.v == 1]
        rows.append(dict(ablation=v, dataset=d, n=len(g), CSR=g.v.mean(), spurious=int(((g.n == 1) & (g.v == 0)).sum()),
                         support_eps_mean=s.support_eps.mean(), single_feature=float((s.support_eps <= 1).mean()), l1_median=s.l1.median(),
                         knn_benign_median=s.knn_benign.median(), margin_logit_median=s.margin_logit.median(),
                         margin_ge_ln3=float((s.margin_logit >= np.log(3)).mean()), oracle_calls_median=s.n_fwd_method.median(),
                         n_bwd_median=s.n_bwd.median(), refine_seconds_median=g.t_restore.median()))
    S = pd.DataFrame(rows); S.to_csv(f"{OUT}/plus_ablation.csv", index=False); pd.set_option("display.width", 250); print(S.round(3).to_string(index=False))
