#!/usr/bin/env python3
"""
scripts/run_coupling.py — time-window coupling sweep (RQ2/RQ3).

One snapshot per tumbling wall-clock interval of Δt seconds, as a streaming detector would build it. For every policy the
detector is RETRAINED (five seeds), and A, B and C are run on random alerts of the windowed test graph:

    python scripts/run_coupling.py --windows 600 60 10 1 --seeds 11 23 37 53 71 --alerts 100 --procs 3

Output: results/coupling_sweep_time.csv (one row per dataset × Δt × seed × variant) and results/coupling_sweep_time_queries.parquet.
"""
import argparse, json, logging, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--windows", nargs="+", type=float, default=[600, 60, 10, 1], help="window lengths Δt in seconds")
ap.add_argument("--policy", choices=["time"], default="time", help="tumbling wall-clock windows of Δt seconds")
ap.add_argument("--seeds", nargs="+", type=int, default=[11, 23, 37, 53, 71]); ap.add_argument("--datasets", nargs="+", default=None)
ap.add_argument("--alerts", type=int, default=100); ap.add_argument("--procs", type=int, default=1)
ap.add_argument("--tag", default="coupling_sweep_time", help="output name (coupling_fullcapture_doh for --datasets doh_full)")
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out

def job(ds, W, seed):
    import torch, flowsce as L
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, args.data_dir, OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"])
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json"))
    EXP = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[{"doh_full": "doh"}.get(ds, ds)])   # full capture: DoH's locked settings
    t0 = time.time()
    Gw = {p: L.build_graph(st["D"][p], st["X"][p], ds, (args.policy, W)) for p in ["train", "val", "test"]}
    m, info = L.train_detector(Gw["train"], Gw["val"], L.DetectorCfg(**({**CFG["detector"], **CFG["tuned"][{"doh_full": "doh"}.get(ds, ds)]} if CFG["main"] == "tuned" else {**CFG["detector"], "kind": CFG["main"]})), seed)
    p = L.predict(m, Gw["test"]); y = Gw["test"]["y"].cpu().numpy(); dm = L.det_metrics(p, y)
    al = np.where(p >= .5)[0]; ks = np.random.default_rng(seed).choice(al, min(args.alerts, len(al)), replace=False)
    rows = []
    for k in ks:
        ga = L.grad_alignment(m, Gw["test"], int(k), sp)
        for v in ["A", "B", "C"]:
            r = L.explain_query(m, Gw["test"], int(k), sp, st["dom"], EXP, v); r.pop("delta", None)
            rows.append(dict(r, **ga, dataset=ds, W=W, seed=seed, query_id=int(k), y_true=int(y[k]), macro_f1=dm["macro_f1"],
                             best_epoch=info["best_epoch"], **{f"g_{a}": b for a, b in L.degree_stats(Gw["test"]).items()}))
    return rows, time.time() - t0

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import flowsce as L
    TAG = args.tag
    stamp = time.strftime("%Y%m%d_%H%M%S"); os.makedirs(f"{OUT}/logs", exist_ok=True)
    log = logging.getLogger("coupling"); log.setLevel(logging.INFO); fm = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(f"{OUT}/logs/{TAG}_{stamp}.log"), logging.StreamHandler(sys.stdout)): h.setFormatter(fm); log.addHandler(h)
    DS = args.datasets or json.load(open(f"{OUT}/protocol/config.json"))["datasets"]
    jobs = [(d, W, s) for d in DS for W in args.windows for s in args.seeds]; out = []; T0 = time.time()
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try: rows, dt = f.result(); out += rows; log.info("[%d/%d] %s done in %.0fs (elapsed %.0fs)", i, len(jobs), futs[f], dt, time.time() - T0)
            except Exception: log.error("FAILED %s\n%s", futs[f], traceback.format_exc())
    Q = pd.DataFrame(out); Q.to_parquet(f"{OUT}/{TAG}_queries.parquet")
    S = []
    for (d, W, s, v), g in Q.groupby(["dataset", "W", "seed", "variant"]):
        S.append(dict(dataset=d, W=W, seed=s, variant=v, macro_f1=g.macro_f1.iloc[0], deg_median=g.g_deg_median.iloc[0],
                      omega_median=g.g_omega_median.iloc[0], **L.rates(g), disc_stale_med=g.disc_stale.median(),
                      disc_stale_max=g.disc_stale.max(), cos_AC_med=g.cos_AC.median(), cos_AC_min=g.cos_AC.min(),
                      updates_med=g.updates.median(), n_fwd_med=g.n_fwd.median()))
    S = pd.DataFrame(S); S.to_csv(f"{OUT}/{TAG}.csv", index=False)
    agg = S.groupby(["dataset", "W", "variant"]).agg(macro_f1=("macro_f1", "mean"), omega=("omega_median", "median"), CSR=("CSR", "mean"),
                                                     spurious=("spurious", "sum"), nominal=("nominal", "sum"), disc_med=("disc_stale_med", "median"),
                                                     disc_max=("disc_stale_max", "max"), cos_min=("cos_AC_min", "min")).reset_index()
    log.info("summary\n%s", agg.round(4).to_string()); log.info("DONE in %.0fs", time.time() - T0)
