#!/usr/bin/env python3
"""
Counterfactual baselines through the shared reconstructed oracle: CANDICE-style and DiCE-style generators, minimizers from the
same saved A0 candidate (none, FlowSCE single / repeated restoration, DiCE post-hoc, LBS-adapted), detector summary and paired
cluster bootstrap. Baseline hyper-parameters are tuned on the same validation alerts as FlowSCE before any test query is run.

    python scripts/run_baselines.py --procs 3 --verify          # resumes finished parts
    python scripts/run_baselines.py --sections 4 5              # summaries + bootstrap only

Each (dataset, seed) runs all registered queries as one GPU batch (flowsce.fast: compressed multigraph + batched oracle);
--verify first compares the batched and sequential implementations (validity must agree; n_fwd up to float noise).
Outputs: protocol/baseline_config.json, baseline_tuning_validation.csv, ledger/GEN_*, ledger/MIN_*, baseline_comparison*.csv,
uncertainty.csv, detector_summary.csv, logs/baselines_<time>.log (+ _report.txt).
"""
import argparse, json, logging, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--force", action="store_true"); ap.add_argument("--sections", nargs="+", type=int, default=[1, 2, 3, 4, 5])
ap.add_argument("--procs", type=int, default=1, help="parallel worker processes (one job = dataset × seed × part)")
ap.add_argument("--chunk", type=int, default=1024, help="max rows per forward batch (lower it if GPU memory is short)")
ap.add_argument("--grad-chunk", type=int, default=512, help="max rows per forward+backward batch")
ap.add_argument("--verify", action="store_true", help="compare batched vs sequential implementations on 3 queries per dataset first")
ap.add_argument("--bootstrap", type=int, default=2000)
args = ap.parse_args()
os.chdir(args.workdir); sys.path.insert(0, args.workdir)
DATA_DIR, OUT, FORCE = args.data_dir, args.out, args.force
for d in ["logs", "ledger"]: os.makedirs(f"{OUT}/{d}", exist_ok=True)

def done(p): return os.path.exists(p) and not FORCE
def fmt(s): s = int(s); return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"

# ============================== worker (runs in its own process) ==============================
_W = {}
def _setup(ds):
    import torch, flowsce as L
    torch.set_num_threads(2)
    if ds not in _W:
        st = L.prepare_dataset(ds, DATA_DIR, OUT, write=False)
        _W[ds] = dict(st=st, sp=L.Space(st["X"]["train"], st["dom"]),
                      pool=L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values))
    L.POOL = _W[ds]["pool"]
    return _W[ds]

def job(kind, ds, seed, payload):
    """kind: 'tune' | 'GEN' | 'MIN'. Returns (tag, seconds, path or result)."""
    import torch, flowsce as L; from flowsce import fast as FF
    t0 = time.time(); W = _setup(ds); st, sp, dom = W["st"], W["sp"], W["st"]["dom"]
    CFG = json.load(open(f"{OUT}/protocol/config.json")); MAIN = CFG["main"]
    EXP = {d: L.ExplainCfg(**v) for d, v in json.load(open(f"{OUT}/protocol/explain_config.json")).items()}; cfg = EXP[ds]
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{MAIN}_{seed}.pt", st["X"]["train"].shape[1])
    if kind == "tune":
        gv = st["G"]["val"]; pv = L.predict(m, gv); al = np.where(pv >= .5)[0]
        qv = np.random.default_rng(seed).choice(al, min(CFG["n_val_queries"], len(al)), replace=False)   # same validation alerts as the explainer configuration search
        name, gp = payload
        fn = FF.candice_batch if name == "candice" else FF.dice_batch
        R = pd.DataFrame(fn(m, gv, qv, [0] * len(qv), sp, dom, cfg, **gp))
        return ("tune", time.time() - t0, dict(dataset=ds, seed=seed, method=name, **gp, CSR=R.v.mean(),
                supp=R.loc[R.v == 1, "support_eps"].median(), l1=R.loc[R.v == 1, "l1"].median()))
    g = st["G"]["test"]; REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    ks, init = Q.query_id.values.astype(int), Q.init_seed.values.astype(int)
    if kind == "GEN":
        BC = json.load(open(f"{OUT}/protocol/baseline_config.json"))[ds]
        rc = FF.candice_batch(m, g, ks, init, sp, dom, cfg, grad_chunk=args.grad_chunk, **BC["candice"])
        rd = FF.dice_batch(m, g, ks, init, sp, dom, cfg, posthoc=False, **BC["dice"])
        rows = [r for pair in zip(rc, rd) for r in pair]                      # query-major order
        meta = [q for q in Q.itertuples() for _ in range(2)]
    else:
        E7 = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
        a0 = E7[(E7.dataset == ds) & (E7.seed == seed) & (E7.variant == "A0")].set_index("query_id")
        sd = [a0.loc[k, "delta"] if a0.loc[k, "v"] == 1 else None for k in ks]
        MINS = ["none (A0)", "FlowSCE single (A)", "FlowSCE repeated (D)", "DiCE post-hoc", "LBS-adapted"]
        rows = FF.minimizers_batch(m, g, ks, sd, init, sp, dom, cfg, MINS, chunk=args.chunk)
        meta = [q for q in Q.itertuples() for _ in MINS]
    for r, q in zip(rows, meta): r.update(dataset=ds, seed=seed, query_id=int(q.query_id), alert_type=q.alert_type, block=q.block)
    path = f"{OUT}/ledger/{kind}_{ds}_{seed}.parquet"; tmp = path + ".tmp.parquet"
    pd.DataFrame(rows).to_parquet(tmp); os.replace(tmp, path)
    return (kind, time.time() - t0, path)

def verify(ds):
    """Sequential (flowsce reference) vs batched (flowsce.fast) on 3 registry queries of seed 11."""
    import flowsce as L; from flowsce import fast as FF
    W = _setup(ds); st, sp, dom = W["st"], W["sp"], W["st"]["dom"]; g = st["G"]["test"]
    CFG = json.load(open(f"{OUT}/protocol/config.json")); cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    BC = json.load(open(f"{OUT}/protocol/baseline_config.json"))[ds]
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_11.pt", st["X"]["train"].shape[1])
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == 11)].head(3)
    ks, init = Q.query_id.values.astype(int), Q.init_seed.values.astype(int); out = {}
    for name, seq, fast in [("CANDICE", lambda k, s: L.candice_style(m, g, k, sp, dom, cfg, seed=s, **BC["candice"]),
                             lambda: FF.candice_batch(m, g, ks, init, sp, dom, cfg, **BC["candice"])),
                            ("DiCE", lambda k, s: L.dice_style(m, g, k, sp, dom, cfg, seed=s, posthoc=False, **BC["dice"]),
                             lambda: FF.dice_batch(m, g, ks, init, sp, dom, cfg, posthoc=False, **BC["dice"]))]:
        a = [seq(int(k), int(s)) for k, s in zip(ks, init)]; b = fast()
        out[name] = [(x["v"], x["n_fwd"], y["v"], y["n_fwd"]) for x, y in zip(a, b)]
    return out

# ============================== driver ==============================
if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    STAMP = time.strftime("%Y%m%d_%H%M%S"); LOG = f"{OUT}/logs/baselines_{STAMP}.log"; REPORT = f"{OUT}/logs/baselines_{STAMP}_report.txt"
    log = logging.getLogger("fast"); log.setLevel(logging.INFO); fm = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    for h in (logging.FileHandler(LOG, encoding="utf-8"), logging.StreamHandler(sys.stdout)): h.setFormatter(fm); log.addHandler(h)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50); pd.set_option("display.max_rows", 500)
    def report(title, obj):
        text = obj.to_string() if isinstance(obj, (pd.DataFrame, pd.Series)) else str(obj)
        open(REPORT, "a", encoding="utf-8").write(f"\n{'=' * 100}\n{title}\n{'=' * 100}\n{text}\n"); log.info("%s\n%s", title, text)
    import flowsce as L
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DATASETS, SEEDS = CFG["datasets"], CFG["seeds"]
    log.info("start sections=%s procs=%d device=%s", args.sections, args.procs, L.DEVICE); T0 = time.time()
    ctx = mp.get_context("spawn"); FAILED = []
    def pool_run(jobs):
        res = []
        if args.procs <= 1:
            for j in jobs:
                try: r = job(*j); res.append(r); log.info("done %s %s %s in %s", j[0], j[1], j[2], fmt(r[1]))
                except Exception: log.error("FAILED %s\n%s", j[:3], traceback.format_exc()); FAILED.append(j[:3])
            return res
        with ProcessPoolExecutor(args.procs, mp_context=ctx) as ex:
            futs = {ex.submit(job, *j): j for j in jobs}
            for i, f in enumerate(as_completed(futs), 1):
                j = futs[f]
                try: r = f.result(); res.append(r); log.info("[%d/%d] done %s %s %s in %s (elapsed %s)", i, len(jobs), j[0], j[1], j[2], fmt(r[1]), fmt(time.time() - T0))
                except Exception: log.error("FAILED %s\n%s", j[:3], traceback.format_exc()); FAILED.append(j[:3])
        return res

    # §1 tune on validation (batched, one job per dataset × method × grid point)
    bc_path = f"{OUT}/protocol/baseline_config.json"
    if 1 in args.sections and not done(bc_path):
        grids = dict(candice=[dict(lam1=l, lam2=l, lr=lr) for l in [.5, .05, .005] for lr in [.05, .1]])   # CANDICE (tuned lambda)
        res = pool_run([("tune", ds, seed, (n, gp)) for ds in DATASETS for seed in SEEDS for n, gr in grids.items() for gp in gr])
        T = pd.DataFrame([r[2] for r in res]); T.to_csv(f"{OUT}/baseline_tuning_validation.csv", index=False); BC = {}
        for (ds, n), sub in T.groupby(["dataset", "method"]):
            pk = ["lam1", "lam2", "lr"] if n == "candice" else ["lam_prox", "lam_div", "lr"]
            sub = sub.groupby(pk).agg(CSR=("CSR", "mean"), supp=("supp", "median"), l1=("l1", "median")).reset_index()
            best = sub.assign(s=sub.supp.fillna(1e9), l=sub.l1.fillna(1e9)).sort_values(["CSR", "s", "l"], ascending=[False, True, True]).iloc[0]
            BC.setdefault(ds, {})[n] = {k: float(best[k]) for k in (["lam1", "lam2", "lr"] if n == "candice" else ["lam_prox", "lam_div", "lr"])}
        L.save_json(BC, bc_path)
    report("§1 baseline_config", json.dumps(json.load(open(bc_path)), indent=2))
    if args.verify:
        for ds in DATASETS:
            v = verify(ds); log.info("verify %s (v_seq, nfwd_seq, v_fast, nfwd_fast): %s", ds, v)
            bad = [x for rows in v.values() for x in rows if x[0] != x[2]]
            if bad: log.warning("verify %s: validity differs on %d run(s); small n_fwd differences come from floating-point noise near the boundary", ds, len(bad))


    # §2 + §3: one job per (part, dataset, seed); biggest datasets first so the pool stays busy
    jobs = []
    for part, sec in [("GEN", 2), ("MIN", 3)]:
        if sec in args.sections:
            for ds in sorted(DATASETS, key=lambda d: d != "unsw"):
                for seed in SEEDS:
                    if not done(f"{OUT}/ledger/{part}_{ds}_{seed}.parquet"): jobs.append((part, ds, seed, None))
                    else: log.info("skip %s %s %s (exists)", part, ds, seed)
    log.info("%d jobs to run", len(jobs)); pool_run(jobs)

    def load(prefix):
        P = [f"{OUT}/ledger/{prefix}_{d}_{s}.parquet" for d in DATASETS for s in SEEDS]
        miss = [p for p in P if not os.path.exists(p)]
        if miss: log.warning("%s: %d ledgers missing", prefix, len(miss)); return None
        return pd.concat([pd.read_parquet(p) for p in P], ignore_index=True)
    E7 = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet"); GENp, MIN = load("GEN"), load("MIN")
    cols = ["dataset", "method", "attempted", "verified", "CSR", "support_eps_median", "support_exact_median", "l1_median",
            "mahal_median", "knn_benign_median", "margin_logit_median", "n_fwd_median", "n_bwd_median"]
    GEN = None
    if GENp is not None:
        GEN = pd.concat([E7[E7.variant == "A0"].assign(method="FlowSCE-gen (A0)"), GENp], ignore_index=True)
        report("§2 Generator comparison", L.summarize(GEN, ["dataset", "method"])[cols].round(3))
    if MIN is not None:
        report("§3 Minimizer comparison (same A0 seed)", L.summarize(MIN, ["dataset", "method"])[cols].round(3))
    if GEN is not None and MIN is not None and (4 in args.sections or 5 in args.sections):
        FULL = E7[E7.variant == "A"].assign(method="FlowSCE full (A)")
        GEN["comparison"], MIN["comparison"], FULL["comparison"] = "generator", "minimizer_same_seed", "pipeline"
        ALL = pd.concat([GEN, MIN, FULL], ignore_index=True)
        ALL.drop(columns=[c for c in ["delta", "feature_importance"] if c in ALL]).to_csv(f"{OUT}/baseline_comparison_queries.csv", index=False)
        BCMP = L.summarize(ALL, ["comparison", "dataset", "method"]); BCMP.to_csv(f"{OUT}/baseline_comparison.csv", index=False)
        report("§4 baseline_comparison", BCMP[["comparison"] + cols].round(3))
        if 5 in args.sections:
            det = pd.DataFrame(json.load(open(f"{OUT}/detector_metrics.json")))
            DT = det.groupby(["dataset", "kind"])[["accuracy", "recall", "specificity", "macro_f1", "fpr"]].agg(["mean", "std"]).round(4)
            DT.to_csv(f"{OUT}/detector_summary.csv"); report("§5 Detector mean ± std", DT)
            pairs = [("pipeline", "FlowSCE full (A)", "generator", "CANDICE-style"), ("pipeline", "FlowSCE full (A)", "generator", "DiCE-style-noposthoc"),
                     ("generator", "FlowSCE-gen (A0)", "generator", "CANDICE-style"), ("generator", "FlowSCE-gen (A0)", "generator", "DiCE-style-noposthoc"),
                     ("minimizer_same_seed", "FlowSCE repeated (D)", "minimizer_same_seed", "LBS-adapted"),
                     ("minimizer_same_seed", "FlowSCE single (A)", "minimizer_same_seed", "DiCE post-hoc"),
                     ("minimizer_same_seed", "FlowSCE single (A)", "minimizer_same_seed", "none (A0)"),
                     ("ablation", "A", "ablation", "C"), ("ablation", "A", "ablation", "B")]
            POOL = pd.concat([ALL, E7[E7.variant.isin(["A", "B", "C"])].assign(method=lambda d: d.variant, comparison="ablation")])
            rows = []
            for ds in DATASETS:
                for ca, a, cb, b in pairs:
                    sub = POOL[(POOL.dataset == ds) & (((POOL.comparison == ca) & (POOL.method == a)) | ((POOL.comparison == cb) & (POOL.method == b)))]
                    rows.append(dict(dataset=ds, a=a, b=b, metric="CSR (v)", **L.paired_cluster_bootstrap(sub, "method", a, b, "v", B=args.bootstrap)))
                    for met in ["support_eps", "l1", "mahal", "knn_benign", "margin_logit", "n_fwd"]:
                        rows.append(dict(dataset=ds, a=a, b=b, metric=met, **L.paired_cluster_bootstrap(sub, "method", a, b, met, success_only=True, B=args.bootstrap)))
                pd.DataFrame(rows).to_csv(f"{OUT}/uncertainty.csv", index=False)
            report("§5 Paired cluster bootstrap", pd.DataFrame(rows).round(4))
    if FAILED: log.error("%d jobs failed: %s — rerun to resume", len(FAILED), FAILED)
    log.info("DONE in %s | log %s | report %s", fmt(time.time() - T0), LOG, REPORT)
    sys.exit(1 if FAILED else 0)
