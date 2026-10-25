#!/usr/bin/env python3
"""
E5 + E6 — generator and minimizer comparisons through the shared reconstructed oracle (Tables T6, T7, A1, A2).

    python scripts/run_comparisons.py --stage tune      # CANDICE (paper configuration): learning rate on validation alerts
    python scripts/run_comparisons.py --stage gen --procs 8
    python scripts/run_comparisons.py --stage min --procs 3
    python scripts/run_comparisons.py --stage budget --procs 3
    python scripts/run_comparisons.py --stage summary

Generators (same alerts, mask, projection/validator, budget of 500 updates / 4096 forward evaluations, no minimizer):
  FlowSCE (A0)          zero-initialized Algorithm 1 (ledger of run_main.py)
  CANDICE               paper configuration (Li et al., 2025, Sections 4.3 and 5.1.2): BCE + 0.5 Mahalanobis + 0.5 L1,
                        Adam, 500 iterations, 5 counterfactuals initialized from N(0, 0.1) with fixed seeds, early stop
                        at the first valid point, training min-max ranges, mutable features; learning rate chosen on
                        validation alerts with the rule of Section 5.4
  CANDICE (tuned λ)     the same objective with λ1 = λ2 and learning rate tuned on validation (protocol/baseline_config.json)
  DiCE                  official dice-ml library, method 'gradient' (flowsce/dice_official.py)
For CANDICE and DiCE the valid counterfactual with the smallest ε-support is the main result; every valid counterfactual is
kept in the ledger (all_cfs).
Minimizers (decoupled protocol of García et al., 2026): applied to the same valid seeds of each generator — none, FlowSCE
single pass (A), FlowSCE repeated (D), DiCE post-hoc sparsity (dice-ml algorithm), LBS-adapted, and RHC with the oracle
budget of LBS on the same seed. Budget ablation: LBS, RHC and D on FlowSCE (A0) seeds with caps of 50/100/200/500 calls.
"""
import argparse, json, os, sys, time, traceback, glob
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--stage", required=True, choices=["tune", "gen", "min", "min-tuned", "budget", "summary"])
ap.add_argument("--procs", type=int, default=3); ap.add_argument("--bootstrap", type=int, default=2000)
ap.add_argument("--chunk", type=int, default=1024)
ap.add_argument("--dice-cpu", action="store_true", help="run the dice-ml generator on CPU workers")
ap.add_argument("--only", choices=["batched", "dice"], default=None, help="gen stage: run only the batched generators or only dice-ml")
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out
GENS = ["FlowSCE (A0)", "CANDICE", "DiCE"]
MINS = ["none (A0)", "FlowSCE single (A)", "FlowSCE repeated (D)", "DiCE post-hoc", "LBS-adapted"]
BUDGETS = [50, 100, 200, 500]
CANDICE_PAPER = dict(n_cf=5, lam1=0.5, lam2=0.5, iters=500, init_noise=0.1)
LR_GRID = [0.005, 0.01, 0.05, 0.1]

_W = {}
def setup(ds):
    import torch, flowsce as L
    torch.set_num_threads(2)
    if ds not in _W:
        st = L.prepare_dataset(ds, "data", OUT, write=False)
        _W[ds] = dict(st=st, sp=L.Space(st["X"]["train"], st["dom"]),
                      pool=L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values))
    L.POOL = _W[ds]["pool"]; return _W[ds]

def ctx(ds, seed):
    import flowsce as L
    W = setup(ds); CFG = json.load(open(f"{OUT}/protocol/config.json"))
    cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", W["st"]["X"]["train"].shape[1])
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    return W, cfg, m, Q, CFG

def save(rows, path):
    tmp = path + ".tmp.parquet"; pd.DataFrame(rows).to_parquet(tmp); os.replace(tmp, path)

# ────────────────────────────── jobs ──────────────────────────────
def job_tune(ds, seed, lr):
    import flowsce as L; from flowsce import fast as FF
    W, cfg, m, Q, CFG = ctx(ds, seed); gv = W["st"]["G"]["val"]
    al = np.where(L.predict(m, gv) >= .5)[0]; qv = np.random.default_rng(seed).choice(al, min(CFG["n_val_queries"], len(al)), replace=False)
    R = pd.DataFrame(FF.candice_batch(m, gv, qv, [0] * len(qv), W["sp"], W["st"]["dom"], cfg, lr=lr, **CANDICE_PAPER))
    ok = R[R.v == 1]
    return dict(dataset=ds, seed=seed, lr=lr, CSR=R.v.mean(), supp=ok.support_eps.median(), l1=ok.l1.median())

def job_gen_batched(ds, seed):
    import flowsce as L; from flowsce import fast as FF
    W, cfg, m, Q, CFG = ctx(ds, seed); g = W["st"]["G"]["test"]; dom, sp = W["st"]["dom"], W["sp"]
    ks, init = Q.query_id.values.astype(int), Q.init_seed.values.astype(int); rows = []
    lr = json.load(open(f"{OUT}/protocol/candice_paper_config.json"))[ds]["lr"]
    BC = json.load(open(f"{OUT}/protocol/baseline_config.json"))[ds]["candice"]
    for name, kw in [("CANDICE", dict(lr=lr, **CANDICE_PAPER)), ("CANDICE (tuned λ)", BC)]:
        t0 = time.time(); R = FF.candice_batch(m, g, ks, init, sp, dom, cfg, method=name, **kw); dt = time.time() - t0
        for r, q in zip(R, Q.itertuples()): r.update(dataset=ds, seed=seed, query_id=int(q.query_id), alert_type=q.alert_type, block=q.block, job_s=dt)
        rows += R
    save(rows, f"{OUT}/ledger/GENB_{ds}_{seed}.parquet"); return len(rows)

def job_gen_dice(ds, seed, part, nparts):
    # dice-ml evaluates one small batch per call: on the compressed graph the CPU is faster than a shared GPU, and many
    # CPU workers run in parallel. The worker must not have imported torch yet (fresh spawn process).
    if args.dice_cpu: os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch; torch.set_num_threads(2)
    from flowsce.dice_official import DiceOfficial
    W, cfg, m, Q, CFG = ctx(ds, seed); st = W["st"]
    D = DiceOfficial(m, st["G"]["test"], st["prep"], st["D"]["train"], st["D"]["train"]["_y"].values, st["dom"], W["sp"], cfg)
    rows = []
    for i, q in enumerate(Q.itertuples()):
        if i % nparts != part: continue
        import numpy as _np, torch as _t
        _np.random.seed(int(q.init_seed)); _t.manual_seed(int(q.init_seed))       # dice-ml draws its initial noise from numpy
        r = D.explain(int(q.query_id)); r.update(dataset=ds, seed=seed, query_id=int(q.query_id), alert_type=q.alert_type, block=q.block)
        rows.append(r)
    save(rows, f"{OUT}/ledger/DICE_{ds}_{seed}_{part}.parquet"); return len(rows)

def gen_seeds(ds, seed, only_a0=False, only_tuned=False):
    """Valid seeds of the three generators for every registered query of (ds, seed): {generator: [delta or None]}."""
    E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet"); Q = ctx_reg(ds, seed)
    a0 = E[(E.dataset == ds) & (E.seed == seed) & (E.variant == "A0")].set_index("query_id")
    if only_a0: return {"FlowSCE (A0)": [np.array(a0.loc[k, "delta"], np.float32) if a0.loc[k, "v"] == 1 else None for k in Q.query_id]}
    gb = pd.read_parquet(f"{OUT}/ledger/GENB_{ds}_{seed}.parquet"); gb = gb[gb.method == "CANDICE"].set_index("query_id")
    dc = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{OUT}/ledger/DICE_{ds}_{seed}_*.parquet")]).set_index("query_id")
    pick = lambda T, k: (np.array(T.loc[k, "delta"], np.float32) if T.loc[k, "v"] == 1 else None)
    if only_tuned:
        gt = pd.read_parquet(f"{OUT}/ledger/GENB_{ds}_{seed}.parquet"); gt = gt[gt.method == "CANDICE (tuned λ)"].set_index("query_id")
        return {"CANDICE (tuned λ)": [pick(gt, k) for k in Q.query_id]}
    return {"FlowSCE (A0)": [pick(a0, k) for k in Q.query_id], "CANDICE": [pick(gb, k) for k in Q.query_id],
            "DiCE": [pick(dc, k) for k in Q.query_id]}

def ctx_reg(ds, seed):
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); return REG[(REG.dataset == ds) & (REG.seed == seed)]

def job_min(ds, seed, tuned=False):
    from flowsce import fast as FF
    W, cfg, m, Q, CFG = ctx(ds, seed); g = W["st"]["G"]["test"]; dom, sp = W["st"]["dom"], W["sp"]
    dml = FF.dice_posthoc_params(W["st"]["prep"], W["st"]["D"]["train"]); seeds = gen_seeds(ds, seed, only_tuned=tuned)
    ks, init = Q.query_id.values.astype(int), Q.init_seed.values.astype(int); rows = []
    for gname, sd in seeds.items():
        R = FF.minimizers_batch(m, g, ks, sd, init, sp, dom, cfg, MINS, chunk=args.chunk, dml=dml)
        meta = [q for q in Q.itertuples() for _ in MINS]
        for r, q in zip(R, meta): r.update(generator=gname, dataset=ds, seed=seed, query_id=int(q.query_id), alert_type=q.alert_type, block=q.block)
        budget = [int(r["n_fwd_method"]) for r in R if r["method"] == "LBS-adapted"]          # RHC gets LBS's calls on the same seed
        R2 = FF.minimizers_batch(m, g, ks, sd, init, sp, dom, cfg, ["RHC"], chunk=args.chunk, budgets={"RHC": budget})
        for r, q, b in zip(R2, Q.itertuples(), budget): r.update(generator=gname, dataset=ds, seed=seed, query_id=int(q.query_id),
                                                           alert_type=q.alert_type, block=q.block, budget=b)
        rows += R + R2
    for r in rows: r["method"] = "none" if r["method"] == "none (A0)" else r["method"]
    save(rows, f"{OUT}/ledger/{'MIN3' if tuned else 'MIN2'}_{ds}_{seed}.parquet"); return len(rows)

def job_budget(ds, seed):
    from flowsce import fast as FF
    W, cfg, m, Q, CFG = ctx(ds, seed); g = W["st"]["G"]["test"]; dom, sp = W["st"]["dom"], W["sp"]
    sd = gen_seeds(ds, seed, only_a0=True)["FlowSCE (A0)"]; ks, init = Q.query_id.values.astype(int), Q.init_seed.values.astype(int); rows = []
    for B in BUDGETS:
        meths = ["LBS-adapted", "RHC", "FlowSCE repeated (D)"]
        R = FF.minimizers_batch(m, g, ks, sd, init, sp, dom, cfg, meths, chunk=args.chunk, budgets={mm: B for mm in meths})
        meta = [q for q in Q.itertuples() for _ in meths]
        for r, q in zip(R, meta): r.update(budget=B, dataset=ds, seed=seed, query_id=int(q.query_id), alert_type=q.alert_type, block=q.block)
        rows += R
    save(rows, f"{OUT}/ledger/MINB_{ds}_{seed}.parquet"); return len(rows)

def run_pool(jobs, fn):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out, T0 = [], time.time()
    if args.procs <= 1:
        for j in jobs: out.append(fn(*j)); print("done", j, flush=True)
        return out
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try: out.append(f.result()); print(f"[{i}/{len(jobs)}] {futs[f]} (elapsed {time.time() - T0:.0f}s)", flush=True)
            except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
    return out

if __name__ == "__main__":
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    if args.stage == "tune":
        res = run_pool([(d, s, lr) for d in DS for s in SEEDS for lr in LR_GRID], job_tune)
        T = pd.DataFrame(res); T.to_csv(f"{OUT}/candice_paper_tuning_validation.csv", index=False); C = {}
        for d, sub in T.groupby("dataset"):
            a = sub.groupby("lr").agg(CSR=("CSR", "mean"), supp=("supp", "mean"), l1=("l1", "mean")).reset_index()
            b = a.assign(s=a.supp.fillna(1e9), l=a.l1.fillna(1e9)).sort_values(["CSR", "s", "l"], ascending=[False, True, True]).iloc[0]
            C[d] = dict(lr=float(b.lr), **{k: v for k, v in CANDICE_PAPER.items()}, rule="Section 5.4 selection rule on validation alerts")
        json.dump(C, open(f"{OUT}/protocol/candice_paper_config.json", "w"), indent=1); print(C)
    elif args.stage == "gen":
        jobs = [(d, s) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/GENB_{d}_{s}.parquet")]
        if args.only != "dice": run_pool(jobs, job_gen_batched)
        if args.only == "batched": sys.exit(0)
        NP = 4
        jobs = [(d, s, p, NP) for d in DS for s in SEEDS for p in range(NP) if not os.path.exists(f"{OUT}/ledger/DICE_{d}_{s}_{p}.parquet")]
        run_pool(jobs, job_gen_dice)
    elif args.stage == "min":
        run_pool([(d, s) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/MIN2_{d}_{s}.parquet")], job_min)
    elif args.stage == "min-tuned":                                     # the same minimizers on the CANDICE (tuned λ) outputs
        run_pool([(d, s, True) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/MIN3_{d}_{s}.parquet")], job_min)
    elif args.stage == "budget":
        run_pool([(d, s) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/MINB_{d}_{s}.parquet")], job_budget)
    else:
        import flowsce as L
        E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
        GB = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/GENB_*.parquet"))], ignore_index=True)
        DC = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/DICE_*.parquet"))], ignore_index=True)
        A0 = E[E.variant == "A0"].assign(method="FlowSCE (A0)")
        GEN = pd.concat([A0, GB, DC], ignore_index=True)
        tcol = {"FlowSCE (A0)": "t_generate", "DiCE": "t_generate"}
        S = L.summarize(GEN, ["dataset", "method"])
        # every valid counterfactual of the multi-counterfactual generators (appendix): mean ε-support over all of them
        allc = []
        for (d, mth), g_ in GEN[GEN.method.isin(["CANDICE", "CANDICE (tuned λ)", "DiCE"])].groupby(["dataset", "method"]):
            sup = [int((np.abs(np.array(c)) > 1e-8).sum()) for cs in g_.all_cfs if cs is not None for c in cs]
            allc.append(dict(dataset=d, method=mth, n_cfs_valid=len(sup), support_eps_mean_all_cfs=float(np.mean(sup)) if sup else np.nan))
        S = S.merge(pd.DataFrame(allc), on=["dataset", "method"], how="left")
        tt = GEN.groupby(["dataset", "method"]).apply(lambda x: float(x["t_generate"].median()) if "t_generate" in x and x["t_generate"].notna().any()
                                                      else float(x["job_s"].iloc[0] / len(x[x.seed == x.seed.iloc[0]]) if "job_s" in x and x["job_s"].notna().any() else np.nan))
        S = S.merge(tt.rename("time_per_alert_s").reset_index(), on=["dataset", "method"], how="left")
        S.to_csv(f"{OUT}/generator_comparison.csv", index=False)
        GEN.drop(columns=[c for c in ["delta", "all_cfs", "feature_importance"] if c in GEN]).to_csv(f"{OUT}/generator_comparison_queries.csv", index=False)
        MN = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/MIN2_*.parquet")) + sorted(glob.glob(f"{OUT}/ledger/MIN3_*.parquet"))], ignore_index=True)
        PL = [pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet")) + sorted(glob.glob(f"{OUT}/ledger/PLUSMIN_*.parquet"))
              + sorted(glob.glob(f"{OUT}/ledger/PLUSMINT_*.parquet"))]
        if PL: MN = pd.concat([MN] + PL, ignore_index=True)                 # FlowSCE (scripts/run_plus.py) on the same seeds
        MS = L.summarize(MN, ["generator", "dataset", "method"])
        oc = MN[MN.v == 1].groupby(["generator", "dataset", "method"]).n_fwd_method.agg(oracle_calls_median="median", oracle_calls_mean="mean").reset_index()
        MS = MS.merge(oc, on=["generator", "dataset", "method"], how="left"); MS.to_csv(f"{OUT}/minimizer_comparison.csv", index=False)
        MN.drop(columns=[c for c in ["delta"] if c in MN]).to_csv(f"{OUT}/minimizer_comparison_queries.csv", index=False)
        MB = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/MINB_*.parquet"))], ignore_index=True)
        BS = L.summarize(MB, ["dataset", "budget", "method"])
        ob = MB[MB.v == 1].groupby(["dataset", "budget", "method"]).n_fwd_method.agg(oracle_calls_median="median", oracle_calls_mean="mean").reset_index()
        BS.merge(ob, on=["dataset", "budget", "method"], how="left").to_csv(f"{OUT}/minimizer_budget.csv", index=False)
        # paired cluster bootstrap (pairs: checkpoint, query, initialization; clusters: time blocks within checkpoint)
        rows = []
        for (gname, d), sub in MN.groupby(["generator", "dataset"]):
            sub = sub.assign(oracle_calls=sub.n_fwd_method)
            for a, b in [("FlowSCE", "LBS-adapted"), ("FlowSCE", "RHC"), ("FlowSCE", "FlowSCE single (A)"), ("FlowSCE", "DiCE post-hoc"),
                         ("FlowSCE single (A)", "LBS-adapted"), ("FlowSCE repeated (D)", "LBS-adapted"), ("LBS-adapted", "RHC")]:
                for met in ["support_eps", "l1", "knn_benign", "margin_logit", "oracle_calls"]:
                    rows.append(dict(generator=gname, dataset=d, a=a, b=b, metric=met,
                                     **L.paired_cluster_bootstrap(sub, "method", a, b, met, success_only=True, B=args.bootstrap)))
        pd.DataFrame(rows).to_csv(f"{OUT}/uncertainty_minimizer.csv", index=False)
        rows = []
        for d, sub in GEN.groupby("dataset"):
            for b in ["CANDICE", "CANDICE (tuned λ)", "DiCE"]:
                s2 = sub[sub.method.isin(["FlowSCE (A0)", b])]
                rows.append(dict(dataset=d, a="FlowSCE (A0)", b=b, metric="CSR (v)", **L.paired_cluster_bootstrap(s2, "method", "FlowSCE (A0)", b, "v", B=args.bootstrap)))
                for met in ["support_eps", "l1", "knn_benign", "margin_logit", "n_fwd"]:
                    rows.append(dict(dataset=d, a="FlowSCE (A0)", b=b, metric=met, **L.paired_cluster_bootstrap(s2, "method", "FlowSCE (A0)", b, met, success_only=True, B=args.bootstrap)))
        PLg = [pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet"))]
        if PLg:                                                         # complete FlowSCE (search + refinement) against each generator
            G2 = pd.concat([GEN] + [x.assign(method="FlowSCE") for x in PLg], ignore_index=True)
            for d, sub in G2.groupby("dataset"):
                for b in ["CANDICE", "CANDICE (tuned λ)", "DiCE"]:
                    s2 = sub[sub.method.isin(["FlowSCE", b])]
                    rows.append(dict(dataset=d, a="FlowSCE", b=b, metric="CSR (v)", **L.paired_cluster_bootstrap(s2, "method", "FlowSCE", b, "v", B=args.bootstrap)))
                    for met in ["support_eps", "l1", "knn_benign", "margin_logit"]:
                        rows.append(dict(dataset=d, a="FlowSCE", b=b, metric=met, **L.paired_cluster_bootstrap(s2, "method", "FlowSCE", b, met, success_only=True, B=args.bootstrap)))
        pd.DataFrame(rows).to_csv(f"{OUT}/uncertainty_generator.csv", index=False)
        rows = []                                                       # matched ablation (Table T4 / A1): A vs C and A vs B
        for d, sub in E.groupby("dataset"):
            sub = sub.assign(method=sub.variant)
            for a_, b_ in [("A", "C"), ("A", "B"), ("A", "A0"), ("D", "A")]:
                s2 = sub[sub.method.isin([a_, b_])]
                rows.append(dict(dataset=d, a=a_, b=b_, metric="CSR (v)", **L.paired_cluster_bootstrap(s2, "method", a_, b_, "v", B=args.bootstrap)))
                for met in ["support_eps", "l1", "knn_benign", "margin_logit", "n_fwd"]:
                    rows.append(dict(dataset=d, a=a_, b=b_, metric=met, **L.paired_cluster_bootstrap(s2, "method", a_, b_, met, success_only=True, B=args.bootstrap)))
        pd.DataFrame(rows).to_csv(f"{OUT}/uncertainty_ablation.csv", index=False)
        print(S.round(3).to_string()); print(MS.round(3).to_string())
