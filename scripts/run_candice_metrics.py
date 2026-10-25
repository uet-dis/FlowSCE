#!/usr/bin/env python3
"""
E7 — explanation metrics of the CANDICE paper (Li et al., 2025, Section 5.4) for FlowSCE (A), CANDICE (paper
configuration), DiCE (dice-ml, with its post-hoc sparsity) and A0 + LBS-adapted (Table A3).

    python scripts/run_candice_metrics.py --stage stability --procs 8
    python scripts/run_candice_metrics.py --stage efficiency --procs 3
    python scripts/run_candice_metrics.py --stage summary

Fidelity (DA@k / ADA, Eq. 10): the top-k explanatory features of an alert are set to zero in the standardised feature
space (as in the CANDICE paper) and the attack probability of the alert is recomputed on its reconstructed graph; ADA@k is
the mean over the registered alerts, k = 1..10. Ranking: change frequency over the counterfactuals of one run for CANDICE
(Eq. 8-9, ties by |delta| of the returned counterfactual) and DiCE; |delta| of the returned counterfactual for FlowSCE and
A0 + LBS. When an explanation changes fewer than k features, all of them are removed.
Stability (IS, Eq. 12, k = 10): five runs with five initialization seeds on a subset of 10 registered alerts per
checkpoint (50 per dataset). FlowSCE initializes at zero and is deterministic (IS = 1 trivially); the table therefore also
reports a FlowSCE variant initialized from N(0, 0.1). A0 + LBS varies through the LBS random generator.
Efficiency: sequential wall-clock time to explain the 500 registered alerts of each dataset (DiCE: from its E5 ledger).
"""
import argparse, glob, json, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--stage", required=True, choices=["stability", "efficiency", "summary"])
ap.add_argument("--procs", type=int, default=4); ap.add_argument("--per-ckpt", type=int, default=10)
ap.add_argument("--cpu", action="store_true", help="stability stage on CPU workers (dice-ml is dominated by small sequential calls)")
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out
RUNS, K = [0, 1, 2, 3, 4], 10
METHODS = ["FlowSCE", "FlowSCE, N(0,0.1) init", "FlowSCE (A)", "CANDICE", "DiCE (post-hoc)", "A0 + LBS-adapted"]

def ranking(delta, freq=None, eps=1e-8):
    d = np.abs(np.asarray(delta, float)) if delta is not None else None
    if d is None: return []
    idx = [j for j in np.argsort(-d, kind="stable") if d[j] > eps]
    if freq is not None:
        f = np.asarray(freq, float); idx = sorted([j for j in range(len(f)) if f[j] > 0], key=lambda j: (-f[j], -d[j]))
    return idx

def setup(ds, seed):
    sys.path.insert(0, args.workdir)
    import torch, flowsce as L
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, "data", OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"])
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1])
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    return L, st, sp, cfg, m, Q

def job_stability(ds, seed):
    if args.cpu: os.environ["CUDA_VISIBLE_DEVICES"] = ""
    L, st, sp, cfg, m, Q = setup(ds, seed); import torch
    from flowsce import fast as FF; from flowsce.dice_official import DiceOfficial
    g, dom = st["G"]["test"], st["dom"]; Q = Q.head(args.per_ckpt); ks = Q.query_id.values.astype(int)
    CP = json.load(open(f"{OUT}/protocol/candice_paper_config.json"))[ds]
    dml = FF.dice_posthoc_params(st["prep"], st["D"]["train"])
    D = DiceOfficial(m, g, st["prep"], st["D"]["train"], st["D"]["train"]["_y"].values, dom, sp, cfg)
    rows = []
    for r in RUNS:
        # CANDICE (paper configuration): five counterfactuals per run, ranking by change frequency
        R = FF.candice_batch(m, g, ks, [1000 * r + int(s) for s in Q.init_seed], sp, dom, cfg, method="CANDICE",
                             lr=CP["lr"], n_cf=5, lam1=0.5, lam2=0.5, iters=500, init_noise=0.1)
        for k, x in zip(ks, R): rows.append(dict(method="CANDICE", run=r, query_id=int(k), top=ranking(x["delta"], x["feature_importance"])[:K]))
        for k in ks:
            k = int(k)
            # FlowSCE (A) and FlowSCE: zero initialization (deterministic) and, for FlowSCE, N(0, 0.1) initialization
            PC = json.load(open(f"{OUT}/protocol/plus_config.json"))[ds]
            for name, xi0 in [("FlowSCE (A)", None), ("FlowSCE", None), ("FlowSCE, N(0,0.1) init",
                              (torch.randn(sp.F, generator=torch.Generator().manual_seed(1000 * r + k)) * 0.1).to(L.DEVICE) * sp.m)]:
                orc = L.QueryOracle(m, g, k, cfg.tau); gen = L.generate(orc, sp, dom, cfg, "A", xi0=xi0); d = gen["delta"]
                if d is not None:
                    d = L.restore(orc, dom, d, cfg, "A")[0] if name == "FlowSCE (A)" else L.plus_from_seed(orc, sp, dom, cfg, d, **L.plus_kwargs(PC))[0]
                rows.append(dict(method=name, run=r, query_id=k, top=ranking(d)[:K]))
            # A0 + LBS-adapted: LBS random generator seeded per run
            orc = L.QueryOracle(m, g, k, cfg.tau); gen = L.generate(orc, sp, dom, cfg, "A")
            x = L.lbs_adapted(m, g, k, sp, dom, cfg, gen["delta"], seed=1000 * r + k)
            rows.append(dict(method="A0 + LBS-adapted", run=r, query_id=k, top=ranking(x["delta"])[:K]))
            # DiCE (dice-ml) with its post-hoc sparsity on every returned counterfactual, ranking by change frequency
            np.random.seed(1000 * r + k); torch.manual_seed(1000 * r + k)
            x = D.explain(k); cfs = []
            for c in (x["all_cfs"] or []):
                o2 = L.QueryOracle(m, g, k, cfg.tau); cx = FF.Ctx(k, o2.a.cpu().numpy(), o2.thr)
                c2, _ = L.drive(o2, FF.dice_posthoc_dml_co(cx, dom, np.array(c, np.float32), cfg, dml), cx); cfs.append(c2)
            if cfs:
                freq = np.mean([(np.abs(c) > 1e-8) for c in cfs], 0)
                best = min(cfs, key=lambda c: ((np.abs(c) > 1e-8).sum(), np.abs(c).sum()))
                rows.append(dict(method="DiCE (post-hoc)", run=r, query_id=k, top=ranking(best, freq)[:K]))
            else: rows.append(dict(method="DiCE (post-hoc)", run=r, query_id=k, top=[]))
    pd.DataFrame([dict(x, dataset=ds, seed=seed, top=json.dumps([int(j) for j in x["top"]])) for x in rows]).to_parquet(f"{OUT}/ledger/STAB_{ds}_{seed}.parquet")
    return len(rows)

def job_efficiency(ds):
    """Sequential reference implementations, 500 registered alerts (five checkpoints)."""
    T = {}; import torch
    for seed in json.load(open(f"{OUT}/protocol/config.json"))["seeds"]:
        L, st, sp, cfg, m, Q = setup(ds, seed); g, dom = st["G"]["test"], st["dom"]
        CP = json.load(open(f"{OUT}/protocol/candice_paper_config.json"))[ds]
        PC = json.load(open(f"{OUT}/protocol/plus_config.json"))[ds]
        for name in ["FlowSCE", "FlowSCE (A)", "CANDICE", "A0 + LBS-adapted"]:
            t0 = time.perf_counter()
            for q in Q.itertuples():
                k = int(q.query_id)
                if name == "FlowSCE (A)": L.explain_query(m, g, k, sp, dom, cfg, "A")
                elif name == "FlowSCE": L.explain_plus(m, g, k, sp, dom, cfg, **L.plus_kwargs(PC))
                elif name == "CANDICE": L.candice_style(m, g, k, sp, dom, cfg, seed=int(q.init_seed), lr=CP["lr"], n_cf=5, lam1=.5, lam2=.5, iters=500, init_noise=.1, method="CANDICE")
                else:
                    orc = L.QueryOracle(m, g, k, cfg.tau); gen = L.generate(orc, sp, dom, cfg, "A")
                    L.lbs_adapted(m, g, k, sp, dom, cfg, gen["delta"], seed=int(q.init_seed))
            if torch.cuda.is_available(): torch.cuda.synchronize()
            T[name] = T.get(name, 0.0) + time.perf_counter() - t0
    return [dict(dataset=ds, method=k, seconds_500_alerts=v, implementation="sequential reference") for k, v in T.items()]

def run_pool(jobs, fn):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out, T0 = [], time.time()
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(fn, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try: out.append(f.result()); print(f"[{i}/{len(jobs)}] {futs[f]} (elapsed {time.time() - T0:.0f}s)", flush=True)
            except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
    return out

if __name__ == "__main__":
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    if args.stage == "stability":
        run_pool([(d, s) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/STAB_{d}_{s}.parquet")], job_stability)
    elif args.stage == "efficiency":
        res = run_pool([(d,) for d in DS], job_efficiency)
        pd.DataFrame([r for x in res for r in x]).to_csv(f"{OUT}/candice_efficiency.csv", index=False)
    else:
        import flowsce as L
        from flowsce import fast as FF
        rows = []
        E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
        GB = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/GENB_*.parquet"))])
        MN = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/MIN2_*.parquet"))])
        DC = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/DICE_*.parquet"))])
        for ds in DS:
            st = L.prepare_dataset(ds, "data", OUT, write=False); g = st["G"]["test"]
            for seed in SEEDS:
                m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1]); bo = FF.BatchOracle(m, g)
                key = lambda T: T[(T.dataset == ds) & (T.seed == seed)].set_index("query_id")
                A = key(E[E.variant == "A"]); C = key(GB[GB.method == "CANDICE"])
                PP = key(pd.concat([pd.read_parquet(p) for p in glob.glob(f"{OUT}/ledger/PLUS_{ds}_{seed}.parquet")]))
                LB = key(MN[(MN.generator == "FlowSCE (A0)") & (MN.method == "LBS-adapted")])
                DP = key(MN[(MN.generator == "DiCE") & (MN.method == "DiCE post-hoc")]); DF = key(DC)
                for k in A.index:
                    a = g["xe"][int(k)].cpu().numpy()
                    rk = {"FlowSCE": ranking(PP.loc[k, "delta"]) if PP.loc[k, "v"] == 1 else [],
                          "FlowSCE (A)": ranking(A.loc[k, "delta"]) if A.loc[k, "v"] == 1 else [],
                          "CANDICE": ranking(C.loc[k, "delta"], C.loc[k, "feature_importance"]) if C.loc[k, "v"] == 1 else [],
                          "A0 + LBS-adapted": ranking(LB.loc[k, "delta"]) if LB.loc[k, "v"] == 1 else [],
                          "DiCE (post-hoc)": ranking(DP.loc[k, "delta"], DF.loc[k, "feature_importance"]) if DP.loc[k, "v"] == 1 else []}
                    for meth, r in rk.items():
                        if not r: rows.append(dict(dataset=ds, seed=seed, query_id=int(k), method=meth, k=0, prob=np.nan, explained=0)); continue
                        ds_ = []
                        for kk in range(1, K + 1):
                            top = r[:kk]; d = np.zeros_like(a); d[top] = -a[top]; ds_.append(d)
                        qs = bo.q_nograd([int(k)] * K, ds_, ["A"] * K)
                        for kk, qv in zip(range(1, K + 1), qs):
                            rows.append(dict(dataset=ds, seed=seed, query_id=int(k), method=meth, k=kk, prob=float(1 / (1 + np.exp(-(qv + bo.thr)))),
                                             explained=1, n_removed=min(kk, len(r))))
        F_ = pd.DataFrame(rows); F_.to_csv(f"{OUT}/candice_fidelity_queries.csv", index=False)
        fid = F_[F_.explained == 1].groupby(["dataset", "method", "k"]).agg(ADA=("prob", "mean"), n=("prob", "size")).reset_index()
        stab = []
        for p in sorted(glob.glob(f"{OUT}/ledger/STAB_*.parquet")):
            S = pd.read_parquet(p)
            for (meth, k), g_ in S.groupby(["method", "query_id"]):
                tops = [set(json.loads(t)) for t in g_.sort_values("run").top]
                vals = [len(tops[i] & tops[j]) / K for i in range(len(tops)) for j in range(i + 1, len(tops))]
                stab.append(dict(dataset=g_.dataset.iloc[0], seed=g_.seed.iloc[0], method=meth, query_id=k, IS=float(np.mean(vals))))
        ST = pd.DataFrame(stab).groupby(["dataset", "method"]).agg(IS=("IS", "mean"), n=("IS", "size")).reset_index() if stab else pd.DataFrame()
        eff = pd.read_csv(f"{OUT}/candice_efficiency.csv") if os.path.exists(f"{OUT}/candice_efficiency.csv") else pd.DataFrame()
        dt = DC.groupby("dataset").t_generate.sum().reset_index().rename(columns={"t_generate": "seconds_500_alerts"}).assign(method="DiCE (post-hoc)", implementation="dice-ml (generation only)")
        eff = pd.concat([eff, dt], ignore_index=True)
        out = fid.pivot_table(index=["dataset", "method"], columns="k", values="ADA").add_prefix("ADA@").reset_index()
        if len(ST): out = out.merge(ST.rename(columns={"n": "IS_n_alerts"}), on=["dataset", "method"], how="outer")
        if len(eff): out = out.merge(eff, on=["dataset", "method"], how="outer")
        out.to_csv(f"{OUT}/candice_metrics.csv", index=False); print(out.round(3).to_string())
