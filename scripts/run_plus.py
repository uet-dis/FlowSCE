#!/usr/bin/env python3
"""
FlowSCE (restoration, single-feature probes, margin-floored repair, margin polishing), the proposed explainer.

    python scripts/run_plus.py --stage test --procs 3     # the 1500 registered alerts, from the saved A0 candidates
    python scripts/run_plus.py --stage seeds --procs 3    # FlowSCE as a minimizer of the CANDICE and DiCE seeds (Table T7)
    python scripts/run_plus.py --stage seeds-tuned --procs 3   # ... and of the CANDICE (tuned lambda) seeds

The configuration in protocol/plus_config.json was fixed on the 50 validation alerts per checkpoint before any test alert was
explained (scripts/validation_compare.py, results/validation/compare_v*.csv); the polishing target m* = ln 3 is fixed a priori.
Outputs: ledger/PLUS_<ds>_<seed>.parquet, ledger/PLUSMIN_<ds>_<seed>.parquet, ledger/PLUSMINT_<ds>_<seed>.parquet.
"""
import argparse, glob, json, os, sys, time, traceback
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--stage", required=True, choices=["test", "seeds", "seeds-tuned"])
ap.add_argument("--procs", type=int, default=3)
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out

def setup(ds, seed):
    import torch, flowsce as L
    torch.set_num_threads(2)
    st = L.prepare_dataset(ds, "data", OUT, write=False); sp = L.Space(st["X"]["train"], st["dom"])
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); cfg = L.ExplainCfg(**json.load(open(f"{OUT}/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", st["X"]["train"].shape[1])
    return L, st, sp, cfg, m, CFG

def job_run(ds, seed, kind):
    L, st, sp, cfg, m, CFG = setup(ds, seed); g, dom = st["G"]["test"], st["dom"]
    P = json.load(open(f"{OUT}/protocol/plus_config.json"))[ds]
    REG = pd.read_csv(f"{OUT}/query_registry.csv"); Q = REG[(REG.dataset == ds) & (REG.seed == seed)]
    if kind == "test":
        E = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet"); a0 = E[(E.dataset == ds) & (E.seed == seed) & (E.variant == "A0")].set_index("query_id")
        seeds = {"FlowSCE (A0)": a0}
    elif kind == "seeds":
        gb = pd.read_parquet(f"{OUT}/ledger/GENB_{ds}_{seed}.parquet"); gb = gb[gb.method == "CANDICE"].set_index("query_id")
        dc = pd.concat([pd.read_parquet(p) for p in glob.glob(f"{OUT}/ledger/DICE_{ds}_{seed}_*.parquet")]).set_index("query_id")
        seeds = {"CANDICE": gb, "DiCE": dc}
    else:
        gt = pd.read_parquet(f"{OUT}/ledger/GENB_{ds}_{seed}.parquet"); seeds = {"CANDICE (tuned λ)": gt[gt.method == "CANDICE (tuned λ)"].set_index("query_id")}
    rows = []
    for gname, T in seeds.items():
        for q in Q.itertuples():
            k = int(q.query_id); sd = np.array(T.loc[k, "delta"], np.float32) if T.loc[k, "v"] == 1 else None
            if sd is None and kind != "test":
                rows.append(dict(method="FlowSCE", generator=gname, dataset=ds, seed=seed, query_id=k, alert_type=q.alert_type, block=q.block, v=0, n=0, returned=0))
                continue
            r = L.explain_plus(m, g, k, sp, dom, cfg, seed_delta=sd, **L.plus_kwargs(P))
            r.update(generator=gname, dataset=ds, seed=seed, query_id=k, alert_type=q.alert_type, block=q.block); rows.append(r)
    tag = {"test": "PLUS", "seeds": "PLUSMIN", "seeds-tuned": "PLUSMINT"}[kind]
    path = f"{OUT}/ledger/{tag}_{ds}_{seed}.parquet"; pd.DataFrame(rows).to_parquet(path + ".tmp.parquet"); os.replace(path + ".tmp.parquet", path)
    return len(rows)

def pool(jobs, fn):
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
    if args.stage == "test":
        pool([(d, s, "test") for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/PLUS_{d}_{s}.parquet")], job_run)
    else:
        tag = {"seeds": "PLUSMIN", "seeds-tuned": "PLUSMINT"}[args.stage]
        pool([(d, s, args.stage) for d in DS for s in SEEDS if not os.path.exists(f"{OUT}/ledger/{tag}_{d}_{s}.parquet")], job_run)
