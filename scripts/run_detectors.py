#!/usr/bin/env python3
"""
E1 — detection comparison (Table T3): the detector variants and baselines that are not trained by run_main.py.

    python scripts/run_detectors.py --procs 4

Same split, preprocessing, five seeds, early stopping on validation, threshold 0.5. Models:
  flow_full, flow_full_gate   FlowSCE detector with the full host embedding (no projection to d_c), without / with gates
  egraphsage                  E-GraphSAGE re-implementation (flowsce/baseline_detectors.py)
  candice_svm / _rf / _mlp    CANDICE-detector re-implementation: beta-TCVAE encoder + SVM / RF / MLP on the flow attributes
The context-shuffle test (rows of X_V permuted across hosts) is run for every graph model that reads context.
Each (dataset, model, seed) writes results/ledger/DET_<dataset>_<model>_<seed>.parquet; the final step merges them into
results/detector_metrics.json and results/detector_predictions.parquet (rows of run_main.py are kept; a 'model' column equals 'kind').
"""
import argparse, json, os, sys, time, traceback
from dataclasses import asdict
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--procs", type=int, default=4)
ap.add_argument("--models", nargs="+", default=["flow_full", "flow_full_gate", "egraphsage", "candice"])
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out
GNN = ["flow_full", "flow_full_gate", "egraphsage"]; CAND = ["candice_svm", "candice_rf", "candice_mlp"]

def path(ds, model, seed): return f"{OUT}/ledger/DET_{ds}_{model}_{seed}.parquet"

def job(ds, model, seed):
    import torch, flowsce as L
    from flowsce import baseline_detectors as BD
    torch.set_num_threads(4); t0 = time.time()
    CFG = json.load(open(f"{OUT}/protocol/config.json"))
    st = L.prepare_dataset(ds, "data", OUT, write=False); G = st["G"]; yte = G["test"]["y"].cpu().numpy(); rows = []
    if model in GNN:
        cfg = L.DetectorCfg(**{**CFG["detector"], "kind": model})
        m, info = L.train_detector(G["train"], G["val"], cfg, seed)
        torch.save(dict(state=m.state_dict(), cfg=asdict(cfg), seed=seed, ds=ds, info={k: v for k, v in info.items() if k != "history"}),
                   f"{OUT}/ckpt/{ds}_{model}_{seed}.pt")
        p = L.predict(m, G["test"]); r = dict(dataset=ds, kind=model, seed=seed, **L.det_metrics(p, yte), best_epoch=info["best_epoch"])
        torch.manual_seed(seed); perm = torch.randperm(G["test"]["N"], device=L.DEVICE)
        ps = L.predict(m, G["test"], xv=G["test"]["xv"][perm])
        r.update(ctx_shuffle_flip=float(((ps >= .5) != (p >= .5)).mean()), ctx_shuffle_macro_f1=L.det_metrics(ps, yte)["macro_f1"])
        rows.append((model, r, p))
    else:                                                   # one beta-TCVAE per seed, three classifiers on its latent mean
        X = st["X"]; ytr = st["D"]["train"]["_y"].values
        vae, info = BD.train_btcvae(X["train"], X["val"], seed)
        torch.save(dict(state=vae.state_dict(), seed=seed, ds=ds, info=info), f"{OUT}/ckpt/{ds}_candice_vae_{seed}.pt")
        Z = {p_: BD.encode_mean(vae, X[p_]) for p_ in ["train", "test"]}
        for name, clf in BD.candice_classifiers(seed).items():
            clf.fit(Z["train"], ytr); p = clf.predict_proba(Z["test"])[:, 1]
            rows.append((name, dict(dataset=ds, kind=name, seed=seed, **L.det_metrics(p, yte), best_epoch=info["best_epoch"]), p))
    for name, r, p in rows:
        r["train_s"] = round(time.time() - t0, 1)
        df = pd.DataFrame(dict(dataset=ds, kind=name, seed=seed, row_id=G["test"]["row_id"], y_true=yte, p_attack=p))
        df.attrs = {}; df["metrics"] = json.dumps(r); df.to_parquet(path(ds, name, seed))
    return [(n, r["macro_f1"]) for n, r, _ in rows], time.time() - t0

if __name__ == "__main__":
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    jobs = [(ds, m, s) for m in args.models for ds in DS for s in SEEDS
            if not os.path.exists(path(ds, "candice_svm" if m == "candice" else m, s))]
    print(f"{len(jobs)} jobs", flush=True); T0 = time.time(); failed = []
    # graph models share the GPU: run them in at most two processes; CANDICE jobs are dominated by the CPU-bound SVM
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try: res, dt = f.result(); print(f"[{i}/{len(jobs)}] {futs[f]} {res} in {dt:.0f}s (elapsed {time.time() - T0:.0f}s)", flush=True)
            except Exception: failed.append(futs[f]); print("FAILED", futs[f], traceback.format_exc(), flush=True)
    # merge
    import glob
    parts = [pd.read_parquet(p) for p in sorted(glob.glob(f"{OUT}/ledger/DET_*.parquet"))]
    MET = json.load(open(f"{OUT}/detector_metrics.json")); have = {(r["dataset"], r["kind"], r["seed"]) for r in MET}
    new = [json.loads(p.metrics.iloc[0]) for p in parts]
    MET = [r for r in MET if (r["dataset"], r["kind"], r["seed"]) not in {(n["dataset"], n["kind"], n["seed"]) for n in new}] + new
    json.dump(MET, open(f"{OUT}/detector_metrics.json", "w"), indent=1)
    P = pd.read_parquet(f"{OUT}/detector_predictions.parquet"); P = P[~P.kind.isin({n["kind"] for n in new})]
    P = pd.concat([P] + [p.drop(columns="metrics") for p in parts], ignore_index=True); P["model"] = P["kind"]
    P.to_parquet(f"{OUT}/detector_predictions.parquet", index=False)
    print(f"merged {len(new)} rows; failed {failed}; done in {time.time() - T0:.0f}s", flush=True)
    sys.exit(1 if failed else 0)
