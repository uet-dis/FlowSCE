#!/usr/bin/env python3
"""
Detection-only study: can the flow-centric FlowSCE detector be configured to match or beat the other detectors?
Separate from the explanation experiments (the explained checkpoints are not changed).

    python scripts/run_detector_search.py --stage search --procs 4    # validation only
    python scripts/run_detector_search.py --stage test --procs 4      # locked configuration, test once

Search space (validation macro-F1, five seeds): context encoder GAT (flow, flow_gate) or GraphSAGE-mean
(flow_sage, flow_sage_gate) x learning rate {1e-3, 3e-3} x dropout {0.1, 0.2} x context layers {1, 2}; every other setting as
in protocol/config.json. Selection per dataset: highest mean validation macro-F1, ties by smaller s.d., then fewer
parameters. The locked configuration is written to protocol/detector_search_config.json before any test prediction.
Outputs: detector_search_validation.csv, detector_search_test.json, ckpt/<ds>_tuned_<seed>.pt (the explained detector),
and rows of model 'tuned' in detector_metrics.json / detector_predictions.parquet.
"""
import argparse, itertools, json, os, sys, time, traceback
from dataclasses import asdict
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", default="results"); ap.add_argument("--stage", required=True, choices=["search", "test"]); ap.add_argument("--procs", type=int, default=4)
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir); OUT = args.out
GRID = [dict(kind=k, lr=lr, dropout=dp, layers=ly) for k, lr, dp, ly in itertools.product(
        ["flow", "flow_gate", "flow_sage", "flow_sage_gate"], [1e-3, 3e-3], [0.1, 0.2], [1, 2])]

def job(ds, ci, seed, test=False):
    import torch, flowsce as L
    torch.set_num_threads(2)
    CFG = json.load(open(f"{OUT}/protocol/config.json")); st = L.prepare_dataset(ds, "data", OUT, write=False); G = st["G"]
    gp = GRID[ci] if not test else json.load(open(f"{OUT}/protocol/detector_search_config.json"))[ds]["config"]
    cfg = L.DetectorCfg(**{**CFG["detector"], **gp}); t0 = time.time()
    m, info = L.train_detector(G["train"], G["val"], cfg, seed)
    r = dict(dataset=ds, cfg_id=ci, seed=seed, **gp, val_macro_f1=info["best_val_macro_f1"], best_epoch=info["best_epoch"],
             n_params=sum(p.numel() for p in m.parameters()), train_s=round(time.time() - t0, 1))
    if test:
        yte = G["test"]["y"].cpu().numpy(); p = L.predict(m, G["test"]); r.update(L.det_metrics(p, yte))
        torch.manual_seed(seed); perm = torch.randperm(G["test"]["N"], device=L.DEVICE); ps = L.predict(m, G["test"], xv=G["test"]["xv"][perm])
        r.update(ctx_shuffle_flip=float(((ps >= .5) != (p >= .5)).mean()))
        torch.save(dict(state=m.state_dict(), cfg=asdict(cfg), seed=seed, ds=ds, info={k: v for k, v in info.items() if k != "history"}),
                   f"{OUT}/ckpt/{ds}_tuned_{seed}.pt")
        r["pred"] = dict(row_id=G["test"]["row_id"].tolist(), y_true=yte.tolist(), p_attack=p.tolist())
    return r

def pool(jobs):
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out, T0 = [], time.time()
    with ProcessPoolExecutor(args.procs, mp_context=mp.get_context("spawn")) as ex:
        futs = {ex.submit(job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try: out.append(f.result())
            except Exception: print("FAILED", futs[f], traceback.format_exc(), flush=True)
            if i % 20 == 0 or i == len(jobs): print(f"[{i}/{len(jobs)}] elapsed {time.time() - T0:.0f}s", flush=True)
    return out

if __name__ == "__main__":
    CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
    if args.stage == "search":
        R = pd.DataFrame(pool([(d, ci, s) for d in DS for ci in range(len(GRID)) for s in SEEDS]))
        R.to_csv(f"{OUT}/detector_search_validation.csv", index=False); C = {}
        for d, g in R.groupby("dataset"):
            a = g.groupby("cfg_id").agg(mean=("val_macro_f1", "mean"), sd=("val_macro_f1", "std"), n_params=("n_params", "first")).reset_index()
            b = a.sort_values(["mean", "sd", "n_params"], ascending=[False, True, True]).iloc[0]
            C[d] = dict(config=GRID[int(b.cfg_id)], val_macro_f1_mean=float(b["mean"]), val_macro_f1_sd=float(b.sd),
                        rule="highest mean validation macro-F1 over five seeds; ties: smaller s.d., then fewer parameters")
        json.dump(C, open(f"{OUT}/protocol/detector_search_config.json", "w"), indent=1); print(json.dumps(C, indent=1))
    else:
        R = pool([(d, -1, s, True) for d in DS for s in SEEDS])
        preds = [pd.DataFrame(dict(dataset=r["dataset"], kind="tuned", seed=r["seed"], **r.pop("pred"))) for r in R]
        json.dump(R, open(f"{OUT}/detector_search_test.json", "w"), indent=1)
        MET = [x for x in json.load(open(f"{OUT}/detector_metrics.json")) if x["kind"] != "tuned"]          # merge as model "tuned"
        keep = ["dataset", "seed", "TP", "TN", "FP", "FN", "accuracy", "precision", "recall", "specificity", "f1_attack", "macro_f1", "fpr",
                "identity_gap", "best_epoch", "train_s", "ctx_shuffle_flip"]
        MET += [dict({k: r[k] for k in keep if k in r}, kind="tuned") for r in R]; json.dump(MET, open(f"{OUT}/detector_metrics.json", "w"), indent=1)
        P = pd.read_parquet(f"{OUT}/detector_predictions.parquet"); P = pd.concat([P[P.kind != "tuned"]] + preds, ignore_index=True); P["model"] = P["kind"]
        P.to_parquet(f"{OUT}/detector_predictions.parquet", index=False)
        T = pd.DataFrame(R).groupby("dataset")[["macro_f1", "accuracy", "recall", "fpr", "ctx_shuffle_flip"]].agg(["mean", "std"])
        print(T.round(4).to_string())
