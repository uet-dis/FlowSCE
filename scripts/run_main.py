#!/usr/bin/env python3
"""
Main experiment: data protocol, detectors, explainer configuration, query registry, RQ1 checks and the A0/A/B/C/D ablation.

    python scripts/run_main.py                      # every stage; a stage is skipped when its output already exists
    python scripts/run_main.py --force              # recompute everything

Stages and outputs (all under results/):
  1  data       stratified group split, train-only preprocessing, manifests     protocol/network_manifest_*.json, splits/
  2  schema     feature schema and raw-space validator checks                   protocol/schema_*.json, protocol/validator_log.json
  3  graphs     snapshot graph manifests and contract checks                    protocol/graph_manifest_*.json
  4  detectors  3 architectures × 5 seeds per dataset                           ckpt/, detector_metrics.json, detector_predictions.parquet
  5  tuning     explainer grid on VALIDATION alerts of all checkpoints          protocol/explain_config.json, explain_tuning_validation.csv
  6  registry   100 test alerts per checkpoint, fixed before any explainer      query_registry.csv
  7  rq1        reconstruction vs rebuild, gradient vs finite differences       reconstruction_gradient_checks.json
  8  ablation   A0 / A / B / C / D on every registered alert                    ledger/E7_*.parquet, counterfactual_queries.*, cf_summary.csv
"""
import argparse, json, logging, os, sys, time
from dataclasses import asdict
import numpy as np, pandas as pd, torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results"); ap.add_argument("--force", action="store_true")
ap.add_argument("--n-queries", type=int, default=100, help="registered test alerts per checkpoint")
ap.add_argument("--n-val", type=int, default=50, help="validation alerts per checkpoint used by every tuning step")
ap.add_argument("--lr", nargs="+", type=float, default=[0.1, 0.2, 0.3])
ap.add_argument("--lam1", nargs="+", type=float, default=[0.001, 0.003, 0.01])
ap.add_argument("--lam2", nargs="+", type=float, default=[0.001, 0.003, 0.01])
ap.add_argument("--margin", nargs="+", type=float, default=[0.5])
ap.add_argument("--until", type=int, default=99, help="stop after this stage (4: detectors only)")
args = ap.parse_args(); os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L                      # noqa: E402
from flowsce import fast as FF           # noqa: E402

OUT = args.out; DATASETS, SEEDS, KINDS, MAIN = ["cic", "doh", "unsw"], L.SEEDS, ["edge", "flow", "flow_gate"], "tuned"
TUNED = f"{OUT}/protocol/detector_search_config.json"      # explained detector: configuration locked on validation (run_detector_search.py)
for d in ["", "/ckpt", "/protocol", "/ledger", "/logs"]: os.makedirs(OUT + d, exist_ok=True)
log = logging.getLogger("main"); log.setLevel(logging.INFO); fm = logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S")
for h in (logging.FileHandler(f"{OUT}/logs/main_{time.strftime('%Y%m%d_%H%M%S')}.log"), logging.StreamHandler(sys.stdout)): h.setFormatter(fm); log.addHandler(h)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
done = lambda p: os.path.exists(p) and not args.force
DET_CFG = L.DetectorCfg(); T0 = time.time()
L.save_json(dict(datasets=DATASETS, seeds=SEEDS, kinds=KINDS, main=MAIN, n_queries=args.n_queries, n_val_queries=args.n_val,
                 detector=asdict(DET_CFG), tuned={d: c["config"] for d, c in json.load(open(TUNED)).items()} if os.path.exists(TUNED) else None,
                 env=L.env_info()), f"{OUT}/protocol/config.json")

# ---- 1-3: data protocol, schema, graphs ----
S = {}
for ds in DATASETS:
    S[ds] = L.prepare_dataset(ds, args.data_dir, OUT, write=True)
    man = json.load(open(f"{OUT}/protocol/network_manifest_{ds}.json"))
    log.info("data %s: split %s | attack rate %s | test vectors also in train after preprocessing %.4f | F = %d", ds, man["split"]["achieved"],
             {k: round(v, 3) for k, v in man["split"]["attack_rate"].items()}, man["split"]["test_processed_duplicates_in_train"], len(S[ds]["prep"].cols))
vlog = {}
for ds in DATASETS:
    dom, Xt = S[ds]["dom"], S[ds]["X"]["test"]; rng = np.random.default_rng(0)
    L.save_json(dom.schema(), f"{OUT}/protocol/schema_{ds}.json")
    idx = rng.choice(len(Xt), 500, replace=False); imm = np.where(~dom.mutable)[0]
    vlog[ds] = dict(n_features=len(dom.names), n_mutable=int(dom.mutable.sum()), n_integer=int(dom.integer.sum()),
                    zero_delta_valid=float(np.mean([dom.check(Xt[i], np.zeros(Xt.shape[1]))[0] for i in idx])),
                    projected_random_valid=float(np.mean([dom.check(Xt[i], dom.project(Xt[i], rng.normal(0, .5, Xt.shape[1]) * dom.mutable))[0] for i in idx])),
                    immutable_edit_rejected=float(np.mean([not dom.check(Xt[i], np.eye(Xt.shape[1])[imm[0]] * .5)[0] for i in idx])) if len(imm) else None,
                    checked=["immutable", "train range (raw)", "integer (raw)", "finite"],
                    not_checked=["coupled statistics", "units", "packet-level realizability"])
L.save_json(vlog, f"{OUT}/protocol/validator_log.json")
for ds in DATASETS:
    gm = dict(policy="snapshot per partition (train/val/test built separately)", direction="source → destination endpoint",
              host_feature="mean of incident processed flow vectors, X_V = D^{-1} B X_E; self-loop counted twice",
              parallel_flows="kept as distinct edges", message_passing="directed src→dst, no added self-loops",
              query_membership="query belongs to the test snapshot whose summaries are reconstructed",
              unseen_hosts="hosts of a partition are built from that partition's flows only", labels_used=False)
    for p, g in S[ds]["G"].items():
        gm[p] = dict(L.check_contract(g), **L.degree_stats(g), xe_sha=L.sha_array(g["xe"].cpu().numpy())); assert gm[p]["ok"], (ds, p)
    L.save_json(gm, f"{OUT}/protocol/graph_manifest_{ds}.json")
log.info("stages 1-3 done (%.0fs)", time.time() - T0)

# ---- 4: detectors ----
met_path = f"{OUT}/detector_metrics.json"
if not done(met_path):
    MET, preds = [], []
    for ds in DATASETS:
        G = S[ds]["G"]; yte = G["test"]["y"].cpu().numpy()
        for kind in KINDS:
            for seed in SEEDS:
                t0 = time.time(); cfg = L.DetectorCfg(**{**asdict(DET_CFG), "kind": kind})
                m, info = L.train_detector(G["train"], G["val"], cfg, seed)
                torch.save(dict(state=m.state_dict(), cfg=asdict(cfg), seed=seed, ds=ds, info={k: v for k, v in info.items() if k != "history"}),
                           f"{OUT}/ckpt/{ds}_{kind}_{seed}.pt")
                p = L.predict(m, G["test"]); r = dict(dataset=ds, kind=kind, seed=seed, **L.det_metrics(p, yte), best_epoch=info["best_epoch"],
                                                      train_s=round(time.time() - t0, 1))
                if kind != "edge":
                    torch.manual_seed(seed); perm = torch.randperm(G["test"]["N"], device=L.DEVICE)
                    ps = L.predict(m, G["test"], xv=G["test"]["xv"][perm])
                    r.update(ctx_shuffle_flip=float(((ps >= .5) != (p >= .5)).mean()), ctx_shuffle_macro_f1=L.det_metrics(ps, yte)["macro_f1"])
                MET.append(r); preds.append(pd.DataFrame(dict(dataset=ds, kind=kind, seed=seed, row_id=G["test"]["row_id"], y_true=yte, p_attack=p)))
                log.info("detector %s %s %s macro-F1 %.4f (%.0fs)", ds, kind, seed, r["macro_f1"], r["train_s"])
    pd.concat(preds).to_parquet(f"{OUT}/detector_predictions.parquet", index=False); L.save_json(MET, met_path)
det = pd.DataFrame(json.load(open(met_path))); assert det.identity_gap.max() < 1e-9
det.groupby(["dataset", "kind"])[["accuracy", "precision", "recall", "specificity", "f1_attack", "macro_f1", "fpr"]].agg(["mean", "std"]).round(4).to_csv(f"{OUT}/detector_table.csv")
if args.until == 4: log.info("stopped after the detectors (--until 4)"); sys.exit(0)
missing = [f"{ds}_{MAIN}_{s}" for ds in DATASETS for s in SEEDS if not os.path.exists(f"{OUT}/ckpt/{ds}_{MAIN}_{s}.pt")]
if missing: sys.exit(f"explained detector checkpoints missing ({missing[:3]}...): run scripts/run_detector_search.py --stage search, then --stage test")
detector = lambda ds, seed: L.load_detector(f"{OUT}/ckpt/{ds}_{MAIN}_{seed}.pt", S[ds]["X"]["train"].shape[1])
SP = {ds: L.Space(S[ds]["X"]["train"], S[ds]["dom"]) for ds in DATASETS}
POOLS = {ds: L.BenignPool(S[ds]["X"]["train"], S[ds]["D"]["train"]["_y"].values, S[ds]["D"]["train"]["row_id"].values) for ds in DATASETS}

# ---- 5: explainer configuration on validation alerts (all checkpoints) ----
cfg_path = f"{OUT}/protocol/explain_config.json"; tune_path = f"{OUT}/explain_tuning_validation.csv"
SELECTION_RULE = ("highest mean validation CSR over the five checkpoints; ties: smallest mean (over checkpoints) of the per-checkpoint "
                  "median s_eps; further ties: smallest mean of the per-checkpoint median L1")

def select_config(T):
    """Selection rule (Section 5.4): (a) highest mean validation CSR; (b) smallest mean s_eps; (c) smallest mean L1."""
    agg = T.groupby(["cfg_id", "lr", "lam1", "lam2", "search_margin"]).agg(CSR=("CSR", "mean"), CSR_min=("CSR", "min"),
                                                                           supp=("supp", "mean"), l1=("l1", "mean")).reset_index()
    agg = agg.assign(s=agg.supp.fillna(1e9), l=agg.l1.fillna(1e9)).sort_values(["CSR", "s", "l"], ascending=[False, True, True])
    return agg

if not done(cfg_path):
    if done(tune_path):                         # the grid was already evaluated on validation alerts: re-apply the rule only
        rows = pd.read_csv(tune_path).to_dict("records")
    else:
        grid = [dict(lr=lr, lam1=l1, lam2=l2, search_margin=k) for lr in args.lr for l1 in args.lam1 for l2 in args.lam2 for k in args.margin]
        rows = []
        for ds in DATASETS:
            L.POOL = POOLS[ds]; gv = S[ds]["G"]["val"]; VQ = {}
            for seed in SEEDS:
                m = detector(ds, seed); al = np.where(L.predict(m, gv) >= .5)[0]
                VQ[seed] = (m, np.random.default_rng(seed).choice(al, min(args.n_val, len(al)), replace=False))
            for gi, gp in enumerate(grid):
                cfg = L.ExplainCfg(**gp, box_proj=True)
                for seed, (m, qv) in VQ.items():
                    R = pd.DataFrame(FF.explain_A_batch(m, gv, qv, SP[ds], S[ds]["dom"], cfg)); ok = R[R.v == 1]
                    rows.append(dict(dataset=ds, seed=seed, cfg_id=gi, **gp, CSR=R.v.mean(), supp=ok.support_eps.median(), l1=ok.l1.median()))
        pd.DataFrame(rows).to_csv(tune_path, index=False)
    EXP, sel = {}, {}
    for ds in DATASETS:
        agg = select_config(pd.DataFrame([r for r in rows if r["dataset"] == ds])); b = agg.iloc[0]
        best = dict(lr=float(b.lr), lam1=float(b.lam1), lam2=float(b.lam2), search_margin=float(b.search_margin))
        EXP[ds] = L.ExplainCfg(**best, box_proj=True)
        sel[ds] = dict(locked=best, top5=agg.head(5)[["cfg_id", "lr", "lam1", "lam2", "CSR", "CSR_min", "supp", "l1"]].to_dict("records"))
        log.info("tuning %s locked %s (mean validation CSR %.3f, mean s_eps %.2f, mean L1 %.3f)", ds, best, b.CSR, b.supp, b.l1)
    L.save_json({ds: asdict(c) for ds, c in EXP.items()}, cfg_path)
    L.save_json(dict(rule=SELECTION_RULE, n_val_alerts_per_checkpoint=args.n_val, datasets=sel), f"{OUT}/protocol/explain_selection.json")
EXP = {ds: L.ExplainCfg(**v) for ds, v in json.load(open(cfg_path)).items()}

# ---- 6: query registry (fixed before any explainer is run on test data) ----
reg_path = f"{OUT}/query_registry.csv"
if not done(reg_path):
    rows = []
    for ds in DATASETS:
        gt = S[ds]["G"]["test"]; yte = gt["y"].cpu().numpy()
        for seed in SEEDS:
            p = L.predict(detector(ds, seed), gt); al = np.where(p >= .5)[0]; rng = np.random.default_rng(seed)
            fp, tp = al[yte[al] == 0], al[yte[al] == 1]
            nfp = min(len(fp), max(10, args.n_queries // 5)); ntp = min(len(tp), args.n_queries - nfp)
            pick = np.concatenate([rng.choice(fp, nfp, replace=False), rng.choice(tp, ntp, replace=False)])
            for i, k in enumerate(pick):
                u, v = gt["src"][k], gt["dst"][k]
                rows.append(dict(dataset=ds, seed=seed, query_id=int(k), row_id=int(gt["row_id"][k]), y_true=int(yte[k]),
                                 alert_type="true_alert" if yte[k] else "false_alert", p_attack=float(p[k]), block=gt["block"][k],
                                 deg_u=float(gt["deg"][u]), deg_v=float(gt["deg"][v]), init_seed=int(seed * 1000 + i)))
    pd.DataFrame(rows).to_csv(reg_path, index=False)
REG = pd.read_csv(reg_path)

# ---- 7: RQ1 on the real test graphs ----
rq1_path = f"{OUT}/reconstruction_gradient_checks.json"
if not done(rq1_path):
    RQ1 = {}
    for ds in DATASETS:
        gt = S[ds]["G"]["test"]; rng = np.random.default_rng(0); m = detector(ds, 11); sp = SP[ds]
        ks = rng.choice(len(gt["src"]), 200, replace=False); q_full = m(gt["xe"], gt["xv"], gt["src"], gt["dst"]).detach()
        qreg = REG[(REG.dataset == ds) & (REG.seed == 11)].query_id.values
        orig = max(abs(L.QueryOracle(m, gt, k).q0(np.zeros(sp.F, np.float32)) - float(q_full[k, 1] - q_full[k, 0])) for k in qreg)
        gch = L.check_gradients(m, gt, qreg[:10], sp, rng)
        RQ1[ds] = dict(rebuild_max_abs=L.check_reconstruction(gt, ks, rng), contract=L.check_contract(gt), original_score_max_abs=orig,
                       tol=dict(reconstruction=1e-12, gradient_fd=1e-6, dtype="float64"), fd_max_abs=float(gch.fd_max_abs.max()),
                       chain_max_abs=float(gch.chain_max_abs.max()), ctx_share_median=float(gch.ctx_share.median()), per_query=gch.to_dict("records"))
        log.info("RQ1 %s rebuild %.1e | original score %.1e | FD %.1e | context share %.1e", ds, RQ1[ds]["rebuild_max_abs"], orig, RQ1[ds]["fd_max_abs"], RQ1[ds]["ctx_share_median"])
    L.save_json(RQ1, rq1_path)

# ---- 8: A0 / A / B / C / D ablation on every registered alert (batched; numerically equivalent to flowsce.explain_query) ----
for ds in DATASETS:
    L.POOL = POOLS[ds]; g = S[ds]["G"]["test"]
    for seed in SEEDS:
        path = f"{OUT}/ledger/E7_{ds}_{seed}.parquet"
        if done(path): continue
        t = time.time(); Q = REG[(REG.dataset == ds) & (REG.seed == seed)].reset_index(drop=True)
        rows = FF.e7_batch(detector(ds, seed), g, Q.query_id.values, SP[ds], S[ds]["dom"], EXP[ds])
        meta = [q for q in Q.itertuples() for _ in range(5)]
        for r, q in zip(rows, meta): r.update(dataset=ds, seed=seed, query_id=int(q.query_id), row_id=int(q.row_id), alert_type=q.alert_type, block=q.block)
        pd.DataFrame(rows).to_parquet(path); log.info("ablation %s seed %s (%.0fs)", ds, seed, time.time() - t)
E7 = pd.concat([pd.read_parquet(f"{OUT}/ledger/E7_{ds}_{s}.parquet") for ds in DATASETS for s in SEEDS], ignore_index=True)
E7.drop(columns=["delta"]).to_csv(f"{OUT}/counterfactual_queries.csv", index=False); E7.to_parquet(f"{OUT}/counterfactual_queries.parquet")
SUM = L.summarize(E7, ["dataset", "variant"]); SUM.to_csv(f"{OUT}/cf_summary.csv", index=False)
log.info("\n%s", SUM[["dataset", "variant", "attempted", "nominal", "verified", "spurious", "CSR", "SVR", "support_eps_median", "l1_median",
                      "margin_logit_median", "n_fwd_median"]].round(3).to_string())
log.info("done in %.0fs", time.time() - T0)
