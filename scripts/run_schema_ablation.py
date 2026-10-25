#!/usr/bin/env python3
"""
Schema / projection ablation: is a low success rate caused by the schema or by detector reliance on immutable features?

    python scripts/run_schema_ablation.py --seeds 11 23 37 53 71

Constraints are relaxed step by step with detector, queries and search configuration fixed (FlowSCE A0, batched):
  S0        real schema (immutable + training range + integer rounding), projection only when a candidate is accepted
  S0p       S0 with the projection onto the domain box after every Adam update (Algorithm 1)
  +key      additionally opens a key immutable group (UNSW-NB15: sttl, dttl, ct_state_ttl; CIC: MIN/MAX_TTL; DoH: top-3 by gradient)
  S1        no range / integer projection, immutables kept
  S1+all    every numeric feature mutable (upper bound; not an admissible configuration)
Per-query diagnostics at δ = 0: share of the gradient on immutable coordinates; whether resetting the key group / all immutable
features to the benign-training median flips the label; whether a failed S0 search reached q < 0 before Π / V_k rejected it.
Outputs: results/schema_ablation.csv, results/schema_ablation_queries.csv, results/logs/schema_ablation_<time>.txt
"""
import argparse, copy, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--datasets", nargs="+", default=None); ap.add_argument("--seeds", nargs="+", type=int, default=[11])
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir)
import flowsce as L; from flowsce import fast as FF
OUT = args.out; os.makedirs(f"{OUT}/logs", exist_ok=True)
TXT = open(f"{OUT}/logs/schema_ablation_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w")
def say(*a):
    s = " ".join(map(str, a)); print(s, flush=True); TXT.write(s + "\n"); TXT.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
CFG = json.load(open(f"{OUT}/protocol/config.json")); DATASETS = args.datasets or CFG["datasets"]
EXP = {d: L.ExplainCfg(**v) for d, v in json.load(open(f"{OUT}/protocol/explain_config.json")).items()}
REG = pd.read_csv(f"{OUT}/query_registry.csv")
KEY = {"unsw": ["sttl", "dttl", "ct_state_ttl"], "cic": ["MIN_TTL", "MAX_TTL"]}

def variant(dom, Xtr, open_idx=(), no_proj=False):
    d = copy.copy(dom); d.mutable = dom.mutable.copy(); d.mutable[list(open_idx)] = True
    d.mask = torch.tensor(d.mutable, dtype=torch.float32)
    if no_proj:
        d.lo = np.full_like(dom.lo, -np.inf, dtype=float); d.hi = np.full_like(dom.hi, np.inf, dtype=float); d.integer = np.zeros_like(dom.integer)
    return d, L.Space(Xtr, d)

summ, perq = [], []
for ds in DATASETS:
    st = L.prepare_dataset(ds, args.data_dir, OUT, write=False); g = st["G"]["test"]; dom = st["dom"]; Xtr = st["X"]["train"]
    L.POOL = L.BenignPool(Xtr, st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    names = dom.names; imm = np.where(~dom.mutable)[0]
    onehot = np.array([("=" in n) for n in names]); numeric_imm = [j for j in imm if not onehot[j]]
    benign_med = np.median(Xtr[st["D"]["train"]["_y"].values == 0], 0).astype(np.float32)
    for seed in args.seeds:
        m = L.load_detector(f"{OUT}/ckpt/{ds}_{CFG['main']}_{seed}.pt", Xtr.shape[1]); cfg = EXP[ds]
        ks = REG[(REG.dataset == ds) & (REG.seed == seed)].query_id.values.astype(int)
        bo = FF.BatchOracle(m, g, cfg.tau)
        # ---- gradient share on immutable coordinates (all coordinates free) ----
        X = torch.zeros(len(ks), len(names), device=L.DEVICE, requires_grad=True)
        G = torch.autograd.grad(bo.q(ks, X, ["A"] * len(ks)).sum(), X)[0].cpu().numpy()
        share = (G[:, imm] ** 2).sum(1) / np.maximum((G ** 2).sum(1), 1e-30)
        top_imm = [imm[j] for j in np.argsort(-(G[:, imm] ** 2).mean(0))[:3]]
        key = [names.index(n) for n in KEY.get(ds, []) if n in names] or [int(j) for j in top_imm]
        say(f"\n######## {ds} seed {seed}: {len(ks)} queries | mutable {int(dom.mutable.sum())}/{len(names)} | key group = {[names[j] for j in key]}")
        say("  top-5 features by mean squared gradient:", [(names[j], round(float(v), 4)) for j, v in
             sorted(enumerate((G ** 2).mean(0)), key=lambda x: -x[1])[:5]], "(immutable:", [names[j] for j in imm if j in np.argsort(-(G ** 2).mean(0))[:5]], ")")
        # ---- reset tests: put immutable / key features at benign-train median ----
        A = g["xe"][torch.as_tensor(ks, device=L.DEVICE)].cpu().numpy()
        def reset(cols):
            D = np.zeros_like(A); D[:, cols] = benign_med[cols] - A[:, cols]
            return bo.q_nograd(ks, D, ["A"] * len(ks)) < 0
        flip_key, flip_imm = reset(key), reset(list(imm))
        # ---- search under schema variants ----
        V = {"S0": variant(dom, Xtr), "S0+key": variant(dom, Xtr, key), "S1": variant(dom, Xtr, (), True),
             "S1+key": variant(dom, Xtr, key, True), "S1+all": variant(dom, Xtr, numeric_imm, True)}
        V["S0p"] = V["S0"]; V["S0p+key"] = V["S0+key"]          # same schema, search with in-loop box projection
        res = {}
        for vn, (dv, spv) in V.items():
            t = time.time(); R = pd.DataFrame(FF.generate_batch(m, g, ks, spv, dv, cfg, box=vn.startswith("S0p"))); res[vn] = R
            s_ok = R[R.v == 1]
            summ.append(dict(dataset=ds, seed=seed, schema=vn, n_mutable=int(dv.mutable.sum()), CSR=R.v.mean(),
                             support_eps_med=s_ok.support_eps.median(), l1_med=s_ok.l1.median(), margin_med=s_ok.margin_logit.median(),
                             share_failed_with_neg_hit=float((R[R.v == 0].neg_hits > 0).mean()) if (R.v == 0).any() else np.nan,
                             seconds=round(time.time() - t, 1)))
            say(f"  {vn:7s} mutable {int(dv.mutable.sum()):2d}  CSR {R.v.mean():.3f}  supp_eps {s_ok.support_eps.median()}  "
                f"failed-but-reached-q<0 {summ[-1]['share_failed_with_neg_hit']:.2f}  ({summ[-1]['seconds']}s)")
        R0 = res["S0"]
        for i, k in enumerate(ks):
            perq.append(dict(dataset=ds, seed=seed, query_id=int(k), grad_imm_share=float(share[i]), reset_key_flip=bool(flip_key[i]),
                             reset_imm_flip=bool(flip_imm[i]), **{f"v_{vn}": int(res[vn].v.iloc[i]) for vn in res},
                             S0_neg_hits=int(R0.neg_hits.iloc[i]), S0_proj_reject=int(R0.proj_reject.iloc[i]),
                             S0_proj_flip=int(R0.proj_flip.iloc[i]), S0_q_min_raw=float(R0.q_min_raw.iloc[i])))
        P = pd.DataFrame([r for r in perq if r["dataset"] == ds and r["seed"] == seed])
        fail = P[P.v_S0 == 0]; ok = P[P.v_S0 == 1]
        say(f"  grad_imm_share median: solved {ok.grad_imm_share.median():.3f} | failed {fail.grad_imm_share.median():.3f}")
        say(f"  reset key group → benign: solved {ok.reset_key_flip.mean():.2f} | failed {fail.reset_key_flip.mean():.2f}; "
            f"reset ALL immutable → benign: failed {fail.reset_imm_flip.mean():.2f}")
        if len(fail):
            say(f"  S0 failures: reached q<0 before Π in {(fail.S0_neg_hits > 0).mean():.2f} of them "
                f"(rejected by V_k {(fail.S0_proj_reject > 0).mean():.2f}, flipped back after Π {(fail.S0_proj_flip > 0).mean():.2f}); "
                f"median best raw q {fail.S0_q_min_raw.median():.3f}")
            say(f"  failed in S0 but solved in: S0p {fail.v_S0p.mean():.2f} | S0p+key {fail['v_S0p+key'].mean():.2f} | S1 {fail.v_S1.mean():.2f} | S0+key {fail['v_S0+key'].mean():.2f} | "
                f"S1+key {fail['v_S1+key'].mean():.2f} | S1+all {fail['v_S1+all'].mean():.2f}")
S = pd.DataFrame(summ); Pq = pd.DataFrame(perq)
S.to_csv(f"{OUT}/schema_ablation.csv", index=False); Pq.to_csv(f"{OUT}/schema_ablation_queries.csv", index=False)
say("\n" + S.round(3).to_string(index=False))
say("""
How to read:
  • CSR(S0p) >> CSR(S0)         → the search needs in-loop projection.
  • CSR(S1) >> CSR(S0)          → range clipping / integer rounding block solutions.
  • CSR(S0+key) >> CSR(S0)      → solutions need a locked feature group: the detector relies on an immutable shortcut.
  • many failed searches that reached q<0 → candidates crossed the boundary but projection / validation pulled them back.
  • q_min_raw far from 0 and reset_key_flip high on failures → mutable features cannot flip the label.
""")
