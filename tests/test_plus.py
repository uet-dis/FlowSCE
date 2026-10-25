#!/usr/bin/env python3
"""
Property test of FlowSCE on real VALIDATION alerts (never test data), for the explained detector:

  • v(FlowSCE) = v(A0)                       validity is never lost
  • s_eps(FlowSCE) <= s_eps(A)               repair never increases the support of single-pass restoration
  • margin(FlowSCE) >= min(margin(A), ln 3)  the margin floor of repair, probing, polishing and shrinking
  • nominal = reconstructed                   no spurious validity

    python tests/test_plus.py --n 3            # 3 validation alerts per dataset (seed 11)
"""
import argparse, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=3); ap.add_argument("--seed", type=int, default=11)
ap.add_argument("--datasets", nargs="+", default=["cic", "doh", "unsw"]); args = ap.parse_args()
os.chdir(ROOT); sys.path.insert(0, ROOT)
import flowsce as L                                                    # noqa: E402

FAIL = []
def check(cond, msg):
    print(("  [ok]   " if cond else "  [FAIL] ") + msg)
    if not cond: FAIL.append(msg)

CFG = json.load(open("results/protocol/config.json")); PC = json.load(open("results/protocol/plus_config.json"))
for ds in args.datasets:
    print(f"== {ds}")
    st = L.prepare_dataset(ds, "data", "results", write=False); sp = L.Space(st["X"]["train"], st["dom"]); dom = st["dom"]
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    cfg = L.ExplainCfg(**json.load(open("results/protocol/explain_config.json"))[ds])
    m = L.load_detector(f"results/ckpt/{ds}_{CFG['main']}_{args.seed}.pt", st["X"]["train"].shape[1]); gv = st["G"]["val"]
    al = np.where(L.predict(m, gv) >= .5)[0]; qv = np.random.default_rng(args.seed).choice(al, 20, replace=False)[: args.n]
    for k in qv:
        k = int(k); a0 = L.explain_query(m, gv, k, sp, dom, cfg, "A0"); sd = a0["delta"] if a0["v"] == 1 else None
        A = L.minimizer_from_seed(m, gv, k, sp, dom, cfg, sd, "FlowSCE single (A)")
        P = L.explain_plus(m, gv, k, sp, dom, cfg, seed_delta=sd, **L.plus_kwargs(PC[ds]))
        tag = f"{ds} q{k}"
        check(P["v"] == a0["v"], f"{tag}: FlowSCE keeps validity ({a0['v']})")
        check(P["n"] == P["v"], f"{tag}: FlowSCE not spurious")
        if a0["v"] == 1:
            check(P["support_eps"] <= A["support_eps"], f"{tag}: s(FlowSCE)={P['support_eps']} <= s(A)={A['support_eps']}")
            fl = min(A["margin_logit"], L.M_TARGET) if PC[ds].get("opts", {}).get("floor_mode") == "target" else A["margin_logit"]
            check(P["margin_logit"] >= fl - 1e-4, f"{tag}: margin(FlowSCE)={P['margin_logit']:.3f} >= floor={fl:.3f} (margin(A)={A['margin_logit']:.3f})")
print(f"\n{'PASS' if not FAIL else 'FAIL'}: {len(FAIL)} failures"); sys.exit(1 if FAIL else 0)
