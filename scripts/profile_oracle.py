#!/usr/bin/env python3
"""
Runtime profile of the oracle and of every explainer: sequential reference implementation vs batched fast path.

    python scripts/profile_oracle.py                    # seed 11, 10 queries per dataset (5 solved by A0, 5 not)
    python scripts/profile_oracle.py --n 20 --seq-n 4

  [A] micro-benchmark: ms per forward and per forward+backward, sequential vs batched oracle (B = 1, 64, 256, 1024)
  [B] every method × dataset with the sequential implementation on seq-n queries: time per query, n_fwd, n_bwd, CSR
  [C] the same methods with the batched implementation on n queries: time per query, speed-up, agreement
Outputs: results/logs/profile_<time>.txt and .csv
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, torch
ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
ap.add_argument("--data-dir", default="data"); ap.add_argument("--out", default="results")
ap.add_argument("--seed", type=int, default=11); ap.add_argument("--n", type=int, default=10); ap.add_argument("--seq-n", type=int, default=4)
ap.add_argument("--datasets", nargs="+", default=None)
args = ap.parse_args(); os.chdir(args.workdir); sys.path.insert(0, args.workdir)
import flowsce as L; from flowsce import fast as FF
OUT = args.out; os.makedirs(f"{OUT}/logs", exist_ok=True); STAMP = time.strftime("%Y%m%d_%H%M%S")
TXT = open(f"{OUT}/logs/profile_{STAMP}.txt", "w")
def say(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); TXT.write(s + "\n"); TXT.flush()
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

CFG = json.load(open(f"{OUT}/protocol/config.json")); DATASETS = args.datasets or CFG["datasets"]; MAIN = CFG["main"]
EXP = {ds: L.ExplainCfg(**v) for ds, v in json.load(open(f"{OUT}/protocol/explain_config.json")).items()}
BC = json.load(open(f"{OUT}/protocol/baseline_config.json"))
REG = pd.read_csv(f"{OUT}/query_registry.csv"); E7 = pd.read_parquet(f"{OUT}/counterfactual_queries.parquet")
say("env", L.env_info())
def sync(): L.sync()

rows_bench, rows_m = [], []
for ds in DATASETS:
    st = L.prepare_dataset(ds, args.data_dir, OUT, write=False); g = st["G"]["test"]; dom = st["dom"]
    sp = L.Space(st["X"]["train"], dom)
    L.POOL = L.BenignPool(st["X"]["train"], st["D"]["train"]["_y"].values, st["D"]["train"]["row_id"].values)
    m = L.load_detector(f"{OUT}/ckpt/{ds}_{MAIN}_{args.seed}.pt", st["X"]["train"].shape[1]); cfg = EXP[ds]
    cg = FF.CompressedGraph(g)
    say(f"\n######## {ds}: N={g['N']} flows={len(g['src'])} unique host pairs={len(cg.src)} F={sp.F} mutable={int(dom.mutable.sum())}")
    # ---------------- [A] micro-benchmark ----------------
    k0 = int(REG[(REG.dataset == ds) & (REG.seed == args.seed)].query_id.iloc[0]); orc = L.QueryOracle(m, g, k0)
    d0 = torch.zeros(sp.F, device=L.DEVICE)
    for _ in range(5): orc.q(d0)
    sync(); t = time.perf_counter(); n = 50
    for _ in range(n): orc.q(d0)
    sync(); f1 = (time.perf_counter() - t) / n * 1e3
    x = d0.clone().requires_grad_(True); sync(); t = time.perf_counter()
    for _ in range(n): torch.autograd.grad(orc.q(x), x)
    sync(); fb1 = (time.perf_counter() - t) / n * 1e3
    rows_bench.append(dict(dataset=ds, impl="sequential (lib)", B=1, ms_per_call=f1, ms_per_row_fwd=f1, ms_per_row_fwd_bwd=fb1))
    bo = FF.BatchOracle(m, g); ks_all = REG[(REG.dataset == ds) & (REG.seed == args.seed)].query_id.values
    for B in [1, 64, 256, 1024]:
        ks = np.resize(ks_all, B); D = torch.zeros(B, sp.F, device=L.DEVICE)
        try:
            for _ in range(3): bo.q(ks, D, ["A"] * B)
            sync(); t = time.perf_counter()
            with torch.no_grad():
                for _ in range(10): bo.q(ks, D, ["A"] * B)
            sync(); fB = (time.perf_counter() - t) / 10 * 1e3
            X = D.clone().requires_grad_(True); sync(); t = time.perf_counter()
            for _ in range(10): torch.autograd.grad(bo.q(ks, X, ["A"] * B).sum(), X)
            sync(); fbB = (time.perf_counter() - t) / 10 * 1e3
            rows_bench.append(dict(dataset=ds, impl="batched+compressed (fast)", B=B, ms_per_call=fB, ms_per_row_fwd=fB / B, ms_per_row_fwd_bwd=fbB / B))
        except RuntimeError as e:
            say(f"  B={B}: {e}"); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    # ---------------- query selection: half where A0 succeeded, half where it failed ----------------
    a0 = E7[(E7.dataset == ds) & (E7.seed == args.seed) & (E7.variant == "A0")].set_index("query_id")
    reg = REG[(REG.dataset == ds) & (REG.seed == args.seed)].set_index("query_id")
    succ, fail = [k for k in reg.index if a0.loc[k, "v"] == 1], [k for k in reg.index if a0.loc[k, "v"] != 1]
    rng = np.random.default_rng(0)
    def pick(n):
        h = n // 2; s = list(rng.choice(succ, min(h, len(succ)), replace=False)); f = list(rng.choice(fail, min(n - len(s), len(fail)), replace=False))
        return [int(k) for k in s + f]
    Qf = pick(args.n); Qs = Qf[: args.seq_n // 2] + Qf[len(Qf) // 2: len(Qf) // 2 + args.seq_n - args.seq_n // 2]
    seeds_d = {k: (a0.loc[k, "delta"] if a0.loc[k, "v"] == 1 else None) for k in Qf}
    init = {k: int(reg.loc[k, "init_seed"]) for k in Qf}; a0ok = {k: int(a0.loc[k, "v"] == 1) for k in Qf}
    MINS = ["FlowSCE single (A)", "FlowSCE repeated (D)", "LBS-adapted"]
    seq_fns = {"CANDICE (tuned λ)": lambda k: L.candice_style(m, g, k, sp, dom, cfg, seed=init[k], **BC[ds]["candice"])}
    for me in MINS: seq_fns[me] = (lambda me: lambda k: L.minimizer_from_seed(m, g, k, sp, dom, cfg, seeds_d[k], me, seed=init[k]))(me)
    fast_fns = {"CANDICE (tuned λ)": lambda ks: FF.candice_batch(m, g, ks, [init[k] for k in ks], sp, dom, cfg, **BC[ds]["candice"])}
    for me in MINS: fast_fns[me] = (lambda me: lambda ks: FF.minimizers_batch(m, g, ks, [seeds_d[k] for k in ks], [init[k] for k in ks], sp, dom, cfg, [me]))(me)
    for me in seq_fns:
        # ---------------- [B] sequential ----------------
        per_q = []
        for k in Qs:
            sync(); t = time.perf_counter(); r = seq_fns[me](k); sync()
            per_q.append(dict(k=k, a0_ok=a0ok[k], s=time.perf_counter() - t, nf=r["n_fwd"], nb=r["n_bwd"], v=r["v"], supp=r["support_eps"]))
        S = pd.DataFrame(per_q)
        # ---------------- [C] batched ----------------
        sync(); t = time.perf_counter(); Rf = pd.DataFrame(fast_fns[me](Qf)); sync(); tf = time.perf_counter() - t
        Rf["k"] = Qf; Rf["a0_ok"] = [a0ok[k] for k in Qf]
        same = Rf.set_index("k").loc[S.k]
        match = float(((same.n_fwd.values == S.nf.values) & (same.v.values == S.v.values)).mean())
        for grp, lab in [(1, "A0 succeeded"), (0, "A0 failed (hard)")]:
            s_ = S[S.a0_ok == grp]; f_ = Rf[Rf.a0_ok == grp]
            rows_m.append(dict(dataset=ds, method=me, queries=lab, seq_s_per_query=s_.s.mean() if len(s_) else np.nan,
                               seq_n_fwd=s_.nf.mean() if len(s_) else np.nan, seq_n_bwd=s_.nb.mean() if len(s_) else np.nan,
                               fast_s_per_query=tf / len(Qf), fast_CSR=f_.v.mean() if len(f_) else np.nan,
                               fast_n_fwd=f_.n_fwd.mean() if len(f_) else np.nan, match_seq_vs_fast=match))
        say(f"  {me:24s} seq {S.s.mean():7.2f}s/q (n_fwd {S.nf.mean():6.0f}) | fast {tf / len(Qf):7.3f}s/q | match {match:.2f}")
B = pd.DataFrame(rows_bench); M = pd.DataFrame(rows_m)
say("\n[A] Oracle micro-benchmark (ms)"); say(B.round(4).to_string(index=False))
say("\n[B+C] Per method / dataset"); say(M.round(3).to_string(index=False))
# ------- extrapolated full-run cost (100 queries × 5 seeds) -------
P_OK = {ds: float(E7[(E7.dataset == ds) & (E7.variant == "A0")].v.mean()) for ds in DATASETS}   # real share of easy queries
def wmean(g, col):
    p = P_OK[g.dataset.iloc[0]]; e = g[g.queries == "A0 succeeded"][col].mean(); h = g[g.queries != "A0 succeeded"][col].mean()
    return np.nansum([p * e, (1 - p) * h]) if not (np.isnan(e) and np.isnan(h)) else np.nan
full = pd.DataFrame([dict(dataset=d, method=me, seq=wmean(g, "seq_s_per_query"), fast=g.fast_s_per_query.mean())
                     for (d, me), g in M.groupby(["dataset", "method"])])
full["seq_h_full"] = full.seq * 500 / 3600; full["fast_h_full"] = full.fast * 500 / 3600; full["speedup"] = full.seq / full.fast
say("\nExtrapolated cost for 100 queries × 5 seeds (hours)"); say(full.round(3).to_string(index=False))
B.to_csv(f"{OUT}/logs/profile_{STAMP}_oracle.csv", index=False); M.to_csv(f"{OUT}/logs/profile_{STAMP}_methods.csv", index=False)
say(f"\nsaved {OUT}/logs/profile_{STAMP}.*")
