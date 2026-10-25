#!/usr/bin/env python3
"""
Every table and figure of the paper, and results/paper/numbers.json (the only source of numbers in the text).

    python scripts/make_paper_tables.py

Main text: T1 data protocol, T3 detection, T4 RQ1-RQ3, F1 coupling figure, T5 ablation quality, T6 generators,
T7 minimizers, T8 runtime (T2, the variant specification, is static text in the paper).
Appendix: A1 bootstrap intervals, A2 minimizer budget, A3 CANDICE-paper metrics, A4 sensitivity, A5 domain relations and
projection ablation, A6 ablation by true / false alert.
"""
import glob, json, math, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
OUT, P = "results", "results/paper"; os.makedirs(P, exist_ok=True)
CFG = json.load(open(f"{OUT}/protocol/config.json")); DS, SEEDS = CFG["datasets"], CFG["seeds"]
NAME = {"cic": "CIC-IDS2018", "doh": "DoHBrw-2020", "unsw": "UNSW-NB15", "doh_full": "DoHBrw-2020 (full capture)"}
NUM = {}

def rd(p):
    p = f"{OUT}/{p}"
    if not os.path.exists(p): print("missing", p); return None
    return pd.read_parquet(p) if p.endswith(".parquet") else pd.read_csv(p)
def f(x, d=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))): return "--"
    return f"{x:.{d}f}"
def sci(x):
    if x == 0: return "0"
    if abs(x) >= 1e-3: return f"{x:.3g}"
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def sci_all(x):                                                       # always a x 10^e (uniform column format)
    if x == 0: return "0"
    m, e = f"{x:.1e}".split("e"); return f"${m}\\times10^{{{int(e)}}}$"
def pm(m, s, d=4): return f"{m:.{d}f} $\\pm$ {s:.{d}f}"
def ci(e, lo, hi, d=3):
    if e is None or (isinstance(e, float) and math.isnan(e)): return "--"
    g = lambda v: f"{v:+.{d}f}" if abs(v) >= 0.5 * 10 ** -d else f"{0:.{d}f}"
    return f"{g(e)} [{g(lo)}, {g(hi)}]"
def key(*a): return "_".join(str(x).replace(" ", "").replace("(", "").replace(")", "").replace("+", "p").replace("$", "").replace("^", "").replace("\\", "")
                            .replace(",", "").replace("λ", "lam").replace(".", "") if i < len(a) - 1 else str(x) for i, x in enumerate(a))
def table(fn, caption, label, header, rows, align, wide=False, size="\\small", note=None, groups=None):
    env = "table*" if wide else "table"
    t = [f"\\begin{{{env}}}[t]", "\\centering" + size, f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         (f"\\resizebox{{\\textwidth}}{{!}}{{%\n\\begin{{tabular}}{{@{{}}{align}@{{}}}}" if wide and len(header) >= 10 else f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}"), "\\toprule"]
    if groups: t.append(groups)
    t += [" & ".join(header) + " \\\\", "\\midrule"]
    for r in rows: t.append("\\midrule" if r == "MID" else " & ".join(map(str, r)) + " \\\\")
    t += ["\\bottomrule", "\\end{tabular}" + ("}" if wide and len(header) >= 10 else "")]
    if note: t.append(f"\\par\\smallskip\\footnotesize {note}")
    t.append(f"\\end{{{env}}}")
    open(f"{P}/{fn}", "w").write("\n".join(t) + "\n"); print("wrote", fn)

E = rd("counterfactual_queries.parquet"); REG = rd("query_registry.csv")
env = CFG["env"]; NUM.update(env_python=float(".".join(env["python"].split(".")[:2])), env_torch=env["torch"].split("+")[0],
                             env_gpu_model=int([w for w in env["gpu"].split() if w.isdigit()][0]))
NUM.update(n_alerts_per_ckpt=CFG.get("n_queries"), n_val_alerts=CFG.get("n_val_queries"), seeds=SEEDS)
for d, c in json.load(open(f"{OUT}/protocol/explain_config.json")).items():
    for k in ["lr", "lam1", "lam2", "search_margin"]: NUM[f"cfg_{d}_{k}"] = c[k]
if os.path.exists(f"{OUT}/protocol/candice_paper_config.json"):
    for d, c in json.load(open(f"{OUT}/protocol/candice_paper_config.json")).items(): NUM[f"cfg_candice_paper_{d}_lr"] = c["lr"]
if os.path.exists(f"{OUT}/protocol/baseline_config.json"):
    for d, c in json.load(open(f"{OUT}/protocol/baseline_config.json")).items():
        NUM.update({f"cfg_candice_tuned_{d}_lam": c["candice"]["lam1"], f"cfg_candice_tuned_{d}_lr": c["candice"]["lr"]})
# protocol constants quoted in Section 5 (set in the scripts named here)
NUM.update(budget_updates=500, budget_fwd=4096, dice_iterations=336, dice_fwd_per_iteration=12, dice_total_cfs=4,     # flowsce/dice_official.py
           candice_n_cf=5, candice_lam=0.5, candice_init_sd=0.1, candice_iters=500, candice_lr_grid=[0.005, 0.01, 0.05, 0.1],  # scripts/run_comparisons.py
           minimizer_caps=[50, 100, 200, 500], stability_alerts_per_ckpt=10, stability_runs=5, ada_k_max=10, is_k=10,            # scripts/run_candice_metrics.py
           dml_quantile=0.1, dml_limit_steps=10000, window_lengths=[600, 60, 10, 1], window_lengths_full=[60, 10, 1],
           val_alerts_per_ckpt=50, block_size=500, bootstrap_B=2000, eps_support=1e-8, knn_q=5, max_restore_trials=2000, tau=0.5)
_tv = rd("explain_tuning_validation.csv")                                                # the executed explainer grid
if _tv is not None:
    NUM.update(grid_size=int(_tv.cfg_id.nunique()), lr_grid=sorted(map(float, _tv.lr.unique())), lam1_grid=sorted(map(float, _tv.lam1.unique())),
               lam2_grid=sorted(map(float, _tv.lam2.unique())), margin_grid=sorted(map(float, _tv.search_margin.unique())))
if os.path.exists(f"{OUT}/protocol/plus_config.json"):                                 # FlowSCE refinement (locked on validation)
    for d, c in json.load(open(f"{OUT}/protocol/plus_config.json")).items():
        NUM.update({f"plus_{d}_T_rep": c["T_rep"], f"plus_{d}_T_polish": c["T_polish"], f"plus_{d}_probe_T": c["opts"]["probe_T"]})
    NUM["plus_m_target"] = float(np.log(3)); NUM["plus_floor_tolerance"] = 1e-6
if os.path.exists(f"{OUT}/protocol/explain_config_previous.json"):
    for d, c in json.load(open(f"{OUT}/protocol/explain_config_previous.json")).items():
        for k in ["lr", "lam1", "lam2"]: NUM[f"cfg_prev_{d}_{k}"] = c[k]

if os.path.exists(f"{OUT}/protocol/network_manifest_doh_full.json"):
    mf = json.load(open(f"{OUT}/protocol/network_manifest_doh_full.json")); NUM["doh_full_flows"] = int(mf["rows"])
# ───────────── T1 data protocol ─────────────
import flowsce as L
gs = rd("graph_stats.csv").set_index("dataset"); rows = []
for d in DS:
    st = L.prepare_dataset(d, "data", OUT, write=False); g = st["G"]["test"]; dom = st["dom"]
    n = {p: len(st["D"][p]) for p in ["train", "val", "test"]}; om = L.degree_stats(g)["omega_median"]
    tr = {r.tobytes() for r in np.round(st["X"]["train"], 6)}; dup = float(np.mean([r.tobytes() in tr for r in np.round(st["X"]["test"], 6)]))
    Fm = int(dom.mutable.sum()); F = len(dom.names); s = gs.loc[d]; flows = sum(n.values())
    rows.append([NAME[d], f"{flows:,}", f"{n['train']:,} / {n['val']:,} / {n['test']:,}", f"{F} / {Fm}", int(g["N"]), int(s.self_loops),
                 f"{int(s.flows_in_parallel_pairs):,}", sci_all(om), f(dup)])
    NUM.update({f"T1_{d}_flows": flows, f"T1_{d}_train": n["train"], f"T1_{d}_val": n["val"], f"T1_{d}_test": n["test"], f"T1_{d}_F": F,
                f"T1_{d}_Fmut": Fm, f"T1_{d}_hosts": int(g["N"]), f"T1_{d}_self_loops": int(s.self_loops),
                f"T1_{d}_parallel_flows": int(s.flows_in_parallel_pairs), f"T1_{d}_parallel_pairs": int(s.parallel_pairs), f"T1_{d}_omega": om,
                f"T1_{d}_dup": dup, f"T1_{d}_dup_pct": round(100 * dup, 1), f"T1_{d}_deg_median": float(s.deg_median), f"T1_{d}_deg_max": float(s.deg_max)})
table("tab_T1_data.tex", "Executed data protocol. Subsets rebuilt row by row from the official releases; stratified group split 70/10/20 "
      "(identical raw vectors never cross partitions, seed 0); preprocessing fitted on the training partition. Test graph: one snapshot of the "
      "test partition. $F/F_{\\rm mut}$: processed / mutable features. Parallel: flows that share their ordered endpoint pair with another flow. "
      "$\\omega$: median coupling over all test flows. Dup.: share of test vectors equal to a training vector after preprocessing.",
      "tab:data", ["Dataset", "Flows", "Train / val / test", "$F/F_{\\rm mut}$", "Hosts", "Self-loops", "Parallel", "$\\omega$", "Dup."],
      rows, "lrrrrrrrr", wide=True)

# ───────────── T3 detection ─────────────
det = pd.DataFrame(json.load(open(f"{OUT}/detector_metrics.json")))
MODELS = [("tuned", "FlowSCE detector (tuned)$^\\ast$"), ("flow", "FlowSCE detector (reduced context)"), ("flow_gate", "Reduced context + gates"), ("flow_full", "Full context"),
          ("flow_full_gate", "Full context + gates"), ("edge", "Edge-only (no context)"), ("egraphsage", "E-GraphSAGE$^\\dagger$"),
          ("candice_svm", "CANDICE-SVM$^\\dagger$"), ("candice_rf", "CANDICE-RF$^\\dagger$"), ("candice_mlp", "CANDICE-MLP$^\\dagger$")]
rows = []
for d in DS:
    for k, lab in MODELS:
        x = det[(det.dataset == d) & (det.kind == k)]
        if x.empty: continue
        mets = ["accuracy", "precision", "recall", "f1_attack", "macro_f1", "fpr"]
        flip = f"{100 * x['ctx_shuffle_flip'].mean():.1f}" if "ctx_shuffle_flip" in x and x["ctx_shuffle_flip"].notna().any() else "--"
        rows.append([NAME[d] if k == "tuned" else "", lab] + [pm(x[mm].mean(), x[mm].std()) for mm in mets] + [flip])
        for mm in mets + ["ctx_shuffle_flip", "ctx_shuffle_macro_f1"]:
            if mm in x and x[mm].notna().any(): NUM[f"T3_{d}_{k}_{mm}"] = float(x[mm].mean()); NUM[f"T3_{d}_{k}_{mm}_sd"] = float(x[mm].std())
        NUM[f"T3_{d}_{k}_ctx_flip_pct"] = round(100 * float(x["ctx_shuffle_flip"].mean()), 1) if "ctx_shuffle_flip" in x and x["ctx_shuffle_flip"].notna().any() else None
    rows.append("MID")
rows = rows[:-1]
NUM["T3_identity_gap_max"] = float(det.identity_gap.max()); NUM["T3_n_ckpts"] = int(len(det)); NUM["T3_n_models"] = int(det.kind.nunique()); NUM["T3_macro_f1_min_tuned"] = float(det[det.kind == "tuned"].macro_f1.min())
NUM["T3_n_checkpoints"] = len(det); NUM["T3_n_models"] = det.kind.nunique()
table("tab_T3_detection.tex", "Detection on the test partitions (mean $\\pm$ s.d.\\ over five seeds, threshold 0.5). Precision, recall, F1 and FPR "
      "refer to the attack class. $^\\dagger$Re-implementations trained on the same split and preprocessing (CANDICE: $\\beta$-TCVAE encoder with "
      "SVM / random forest / MLP on the flow attributes; E-GraphSAGE: mean aggregation over hosts). The explanations of Section~\\ref{sec:results} "
      "use the first model ($^\\ast$architecture and training settings selected on validation macro-F1: GAT context with one layer, "
      "learning rate $3\\times10^{-3}$ and dropout 0.1 on CIC-IDS2018 and DoHBrw-2020; GraphSAGE-mean context with gates on UNSW-NB15). "
      "Flip: \\% of test predictions that change when the rows of $X_V$ are permuted across hosts (models that read host context).", "tab:detector",
      ["Dataset", "Model", "Accuracy", "Precision", "Recall", "F1", "Macro-F1", "FPR", "Flip (\\%)"], rows, "llrrrrrrr", wide=True, size="\\footnotesize")

# ───────────── T4 RQ1-RQ3 ─────────────
rq1 = json.load(open(f"{OUT}/reconstruction_gradient_checks.json")); un = rd("uncertainty_ablation.csv")
TW = rd("coupling_sweep_time_queries.parquet"); FC = rd("coupling_fullcapture_doh_queries.parquet")
def sweep_summary(Q, tag):
    out = {}
    for (d, W), q in Q.groupby(["dataset", "W"]):
        A_, B_, C_ = (q[q.variant == v] for v in "ABC")
        bt = L.paired_cluster_bootstrap(q[q.variant.isin(["A", "C"])].assign(qc=q.query_id), "variant", "A", "C", "v", cluster_col="qc", ckpt_col="seed", B=2000)
        r = dict(spur=int(((B_.n == 1) & (B_.v == 0)).sum()), nom=int(B_.n.sum()), csrA=A_.v.mean(), csrB=B_.v.mean(), csrC=C_.v.mean(),
                 dAC=bt["estimate"], lo=bt["ci_low"], hi=bt["ci_high"], om=float(q.g_omega_median.median()),
                 mf1=float(q.groupby("seed").macro_f1.first().mean()), mf1_min=float(q.groupby("seed").macro_f1.first().min()),
                 dmean=float(B_.disc_stale.mean()), dp90=float(np.nanpercentile(B_.disc_stale, 90)), cosmin=float(q.cos_AC.min()), n=len(A_),
                 spurAC=int(((q.n == 1) & (q.v == 0) & q.variant.isin(["A", "C"])).sum()))
        r["svr"] = r["spur"] / r["nom"] if r["nom"] else float("nan")
        out[(d, W)] = r
        for k_, v_ in r.items(): NUM[f"{tag}_{d}_W{W:g}_{k_}"] = v_
    return out
SW = sweep_summary(TW, "swt") if TW is not None else {}
SF = sweep_summary(FC, "swf") if FC is not None else {}
rows = []
for d in DS:
    e = E[E.dataset == d]; B = e[e.variant == "B"]; sp_f, nm_f = int(((B.n == 1) & (B.v == 0)).sum()), int(B.n.sum())
    u = un[(un.dataset == d) & (un.a == "A") & (un.b == "C") & (un.metric == "CSR (v)")].iloc[0]
    w = {k: v for k, v in SW.items() if k[0] == d}; sw, nw = sum(v["spur"] for v in w.values()), sum(v["nom"] for v in w.values())
    best = max(w.items(), key=lambda kv: (kv[1]["dAC"], kv[1]["lo"])) if w else None
    zero = all(abs(v["dAC"]) < 5e-4 and abs(v["lo"]) < 5e-4 and abs(v["hi"]) < 5e-4 for v in w.values())
    lg = rq1[d].get("legacy_update", {}); lr_ = lg.get("random")
    rows.append([NAME[d], sci(rq1[d]["rebuild_max_abs"]), sci(rq1[d]["fd_max_abs"]),
                 "no self-loop" if not lg.get("self_loops") else f"{sci(lr_['xv_max_abs'])} / {sci(lr_['dq_max_abs'])}",
                 f"{sp_f}/{nm_f}", f"{sw}/{nw}", ci(u.estimate, u.ci_low, u.ci_high),
                 "0.000 at every $\\Delta t$" if zero else ci(best[1]["dAC"], best[1]["lo"], best[1]["hi"]) + f" ($\\Delta t{{=}}{best[0][1]:g}$\\,s)"])
    NUM.update({f"T4_{d}_rebuild": rq1[d]["rebuild_max_abs"], f"T4_{d}_fd": rq1[d]["fd_max_abs"], f"T4_{d}_chain": rq1[d]["chain_max_abs"],
                f"T4_{d}_orig_score": rq1[d]["original_score_max_abs"], f"T4_{d}_ctx_share": rq1[d]["ctx_share_median"],
                f"T4_{d}_spur_full": sp_f, f"T4_{d}_nom_full": nm_f, f"T4_{d}_spur_win": sw, f"T4_{d}_nom_win": nw,
                f"T4_{d}_dAC_full": float(u.estimate), f"T4_{d}_omega_reg": float(e.omega.median()),
                f"T4_{d}_disc_med": float(B.disc_stale.median()), f"T4_{d}_disc_max": float(B.disc_stale.max()),
                f"T4_{d}_disc_mean": float(B.disc_stale.mean()), f"T4_{d}_cosmin": float(e.cos_AC.min()) if "cos_AC" in e else None})
    if lr_: NUM.update({f"T4_{d}_legacy_xv": lr_["xv_max_abs"], f"T4_{d}_legacy_dq_max": lr_["dq_max_abs"], f"T4_{d}_legacy_dq_med": lr_["dq_median_abs"],
                        f"T4_{d}_legacy_flips": lr_["label_flips"], f"T4_{d}_legacy_n": lr_["n"]})
    NUM[f"T4_{d}_self_loops"] = lg.get("self_loops", 0)
    if best: NUM.update({f"T4_{d}_dAC_win_best": best[1]["dAC"], f"T4_{d}_dAC_win_best_lo": best[1]["lo"], f"T4_{d}_dAC_win_best_hi": best[1]["hi"], f"T4_{d}_dAC_win_best_W": best[0][1]})
if SF:
    sw, nw = sum(v["spur"] for v in SF.values()), sum(v["nom"] for v in SF.values()); best = max(SF.items(), key=lambda kv: (kv[1]["dAC"], kv[1]["lo"]))
    rows.append([NAME["doh_full"], "--", "--", "--", "--", f"{sw}/{nw}", "--", ci(best[1]["dAC"], best[1]["lo"], best[1]["hi"]) + f" ($\\Delta t{{=}}{best[0][1]:g}$\\,s)"])
    NUM.update(T4_dohfull_spur_win=sw, T4_dohfull_nom_win=nw, T4_dohfull_dAC_best=best[1]["dAC"], T4_dohfull_dAC_best_lo=best[1]["lo"], T4_dohfull_dAC_best_hi=best[1]["hi"])
NUM["T4_spur_win_total"] = sum(v["spur"] for v in SW.values()); NUM["T4_nom_win_total"] = sum(v["nom"] for v in SW.values())
NUM["T4_spurAC_win_total"] = sum(v["spurAC"] for v in SW.values()) + sum(v["spurAC"] for v in SF.values())
NUM["T4_svr_win_max"] = max(v["svr"] for v in SW.values() if not math.isnan(v["svr"])) if SW else None
NUM["T4_mf1_win_min"] = min(v["mf1_min"] for v in SW.values()) if SW else None
table("tab_T4_main.tex", "RQ1--RQ3 on the real detectors and alerts. RQ1: maximum absolute difference of the rank-one update from an "
      "independent rebuild (Rebuild) and of the total gradient from central finite differences (FD), float64; Legacy: a naive update rule that ignores "
      "the double incidence of a self-loop (Section~\\ref{sec:verification-design}), on every self-loop of the test graph, $\\max|\\Delta X_V|$ / $\\max|\\Delta q|$. RQ2: stale-context candidates rejected by "
      "reconstruction / nominal successes, on the full snapshot (500 registered alerts) and over the wall-clock windows "
      "$\\Delta t\\in\\{600,60,10,1\\}$\\,s (detector retrained per window, five seeds, 100 alerts each). RQ3: paired cluster-bootstrap "
      "difference in CSR between A and C with 95\\% interval; for the windows, the window with the largest difference. A and C never return a "
      "spurious candidate.", "tab:main",
      ["Dataset", "Rebuild", "FD", "Legacy", "Full", "Windows", "Full snapshot", "Windows (largest)"], rows, "lrrrrrll", wide=True, size="\\footnotesize",
      groups="& \\multicolumn{3}{c}{RQ1: reconstruction} & \\multicolumn{2}{c}{RQ2: spurious B} & \\multicolumn{2}{c}{RQ3: $\\Delta$CSR A$-$C} \\\\\n"
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(l){7-8}")

# ───────────── F1 coupling figure ─────────────
col = {"cic": "blue", "doh": "orange", "unsw": "teal", "doh_full": "red"}; mk = {"cic": "*", "doh": "square*", "unsw": "triangle*", "doh_full": "diamond*"}; plots = []
cic_max = max([float(E[(E.dataset == "cic") & (E.variant == "B")].disc_stale.max())] + ([float(TW[(TW.dataset == "cic") & (TW.variant == "B")].disc_stale.max())] if TW is not None else []))
for d, Q in [("cic", TW), ("doh", TW), ("unsw", TW), ("doh_full", FC)]:
    if Q is None: continue
    x = Q[(Q.dataset == d) & (Q.variant == "B")]
    pts = [] if d == "doh_full" else [(float(E[(E.dataset == d) & (E.variant == "B")].omega.median()), float(E[(E.dataset == d) & (E.variant == "B")].disc_stale.mean()))]
    pts += [(float(q.g_omega_median.median()), float(q.disc_stale.mean())) for _, q in sorted(x.groupby("W"), key=lambda t: -t[0])]
    pts = [p for p in pts if p[1] > 0]
    plots.append(f"\\addplot[{col[d]},thick,mark={mk[d]}] coordinates {{" + "".join(f"({a:.4g},{b:.4g})" for a, b in pts) + f"}};\n\\addlegendentry{{{NAME[d]}}}")
open(f"{P}/fig_F1_coupling.tex", "w").write(
    "\\begin{figure}[t]\n\\centering\n\\begin{tikzpicture}\n\\begin{axis}[width=.97\\linewidth,height=5.4cm,xmode=log,ymode=log,"
    "xlabel={median coupling $\\omega_{\\pi,k}$},ylabel={mean $|q_{\\rm rec}-q_{\\rm stale}|$ of B},tick label style={font=\\small},"
    "label style={font=\\small},ymajorgrids,grid style={black!10},legend style={at={(0.02,0.98)},anchor=north west,draw=none,font=\\scriptsize,fill=none}]\n"
    + "\n".join(plots) + "\n\\end{axis}\n\\end{tikzpicture}\n\\caption{Stale-context score discrepancy grows with coupling on real alerts. "
    "Points, in order of decreasing window length: the full test snapshot (median coupling over the registered alerts), then wall-clock windows "
    "of $\\Delta t\\in\\{600,60,10,1\\}$\\,s, each with a detector retrained on five seeds and 100 alerts per seed. The full-capture curve uses the "
    "complete DoHBrw-2020 layer-2 capture with $\\Delta t\\in\\{60,10,1\\}$\\,s. On UNSW-NB15 the 10\\,s and 1\\,s windows share the "
    "median coupling $\\sqrt2$ (both endpoints of degree one), so their points are stacked.}\n\\label{fig:coupling}\n\\end{figure}\n")
NUM["F1_cic_disc_max"] = cic_max

# ───────────── T5 ablation quality ─────────────
A0NF = E[E.variant == "A0"].set_index(["dataset", "seed", "query_id"]).n_fwd   # D and FlowSCE start from the saved A0 output: add its search cost
A0NB = E[E.variant == "A0"].set_index(["dataset", "seed", "query_id"]).n_bwd
_pl = sorted(glob.glob(f"{OUT}/ledger/PLUS_*.parquet")); PLUS = pd.concat([pd.read_parquet(p) for p in _pl], ignore_index=True) if _pl else None
rows = []
for d in DS:
    for v, lab in [("A0", "A0: search only"), ("A", "A: one restoration pass"), ("C", "C: detached gradient"), ("D", "D: repeated restoration"), ("Plus", "\\textbf{FlowSCE}")]:
        x = (PLUS[PLUS.dataset == d] if v == "Plus" else E[(E.dataset == d) & (E.variant == v)]) if (v != "Plus" or PLUS is not None) else None
        if x is None: continue
        s = x[x.v == 1]
        if v in ("D", "Plus"):
            ix = pd.MultiIndex.from_frame(s[["dataset", "seed", "query_id"]])
            s = s.assign(n_fwd=s.n_fwd.values + A0NF.reindex(ix).values, n_bwd=s.n_bwd.values + A0NB.reindex(ix).values)
        rows.append([NAME[d] if v == "A0" else "", lab, f(x.v.mean()), f(s.support_eps.median(), 0), f(s.support_eps.mean(), 2),
                     f(100 * float((s.support_eps <= 1).mean()), 1), f(s.l1.median(), 2), f(s.knn_benign.median(), 2), f(s.margin_logit.median()),
                     f(s.n_fwd.median(), 0), f(s.n_bwd.median(), 0)])
        NUM.update({f"T5_{d}_{v}_single_pct": 100 * float((s.support_eps <= 1).mean()), f"T5_{d}_{v}_nbwd": float(s.n_bwd.median())})
        NUM.update({f"T5_{d}_{v}_CSR": x.v.mean(), f"T5_{d}_{v}_supp": s.support_eps.median(), f"T5_{d}_{v}_supp_mean": s.support_eps.mean(),
                    f"T5_{d}_{v}_l1": s.l1.median(), f"T5_{d}_{v}_knn": s.knn_benign.median(), f"T5_{d}_{v}_margin": s.margin_logit.median(),
                    f"T5_{d}_{v}_nfwd": s.n_fwd.median(), f"T5_{d}_{v}_spurious": int(((x.n == 1) & (x.v == 0)).sum())})
        if v == "D": NUM[f"T5_{d}_D_budget_stops"] = int((x.restore_stop == "budget_stop").sum())
        if v == "Plus":
            NUM.update({f"T5_{d}_Plus_reach_pct": round(100 * float((s.margin_logit > np.log(3)).mean()), 1), f"T5_{d}_Plus_polished_pct": round(100 * float(s.polished.mean()), 1),
                        f"T5_{d}_Plus_supp_le_A": float((s.support_eps <= s.support_A).mean()), f"T5_{d}_Plus_margin_ge_A": float((s.margin_logit >= s.margin_A - 1e-4).mean()),
                        f"T5_{d}_Plus_supp1_pct": round(100 * float((s.support_eps <= 1).mean()), 1), f"T5_{d}_Plus_n_valid": int(len(s)),
                        f"T5_{d}_Plus_floor_ok_pct": round(100 * float((s.margin_logit >= np.minimum(s.margin_A, np.log(3)) - 1e-4).mean()), 1),
                        f"T5_{d}_Plus_repair_nochange_pct": round(100 * float((s.repair_stop == "no_change").mean()), 1),
                        f"T5_{d}_Plus_polish_budgetstop_pct": round(100 * float((s.polish_stop == "budget_stop").mean()), 1)})
            a0 = E[(E.dataset == d) & (E.variant == "A0")].set_index(["seed", "query_id"])[["support_eps", "l1", "margin_logit"]]
            jj = s.set_index(["seed", "query_id"]).join(a0, rsuffix="_A0"); jj = jj[jj.support_eps_A0 > 0]
            if len(jj):                                  # FlowSCE case study: same selection as scripts/run_analysis.py (iii)
                med = jj.support_eps.median(); c = jj[jj.support_eps == jj.support_eps.iloc[(jj.support_eps - med).abs().argmin()]]
                (cs, ck), r = next(iter(c.sort_values(["support_eps_A0", "l1"], ascending=[False, True]).iterrows()))
                NUM.update({f"caseP_{d}_query": int(ck), f"caseP_{d}_seed": int(cs), f"caseP_{d}_supp_A0": float(r.support_eps_A0), f"caseP_{d}_l1_A0": float(r.l1_A0),
                            f"caseP_{d}_margin_A0": float(r.margin_logit_A0), f"caseP_{d}_supp": float(r.support_eps), f"caseP_{d}_l1": float(r.l1),
                            f"caseP_{d}_margin": float(r.margin_logit), f"caseP_{d}_calls": float(r.n_fwd_method),
                            f"caseP_{d}_supp_A": float(r.support_A), f"caseP_{d}_margin_A": float(r.margin_A)})
    a0, a1, dd = (E[(E.dataset == d) & (E.variant == v)].set_index(["seed", "query_id"]) for v in ["A0", "A", "D"])
    j = (a0.v == 1) & (a1.v == 1)
    NUM[f"T5_{d}_supp_reduction_pct"] = round(100 * (1 - a1.support_eps[j].median() / a0.support_eps[j].median()), 0)
    NUM[f"T5_{d}_margin_lower_pct"] = round(100 * float((a1.margin_logit[j] < a0.margin_logit[j]).mean()), 0)
    jd = (dd.v == 1) & (a1.v == 1); NUM[f"T5_{d}_D_fewer_pct"] = round(100 * float((dd.support_eps[jd] < a1.support_eps[jd]).mean()), 0)
    jj = a0[["v", "support_eps", "margin_logit"]].join(a1[["v", "support_eps", "margin_logit"]], lsuffix="_A0", rsuffix="_A").query("v_A0 == 1 and v_A == 1")
    jj = jj.assign(drop=jj.margin_logit_A0 - jj.margin_logit_A).sort_values("drop", ascending=False)   # case study (ii) of results/case_studies.md
    (cs_seed, cs_k), cr = next(iter(jj.iterrows()))
    NUM.update({f"case_{d}_query": int(cs_k), f"case_{d}_seed": int(cs_seed), f"case_{d}_supp_A0": cr.support_eps_A0, f"case_{d}_supp_A": cr.support_eps_A,
                f"case_{d}_margin_A0": cr.margin_logit_A0, f"case_{d}_margin_A": cr.margin_logit_A})
    sb = E[(E.dataset == d) & (E.variant == "B") & (E.n == 1) & (E.v == 0)]
    if len(sb): NUM.update({f"case_{d}_spur_query": int(sb.iloc[0].query_id), f"case_{d}_spur_qstale": float(sb.iloc[0].q_nominal), f"case_{d}_spur_qrec": float(sb.iloc[0].q_reconstructed)})
    rows.append("MID")
table("tab_T5_quality.tex", "Matched ablation on the 500 registered alerts per dataset. CSR over all alerts; $s_\\epsilon$ (number of changed features, "
      "$\\epsilon=10^{-8}$), $L_1$ deviation, distance to the five nearest training-benign flows (kNN) and logit margin are medians (mean $s_\\epsilon$ "
      "also shown) over verified successes; 1-feat.: \\% of verified counterfactuals with a single changed feature; $N_f$ / $N_b$: median forward / backward "
      "evaluations of the whole explanation from the alert (for D and FlowSCE including the search that produced their starting point). The stale-context variant B is reported in Table~\\ref{tab:main}.",
      "tab:quality", ["Dataset", "Variant", "CSR", "$s_\\epsilon$", "mean $s_\\epsilon$", "1-feat.\\ (\\%)", "$L_1$", "kNN", "Margin", "$N_f$", "$N_b$"], rows[:-1], "llrrrrrrrrr", wide=True)

# ───────────── T6 generators ─────────────
G = rd("generator_comparison.csv")
if G is not None:
    rows = []; GM = [("FlowSCE (A0)", "FlowSCE search (A0)"), ("CANDICE", "CANDICE"), ("CANDICE (tuned λ)", "CANDICE (tuned $\\lambda$)"), ("DiCE", "DiCE (dice-ml)")]
    for d in DS:
        for m_, lab in GM:
            r = G[(G.dataset == d) & (G.method == m_)]
            if r.empty: continue
            r = r.iloc[0]
            rows.append([NAME[d] if m_ == GM[0][0] else "", lab, f(r.CSR), f(r.get("support_eps_median"), 0), f(r.get("l1_median"), 2), f(r.get("knn_benign_median"), 2),
                         f(r.get("margin_logit_median")), f(r.get("n_fwd_median"), 0)])
            for c in ["CSR", "support_eps_median", "support_eps_mean", "l1_median", "knn_benign_median", "margin_logit_median", "n_fwd_median", "time_per_alert_s",
                      "support_eps_mean_all_cfs", "verified"]:
                if c in r and pd.notna(r[c]): NUM[key("T6", d, m_, c)] = float(r[c])
        if PLUS is not None:                                            # complete FlowSCE: search (A0) + refinement, on the same alerts
            x = PLUS[PLUS.dataset == d]; s_ = x[x.v == 1]
            nf = s_.n_fwd.values + A0NF.reindex(pd.MultiIndex.from_frame(s_[["dataset", "seed", "query_id"]])).values
            rows.append(["", "\\textbf{FlowSCE} (search + refinement)", f(x.v.mean()), f(s_.support_eps.median(), 0), f(s_.l1.median(), 2), f(s_.knn_benign.median(), 2),
                         f(s_.margin_logit.median()), f(float(np.median(nf)), 0)])
            NUM[key("T6", d, "FlowSCE", "n_fwd_median")] = float(np.median(nf))
        rows.append("MID")
    table("tab_T6_generators.tex", "Generators through the shared reconstructed oracle, the same 500 registered alerts per dataset, mask, projection, "
          "validator and budget (500 updates, 4096 forward evaluations); no minimizer, except the last row of each dataset, the complete FlowSCE "
          "explainer (search followed by the refinement of Section~\\ref{sec:refine}). CANDICE: configuration of the original paper "
          "($\\lambda_1=\\lambda_2=0.5$, five counterfactuals from $\\mathcal N(0,0.1)$, learning rate chosen on validation alerts); CANDICE (tuned "
          "$\\lambda$): penalties tuned on validation alerts; DiCE: official library, gradient method, defaults except the iteration count. For CANDICE "
          "and DiCE the valid counterfactual with the smallest $s_\\epsilon$ is reported. Quality columns: medians over verified successes; $N_f$: "
          "median forward evaluations of successful queries (for FlowSCE, search and refinement together). Sequential run times are compared in Appendix~\\ref{app:candice}.", "tab:generators",
          ["Dataset", "Generator", "CSR", "$s_\\epsilon$", "$L_1$", "kNN", "Margin", "$N_f$"], rows[:-1], "llrrrrrr", wide=True)

# ───────────── T7 minimizers ─────────────
MN = rd("minimizer_comparison.csv")
MM = [("none", "none"), ("FlowSCE single (A)", "One restoration pass (A)"), ("FlowSCE repeated (D)", "Repeated restoration (D)"), ("FlowSCE", "\\textbf{FlowSCE}"),
      ("DiCE post-hoc", "DiCE post-hoc"), ("LBS-adapted", "LBS-adapted"), ("RHC", "RHC")]
if MN is not None:
    rows = []
    GN = [g_ for g_ in ["FlowSCE (A0)", "CANDICE (tuned λ)", "DiCE"] if g_ in set(MN.generator)]
    for gname in GN:
        for m_, lab in MM:
            cells = [{"FlowSCE (A0)": "FlowSCE search (A0)", "CANDICE (tuned λ)": "CANDICE (tuned $\\lambda$)"}.get(gname, gname) if m_ == "none" else "", lab]
            for d in DS:
                r = MN[(MN.generator == gname) & (MN.dataset == d) & (MN.method == m_)]
                if r.empty: cells += ["--"] * 5; continue
                r = r.iloc[0]
                cells += [f"{int(r.verified)}", f"{f(r.get('support_eps_median'), 0)} / {f(r.get('support_eps_mean'), 1)}", f(r.get("l1_median"), 2),
                          f(r.get("margin_logit_median"), 2), f(r.get("oracle_calls_median"), 0)]
                for c in ["verified", "support_eps_median", "support_eps_mean", "l1_median", "knn_benign_median", "margin_logit_median", "oracle_calls_median", "oracle_calls_mean"]:
                    if c in r and pd.notna(r[c]): NUM[key("T7", gname, d, m_, c)] = float(r[c])
            rows.append(cells)
        rows.append("MID")
    GLAB = {"FlowSCE (A0)": "FlowSCE search", "CANDICE (tuned λ)": "CANDICE (tuned $\\lambda$)"}
    lost = MN[(MN.method != "none") & (MN.verified < MN.merge(MN[MN.method == "none"][["generator", "dataset", "verified"]], on=["generator", "dataset"], suffixes=("", "_seed")).verified_seed.values)]
    for r in lost.itertuples(): NUM[key("T7", r.generator, r.dataset, r.method, "spurious")] = float(r.spurious)
    vtxt = ("Validity is that of the generator and is preserved by every minimizer." if lost.empty else
            "Validity is that of the generator and is preserved by every minimizer except " +
            "; ".join(f"{r.method} on {int(r.spurious)} {NAME[r.dataset]} {'alert' if int(r.spurious) == 1 else 'alerts'} of the "
                      f"{GLAB.get(r.generator, r.generator)} outputs" for r in lost.itertuples()) +
            ", whose output sits at the decision threshold and is rejected by the reconstructed check.")
    hdr = " & ".join(f"\\multicolumn{{5}}{{c}}{{{NAME[d]}}}" for d in DS)
    table("tab_T7_minimizers.tex", "Minimizers applied to the same valid counterfactuals of each generator (decoupled protocol of the LBS paper). "
          "$n$: valid seeds (the generator's verified successes among 500 alerts); $s_\\epsilon$: median / mean changed features; $L_1$ deviation and logit margin: medians; "
          "OC: median oracle calls of the minimizer itself. "
          "RHC uses the oracle calls of LBS on the same seed. " + vtxt,
          "tab:minimizers", ["Seed", "Minimizer"] + ["$n$", "$s_\\epsilon$", "$L_1$", "Marg.", "OC"] * 3, rows[:-1], "ll" + "rrrrr" * 3, wide=True, size="\\scriptsize",
          groups="& & " + hdr + " \\\\\n" + "".join(f"\\cmidrule(lr){{{3 + 5 * i}-{7 + 5 * i}}}" for i in range(3)))

# ───────────── T8 runtime ─────────────
rt = rd("runtime_scaling.csv")
if rt is not None:
    rows = []
    for d in DS:
        x = rt[rt.dataset == d]; fmax = x.F_mut.max()
        cells = [(fr, fmax) for fr in [.25, .5, 1.]] + [(1., fm) for fm in sorted(x.F_mut.unique()) if fm != fmax]
        for fr, fm in cells:
            c = x[(x.frac_flows == fr) & (x.F_mut == fm)]
            if c.empty: continue
            lat = c.latency_s
            rows.append([NAME[d] if (fr, fm) == cells[0] else "", int(c.N.iloc[0]), f"{int(c.M.iloc[0]):,}", int(fm), f(lat.mean()), f(lat.median()), f(np.percentile(lat, 95)),
                         f(c.t_generate.median()), f(c.t_restore.median()), f(c.t_assess.median(), 4), f(c.gpu_peak_mb.max(), 0), f(c.cpu_rss_mb.max(), 0),
                         f(c.v.mean(), 2), f"{c.n_fwd.median():.0f} / {c.n_bwd.median():.0f}"])
            k_ = f"T8_{d}_f{int(fr * 100)}_m{int(fm)}"
            NUM.update({k_ + "_mean": lat.mean(), k_ + "_median": lat.median(), k_ + "_p95": float(np.percentile(lat, 95)), k_ + "_gpu": c.gpu_peak_mb.max(),
                        k_ + "_rss": c.cpu_rss_mb.max(), k_ + "_gen": c.t_generate.median(), k_ + "_rest": c.t_restore.median(), k_ + "_assess": c.t_assess.median(),
                        k_ + "_csr": c.v.mean(), k_ + "_nf": c.n_fwd.median(), k_ + "_nb": c.n_bwd.median(), k_ + "_M": int(c.M.iloc[0]), k_ + "_N": int(c.N.iloc[0])})
        rows.append("MID")
    NUM["T8_gpu_max"] = float(rt.gpu_peak_mb.max())
    table("tab_T8_runtime.tex", "Sequential latency of FlowSCE (variant A, seed-11 checkpoint, 20 alerts per cell, one warm-up excluded, GPU-synchronized): "
          "test graphs from 25/50/100\\% of the test time blocks with all mutable features, and the full graph with the first 25/50\\% of them in schema order. "
          "Rows with a reduced mutable set mostly time failed searches that exhaust the update budget (cost of failure, not latency of an explanation). Latency in seconds; generate / restore / assess: medians; GPU: peak allocated MB; RSS: peak CPU resident MB of the process; "
          "$N_f/N_b$: median forward / backward evaluations.", "tab:runtime",
          ["Dataset", "$N$", "$M$", "$F_{\\rm mut}$", "Mean", "Median", "p95", "Gen.", "Rest.", "Assess", "GPU", "RSS", "CSR", "$N_f/N_b$"], rows[:-1],
          "lrrrrrrrrrrrrr", wide=True, size="\\scriptsize")

# ───────────── A1 bootstrap intervals ─────────────
rows = []; MET = {"CSR (v)": "CSR", "support_eps": "$s_\\epsilon$", "l1": "$L_1$", "knn_benign": "kNN", "margin_logit": "Margin", "n_fwd": "$N_f$", "oracle_calls": "OC"}
def add_ci(src, a, b, label, filt=None, tag=""):
    T = rd(src)
    if T is None: return
    if filt is not None: T = T[filt(T)]
    for d in DS:
        x = T[(T.dataset == d) & (T.a == a) & (T.b == b)]
        if x.empty: continue
        cells = [NAME[d], label]
        for met in MET:
            r = x[x.metric == met]
            if r.empty or pd.isna(r.estimate.iloc[0]): cells.append("--"); continue
            r = r.iloc[0]
            if met != "CSR (v)" and r.n_pairs < 20: cells.append(f"($n={int(r.n_pairs)}$)"); continue      # too few jointly solved alerts
            cells.append(ci(r.estimate, r.ci_low, r.ci_high, 3 if met in ("CSR (v)", "support_eps") else 2))
            NUM.update({key("A1", tag, d, a, b, met): r.estimate, key("A1", tag, d, a, b, met) + "_lo": r.ci_low, key("A1", tag, d, a, b, met) + "_hi": r.ci_high,
                        key("A1", tag, d, a, b, met) + "_n": r.n_pairs})
        rows.append(cells)
add_ci("uncertainty_ablation.csv", "A", "C", "A $-$ C", tag="abl"); add_ci("uncertainty_ablation.csv", "A", "B", "A $-$ B", tag="abl")
add_ci("uncertainty_ablation.csv", "A", "A0", "A $-$ A0", tag="abl"); add_ci("uncertainty_ablation.csv", "D", "A", "D $-$ A", tag="abl")
LAM = "$\\lambda$"
for b in ["CANDICE", "CANDICE (tuned λ)", "DiCE"]: add_ci("uncertainty_generator.csv", "FlowSCE (A0)", b, "A0 $-$ " + b.replace("λ", LAM), tag="gen")
for b in ["CANDICE (tuned λ)", "DiCE"]: add_ci("uncertainty_generator.csv", "FlowSCE", b, "FlowSCE $-$ " + b.replace("λ", LAM), tag="gen")
for a, b in [("FlowSCE", "LBS-adapted"), ("FlowSCE", "RHC"), ("FlowSCE", "FlowSCE single (A)"), ("FlowSCE single (A)", "LBS-adapted"), ("LBS-adapted", "RHC")]:
    la = {"FlowSCE": "FlowSCE", "FlowSCE single (A)": "A", "FlowSCE repeated (D)": "D"}.get(a, a); lb = {"FlowSCE single (A)": "A"}.get(b, b)
    add_ci("uncertainty_minimizer.csv", a, b, f"{la} $-$ {lb} (A0 seeds)", filt=lambda T: T.generator == "FlowSCE (A0)", tag="min")
for b in ["LBS-adapted", "RHC"]:
    add_ci("uncertainty_minimizer.csv", "FlowSCE", b, f"FlowSCE $-$ {b} (CANDICE tuned $\\lambda$ seeds)", filt=lambda T: T.generator == "CANDICE (tuned λ)", tag="minCT")
    add_ci("uncertainty_minimizer.csv", "FlowSCE", b, f"FlowSCE $-$ {b} (DiCE seeds)", filt=lambda T: T.generator == "DiCE", tag="minDiCE")
table("tab_A1_bootstrap.tex", "Paired cluster-bootstrap differences $a-b$ with 95\\% percentile intervals (2000 replicates; pairs: checkpoint, query, "
      "initialization; clusters: blocks of 500 test flows resampled within each checkpoint). CSR uses every alert; the other metrics use the alerts "
      "both methods solve (cells with fewer than 20 such alerts show only their number). $N_f$: forward evaluations; OC: oracle calls of the minimizer.", "tab:a-bootstrap", ["Dataset", "Contrast"] + list(MET.values()), rows, "ll" + "r" * len(MET),
      wide=True, size="\\scriptsize")

# ───────────── A2 minimizer budget ─────────────
MB = rd("minimizer_budget.csv")
if MB is not None:
    rows = []
    for d in DS:
        for B in sorted(MB.budget.unique()):
            cells = [NAME[d] if B == MB.budget.min() else "", int(B)]
            for m_ in ["LBS-adapted", "RHC", "FlowSCE repeated (D)"]:
                r = MB[(MB.dataset == d) & (MB.budget == B) & (MB.method == m_)]
                if r.empty: cells += ["--"] * 2; continue
                r = r.iloc[0]; cells += [f"{f(r.support_eps_median, 0)} / {f(r.support_eps_mean, 2)}", f(r.oracle_calls_median, 0)]
                NUM[key("A2", d, B, m_, "supp_mean")] = float(r.support_eps_mean); NUM[key("A2", d, B, m_, "oc")] = float(r.oracle_calls_median)
            rows.append(cells)
        rows.append("MID")
    table("tab_A2_budget.tex", "Minimizer budget on FlowSCE (A0) seeds: cap on the oracle calls of the minimizer. $s_\\epsilon$: median / mean; OC: median "
          "oracle calls used.", "tab:a-budget", ["Dataset", "Cap", "LBS $s_\\epsilon$", "LBS OC", "RHC $s_\\epsilon$", "RHC OC", "D $s_\\epsilon$", "D OC"],
          rows[:-1], "lrrrrrrr", size="\\footnotesize")

# ───────────── A3 CANDICE-paper metrics ─────────────
CM = rd("candice_metrics.csv")
if CM is not None:
    rows = []
    for d in DS:
        for m_ in ["FlowSCE", "FlowSCE, N(0,0.1) init", "FlowSCE (A)", "CANDICE", "DiCE (post-hoc)", "A0 + LBS-adapted"]:
            r = CM[(CM.dataset == d) & (CM.method == m_)]
            if r.empty: continue
            r = r.iloc[0]
            g_ = lambda c: r[c] if c in r and pd.notna(r[c]) else float("nan")
            rows.append([NAME[d] if m_ == "FlowSCE" else "", {"FlowSCE (A)": "One restoration pass (A)"}.get(m_, m_).replace("N(0,0.1)", "$\\mathcal N(0,0.1)$"), f(g_("ADA@1")), f(g_("ADA@2")), f(g_("ADA@5")), f(g_("ADA@10")),
                         f(g_("IS")), f(g_("seconds_500_alerts"), 0)])
            for c in ["ADA@1", "ADA@2", "ADA@5", "ADA@10", "IS", "seconds_500_alerts"]:
                if pd.notna(g_(c)): NUM[key("A3", d, m_, c.replace("@", ""))] = float(g_(c))
            if pd.notna(g_("seconds_500_alerts")): NUM[key("A3", d, m_, "seconds_per_alert")] = float(g_("seconds_500_alerts")) / 500
        fx, lx = CM[(CM.dataset == d) & (CM.method == "FlowSCE")], CM[(CM.dataset == d) & (CM.method == "A0 + LBS-adapted")]
        if len(fx) and len(lx): NUM[key("A3", d, "time_ratio_LBS")] = float(fx.seconds_500_alerts.iloc[0] / lx.seconds_500_alerts.iloc[0])
        rows.append("MID")
    table("tab_A3_candice_metrics.tex", "Explanation metrics of the CANDICE paper. ADA@$k$: mean attack probability after the top-$k$ explanatory "
          "features of each alert are set to zero in the standardized space (lower is more faithful). IS: mean intersection of the top-10 features "
          "over five runs (50 alerts per dataset); FlowSCE initializes at zero and is deterministic, so a variant initialized from $\\mathcal N(0,0.1)$ "
          "is also shown. Time: sequential seconds to explain the 500 registered alerts (DiCE: generation only, from its ledger).",
          "tab:a-candice", ["Dataset", "Method", "ADA@1", "ADA@2", "ADA@5", "ADA@10", "IS", "Time (s)"], rows[:-1], "llrrrrrr", wide=True, size="\\footnotesize")

# ───────────── A4 sensitivity ─────────────
se = rd("sensitivity.csv")
if se is not None:
    rows = []
    for d in DS:
        x = se[se.dataset == d]
        for r in x.itertuples():
            rows.append([NAME[d] if r.Index == x.index[0] else "", {"lam1": "$\\lambda_1$", "lam2": "$\\lambda_2$", "search_margin": "$\\kappa$",
                         "eps_support": "$\\epsilon$", "min_margin": "$\\mu_{\\min}$", "T": "$T$"}.get(r.factor, r.factor), f"{r.value:g}", f(r.CSR),
                         f(getattr(r, "support_eps_median", float("nan")), 0), f(getattr(r, "l1_median", float("nan")), 2), f(getattr(r, "margin_logit_median", float("nan")))])
            for c in ["CSR", "support_eps_median", "l1_median", "margin_logit_median"]:
                v_ = getattr(r, c, None)
                if v_ is not None and pd.notna(v_): NUM[key("A4", d, r.factor, f"{r.value:g}", c)] = float(v_)
        rows.append("MID")
    table("tab_A4_sensitivity.tex", "One-factor-at-a-time sensitivity of FlowSCE (A) on validation alerts of the seed-11 checkpoint. $\\mu_{\\min}$: minimum "
          "acceptance margin in search and restoration ($\\ln 3$ corresponds to an attack probability of at most 0.25).", "tab:a-sensitivity",
          ["Dataset", "Factor", "Value", "CSR", "$s_\\epsilon$", "$L_1$", "Margin"], rows[:-1], "llrrrrr", size="\\footnotesize")

# ───────────── A5 domain relations and projection ablation ─────────────
dc, sa = rd("domain_checks.csv"), rd("schema_ablation.csv")
rows = []
if dc is not None:
    for r in dc.itertuples():
        txt = f(getattr(r, "share_holding", float("nan"))) if r.step == "raw data" else (f(getattr(r, "share_violating", float("nan"))) if r.relation != "skipped" else "n/a")
        rel = "\\texttt{" + r.relation.replace("_", "\\_").replace("*", "$\\cdot$") + "}" if r.relation != "skipped" else "--"
        rows.append([NAME[r.dataset], r.step, "--" if r.method == "—" else r.method, rel, int(r.n) if pd.notna(r.n) else "--", txt])
        NUM[key("A5", r.dataset, r.step, r.method)] = float(getattr(r, "share_holding", float("nan")) if r.step == "raw data" else getattr(r, "share_violating", float("nan")))
    table("tab_A5_domain.tex", "Byte--duration--rate relations (diagnostic; not enforced by the validator). Raw data: share of flows on which the "
          "documented relation holds (relative tolerance 1\\%). Counterfactuals: share of valid counterfactuals that violate it, where the relation "
          "holds on every flow and all its fields are features (n/a: the relation does not hold on every raw flow, so counterfactuals are not checked).", "tab:a-domain", ["Dataset", "Step", "Method", "Relation", "$n$", "Share"], rows,
          "llllrr", wide=True, size="\\footnotesize")
if sa is not None:
    rows = []
    for d in DS:
        for s_ in sa.schema.unique():
            x = sa[(sa.dataset == d) & (sa.schema == s_)]
            if x.empty: continue
            rows.append([NAME[d] if s_ == sa.schema.unique()[0] else "", s_, int(x.n_mutable.iloc[0]), f(x.CSR.mean()), f(x.support_eps_med.median(), 0)])
            NUM[key("A5s", d, s_, "CSR")] = float(x.CSR.mean())
        rows.append("MID")
    table("tab_A5_projection.tex", "Schema and projection ablation of the generation step on the registered alerts (mean CSR over five checkpoints). "
          "S0: declared schema without in-loop projection; S0p: with projection after every update (Algorithm~\\ref{alg:flowsce}); "
          "+key / S1 / +all: additionally opening key immutable fields.", "tab:a-projection", ["Dataset", "Schema", "$F_{\\rm mut}$", "CSR", "$s_\\epsilon$"],
          rows[:-1], "llrrr", size="\\footnotesize")

# ───────────── A6 true / false alerts ─────────────
EE = pd.concat([E] + ([PLUS.assign(variant="FlowSCE")] if PLUS is not None else []), ignore_index=True)
EE = EE.merge(REG[["dataset", "seed", "query_id", "alert_type"]], on=["dataset", "seed", "query_id"], how="left", suffixes=("", "_reg"))
at = "alert_type_reg" if "alert_type_reg" in EE else "alert_type"
rows = []
for d in DS:
    for t in sorted(EE[at].dropna().unique()):
        for v in ["A0", "A", "C", "D", "FlowSCE"]:
            x = EE[(EE.dataset == d) & (EE[at] == t) & (EE.variant == v)]; s = x[x.v == 1]
            rows.append([NAME[d] if (t, v) == (sorted(EE[at].dropna().unique())[0], "A0") else "", t.replace("_alert", "") if v == "A0" else "", v, len(x), f(x.v.mean()),
                         f(s.support_eps.median(), 0), f(s.l1.median(), 2), f(s.margin_logit.median())])
            NUM.update({key("A6", d, t, v, "n"): len(x), key("A6", d, t, v, "CSR"): x.v.mean(), key("A6", d, t, v, "supp"): s.support_eps.median()})
    rows.append("MID")
table("tab_A6_alerttype.tex", "Matched ablation and FlowSCE by alert type (registered true and false alerts). Rows with $n<20$ (the single false alert of CIC-IDS2018) are not interpretable.", "tab:a-alerttype",
      ["Dataset", "Alert", "Variant", "$n$", "CSR", "$s_\\epsilon$", "$L_1$", "Margin"], rows[:-1], "lllrrrrr", size="\\footnotesize")


# ───────────── A7 per-window RQ2/RQ3 ─────────────
rows = []
for d, pre, Ws in [("cic", "swt", [600, 60, 10, 1]), ("doh", "swt", [600, 60, 10, 1]), ("unsw", "swt", [600, 60, 10, 1]), ("doh_full", "swf", [60, 10, 1])]:
    for W in Ws:
        k_ = f"{pre}_{d}_W{W}_"
        if k_ + "nom" not in NUM: continue
        g_ = lambda c: NUM.get(k_ + c)
        rows.append([({"doh_full": "DoHBrw-2020 (full)"}.get(d) or NAME[d]) if W == Ws[0] else "", W, sci(g_("om")), f(g_("mf1"), 4),
                     f"{int(g_('spur'))}/{int(g_('nom'))}", f(100 * g_("svr"), 1), f(g_("csrA")), f(g_("csrC")),
                     ci(g_("dAC"), g_("lo"), g_("hi"), 3), f(g_("cosmin"), 2)])
    rows.append("MID")
if rows:
    table("tab_A7_windows.tex", "Wall-clock window policies: detector retrained per window length and seed (five seeds, 100 alerts each). "
          "$\\omega$: median coupling of the registered alerts; Macro-F1: mean test macro-F1 of the retrained detectors; B spurious: stale-context "
          "candidates rejected by reconstruction / nominal successes (SVR in \\%); $\\Delta$CSR: A$-$C with paired cluster-bootstrap 95\\% interval; "
          "cos: minimum cosine between the search directions of A and C.", "tab:a-windows",
          ["Dataset", "$\\Delta t$ (s)", "$\\omega$", "Macro-F1", "B spurious", "SVR (\\%)", "CSR A", "CSR C", "$\\Delta$CSR A$-$C", "cos"], rows[:-1],
          "lrrrrrrrrr", wide=True, size="\\footnotesize")

# ───────────── A8 plausibility ─────────────
PZ = rd("plausibility.csv")
if PZ is not None:
    rows = []
    for d in DS:
        x = PZ[PZ.dataset == d]; ref = x[x.kind == "reference"].set_index("method").knn_median; pc = x[x.kind == "percentile"].set_index("method")
        for m_ in ["FlowSCE", "LBS-adapted", "CANDICE (tuned λ)"]:
            if m_ not in pc.index: continue
            r = pc.loc[m_]
            rows.append([NAME[d] if m_ == "FlowSCE" else "", m_.replace("λ", "$\\lambda$"), int(r.changed_values), f(100 * r.share_inside_p1_p99, 1),
                         f(100 * r.share_at_extremes, 1), f(ref.get("registered alerts (original)"), 2) if m_ == "FlowSCE" else "",
                         f(ref.get("benign test flows"), 2) if m_ == "FlowSCE" else ""])
            NUM.update({key("A8", d, m_, "inside"): 100 * r.share_inside_p1_p99, key("A8", d, m_, "extreme"): 100 * r.share_at_extremes})
        NUM.update({key("A8", d, "knn_alert"): ref.get("registered alerts (original)"), key("A8", d, "knn_benign"): ref.get("benign test flows")})
        fs = x[x.kind == "feature_share"]
        for i_, r_ in enumerate(fs.itertuples()): NUM[key("A8", d, "feat", i_ + 1)] = 100 * r_.share_of_explanations
        if len(fs): NUM[key("A8", d, "distinct_features")] = float(fs.distinct_features.iloc[0])
        rows.append("MID")
    table("tab_A8_plausibility.tex", "Where the counterfactual values lie. For every changed feature of a valid counterfactual on the registered "
          "alerts: share of raw counterfactual values inside the 1st--99th percentile of the training distribution of that feature, and share at the "
          "extremes ($\\le$0.1th or $\\ge$99.9th percentile). Reference scale of the kNN distance (median distance to the five nearest benign training "
          "flows): the registered alerts themselves and 500 benign test flows.", "tab:a-plausibility",
          ["Dataset", "Method", "Changed values", "Inside p1--p99 (\\%)", "At extremes (\\%)", "kNN alerts", "kNN benign"], rows[:-1], "llrrrrr", size="\\footnotesize")

# ───────────── T9 case table ─────────────
CT = rd("case_table.csv")
if CT is not None:
    fmtv = lambda v: "0" if abs(v) < 1e-6 else (f"{v:,.0f}" if abs(v) >= 100 else f"{v:.3g}")
    rows = []
    for r in CT.itertuples():
        rows.append([NAME[r.dataset], f"{r.alert} ({r.attack_type})", "\\texttt{" + str(r.feature).replace("_", "\\_") + "}", r.unit,
                     f"{fmtv(r.raw_original)} $\\to$ {fmtv(r.raw_counterfactual)}", f"{int(r.support_A0)} $\\to$ {int(r.support)}", f(r.margin, 2)])
        NUM.update({key("T9", r.dataset, r.alert, "margin"): r.margin, key("T9", r.dataset, r.alert, "orig"): r.raw_original,
                    key("T9", r.dataset, r.alert, "cf"): r.raw_counterfactual, key("T9", r.dataset, r.alert, "suppA0"): r.support_A0})
    table("tab_T9_cases.tex", "FlowSCE explanations of one true and one false alert per dataset (the alert whose support equals the median, "
          "largest search support first). Raw values in the units of the extractor documentation (``unknown'': not documented); "
          "support: changed features of the search output $\\to$ FlowSCE; margin: logit margin of the benign decision. The CIC-IDS2018 and DoHBrw-2020 "
          "true-alert changes leave the related byte and duration fields unchanged and are therefore not realizable traffic.", "tab:cases",
          ["Dataset", "Alert (class)", "Changed feature", "Unit", "Original $\\to$ counterfactual", "Support", "Margin"], rows, "lllllrr", wide=True, size="\\footnotesize")

# ───────────── A9 refinement ablation ─────────────
AB = rd("plus_ablation.csv")
if AB is not None:
    rows = []
    LAB = [("full", "complete refinement"), ("no_probe", "without single-feature probes"), ("no_repair", "without repair"), ("no_polish", "without polishing"),
           ("no_shrink", "without $L_1$ shrinking"), ("cap1000", "forward budget 1000"), ("cap500", "forward budget 500"), ("cap200", "forward budget 200"),
           ("probe_only", "probes only, no generator")]
    for d in DS:
        for v, lab in LAB:
            r = AB[(AB.dataset == d) & (AB.ablation == v)]
            if r.empty: continue
            r = r.iloc[0]
            rows.append([NAME[d] if v == "full" else "", lab, f(r.CSR), f(r.support_eps_mean, 3), f(100 * r.single_feature, 1), f(r.l1_median, 2),
                         f(r.knn_benign_median, 2), f(r.margin_logit_median), f(100 * r.margin_ge_ln3, 1), f(r.oracle_calls_median, 0), f(r.n_bwd_median, 0)])
            for c in ["CSR", "support_eps_mean", "single_feature", "l1_median", "knn_benign_median", "margin_logit_median", "margin_ge_ln3", "oracle_calls_median", "n_bwd_median", "refine_seconds_median"]:
                NUM[key("A9", d, v, c)] = float(r[c])
        rows.append("MID")
    table("tab_A9_ablation.tex", "Ablation of the FlowSCE refinement on the registered alerts, starting from the same search outputs (A0): one step "
          "removed at a time, the complete refinement under a reduced forward budget (compare with the oracle calls of LBS-adapted in "
          "Table~\\ref{tab:minimizers}), and the single-feature probes run from the alert itself without a generator (CSR over all alerts; "
          "the other columns over its successes). Quality columns over verified successes; margin $\\ge\\ln3$: share reaching the polishing target; "
          "OC / $N_b$: median forward / backward evaluations of the refinement.", "tab:a-ablation",
          ["Dataset", "Refinement", "CSR", "mean $s_\\epsilon$", "1-feat.\\ (\\%)", "$L_1$", "kNN", "Margin", "$\\ge\\ln3$ (\\%)", "OC", "$N_b$"], rows[:-1],
          "llrrrrrrrrr", wide=True, size="\\footnotesize")

# ───────────── A10 per-alert coupling, stale search with a final check, false vs true alerts ─────────────
AA = rd("alert_analysis.csv")
if AA is not None:
    rows = []
    for r in AA[AA.analysis == "coupling"].itertuples():
        lab = f"$\\omega\\in({r.lo:g},{r.hi:g}]$".replace("inf", "\\infty")
        rows.append(["per-alert coupling", lab, f"{int(r.spurious)}/{int(r.nominal)}", f(100 * r.rate, 1)])
        NUM[key("A10", "omega", r.lo, "rate")] = 100 * r.rate; NUM[key("A10", "omega", r.lo, "spur")] = r.spurious; NUM[key("A10", "omega", r.lo, "nom")] = r.nominal
    for r in AA[AA.analysis == "stale_margin"].itertuples():
        lab = f"stale margin $\\in({r.lo:g},{r.hi:g}]$".replace("inf", "\\infty")
        rows.append(["stale margin of B", lab, f"{int(r.spurious)}/{int(r.nominal)}", f(100 * r.rate, 1)])
        NUM[key("A10", "smargin", r.lo, "rate")] = 100 * r.rate
    for r in AA[AA.analysis == "coupling_by_dataset"].itertuples():
        NUM[key("A10ds", r.dataset, r.lo, "rate")] = 100 * r.rate; NUM[key("A10ds", r.dataset, r.lo, "spur")] = r.spurious; NUM[key("A10ds", r.dataset, r.lo, "nom")] = r.nominal
    cd = AA[AA.analysis == "coupling_by_dataset"]
    if len(cd):
        for ds_ in ["cic", "doh", "doh_full", "unsw"]:
            x = cd[cd.dataset == ds_]
            if x.empty: continue
            rows.append([({"doh_full": "DoHBrw-2020 (full)"}.get(ds_) or NAME[ds_]), "by coupling bin", " / ".join(f"{int(r.spurious)}/{int(r.nominal)}" for r in x.itertuples()), "--"])
    table("tab_A10_coupling.tex", "Spurious validity of the stale-context variant B among its nominal successes, pooled over all wall-clock window "
          "policies (subsets and complete DoHBrw-2020 capture), by the coupling $\\omega_{\\pi,k}$ of the individual alert and by B's own stale logit margin; "
          "last rows: the same coupling bins ($\\le$0.01, 0.01--0.1, 0.1--0.5, $>$0.5) per dataset.",
          "tab:a-coupling", ["Grouping", "Bin", "Spurious / nominal", "Rate (\\%)"], rows, "llrr", size="\\footnotesize")
    rows = []
    for r in AA[AA.analysis == "window"].sort_values(["dataset", "W"], ascending=[True, False]).itertuples():
        rows.append([{"doh_full": "DoHBrw-2020 (full)"}.get(r.dataset) or NAME[r.dataset], f"{r.W:g}", f(r.csr_B_checked), f(r.csr_A),
                     ci(r.dAC, r.bonf_lo, r.bonf_hi, 3), f(r.LV_ratio_median, 3)])
        k_ = f"A10w_{r.dataset}_W{int(r.W)}_"
        NUM.update({k_ + "csrB": r.csr_B_checked, k_ + "csrA": r.csr_A, k_ + "bonf_lo": r.bonf_lo, k_ + "bonf_hi": r.bonf_hi, k_ + "LV": r.LV_ratio_median})
        NUM[k_ + "gap_pts"] = 100 * (r.csr_A - r.csr_B_checked)
    table("tab_A10_windows.tex", "Window policies: CSR of a stale search followed by a check on the reconstructed graph (B checked) and of FlowSCE's "
          "search (A); A$-$C CSR difference with a Bonferroni-adjusted 95\\% bootstrap interval over the 15 policies (alerts resampled within "
          "checkpoints); and the median ratio $|q_{\\rm rec}-q_{\\rm stale}|/(\\omega_{\\pi,k}\\norm{\\delta}_2)$ of B's candidates, an empirical "
          "counterpart of $L_V$ in Proposition~\\ref{prop:mismatch}.", "tab:a-windows2",
          ["Dataset", "$\\Delta t$ (s)", "CSR B checked", "CSR A", "$\\Delta$CSR A$-$C (Bonferroni)", "$\\hat L_V$"], rows, "lrrrrr", size="\\footnotesize")
    for r in AA[AA.analysis == "floor"].itertuples(): NUM[key("A10", r.dataset, "floor_below_ln3")] = 100 * r.share_floor_below_ln3
    for r in AA[AA.analysis == "false_vs_true"].itertuples(): NUM[key("A10", r.dataset, "auroc", r.score)] = r.auroc
MNr = rd("minimizer_comparison.csv")
if MNr is not None:
    for gname in ["FlowSCE (A0)", "CANDICE (tuned λ)"]:
        for d in DS:
            fl = MNr[(MNr.generator == gname) & (MNr.dataset == d) & (MNr.method == "FlowSCE")]; lb = MNr[(MNr.generator == gname) & (MNr.dataset == d) & (MNr.method == "LBS-adapted")]
            if len(fl) and len(lb): NUM[key("OCratio", gname, d)] = float(fl.oracle_calls_median.iloc[0] / lb.oracle_calls_median.iloc[0])

# ───────────── A11 percentile-constrained domain ─────────────
CS = rd("constrained.csv")
if CS is not None:
    rows = []
    for d in DS:
        for m_, lab in [("FlowSCE (constrained)", "FlowSCE"), ("A0 + LBS-adapted (constrained)", "search + LBS-adapted")]:
            r = CS[(CS.dataset == d) & (CS.method == m_)]
            if r.empty: continue
            r = r.iloc[0]
            rows.append([NAME[d] if lab == "FlowSCE" else "", lab, f(r.CSR), f(r.support_eps_mean, 2), f(100 * r.single_feature, 1), f(r.l1_median, 2),
                         f(r.knn_benign_median, 2), f(r.margin_logit_median), f(r.oracle_calls_median, 0)])
            for c in ["CSR", "support_eps_mean", "single_feature", "l1_median", "knn_benign_median", "margin_logit_median", "oracle_calls_median"]:
                NUM[key("A11", d, lab, c)] = float(r[c])
        bt = CS[(CS.dataset == d) & (CS.method == "FlowSCE - LBS (paired)")].set_index("metric")
        if len(bt):
            cells = ["", "FlowSCE $-$ LBS (paired mean)", "--"]
            for met, dd in [("support_eps", 2), (None, None), ("l1", 2), ("knn_benign", 2), ("margin_logit", 2), ("n_fwd_method", 0)]:
                if met is None: cells.append("--"); continue
                r = bt.loc[met]; cells.append(ci(r.estimate, r.ci_low, r.ci_high, dd))
                NUM.update({key("A11", d, "diff", met): r.estimate, key("A11", d, "diff", met) + "_lo": r.ci_low, key("A11", d, "diff", met) + "_hi": r.ci_high})
            rows.append(cells)
        rows.append("MID")
    table("tab_A11_constrained.tex", "Percentile-constrained domain: the training range of every feature is narrowed to its 1st--99th training "
          "percentile in projection, validator, and search box; locked settings otherwise. Both methods start from the same constrained search. "
          "CSR over the 500 registered alerts; quality medians over verified successes; last row: paired cluster-bootstrap mean difference with 95\\% interval.",
          "tab:a-constrained", ["Dataset", "Method", "CSR", "mean $s_\\epsilon$", "1-feat.\\ (\\%)", "$L_1$", "kNN", "Margin", "OC"], rows[:-1],
          "llrrrrrrr", wide=True, size="\\footnotesize")

# ───────────── A12 statistics quoted in the text ─────────────
def g3(fmt, *ks):
    v = [NUM.get(k) for k in ks]
    return " / ".join("--" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x) for x in v)
rows = [
 ["Repeated restoration removes further features (\\% of alerts)", g3("{:.0f}", *[f"T5_{d}_D_fewer_pct" for d in DS])],
 ["Single-pass restoration lowers the margin (\\% of alerts)", g3("{:.0f}", *[f"T5_{d}_margin_lower_pct" for d in DS])],
 ["Repair ends with a no-change pass (\\%)", g3("{:.1f}", *[f"T5_{d}_Plus_repair_nochange_pct" for d in DS])],
 ["Polishing reaches $q<-\\ln3$ (\\%)", g3("{:.1f}", *[f"T5_{d}_Plus_reach_pct" for d in DS])],
 ["Margin floor $\\min(\\mu_A,\\ln3)$ below $\\ln3$ (\\%)", g3("{:.1f}", *[key("A10", d, "floor_below_ln3") for d in DS])],
 ["FlowSCE output with one changed feature (\\%)", g3("{:.1f}", *[f"T5_{d}_Plus_supp1_pct" for d in DS])],
 ["Most frequently changed feature (\\% of explanations)", g3("{:.1f}", *[key("A8", d, "feat", 1) for d in DS])],
 ["Distinct features changed by FlowSCE", g3("{:.0f}", *[key("A8", d, "distinct_features") for d in DS])],
 ["Context share of the search gradient, full snapshot (max)", g3("{:.1e}", *[f"T4_{d}_ctx_share" for d in DS])],
 ["Min.\\ cosine of A and C search directions, full snapshot", g3("{:.5f}", *[f"T4_{d}_cosmin" for d in DS])],
 ["AUROC false vs.\\ true alerts: $L_1$ of FlowSCE (DoH / UNSW)", g3("{:.3f}", key("A10", "doh", "auroc", "l1"), key("A10", "unsw", "auroc", "l1"))],
 ["AUROC false vs.\\ true alerts: detector score (DoH / UNSW)", g3("{:.3f}", key("A10", "doh", "auroc", "q_orig"), key("A10", "unsw", "auroc", "q_orig"))],
 ["Refinement time per alert, five parallel processes: complete (s)", g3("{:.1f}", *[key("A9", d, "full", "refine_seconds_median") for d in DS])],
 ["Refinement time per alert, five parallel processes: without probes (s)", g3("{:.1f}", *[key("A9", d, "no_probe", "refine_seconds_median") for d in DS])],
]
table("tab_A12_stats.tex", "Statistics quoted in the text that are not columns of other tables (CIC-IDS2018 / DoHBrw-2020 / UNSW-NB15 unless stated).",
      "tab:a-stats", ["Quantity", "Value"], rows, "p{0.62\\linewidth}l", size="\\footnotesize")

# ───────────── reference verification on constructed inputs (reference/run_reference_verification.py) ─────────────
RV = f"{OUT}/reference_verification/results.json"
if os.path.exists(RV):
    rv = json.load(open(RV)); cases = {t["name"]: t["cases"] for t in rv["tests"]}
    NUM.update(ref_suite_groups=rv["groups_passed"], ref_suite_cases=rv["cases_passed"], ref_unittests=rv.get("contract_tests_run"),
               ref_graphs=cases.get("exact_snapshot_vs_independent_rebuild"), ref_excluded_cases=cases.get("fixed_reference_invariance"),
               ref_fd_predictors=cases.get("total_gradient_finite_difference_attention_oracle"),
               ref_restore_cases=cases.get("restoration_validity_support_lp_joint_deviation"),
               ref_rebuild_maxdev=rv["metrics"]["max_snapshot_abs_error"], ref_fd_maxdev=rv["metrics"]["max_gradient_fd_abs_error"],
               ref_fd_h=rv.get("fd_step"), ref_chain_maxdev=rv["metrics"]["max_gradient_chain_abs_error"],
               ref_excluded_maxdev=rv.get("excluded_reference_max_abs_error"),
               ref_all_pass=bool(all(t["status"] == "PASS" for t in rv["tests"]) and rv.get("contract_tests_ok", False)),
               ref_linear_coupled_updates=rv["metrics"]["linear_search"]["coupled_updates"],
               ref_linear_detached_updates=rv["metrics"]["linear_search"]["detached_updates"],
               ref_counterexample_frozen=rv["metrics"]["counterexample"]["frozen_score"],
               ref_counterexample_rebuilt=rv["metrics"]["counterexample"]["reconstructed_score"])
if PLUS is not None: NUM["plus_total_valid"] = int(PLUS.v.sum()); NUM["plus_total_alerts"] = int(len(PLUS))
json.dump({k: (v.item() if hasattr(v, "item") else v) for k, v in NUM.items()}, open(f"{P}/numbers.json", "w"), indent=1, default=str)
print("numbers.json:", len(NUM), "values")
