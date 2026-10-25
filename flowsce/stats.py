"""Statistics: detector summary and paired cluster bootstrap."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

# ══════════════════════════════ Statistics ══════════════════════════════
def detector_table(metrics_rows):
    df = pd.DataFrame(metrics_rows)
    cols = ["accuracy", "precision", "recall", "specificity", "f1_attack", "macro_f1", "fpr"]
    agg = df.groupby(["dataset", "kind"])[cols].agg(["mean", "std"])
    return agg

def paired_cluster_bootstrap(df, method_col, a, b, metric, cluster_col="block", ckpt_col="seed", B=2000, seed=0,
                             success_only=False):
    """Difference metric(a) − metric(b), paired on (checkpoint, query). Clusters (time blocks) are resampled within
    each checkpoint; checkpoints are kept (hierarchy preserved) and averaged. For quality metrics use
    success_only=True → intersection of jointly verified queries (denominator reported)."""
    rng = np.random.default_rng(seed)
    wide = df.pivot_table(index=[ckpt_col, "query_id", cluster_col], columns=method_col, values=metric, aggfunc="first")
    if a not in wide.columns or b not in wide.columns: return dict(n_pairs=0, estimate=None, ci_low=None, ci_high=None)   # no value of a or b
    if success_only:
        vv = df.pivot_table(index=[ckpt_col, "query_id", cluster_col], columns=method_col, values="v", aggfunc="first")
        vv = vv.reindex(wide.index); wide = wide[((vv[a] == 1) & (vv[b] == 1)).values]
    wide = wide[[a, b]].dropna(); diff = (wide[a] - wide[b]).rename("d").reset_index()
    if diff.empty: return dict(n_pairs=0, estimate=None, ci_low=None, ci_high=None)
    groups = {c: [g.d.values for _, g in sub.groupby(cluster_col)] for c, sub in diff.groupby(ckpt_col)}
    est = float(np.mean([np.concatenate(v).mean() for v in groups.values()]))
    boots = []
    for _ in range(B):
        vals = []
        for v in groups.values():
            pick = rng.integers(0, len(v), len(v)); vals.append(np.concatenate([v[i] for i in pick]).mean())
        boots.append(np.mean(vals))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return dict(n_pairs=len(diff), n_checkpoints=len(groups), estimate=est, ci_low=float(lo), ci_high=float(hi))
