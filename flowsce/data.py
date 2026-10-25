"""Data loading, deduplicated group split, time blocks and train-only preprocessing."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DATASETS, META_COLUMNS

# ══════════════════════════════ Data & dedup split ══════════════════════════════
def fetch(ds: str, data_dir: str) -> str:
    path = os.path.join(data_dir, DATASETS[ds]["file"])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: build the subsets from the official releases with scripts/prepare_data.py")
    return path

def load(ds: str, data_dir: str) -> pd.DataFrame:
    """Loads parquet; adds row_id (row index in file), _y (0 benign / 1 attack), _t (int time), _strat."""
    c = DATASETS[ds]; df = pd.read_parquet(fetch(ds, data_dir)).reset_index(drop=True)
    if "row_id" not in df: df.insert(0, "row_id", np.arange(len(df)))
    assert (df["row_id"].values == np.arange(len(df))).all(), "row_id must equal the row index"
    lab = df[c["label"]]
    df["_y"] = (lab.astype(str).str.strip() != "Benign").astype(int) if c["label_map"] == "benign_string" else lab.astype(int)
    t = df[c["time"]]
    df["_t"] = pd.to_datetime(t.astype(str)).astype("int64") if t.dtype == object or str(t.dtype).startswith("str") else t.astype("int64")
    st = df[c["strat"]].astype(object); st = st.where(st.notna(), "Normal")          # UNSW: benign rows have no attack_cat
    df["_strat"] = st.astype(str).str.strip().replace({"nan": "Normal", "None": "Normal", "<NA>": "Normal", "": "Normal"})
    for k in ("src", "dst"): df[c[k]] = df[c[k]].astype(str)
    return df

def feature_frame(df, ds):
    c = DATASETS[ds]
    meta = META_COLUMNS + ["_y", "_t", "_strat", "_part", "_group", "_block", c["label"], c["src"], c["dst"]] + c["drop"]
    return df.drop(columns=[m for m in meta if m in df.columns])

def dedup_split(df, ds, seed=0, ratios=(.7, .1, .2)):
    """Stratified GROUP split: rows with identical raw feature vectors form one group and land in
    one partition (removes exact train/test duplicates). Groups are keyed by their majority attack type."""
    groups = pd.util.hash_pandas_object(feature_frame(df, ds).astype(str), index=False).values
    g = pd.DataFrame({"g": groups, "s": df["_strat"].values})
    gs = g.groupby("g").agg(s=("s", lambda x: x.value_counts().index[0]), n=("s", "size")).reset_index()
    rng = np.random.default_rng(seed); assign = {}
    for _, sub in gs.groupby("s"):
        sub = sub.sample(frac=1, random_state=int(rng.integers(1 << 30))); cum = sub.n.cumsum() / sub.n.sum()
        lab = np.where(cum <= ratios[0], 0, np.where(cum <= ratios[0] + ratios[1], 1, 2))
        if (lab == 2).sum() == 0: lab[-1] = 2
        if (lab == 1).sum() == 0 and len(lab) > 2: lab[-2] = 1
        assign.update(dict(zip(sub.g, lab)))
    part = np.array(["train", "val", "test"])[g.g.map(assign).values]
    return part, groups

def add_time_blocks(df, size=500):
    """Contiguous time blocks inside each partition: bootstrap clusters ('graph/session blocks')."""
    blk = np.empty(len(df), dtype=object)
    for p, sub in df.groupby("_part"):
        order = sub.sort_values("_t", kind="stable").index.values
        blk[order] = [f"{p}-{i // size}" for i in range(len(order))]
    df["_block"] = blk
    return df

# ══════════════════════════════ Preprocessing (fit on train only) ══════════════════════════════
class Prep:
    """Constant drop → median fill → |corr|>0.9 drop → log1p on skewed non-negative → standardize.
    Every statistic is fitted on the training partition only."""
    def fit(self, df, ds, corr_thr=0.9):
        c = DATASETS[ds]; X = feature_frame(df[df._part == "train"], ds)
        self.cats = {k: X[k].astype(str).value_counts().index[:8].tolist() for k in c["cat"]}
        X = self._onehot(X).replace([np.inf, -np.inf], np.nan).astype(float)
        self.const = [k for k in X.columns if X[k].nunique(dropna=True) <= 1]
        X = X.drop(columns=self.const); self.median = X.median(); X = X.fillna(self.median)
        corr = X.corr().abs().fillna(0); up = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
        self.corrdrop = [k for k in up.columns if (up[k] > corr_thr).any()]
        X = X.drop(columns=self.corrdrop)
        self.log = [k for k in X.columns if X[k].nunique() > 20 and X[k].skew() > 2 and X[k].max() > 100 and (X[k] >= 0).all()]
        X[self.log] = np.log1p(X[self.log])
        self.cols = X.columns.tolist(); self.mu = X.mean(); self.sd = X.std().replace(0, 1.0)
        self.ds = ds
        return self
    def _onehot(self, X):
        X = X.copy()
        for k, vals in self.cats.items():
            v = X[k].astype(str)
            for u in vals: X[f"{k}={u}"] = (v == u).astype(float)
            X = X.drop(columns=k)
        return X
    def raw_frame(self, df):
        X = self._onehot(feature_frame(df, self.ds)).replace([np.inf, -np.inf], np.nan).astype(float)
        return X[self.cols].fillna(self.median)
    def transform(self, df):
        X = self.raw_frame(df); X[self.log] = np.log1p(X[self.log].clip(lower=0))
        return ((X - self.mu) / self.sd).values.astype(np.float32)
    def summary(self):
        return dict(n_features=len(self.cols), dropped_constant=self.const, dropped_corr=self.corrdrop,
                    log1p=self.log, onehot=self.cats)
