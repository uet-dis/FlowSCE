"""One-call deterministic dataset preparation (load → split → preprocessing → schema → graphs)."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DATASETS, sha256, sha_array, env_info, save_json
from .data import load, fetch, dedup_split, add_time_blocks, Prep
from .schema import Domain
from .graph import build_graph

# ══════════════════════════════ One-call dataset preparation ══════════════════════════════
def prepare_dataset(ds, data_dir, out_dir, split_seed=0, block_size=500, write=True):
    """Deterministic: load → dedup split → blocks → train-fit preprocessing → schema → full-snapshot graphs.
    Writes splits/<ds>_split.csv and protocol/network_manifest_<ds>.json."""
    df = load(ds, data_dir); part, grp = dedup_split(df, ds, split_seed); df["_part"] = part; df["_group"] = grp
    reg = f"{out_dir}/splits/{ds}_split.csv"            # the registered split is fixed: refuse to continue on any other split
    if os.path.exists(reg):
        R = pd.read_csv(reg, usecols=["row_id", "part"]).set_index("row_id").part
        mism = int((R.reindex(df["row_id"]).values != df["_part"].values).sum())
        if mism: raise RuntimeError(f"{ds}: recomputed split differs from {reg} on {mism} rows; check the software environment "
                                    "(results/protocol/config.json lists the versions that produced the registered split)")
    df = add_time_blocks(df, block_size)
    prep = Prep().fit(df, ds); dom = Domain(prep, df[df._part == "train"], ds)
    X = {p: prep.transform(df[df._part == p]) for p in ["train", "val", "test"]}; D = {p: df[df._part == p] for p in X}
    G = {p: build_graph(D[p], X[p], ds) for p in X}
    if write:
        os.makedirs(f"{out_dir}/splits", exist_ok=True); os.makedirs(f"{out_dir}/protocol", exist_ok=True)
        split_path = f"{out_dir}/splits/{ds}_split.csv"
        df[["row_id", "_part", "_group", "_block"]].rename(columns=lambda c: c.strip("_")).to_csv(split_path, index=False)
        tr_keys = set(map(bytes, np.round(X["train"], 5)))
        c = DATASETS[ds]
        man = dict(dataset=ds, source=c["source"], file=c["file"], file_sha256=sha256(fetch(ds, data_dir)), rows=len(df),
                   label_column=c["label"], label_mapping=c["label_map"], attack_type_column=c["strat"],
                   endpoint_columns=[c["src"], c["dst"]], time_column=c["time"],
                   split=dict(rule="stratified group split; group = identical raw feature vector; strata = attack type",
                              seed=split_seed, ratios=[.7, .1, .2], file=split_path, file_sha256=sha256(split_path),
                              achieved={p: int((df._part == p).sum()) for p in ["train", "val", "test"]},
                              classes={p: D[p]["_strat"].value_counts().to_dict() for p in D},
                              attack_rate={p: float(D[p]["_y"].mean()) for p in D},
                              test_processed_duplicates_in_train=float(np.mean([bytes(r) in tr_keys for r in np.round(X["test"], 5)]))),
                   preprocessing=dict(prep.summary(), fitted_on="train", median_sha=sha_array(prep.median.values),
                                      mu_sha=sha_array(prep.mu.values), sd_sha=sha_array(prep.sd.values)),
                   bootstrap_cluster=f"contiguous time blocks of {block_size} flows within each partition",
                   env=env_info())
        save_json(man, f"{out_dir}/protocol/network_manifest_{ds}.json")
    return dict(ds=ds, df=df, prep=prep, dom=dom, X=X, D=D, G=G)
