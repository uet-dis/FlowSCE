#!/usr/bin/env python3
"""
E4 (supplement): the complete layer-2 capture of CIRA-CIC-DoHBrw-2020 (l2-benign.csv + l2-malicious.csv), parsed exactly
like the 49,807-flow subset, for the time-window coupling sweep on naturally dense traffic.

    python scripts/prepare_doh_full.py --archives /path/to/downloads      # writes data/cira_cic_dohbrw2020_l2_full.parquet

The dataset is registered as "doh_full" in flowsce/config.py (same columns, schema and immutable fields as "doh").
"""
import argparse, hashlib, os, re, sys, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--archives", required=True); ap.add_argument("--out", default="data"); args = ap.parse_args()
archives = os.path.abspath(args.archives); os.chdir(ROOT)
src = open("scripts/prepare_data.py").read()
exec(src[src.index("def parse_col"):src.index("os.makedirs(args.out")])          # the same value parser as the subsets
z = next(zipfile.ZipFile(os.path.join(archives, f)) for f in os.listdir(archives) if f.endswith(".zip")
         and any(n.endswith("l2-malicious.csv") for n in zipfile.ZipFile(os.path.join(archives, f)).namelist()))
parts = []
for name in ["l2-benign.csv", "l2-malicious.csv"]:
    mem = next(n for n in z.namelist() if n.endswith(name))
    R = pd.read_csv(z.open(mem), dtype=str, encoding="utf-8-sig", keep_default_na=False, low_memory=False)
    parts.append(R.assign(source_file=name, source_row=np.arange(len(R))))
R = pd.concat(parts, ignore_index=True)
cols = [c for c in R.columns if c not in ("source_file", "source_row")]
C = pd.DataFrame({"row_id": np.arange(len(R), dtype="int64"), "dataset": "doh_full"})
for c in cols: C[c] = parse_col(R[c])
C["TimeStamp"] = pd.to_datetime(C["TimeStamp"].astype(str))
C["attack_type"] = C["Label"].astype(str); C["label"] = (C["Label"].astype(str).str.strip() != "Benign").astype("int8")
C["source_file"] = R.source_file.astype("string"); C["source_row"] = R.source_row.astype("int64")
path = os.path.join(args.out, "cira_cic_dohbrw2020_l2_full.parquet"); C.to_parquet(path, index=False)
print(f"doh_full: {len(C):,} rows -> {path}; classes {C.attack_type.value_counts().to_dict()}")
ref = dict(l.split()[::-1] for l in open("data/SHA256SUMS") if l.strip())
h = hashlib.sha256(pd.read_parquet(path).to_csv(index=False, float_format="%.17g").encode()).hexdigest()
want = ref.get("cira_cic_dohbrw2020_l2_full.csv")
print("checksum", "OK" if h == want else f"MISMATCH ({h} != {want})"); sys.exit(0 if h == want else 1)
