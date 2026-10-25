#!/usr/bin/env python3
"""
Build the three experiment datasets from the OFFICIAL releases, row by row, and verify them against data/SHA256SUMS.

Download the archives (see README) into one folder, then

    python scripts/prepare_data.py --archives /path/to/downloads

  archive                       release                                   members used
  data_2018.zip                 NF-CICIDS2018-v3, UQ eSpace (UQ:ece9b83)    data/NF-CICIDS2018-v3.csv
  Total_CSVs.zip                CIRA-CIC-DoHBrw-2020, UNB CIC             l2-benign.csv, l2-malicious.csv
  <UNSW-NB15 CSV folder>.zip    UNSW-NB15, UNSW Canberra                  UNSW-NB15_1..4.csv, NUSW-NB15_features.csv

The rows to take are listed in data/source_rows/source_rows_<ds>.csv.gz (row_id → source file, 0-based data row). Values are parsed
from the source text: integer columns become nullable Int64, other numbers float64, text is stripped, hexadecimal ports become
integers, and DoH TimeStamp becomes datetime64. Added columns: row_id, dataset, label (0 benign / 1 attack), attack_type,
source_file, source_row. Archives are read as streams; nothing is extracted to disk.
"""
import argparse, glob, hashlib, os, sys, zipfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--archives", required=True, help="folder with the downloaded zip archives")
ap.add_argument("--out", default="data"); ap.add_argument("--chunk", type=int, default=250_000); args = ap.parse_args()
archives = os.path.abspath(args.archives); os.chdir(ROOT)

SPEC = {
 "cic": dict(name="nf_cicids2018_v3_4class_60k", archive="data_2018.zip", header=0, strat="Attack", label="Label", member_suffix="data/NF-CICIDS2018-v3.csv"),
 "doh": dict(name="cira_cic_dohbrw2020_l2_49807", archive="Total_CSVs.zip", header=0, strat="Label", label="Label"),
 "unsw": dict(name="unsw_nb15_60k", archive=None, header=None, strat="attack_cat", label="Label", docs_suffix="NUSW-NB15_features.csv"),
}
TIME = {"doh": ["TimeStamp"]}

def find_archive(ds):
    sp = SPEC[ds]
    if sp["archive"] and os.path.exists(os.path.join(archives, sp["archive"])): return os.path.join(archives, sp["archive"])
    for z in glob.glob(os.path.join(archives, "*.zip")):                     # UNSW-NB15 is distributed as a folder download
        names = zipfile.ZipFile(z).namelist()
        if ds == "unsw" and any(n.endswith("UNSW-NB15_1.csv") for n in names): return z
        if ds == "cic" and any(n.endswith(sp["member_suffix"]) for n in names): return z
        if ds == "doh" and any(n.endswith("l2-malicious.csv") for n in names): return z
    sys.exit(f"archive for {ds} not found in {archives}")

def parse_col(t: pd.Series) -> pd.Series:
    t = t.astype(str).str.strip(); t = t.where(~t.str.lower().isin({"", "nan", "none", "null"}), None)
    hexm = t.fillna("").str.lower().str.match(r"^0x[0-9a-f]+$")
    num = pd.to_numeric(t.where(~hexm), errors="coerce"); num[hexm] = t[hexm].map(lambda x: int(x, 16))
    present = t.notna()
    if present.any() and num[present].notna().all():
        if (num[present] == np.floor(num[present])).all() and num[present].abs().max() < 2 ** 62 and not t[present].str.contains(r"[.eE]").any():
            return num.round().astype("Int64")
        return num.astype("float64")
    return t.astype("string")

os.makedirs(args.out, exist_ok=True)
for ds, sp in SPEC.items():
    z = zipfile.ZipFile(find_archive(ds)); names = None; members = z.namelist()
    M = pd.read_csv(f"data/source_rows/source_rows_{ds}.csv.gz")
    if sp["header"] is None:
        doc = pd.read_csv(z.open(next(n for n in members if n.endswith(sp["docs_suffix"]))), encoding="latin-1")
        names = [str(n).strip() for n in doc["Name"]]
        names = [{"ct_src_ltm": "ct_src_ ltm"}.get(n, n) for n in names]                # column name as written in the release files
    parts = []
    for src, g in M.groupby("source_file"):
        mem = next(n for n in members if n.endswith(src.split("/")[-1]))
        want = pd.Series(g.row_id.values, index=g.source_row.astype(int).values); seen = 0
        for ch in pd.read_csv(z.open(mem), header=sp["header"], names=names, dtype=str, chunksize=args.chunk, encoding="utf-8-sig",
                              keep_default_na=False, low_memory=False):
            ch.index = np.arange(seen, seen + len(ch)); seen += len(ch)
            hit = ch.loc[ch.index.intersection(want.index)]
            if len(hit): parts.append(hit.assign(row_id=want.loc[hit.index].values, source_file=src.split("/")[-1], source_row=hit.index))
    R = pd.concat(parts).sort_values("row_id").reset_index(drop=True); assert (R.row_id.values == np.arange(len(M))).all()
    cols = [c for c in R.columns if c not in ("row_id", "source_file", "source_row")]
    C = pd.DataFrame({"row_id": R.row_id.astype("int64"), "dataset": ds})
    for c in cols: C[c] = parse_col(R[c])
    for c in TIME.get(ds, []): C[c] = pd.to_datetime(C[c].astype(str))
    strat = C[sp["strat"]].astype("string").str.strip().fillna("")
    C["attack_type"] = strat.replace({"": "Benign", "Normal": "Benign", "Backdoors": "Backdoor"}) if ds != "doh" else C["Label"].astype(str)
    C["label"] = (C["Label"].astype(str).str.strip() != "Benign").astype("int8") if ds == "doh" else C["Label"].astype("int8")
    C["source_file"] = R.source_file.astype("string"); C["source_row"] = R.source_row.astype("int64")
    path = os.path.join(args.out, sp["name"] + ".parquet"); C.to_parquet(path, index=False)
    print(f"{ds}: {len(C):,} rows × {len(C.columns)} columns → {path}; classes {C.attack_type.value_counts().to_dict()}", flush=True)

# content check: SHA-256 of a canonical CSV rendering (independent of parquet library versions)
ref = dict(l.split()[::-1] for l in open("data/SHA256SUMS") if l.strip())
ok = True
for ds, sp in SPEC.items():
    d = pd.read_parquet(os.path.join(args.out, sp["name"] + ".parquet"))
    h = hashlib.sha256(d.to_csv(index=False, float_format="%.17g").encode()).hexdigest(); want = ref.get(sp["name"] + ".csv")
    print(f"{ds}: content SHA-256 {h} {'OK' if h == want else 'MISMATCH (expected ' + str(want) + ')'}"); ok &= h == want
sys.exit(0 if ok else 1)
