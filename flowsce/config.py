"""Configuration, dataset registry and small utilities."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

# ══════════════════════════════ Config & utilities ══════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [11, 23, 37, 53, 71]
# Columns added by scripts/prepare_data.py for provenance; they are never model features.
META_COLUMNS = ["row_id", "dataset", "label", "attack_type", "source_file", "source_row"]

DATASETS = {
    "cic": dict(file="nf_cicids2018_v3_4class_60k.parquet", source="https://espace.library.uq.edu.au/view/UQ:ece9b83 (NF-CICIDS2018-v3)", time="FLOW_START_MILLISECONDS", tunit=1e-3, src="IPV4_SRC_ADDR",
                dst="IPV4_DST_ADDR", label="Label", label_map="int", strat="Attack",
                drop=["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Attack"], cat=[],
                immutable=["L4_SRC_PORT", "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
                           "SERVER_TCP_FLAGS", "MIN_TTL", "MAX_TTL", "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID",
                           "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE"]),
    "doh": dict(file="cira_cic_dohbrw2020_l2_49807.parquet", source="https://www.unb.ca/cic/datasets/dohbrw-2020.html (l2-benign.csv, l2-malicious.csv)", time="TimeStamp", tunit=1e-9, src="SourceIP", dst="DestinationIP",
                label="Label", label_map="benign_string", strat="Label", drop=["TimeStamp"], cat=[],
                immutable=["SourcePort", "DestinationPort"]),
    # complete layer-2 capture of DoHBrw-2020 (scripts/prepare_doh_full.py), used only by the time-window sweep on dense traffic
    "doh_full": dict(file="cira_cic_dohbrw2020_l2_full.parquet", source="https://www.unb.ca/cic/datasets/dohbrw-2020.html (l2-benign.csv, l2-malicious.csv, all rows)",
                     time="TimeStamp", tunit=1e-9, src="SourceIP", dst="DestinationIP", label="Label", label_map="benign_string",
                     strat="Label", drop=["TimeStamp"], cat=[], immutable=["SourcePort", "DestinationPort"]),
    "unsw": dict(file="unsw_nb15_60k.parquet", source="https://research.unsw.edu.au/projects/unsw-nb15-dataset (UNSW-NB15_1..4.csv)", time="Stime", tunit=1.0, src="srcip", dst="dstip", label="Label",
                 label_map="int", strat="attack_cat", drop=["Stime", "Ltime", "attack_cat"],
                 cat=["proto", "state", "service"],
                 # ct_* are host/connection-window aggregates computed by the extractor: they are derived
                 # context, so they are frozen (not editable through a single-flow intervention).
                 immutable=["sport", "dsport", "sttl", "dttl", "swin", "dwin", "stcpb", "dtcpb", "is_sm_ips_ports",
                            "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
                            "ct_srv_dst", "ct_dst_ltm", "ct_src_ ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm",
                            "ct_dst_src_ltm"]),
}

@dataclass
class DetectorCfg:
    kind: str = "flow"          # 'edge' | 'flow' | 'flow_gate' | 'flow_full' | 'flow_full_gate' | 'egraphsage'
    d_e: int = 128              # flow-branch width
    d_c: int = 32               # endpoint-context width (d_c << d_e)
    hid: int = 64               # GAT per-head width
    heads: int = 4
    layers: int = 2
    mlp: tuple = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 300
    patience: int = 40

@dataclass
class ExplainCfg:
    tau: float = 0.5            # decision threshold
    lr: float = 0.1             # Adam on ξ
    lam1: float = 0.05          # covariance (Mahalanobis) penalty in ξ coordinates
    lam2: float = 0.02          # L1 penalty
    search_margin: float = 0.0  # hinge target: [q + κ]_+  (κ = 0 is the paper's Eq. searchloss)
    T: int = 500                # gradient-update budget per query
    max_fwd: int = 4096         # predictor forward budget per query (generation + restoration)
    eps_support: float = 1e-8   # ε for thresholded support
    max_restore_trials: int = 2000
    cov_ridge: float = 1e-3
    knn_q: int = 5
    box_proj: bool = True       # Algorithm 1: project ξ onto the domain box after every update (False: no in-loop projection; schema ablation)
    min_margin: float = 0.0     # minimum acceptance margin m: search and restoration accept a candidate only if q < -m (E9)

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""): h.update(b)
    return h.hexdigest()

def sha_array(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

def env_info() -> dict:
    import sklearn
    return dict(python=platform.python_version(), torch=torch.__version__, numpy=np.__version__,
                pandas=pd.__version__, sklearn=sklearn.__version__, device=str(DEVICE),
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    def conv(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, float) and not math.isfinite(o): return None
        return str(o)
    with open(path, "w") as f: json.dump(obj, f, indent=2, default=conv)

def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def _np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
