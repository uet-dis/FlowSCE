"""Flow-centric GAT detector (pure torch), training, loading and detector metrics."""
from __future__ import annotations
import os, json, time, hashlib, platform, copy, math, random
from dataclasses import dataclass, field, asdict
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE, DetectorCfg, set_seed

# ══════════════════════════════ Detector (pure-torch GAT, no PyG needed) ══════════════════════════════
class GATLayer(nn.Module):
    """Same computation as PyG GATConv(add_self_loops=False, concat=True); messages flow src → dst."""
    def __init__(s, i, d, H, dropout):
        super().__init__(); s.H, s.d, s.p = H, d, dropout
        s.lin = nn.Linear(i, H * d, bias=False)
        s.a_s = nn.Parameter(torch.empty(H, d)); s.a_d = nn.Parameter(torch.empty(H, d)); s.b = nn.Parameter(torch.zeros(H * d))
        nn.init.xavier_uniform_(s.a_s); nn.init.xavier_uniform_(s.a_d)
    def forward(s, h, src, dst):
        N = h.shape[0]; Wh = s.lin(h).view(N, s.H, s.d)
        e = F.leaky_relu((Wh * s.a_s).sum(-1)[src] + (Wh * s.a_d).sum(-1)[dst], 0.2)
        m = torch.full((N, s.H), -torch.finfo(h.dtype).max, dtype=h.dtype, device=h.device)
        m = m.scatter_reduce(0, dst[:, None].expand(-1, s.H), e, "amax", include_self=True)
        ex = torch.exp(e - m[dst]); den = torch.zeros(N, s.H, dtype=h.dtype, device=h.device).index_add(0, dst, ex)
        al = F.dropout(ex / den[dst].clamp(min=1e-16), s.p, s.training)
        out = torch.zeros(N, s.H, s.d, dtype=h.dtype, device=h.device).index_add(0, dst, al.unsqueeze(-1) * Wh[src])
        return out.reshape(N, -1) + s.b

class Detector(nn.Module):
    """Flow-centric detector of Section 4.1: z_k = ReLU(W_e a_k), c^s, c^d = ReLU(W h^{(L)}), optional gates.
    kind: 'edge' (no context) | 'flow' (reduced context, d_c) | 'flow_gate' | 'flow_full' (host embeddings h^{(L)} used
    directly, without the projection to d_c) | 'flow_full_gate' | 'flow_sage' / 'flow_sage_gate' (the GAT context encoder replaced
    by GraphSAGE-mean layers over the hosts, h_v <- ReLU(W [h_v ; mean_{u -> v} h_u]); selected for UNSW-NB15 by the detector search)."""
    def __init__(s, n_feat, cfg: DetectorCfg):
        super().__init__(); s.cfg, s.kind, s.p = cfg, cfg.kind, cfg.dropout
        s.ep = nn.Linear(n_feat, cfg.d_e)
        inn = cfg.d_e
        if s.kind != "edge":
            s.ne = nn.Linear(n_feat, cfg.hid)
            if s.kind.startswith("flow_sage"):
                s.sages = nn.ModuleList([nn.Linear(2 * (cfg.hid if l == 0 else cfg.hid * cfg.heads), cfg.hid * cfg.heads) for l in range(cfg.layers)])
            else:
                s.gats = nn.ModuleList([GATLayer(cfg.hid if l == 0 else cfg.hid * cfg.heads, cfg.hid, cfg.heads, cfg.dropout)
                                        for l in range(cfg.layers)])
            s.full = s.kind.startswith("flow_full")
            if s.full: inn += 2 * cfg.hid * cfg.heads
            else:
                s.ps = nn.Linear(cfg.hid * cfg.heads, cfg.d_c, bias=False); s.pd = nn.Linear(cfg.hid * cfg.heads, cfg.d_c, bias=False)
                inn += 2 * cfg.d_c
            if s.kind.endswith("_gate"): s.gate = nn.Linear(cfg.d_e, 2)
        layers, prev = [], inn
        for w in cfg.mlp: layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.ReLU(), nn.Dropout(cfg.dropout)]; prev = w
        s.mlp = nn.Sequential(*layers, nn.Linear(prev, 2))
    def node_emb(s, xv, src, dst):
        h = F.dropout(F.relu(s.ne(xv)), s.p, s.training)
        if hasattr(s, "sages"):                                   # mean over incoming messages (src -> dst)
            cnt = torch.zeros(h.shape[0], 1, dtype=h.dtype, device=h.device).index_add_(0, dst, torch.ones(len(dst), 1, dtype=h.dtype, device=h.device))
            for lin in s.sages:
                agg = torch.zeros_like(h).index_add_(0, dst, h[src]) / cnt.clamp(min=1)
                h = F.dropout(F.relu(lin(torch.cat([h, agg], 1))), s.p, s.training)
            return h
        for g in s.gats: h = F.dropout(F.elu(g(h, src, dst)), s.p, s.training)
        return h
    def forward(s, xe, xv, src, dst, eidx=None):
        e = slice(None) if eidx is None else eidx
        z = F.relu(s.ep(xe[e])); parts = [z]
        if s.kind != "edge":
            h = s.node_emb(xv, src, dst)
            cs, cd = (h[src[e]], h[dst[e]]) if s.full else (F.relu(s.ps(h[src[e]])), F.relu(s.pd(h[dst[e]])))
            if s.kind.endswith("_gate"):
                gt = torch.sigmoid(s.gate(z)); cs, cd = gt[:, :1] * cs, gt[:, 1:] * cd
            parts += [cs, cd]
        return s.mlp(torch.cat(parts, -1))

@torch.no_grad()
def predict(m, g, xv=None):
    m.eval(); lg = m(g["xe"], g["xv"] if xv is None else xv, g["src"], g["dst"])
    return torch.softmax(lg, 1)[:, 1].cpu().numpy()

def train_detector(gtr, gva, cfg: DetectorCfg, seed: int, verbose=False):
    set_seed(seed)
    m = build_detector(gtr["xe"].shape[1], cfg).to(DEVICE)
    opt = torch.optim.AdamW(m.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10, min_lr=1e-6)
    y = gtr["y"]; cnt = torch.bincount(y, minlength=2).float(); w = len(y) / (2 * cnt)   # M_tr / (2 n_c)
    best, state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        m.train(); opt.zero_grad()
        loss = F.cross_entropy(m(gtr["xe"], gtr["xv"], gtr["src"], gtr["dst"]), y, weight=w)
        loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        f1 = det_metrics(predict(m, gva), gva["y"].cpu().numpy())["macro_f1"]; sch.step(f1); hist.append((ep, loss.item(), f1))
        if f1 > best: best, state, bad, best_ep = f1, copy.deepcopy(m.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= cfg.patience: break
    m.load_state_dict(state); m.eval()
    return m, dict(best_val_macro_f1=best, best_epoch=best_ep, epochs_run=ep + 1, history=hist)

def load_detector(path, n_feat):
    ck = torch.load(path, map_location=DEVICE)
    m = build_detector(n_feat, DetectorCfg(**ck["cfg"])).to(DEVICE); m.load_state_dict(ck["state"]); m.eval(); return m

def build_detector(n_feat, cfg: DetectorCfg):
    if cfg.kind == "egraphsage":
        from .baseline_detectors import EGraphSAGE
        return EGraphSAGE(n_feat, cfg)
    return Detector(n_feat, cfg)

def det_metrics(p, y, tau=0.5):
    yh = (np.asarray(p) >= tau).astype(int); y = np.asarray(y)
    tp = int(((yh == 1) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    fp = int(((yh == 1) & (y == 0)).sum()); fn = int(((yh == 0) & (y == 1)).sum())
    d = lambda a, b: a / b if b else float("nan")
    rec, spec, prec, npv = d(tp, tp + fn), d(tn, tn + fp), d(tp, tp + fp), d(tn, tn + fn)
    f1a, f1b = d(2 * prec * rec, prec + rec), d(2 * npv * spec, npv + spec)
    acc = d(tp + tn, len(y)); rho = d(tp + fn, len(y))
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, accuracy=acc, precision=prec, recall=rec, specificity=spec,
                f1_attack=f1a, macro_f1=(f1a + f1b) / 2, fpr=d(fp, fp + tn),
                identity_gap=abs(acc - (rho * rec + (1 - rho) * spec)))   # Eq. (accuracy-consistency)
