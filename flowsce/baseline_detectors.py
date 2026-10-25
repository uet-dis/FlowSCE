"""Detector baselines for the detection comparison (Table T3). Both are re-implementations, written for the split,
preprocessing and graphs of this repository.

E-GraphSAGE (Lo et al., 2022): GraphSAGE with a mean aggregator over the hosts. Node features are initialized to ones;
at every layer a host aggregates the mean of the attributes of its incident flows (the snapshot summary X_V) and
concatenates it with its own state, h_v^l = ReLU(W^l [h_v^{l-1} ; mean_{e ∋ v} x_e]). The flow embedding is
[h_u^L ; h_v^L] and an MLP classifies it. Trained with the same loop as the FlowSCE detector (class-weighted cross-entropy,
AdamW, early stopping on validation macro-F1).

CANDICE-detector (Li et al., 2025, Sections 4.2 and 5.1.2): a beta-TCVAE with a symmetric MLP encoder-decoder
(hidden 128-64-32-16, latent 8, ReLU), beta = 2 (alpha = gamma = 1), Adam with learning rate 1e-4, batch 256, at most
100 epochs with early stopping on the validation loss (the original uses 5-fold cross-validation; here the repository's
validation split plays that role). The encoder mean is the disentangled representation; SVM (RBF, probability=True),
random forest (100 trees) and MLP (two layers of 8 units) classifiers are trained on it. The classifiers use flow
attributes only (no graph).
"""
from __future__ import annotations
import copy, math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .config import DEVICE, DetectorCfg, set_seed


class EGraphSAGE(nn.Module):
    def __init__(s, n_feat, cfg: DetectorCfg):
        super().__init__(); s.cfg, s.kind, s.p = cfg, "egraphsage", cfg.dropout
        H = cfg.hid * cfg.heads                       # same host-embedding width as the FlowSCE detector
        s.layers = nn.ModuleList([nn.Linear((1 if l == 0 else H) + n_feat, H) for l in range(cfg.layers)])
        layers, prev = [], 2 * H
        for w in cfg.mlp: layers += [nn.Linear(prev, w), nn.LayerNorm(w), nn.ReLU(), nn.Dropout(cfg.dropout)]; prev = w
        s.mlp = nn.Sequential(*layers, nn.Linear(prev, 2))
    def forward(s, xe, xv, src, dst, eidx=None):
        e = slice(None) if eidx is None else eidx
        h = torch.ones(xv.shape[0], 1, dtype=xv.dtype, device=xv.device)
        for lin in s.layers: h = F.dropout(F.relu(lin(torch.cat([h, xv], 1))), s.p, s.training)
        return s.mlp(torch.cat([h[src[e]], h[dst[e]]], 1))


# ────────────────────────────── CANDICE-detector ──────────────────────────────
class BetaTCVAE(nn.Module):
    def __init__(s, n_feat, hidden=(128, 64, 32, 16), latent=8):
        super().__init__()
        enc, prev = [], n_feat
        for w in hidden: enc += [nn.Linear(prev, w), nn.ReLU()]; prev = w
        s.enc = nn.Sequential(*enc); s.mu = nn.Linear(prev, latent); s.logvar = nn.Linear(prev, latent)
        dec, prev = [], latent
        for w in reversed(hidden): dec += [nn.Linear(prev, w), nn.ReLU()]; prev = w
        s.dec = nn.Sequential(*dec, nn.Linear(prev, n_feat))
    def encode(s, x):
        h = s.enc(x); return s.mu(h), s.logvar(h)
    def forward(s, x):
        mu, lv = s.encode(x); z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return s.dec(z), z, mu, lv

def _log_density_gaussian(z, mu, lv):
    return -0.5 * (math.log(2 * math.pi) + lv + (z - mu) ** 2 * torch.exp(-lv))

def btcvae_loss(x, xr, z, mu, lv, n_data, alpha=1.0, beta=2.0, gamma=1.0):
    """Negative of Eq. (3) of the CANDICE paper (Chen et al., 2018), minibatch-weighted sampling estimator;
    Gaussian reconstruction (squared error) on the standardised features."""
    B, D = z.shape
    rec = F.mse_loss(xr, x, reduction="sum") / B
    logqz_x = _log_density_gaussian(z, mu, lv).sum(1)
    logpz = _log_density_gaussian(z, torch.zeros_like(z), torch.zeros_like(z)).sum(1)
    mat = _log_density_gaussian(z[:, None, :], mu[None, :, :], lv[None, :, :])          # [B, B, D]
    logqz = torch.logsumexp(mat.sum(2), 1) - math.log(B * n_data)
    logqz_prod = (torch.logsumexp(mat, 1) - math.log(B * n_data)).sum(1)
    mi, tc, dkl = (logqz_x - logqz).mean(), (logqz - logqz_prod).mean(), (logqz_prod - logpz).mean()
    return rec + alpha * mi + beta * tc + gamma * dkl

def train_btcvae(Xtr, Xva, seed, epochs=100, batch=256, lr=1e-4, beta=2.0, patience=10):
    set_seed(seed); n = Xtr.shape[0]
    m = BetaTCVAE(Xtr.shape[1]).to(DEVICE); opt = torch.optim.Adam(m.parameters(), lr=lr)
    Xt = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE); Xv = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    g = torch.Generator(device="cpu").manual_seed(seed); best, state, bad = float("inf"), None, 0
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch].to(DEVICE)]
            if len(xb) < 2: continue
            xr, z, mu, lv = m(xb); loss = btcvae_loss(xb, xr, z, mu, lv, n, beta=beta)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(np.mean([btcvae_loss(xb, *m(xb), Xv.shape[0], beta=beta).item()
                                for xb in torch.split(Xv, batch) if len(xb) > 1]))
        if vl < best: best, state, bad, best_ep = vl, copy.deepcopy(m.state_dict()), 0, ep
        else:
            bad += 1
            if bad >= patience: break
    m.load_state_dict(state); m.eval()
    return m, dict(best_val_loss=best, best_epoch=best_ep, epochs_run=ep + 1)

@torch.no_grad()
def encode_mean(m, X, batch=8192):
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    return torch.cat([m.encode(xb)[0] for xb in torch.split(X, batch)]).cpu().numpy()

def candice_classifiers(seed):
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    return {"candice_svm": SVC(kernel="rbf", probability=True, random_state=seed),
            "candice_rf": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=4),
            "candice_mlp": MLPClassifier(hidden_layer_sizes=(8, 8), early_stopping=True, random_state=seed, max_iter=500)}
