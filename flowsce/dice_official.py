"""DiCE through the official dice-ml library (Mothilal et al., 2020), method 'gradient' (DiceTorch), with the
reconstructed oracle wrapped as a PyTorch model.

dice-ml works in raw feature units and min-max normalises them internally with the training ranges. The wrapper maps
DiCE's normalised input back to raw units, applies the frozen preprocessing (log1p where fitted, standardisation),
replaces flow k of the test snapshot by it, reconstructs the host summaries (exact affine update) and returns the
attack probability of flow k. Library defaults are kept (proximity weight 0.5, diversity weight 1.0, hinge y-loss,
inverse-MAD feature weights, Adam with learning rate 0.05, initialization near the query), with total_CFs = 4,
features_to_vary = the mutable features and posthoc_sparsity_param = None (no post-hoc step in the generator table).
The only change to the defaults is the iteration count: dice-ml evaluates the model 12 times per iteration for four
counterfactuals and its default min_iter = 500 alone would need about 6000 forward evaluations, so
min_iter = max_iter = 336 keeps every query within the shared budget of 4096 forward evaluations (and 500 updates). Every returned
counterfactual is projected onto the schema and judged by the same validator and reconstructed oracle as all methods.
"""
from __future__ import annotations
import time, warnings
import numpy as np, pandas as pd, torch, torch.nn as nn
from .config import DEVICE, ExplainCfg
from .oracle import QueryOracle
from .metrics import assess


class _OracleModule(nn.Module):
    """Evaluates the CFs that dice-ml passes with the batched oracle (flowsce.fast.BatchOracle: parallel edges merged,
    numerically equivalent to QueryOracle); every row counts as one forward evaluation of the query's oracle."""
    def __init__(s, orc: QueryOracle, bo, dom, lo, hi):
        super().__init__(); s.orc, s.bo, s.dom = orc, bo, dom
        s.lo = torch.tensor(lo, dtype=torch.float64, device=DEVICE); s.span = torch.tensor(hi - lo, dtype=torch.float64, device=DEVICE)
        s.mu = torch.tensor(dom.mu, dtype=torch.float64, device=DEVICE); s.sd = torch.tensor(dom.sd, dtype=torch.float64, device=DEVICE)
        s.islog = torch.tensor(dom.is_log, device=DEVICE); s.mut = torch.tensor(dom.mutable, dtype=torch.float64, device=DEVICE)
        s.a = orc.a.double()
    def forward(s, z):
        z = torch.as_tensor(z).to(DEVICE, torch.float64).reshape(-1, s.lo.shape[0])
        raw = s.lo + z * s.span
        y = torch.where(s.islog, torch.log1p(torch.clamp(raw, min=0)), raw)
        x = (y - s.mu) / s.sd
        d = ((x - s.a[None]) * s.mut[None]).float(); n = d.shape[0]
        q = s.bo.q([s.orc.k] * n, d, ["A"] * n); s.orc.nf += n
        return torch.sigmoid(q + s.bo.thr).reshape(-1, 1).float().cpu()


class DiceOfficial:
    """One instance per (dataset, checkpoint); explain(k) runs dice-ml on test flow k."""
    def __init__(self, model, g, prep, df_train, y_train, dom, sp, cfg: ExplainCfg, max_iter=336, total_cfs=4):
        import dice_ml
        self.dice_ml, self.model, self.g, self.prep, self.dom, self.sp, self.cfg = dice_ml, model, g, prep, dom, sp, cfg
        R = prep.raw_frame(df_train).reset_index(drop=True); self.cols = list(R.columns)
        self.lo, self.hi = R.min().values.astype(np.float64), R.max().values.astype(np.float64)
        self.hi = np.where(self.hi > self.lo, self.hi, self.lo + 1.0)
        R["attack"] = np.asarray(y_train).astype(int)
        from .fast import decimal_precision                     # dice-ml's precision rule, robust to scientific notation
        self.data = dice_ml.Data(dataframe=R, continuous_features=self.cols, outcome_name="attack",
                                 continuous_features_precision={c: decimal_precision(R[c]) for c in self.cols})
        self.vary = [c for c, m in zip(self.cols, dom.mutable) if m]
        self.max_iter, self.total_cfs = max_iter, total_cfs
        from .fast import BatchOracle
        self.bo = BatchOracle(model, g, cfg.tau)

    def explain(self, k, method="DiCE"):
        orc = QueryOracle(self.model, self.g, k, self.cfg.tau); t0 = time.perf_counter()
        a = orc.a.cpu().numpy(); raw_a = self.dom.to_raw(a)
        net = _OracleModule(orc, self.bo, self.dom, self.lo, self.hi)
        m = self.dice_ml.Model(model=net, backend="PYT")
        exp = self.dice_ml.Dice(self.data, m, method="gradient")
        q = pd.DataFrame([raw_a], columns=self.cols); cfs, err = [], ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = exp.generate_counterfactuals(q, total_CFs=self.total_cfs, desired_class="opposite",
                                                   features_to_vary=self.vary, posthoc_sparsity_param=None,
                                                   min_iter=self.max_iter, max_iter=self.max_iter, verbose=False)
            df = res.cf_examples_list[0].final_cfs_df
            if df is not None and len(df):
                for _, r in df[self.cols].iterrows():
                    raw = r.values.astype(np.float64)
                    d = (self.dom.from_raw(raw) - a).astype(np.float32); d[~self.dom.mutable] = 0.0
                    d[np.isclose(raw, raw_a, rtol=0, atol=1e-9)] = 0.0
                    dp = self.dom.project(a, d)
                    if self.dom.check(a, dp)[0] and orc.q0(dp) < 0: cfs.append(dp)
        except Exception as e:                                  # recorded as a failed query, never dropped
            err = f"{type(e).__name__}: {str(e)[:200]}"
        t_gen = time.perf_counter() - t0; n_gen = orc.nf
        best = min(cfs, key=lambda c: ((np.abs(c) > self.cfg.eps_support).sum(), np.abs(c).sum())) if cfs else None
        row = dict(method=method, n_fwd_method=n_gen); row.update(assess(orc, self.dom, self.sp, best, "A", self.cfg))
        fi = (np.mean([(c != 0) for c in cfs], 0) if cfs else np.zeros(self.sp.F)).tolist()
        row.update(n_fwd=orc.nf, n_bwd=np.nan, n_valid_cfs=len(cfs), all_cfs=[c.tolist() for c in cfs], feature_importance=fi,
                   t_generate=t_gen, dice_error=err)
        return row
