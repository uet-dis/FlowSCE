"""Prediction-/query-level reporting for FlowSCE. No benchmark scores are imputed.

CLI computes detector metrics only from supplied prediction CSVs. Missing
outcomes remain failures/NA; constructed fixtures must not be labeled traffic.
"""
from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence
import numpy as np


def ratio(num: float, den: float) -> float | None:
    return float(num / den) if den else None


def detector_metrics(y: Sequence[int], p: Sequence[float], threshold: float = 0.5) -> dict:
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Nonempty equally sized one-dimensional prediction arrays required.')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Labels must be binary and probabilities finite in [0,1].')
    if not 0 < threshold < 1:
        raise ValueError('The fixed decision threshold must be in (0,1).')
    pred = p >= threshold
    tp = int(((y == 1) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
    fp = int(((y == 0) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
    f1_attack = ratio(2*tp, 2*tp+fp+fn); f1_benign = ratio(2*tn, 2*tn+fp+fn)
    res = dict(n=len(y), TP=tp, TN=tn, FP=fp, FN=fn,
        accuracy=(tp+tn)/len(y), precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
        specificity=ratio(tn,tn+fp), FPR=ratio(fp,fp+tn), F1_attack=f1_attack,
        F1_benign=f1_benign, macro_F1=(f1_attack+f1_benign)/2 if f1_attack is not None and f1_benign is not None else None,
        threshold=threshold, zero_denominator='NA; no implicit zero or one fill')
    if res['recall'] is not None and res['specificity'] is not None:
        rho = (tp+fn)/len(y)
        assert abs(res['accuracy'] - rho*res['recall'] - (1-rho)*res['specificity']) < 1e-12
    return res


def query_summary(records: list[dict[str, Any]], epsilon: float = 1e-8) -> dict:
    """One eligible originally-alerting query per record, failures included.

    A 'returned' candidate has scalar nominal and reconstructed shifted logit
    scores q; q<0 is benign. Constraint validity is independently logged. Nominal
    validity requires both the nominal score and the same domain validator.
    Records include F and the *complete* delta even for nominally valid stale
    candidates that fail reconstruction. No tiny coordinate is rounded away.
    """
    if epsilon < 0 or not math.isfinite(epsilon) or not records:
        raise ValueError('Nonnegative finite epsilon and at least one eligible query required.')
    ids=[r['query_id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate queries: summarize one dataset/checkpoint/seed/variant cell at a time.')
    valid=[]; nominal=[]; spurious=[]; quality=[]
    for r in records:
        if r['status'] not in ('returned','failure'):
            raise ValueError('NotAlert/configuration errors must be recorded separately, not mixed with eligible queries.')
        n=v=False
        if r['status']=='returned':
            de=np.asarray(r['delta'],dtype=float); F=int(r['F'])
            if de.shape!=(F,) or not np.isfinite(de).all() or F<=0:
                raise ValueError('A complete finite perturbation in the declared F-dimensional space is required.')
            qn=float(r['nominal_score']); qr=float(r['reconstructed_score'])
            if not (math.isfinite(qn) and math.isfinite(qr)):
                raise ValueError('Every returned candidate needs both independently computed scores.')
            admissible=r['admissible']
            if not isinstance(admissible,bool): raise ValueError('admissible must be boolean.')
            n = qn < 0 and admissible
            v = qr < 0 and admissible
            if v:
                quality.append(dict(query_id=r['query_id'],F=F,support_exact=int(np.count_nonzero(de)),
                    support_epsilon=int((np.abs(de)>epsilon).sum()),sparsity=1-float((np.abs(de)>epsilon).sum())/F,
                    L1=float(np.linalg.norm(de,ord=1)),L2=float(np.linalg.norm(de)),logit_margin=-qr))
        nominal.append(n); valid.append(v); spurious.append(n and not v)
    N=len(records); Ns=sum(nominal); Nv=sum(valid); Ne=sum(spurious)
    # Only reconstructed-successful queries enter quality summaries; denominator
    # is explicit and the full ledger still contains every failure.
    stats={}
    for key in ['support_exact','support_epsilon','sparsity','L1','L2','logit_margin']:
        values=np.array([r[key] for r in quality])
        stats[key]=dict(n=len(values),mean=float(values.mean()),median=float(np.median(values)),
            q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75))) if len(values) else dict(n=0,mean=None,median=None,q25=None,q75=None)
    return dict(attempted=N,successful=Nv,failed=N-Nv,nominally_valid=Ns,spurious=Ne,
        CSR=Nv/N,nominal_yield=Ns/N,SVR=ratio(Ne,Ns),post_reconstruction_failure_all=Ne/N,
        epsilon=epsilon,quality_population='reconstructed and domain-valid successes only',
        quality_summary=stats,per_query_quality=quality,success_flags=valid)


def paired_cluster_bootstrap(x: Sequence[float], y: Sequence[float], clusters: Sequence[str],
                             repetitions: int = 2000, seed: int = 20260921) -> dict:
    """Paired cluster percentile CI for mean(x-y), conditional on supplied blocks.

    Caller must align exact (checkpoint, query, initialization) keys and state
    whether values represent all-query success or joint-success-only quality.
    This routine does not manufacture replication across trained checkpoints.
    """
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); c=np.asarray(clusters,dtype=str)
    if x.ndim!=1 or y.shape!=x.shape or c.shape!=x.shape or not len(x) or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Aligned finite paired values and block IDs are required.')
    if repetitions<100: raise ValueError('Use at least 100 bootstrap replicates.')
    labels=np.unique(c)
    if len(labels)<2: raise ValueError('At least two independent sampling blocks required; no CI for a single constructed fixture.')
    groups=[np.flatnonzero(c==a) for a in labels]; rng=np.random.default_rng(seed); diff=x-y
    draws=np.empty(repetitions)
    for b in range(repetitions):
        take=rng.integers(0,len(groups),len(groups)); ix=np.concatenate([groups[t] for t in take]);draws[b]=diff[ix].mean()
    return dict(paired_difference=float(diff.mean()),ci95=[float(v) for v in np.quantile(draws,[.025,.975])],
        n_queries=len(x),n_blocks=len(groups),bootstrap_repetitions=repetitions,bootstrap_seed=seed,
        interval='paired cluster percentile; conditional on the supplied blocks/checkpoint')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--predictions',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--threshold',type=float,default=.5)
    a=ap.parse_args()
    with a.predictions.open(newline='') as f: rows=list(csv.DictReader(f))
    fields={'dataset','seed','row_id','y_true','p_attack'}
    if not rows or not fields.issubset(rows[0]):raise ValueError('CSV requires '+','.join(sorted(fields)))
    groups=defaultdict(list)
    for r in rows:groups[(r['dataset'],r['seed'])].append(r)
    report=[]
    for (dataset,seed),rr in groups.items():
        if len({r['row_id'] for r in rr})!=len(rr):raise ValueError('Duplicate row IDs within a detector run.')
        report.append(dict(dataset=dataset,seed=seed,**detector_metrics([int(r['y_true']) for r in rr],[float(r['p_attack']) for r in rr],a.threshold)))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2,allow_nan=False))
if __name__=='__main__':main()
