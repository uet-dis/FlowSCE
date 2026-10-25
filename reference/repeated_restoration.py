"""Variant D: complete passes until no accepted coordinate restoration.

A caller enforcing a query budget must report budget-stop rather than assert
irreducibility before a complete no-change pass has finished.
"""
from __future__ import annotations
import torch
from flowsce_reference import restoration, QueryOracle, Validator


def repeated_restoration(oracle:QueryOracle,delta:torch.Tensor,mask:torch.Tensor,
                         epsilon:float=1e-8,validator:Validator|None=None):
    current=delta.detach().clone();full_trace=[]
    s0=int((delta.abs()>epsilon).sum());max_passes=s0+1
    for pass_id in range(max_passes):
        after,trace=restoration(oracle,current,mask,epsilon,validator)
        full_trace.extend([dict(pass_id=pass_id+1,**r) for r in trace])
        changed=any(r['accepted'] for r in trace);current=after
        if not changed:return current,full_trace
    raise RuntimeError('Finite support-decrease invariant violated.')
