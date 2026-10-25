"""Specification-level reference implementation of FlowSCE on constructed inputs (independent of flowsce/).

All arithmetic uses caller-supplied processed features. No dataset, pretrained
weights, feature schema or experimental hyperparameters are inferred.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable
import math
import torch

Tensor=torch.Tensor
Validator=Callable[[Tensor], bool]

@dataclass(frozen=True)
class AffinePolicy:
    C: Tensor
    H: Tensor
    name: str

    def rebuild(self, edges: Tensor) -> Tensor:
        if edges.ndim != 2 or self.C.ndim != 2 or self.H.ndim != 2:
            raise ValueError('Expected matrices for edges, C and H.')
        if self.C.shape[1] != edges.shape[0] or self.H.shape != (self.C.shape[0],edges.shape[1]):
            raise ValueError('Incompatible policy dimensions.')
        if not all(bool(torch.isfinite(a).all()) for a in [edges,self.C,self.H]):
            raise ValueError('Non-finite graph or policy input.')
        return self.C @ edges + self.H

    @classmethod
    def snapshot(cls, edge_index: Tensor, n_nodes: int, n_features: int) -> 'AffinePolicy':
        if edge_index.ndim != 2 or edge_index.shape[0]!=2 or edge_index.dtype!=torch.long:
            raise ValueError('edge_index must be a 2 by M int64 tensor.')
        if n_nodes<=0 or n_features<=0 or bool(((edge_index<0)|(edge_index>=n_nodes)).any()):
            raise ValueError('Invalid node/feature dimensions or endpoint index.')
        B=torch.zeros((n_nodes,edge_index.shape[1]),dtype=torch.float64,device=edge_index.device)
        cols=torch.arange(edge_index.shape[1],device=edge_index.device)
        for ends in edge_index:B.index_put_((ends,cols),torch.ones_like(cols,dtype=torch.float64),accumulate=True)
        C=B/B.sum(dim=1,keepdim=True).clamp(min=1)
        return cls(C,torch.zeros((n_nodes,n_features),dtype=torch.float64,device=edge_index.device),'snapshot')

    @classmethod
    def fixed_reference(cls, nodes: Tensor, n_edges: int) -> 'AffinePolicy':
        return cls(torch.zeros((nodes.shape[0],n_edges),dtype=nodes.dtype,device=nodes.device),nodes.detach().clone(),'excluded_fixed_reference')


class QueryOracle:
    """Frozen score interface. score_fn must be deterministic and return a scalar.

    Negative controls use `mode`; production queries must use 'consistent'.
    A neural module supplied as score_fn is placed in eval mode and its weights
    are frozen. This is not a runtime proof that an arbitrary callable is pure.
    """
    def __init__(self,edges:Tensor,nodes:Tensor,policy:AffinePolicy,k:int,
                 score_fn:Callable[[Tensor,Tensor],Tensor],atol:float=1e-10):
        if not 0<=k<edges.shape[0]:raise ValueError('Query index outside graph.')
        rebuilt=policy.rebuild(edges)
        if nodes.shape!=rebuilt.shape or not torch.allclose(rebuilt,nodes,atol=atol,rtol=0):
            raise ValueError('Policy does not reconstruct the ORIGINAL graph; refusing query.')
        self.edges=edges.detach().clone();self.nodes=nodes.detach().clone();self.policy=policy;self.k=k;self.score_fn=score_fn
        self.forward_calls=0
        if isinstance(score_fn,torch.nn.Module):
            score_fn.eval()
            for p in score_fn.parameters():p.requires_grad_(False)

    def state(self,delta:Tensor,mode:str='consistent')->tuple[Tensor,Tensor]:
        if delta.shape != (self.edges.shape[1],) or not bool(torch.isfinite(delta).all()):
            raise ValueError('Invalid perturbation.')
        selector=torch.zeros((self.edges.shape[0],1),dtype=self.edges.dtype,device=self.edges.device);selector[self.k,0]=1
        edges=self.edges+selector@delta.unsqueeze(0)
        if mode=='consistent':nodes=self.policy.rebuild(edges)
        elif mode=='detached':nodes=self.policy.rebuild(edges).detach()
        elif mode=='frozen':nodes=self.nodes
        else:raise ValueError('Unknown reconstruction mode.')
        return edges,nodes

    def residual(self,delta:Tensor,mode:str='consistent')->float:
        """Input mismatch only; this does not certify an arbitrary derivative."""
        edges,nodes=self.state(delta,mode)
        with torch.no_grad():return float(torch.linalg.matrix_norm(nodes-self.policy.rebuild(edges)))

    def score(self,delta:Tensor,mode:str='consistent')->Tensor:
        edges,nodes=self.state(delta,mode);q=self.score_fn(edges,nodes)
        if q.numel()!=1 or not bool(torch.isfinite(q).all()):raise ValueError('Score must be finite and scalar.')
        self.forward_calls+=1
        return q.reshape(())

@dataclass
class Explanation:
    status:str
    delta:list[float]|None
    score:float|None
    updates:int
    forward_calls:int
    accepted_restorations:int
    trace:list[dict]
    note:str='Reference implementation on constructed inputs.'


def restoration(oracle:QueryOracle,delta:Tensor,mask:Tensor,epsilon:float=1e-8,
                validator:Validator|None=None)->tuple[Tensor,list[dict]]:
    if epsilon<0 or not math.isfinite(epsilon):raise ValueError('epsilon must be finite and nonnegative.')
    mask=mask.to(torch.bool)
    if mask.shape!=delta.shape or bool((delta[~mask]!=0).any()):raise ValueError('Immutable coordinate changed.')
    accepts=validator or (lambda d:True)
    with torch.no_grad():
        if not bool(oracle.score(delta)<0) or not accepts(delta):raise ValueError('Restoration requires an admissible valid candidate.')
        current=delta.detach().clone();trace=[]
        J=[j for j in range(delta.numel()) if bool(mask[j]) and abs(float(current[j]))>epsilon]
        J.sort(key=lambda j:(abs(float(current[j])),j))
        for j in J:
            trial=current.clone();trial[j]=0;q=float(oracle.score(trial))
            accepted=q<0 and bool(accepts(trial))
            if accepted:current=trial
            trace.append({'coordinate':j,'score':q,'accepted':accepted,'support':int((current.abs()>epsilon).sum())})
    return current,trace


def search(oracle:QueryOracle,mask:Tensor,scale:Tensor,*,max_updates:int=100,lr:float=.05,
           lambda_prox:float=.0,lambda_l1:float=.0,epsilon:float=1e-8,
           precision:Tensor|None=None,validator:Validator|None=None,prune:bool=True,
           gradient_mode:str='consistent')->Explanation:
    """Finite-budget masked Adam; post-update candidates are always checked.

    `gradient_mode='detached'` is an intentional negative control: forward
    states remain consistent, but node-feature gradients are omitted. Never
    advertise that variant as FlowSCE. The normal setting is 'consistent'.
    """
    F=oracle.edges.shape[1];dtype=oracle.edges.dtype;device=oracle.edges.device
    if max_updates<0 or not isinstance(max_updates,int) or lr<=0 or not math.isfinite(lr):raise ValueError('Invalid search budget/rate.')
    if mask.shape!=(F,) or scale.shape!=(F,) or not bool(torch.isfinite(scale).all()) or bool((scale<0).any()):raise ValueError('Invalid mask/scale.')
    if gradient_mode not in ['consistent','detached']:raise ValueError('Search final validity must use consistent forward states.')
    if min(lambda_prox,lambda_l1)<0:raise ValueError('Negative penalty.')
    active=mask.to(torch.bool)&(scale>0)
    Q=torch.eye(F,dtype=dtype,device=device) if precision is None else precision
    if Q.shape!=(F,F) or not bool(torch.isfinite(Q).all()) or not torch.allclose(Q,Q.T,atol=1e-12) or float(torch.linalg.eigvalsh(Q).min()) < -1e-10:
        raise ValueError('Precision must be symmetric positive semidefinite in optimization coordinates.')
    accepts=validator or (lambda d: True);start_calls=oracle.forward_calls
    xi=torch.zeros(F,dtype=dtype,device=device,requires_grad=True)
    opt=torch.optim.Adam([xi],lr=lr);trace=[];found=None;updates=0
    for t in range(max_updates+1):
        delta=xi*scale*active
        q=oracle.score(delta,gradient_mode)
        qvalue=float(q.detach());trace.append({'step':t,'score':qvalue,'delta':delta.detach().tolist()})
        if t==0 and qvalue<0:
            return Explanation('NotAlert',None,qvalue,0,oracle.forward_calls-start_calls,0,trace)
        if qvalue<0 and accepts(delta.detach()):found=delta.detach().clone();break
        if t==max_updates or not bool(active.any()):break
        # Penalize only actual optimization coordinates. Otherwise covariance
        # cross-terms could move an inactive latent variable to lower the
        # penalty even though its flow coordinate cannot change.
        xi_active=xi*active
        loss=torch.relu(q)+lambda_prox*torch.sqrt(xi_active@Q@xi_active+1e-12)+lambda_l1*xi_active.abs().sum()
        opt.zero_grad();loss.backward()
        if xi.grad is None or not bool(torch.isfinite(xi.grad).all()):break
        opt.step();updates+=1
    if found is None:return Explanation('Failure',None,None,updates,oracle.forward_calls-start_calls,0,trace)
    rtrace=[]
    if prune:found,rtrace=restoration(oracle,found,mask,epsilon,accepts)
    # Final recheck also catches inadmissible external mutations and tiny residuals.
    with torch.no_grad():qfinal=float(oracle.score(found))
    if qfinal>=0 or not accepts(found):raise RuntimeError('Final explanation failed revalidation.')
    trace+=[{'restoration':x} for x in rtrace]
    return Explanation('Valid',found.tolist(),qfinal,updates,oracle.forward_calls-start_calls,sum(x['accepted'] for x in rtrace),trace)
