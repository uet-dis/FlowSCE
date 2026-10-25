#!/usr/bin/env python3
"""Deterministic reference verification on constructed inputs (CPU, float64, one thread).

    python reference/run_reference_verification.py --out results/reference_verification

Checks the rank-one host update against an independent sum-and-count rebuild (parallel flows, self-loops counted twice,
isolated hosts), excluded-reference invariance, total gradients against central finite differences on untrained attention
predictors, restoration invariants, the stale-context counterexample and boundary cases. Writes results.json and
execution.json to --out.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from flowsce_reference import AffinePolicy, QueryOracle, search, restoration

torch.set_default_dtype(torch.float64)
torch.set_num_threads(1)
D=torch.float64


def rebuilt_means(edges,idx,N):
    # Independent, direct reference; does not use the coefficient matrix C.
    sums=torch.zeros((N,edges.shape[1]));counts=torch.zeros(N)
    for j in range(edges.shape[0]):
        for v in idx[:,j].tolist():sums[v]+=edges[j];counts[v]+=1
    return sums/counts.clamp(min=1).unsqueeze(1)


def fixture(seed,F=4):
    g=torch.Generator().manual_seed(seed);N=8;M=16
    idx=torch.randint(0,N-1,(2,M),generator=g)
    idx[:,0]=torch.tensor([0,0]);idx[:,1]=torch.tensor([1,2]);idx[:,2]=torch.tensor([1,2])
    edges=torch.randn((M,F),generator=g)*.3
    policy=AffinePolicy.snapshot(idx,N,F)
    return edges,idx,policy,g


class TinyAttentionScore(torch.nn.Module):
    """Two fixed random attention heads; a smooth, untrained test oracle."""
    def __init__(self,idx,N,F,k,g):
        super().__init__();self.idx=idx;self.N=N;self.k=k
        self.register_buffer('W',torch.randn((2,F,3),generator=g)*.3)
        self.register_buffer('av',torch.randn((2,6),generator=g)*.2)
        self.register_buffer('ev',torch.randn(F,generator=g)*.2)
        self.register_buffer('out',torch.randn(12,generator=g)*.3)
        # Explicit directed message neighborhood plus computational self connections.
        self.neighbors=[]
        for i in range(N):
            self.neighbors.append(sorted(set([i]+idx[0,idx[1]==i].tolist())))
    def forward(self,E,V):
        hs=[]
        for h in range(2):
            H=torch.tanh(V@self.W[h]);rows=[]
            for i,neigh in enumerate(self.neighbors):
                nb=H[neigh];central=H[i].expand_as(nb)
                logits=torch.tanh(torch.cat([central,nb],1)@self.av[h])
                rows.append((torch.softmax(logits,0).unsqueeze(1)*nb).sum(0))
            hs.append(torch.stack(rows))
        Z=torch.cat(hs,1);u,v=self.idx[:,self.k].tolist()
        return torch.sin(E[self.k]@self.ev)+torch.tanh(torch.cat([Z[u],Z[v]])@self.out)


def execute(stage):
    records=[];metrics={};raw={}
    def check(name,fn,cases=1):
        fn();records.append({'name':name,'status':'PASS','cases':cases})
    err=0.
    for seed in range(100):
        E,idx,p,g=fixture(1000+seed);k=seed%E.shape[0];delta=torch.randn(E.shape[1],generator=g)
        X=p.rebuild(E);E2=E.clone();E2[k]+=delta
        X2=X+p.C[:,k:k+1]@delta.unsqueeze(0)
        error=float((X2-rebuilt_means(E2,idx,X.shape[0])).abs().max());err=max(err,error)
        assert error<1e-12
    records.append({'name':'exact_snapshot_vs_independent_rebuild','status':'PASS','cases':100})
    metrics['max_snapshot_abs_error']=err
    for seed in range(30):
        E,idx,p,g=fixture(2000+seed);X=p.rebuild(E);fixed=AffinePolicy.fixed_reference(X,E.shape[0]);E2=E+torch.randn(E.shape,generator=g)
        assert torch.equal(fixed.rebuild(E2),X)
    records.append({'name':'fixed_reference_invariance','status':'PASS','cases':30})
    E,idx,p,g=fixture(123);X=p.rebuild(E);bad=False
    try:QueryOracle(E,X+1,p,0,lambda A,B:A.sum()+B.sum())
    except ValueError:bad=True
    check('reject_original_graph_policy_mismatch',lambda:assert_true(bad))
    check('independent_original_rebuild',lambda:assert_true(torch.allclose(X,rebuilt_means(E,idx,X.shape[0]),atol=1e-12)))
    check('isolated_node_zero_summary',lambda:assert_true(torch.equal(X[-1],torch.zeros(E.shape[1]))))
    check('self_loop_double_incidence',lambda:assert_true(float(p.C[0,0])>0))
    if stage==1:return {'tests':records,'metrics':metrics,'raw':raw}

    errors=[];chain_errors=[]
    for seed in range(24):
        E,idx,p,g=fixture(3000+seed);k=seed%E.shape[0];X=p.rebuild(E)
        model=TinyAttentionScore(idx,X.shape[0],E.shape[1],k,g)
        oracle=QueryOracle(E,X,p,k,model);delta=(torch.randn(E.shape[1],generator=g)*.1).requires_grad_(True)
        auto=torch.autograd.grad(oracle.score(delta),delta)[0]
        fd=[];eps=1e-6
        for j in range(delta.numel()):
            plus=delta.detach().clone();minus=plus.clone();plus[j]+=eps;minus[j]-=eps
            fd.append(float((oracle.score(plus)-oracle.score(minus))/(2*eps)))
        errors.append(float((auto-torch.tensor(fd)).abs().max()))
        Ep,Np=oracle.state(delta.detach());Ep=Ep.detach().requires_grad_(True);Np=Np.detach().requires_grad_(True)
        ge,gn=torch.autograd.grad(model(Ep,Np),(Ep,Np));chain=ge[k]+(p.C[:,k:k+1]*gn).sum(0)
        chain_errors.append(float((auto-chain).abs().max()))
    assert max(errors)<1e-6 and max(chain_errors)<1e-12
    records.append({'name':'total_gradient_finite_difference_attention_oracle','status':'PASS','cases':24})
    records.append({'name':'total_gradient_chain_identity','status':'PASS','cases':24})
    metrics['max_gradient_fd_abs_error']=max(errors);metrics['max_gradient_chain_abs_error']=max(chain_errors)

    E=torch.ones((1,1));p=AffinePolicy.snapshot(torch.tensor([[0],[1]],dtype=torch.long),2,1);X=p.rebuild(E)
    f=lambda A,B:B.sum()-A[0,0]-.5
    o=QueryOracle(E,X,p,0,f);delta=torch.tensor([1.],requires_grad=True)
    qf=float(o.score(delta,'frozen').detach());qc=float(o.score(delta).detach());gtrue=float(torch.autograd.grad(o.score(delta),delta)[0].detach());gdet=float(torch.autograd.grad(o.score(delta,'detached'),delta)[0].detach())
    assert (qf,qc,gtrue,gdet)==(-.5,1.5,1.,-1.)
    metrics['counterexample']={'frozen_score':qf,'reconstructed_score':qc,'coupled_gradient':gtrue,'detached_gradient':gdet,'score_discrepancy':qc-qf}
    check('linear_spurious_validity_and_bound_equality',lambda:assert_true(qc-qf==2))
    check('detached_gradient_negative_control',lambda:assert_true(gtrue==-gdet))
    rgood=search(QueryOracle(E,X,p,0,f),torch.ones(1,dtype=torch.bool),torch.ones(1),max_updates=20,lr=.1)
    rbad=search(QueryOracle(E,X,p,0,f),torch.ones(1,dtype=torch.bool),torch.ones(1),max_updates=20,lr=.1,gradient_mode='detached')
    assert rgood.status=='Valid' and rbad.status=='Failure'
    check('matched_constructed_search_gradient_control',lambda:assert_true(rgood.score<0 and rbad.delta is None))
    raw['linear_search_coupled']=asdict(rgood);raw['linear_search_detached']=asdict(rbad)
    metrics['linear_search']={'coupled_status':rgood.status,'coupled_updates':rgood.updates,'coupled_final_score':rgood.score,'detached_status':rbad.status,'detached_updates':rbad.updates}
    degree=[]
    for d in [1,2,4,8,16,32]:
        # Query is between hosts 0 and 1. Distinct filler flows maintain endpoint degree d.
        ends=[(0,1)]+[(0,2)]*(d-1)+[(1,3)]*(d-1)
        ei=torch.tensor(ends,dtype=torch.long).T;edges=torch.ones((len(ends),1));pol=AffinePolicy.snapshot(ei,4,1);nodes=pol.rebuild(edges)
        score=lambda A,B:B[0,0]+B[1,0]-A[0,0]-.5
        oracle=QueryOracle(edges,nodes,pol,0,score);de=torch.ones(1)
        vf=float(oracle.score(de,'frozen'));vc=float(oracle.score(de));bound=2/d
        assert abs((vc-vf)-bound)<1e-12
        degree.append({'endpoint_degree':d,'frozen_score':vf,'reconstructed_score':vc,'discrepancy_bound':bound,'valid_after_rebuild':vc<0})
    records.append({'name':'degree_conditioned_sharp_bound_family','status':'PASS','cases':6});raw['degree_family']=degree
    for seed in range(64):
        E,idx,p,g=fixture(4000+seed,F=6);k=seed%E.shape[0];X=p.rebuild(E)
        coeff=torch.rand(6,generator=g)+.1;coeff[0]=0
        # Include dependent node features but use an exact affine score in changes.
        baseX=X.clone();baseE=E.clone()
        score=lambda A,B: .5-((A[k]-baseE[k])*coeff).sum()-.1*(B-baseX).sum()
        oracle=QueryOracle(E,X,p,k,score);delta=torch.ones(6);mask=torch.ones(6,dtype=torch.bool)
        d2,tr=restoration(oracle,delta,mask)
        assert float(oracle.score(d2))<0
        assert int((d2.abs()>1e-8).sum()) <= 6
        for order in [1,2,float('inf')]:assert float(torch.linalg.vector_norm(d2,ord=order))<=float(torch.linalg.vector_norm(delta,ord=order))+1e-12
        joint=lambda de:float(de@de+(p.C[:,k:k+1]@de.unsqueeze(0)).square().sum())
        assert joint(d2)<=joint(delta)+1e-12
        assert all(tr[i]['support']<=tr[i-1]['support'] for i in range(1,len(tr)))
    records.append({'name':'restoration_validity_support_lp_joint_deviation','status':'PASS','cases':64})
    Q=torch.tensor([[1.,-.9],[-.9,1.]]);d=torch.ones(2);dp=torch.tensor([0.,1.]);before=float(d@Q@d);after=float(dp@Q@dp)
    assert abs(before-.2)<1e-12 and after==1
    metrics['mahalanobis_counterexample']={'before_squared':before,'after_squared':after}
    check('mahalanobis_nonmonotonicity_counterexample',lambda:assert_true(after>before))
    if stage==2:return {'tests':records,'metrics':metrics,'raw':raw}
    # Boundary and rejection tests.
    def simple(F=1):
        edges=torch.zeros((1,F));pol=AffinePolicy.fixed_reference(torch.zeros((2,F)),1)
        return edges,pol,pol.rebuild(edges)
    E,p,X=simple();mask=torch.ones(1,dtype=torch.bool);s=torch.ones(1)
    result=search(QueryOracle(E,X,p,0,lambda A,B:.01-A[0,0]),mask,s,max_updates=1,lr=.02,prune=False)
    check('last_post_update_candidate_checked',lambda:assert_true(result.status=='Valid' and result.updates==1))
    result=search(QueryOracle(E,X,p,0,lambda A,B:1+A.sum()*0),mask,s,max_updates=0)
    check('zero_budget_failure_not_fabricated_counterfactual',lambda:assert_true(result.status=='Failure' and result.delta is None))
    result=search(QueryOracle(E,X,p,0,lambda A,B:1+A.sum()*0),mask,s,max_updates=3)
    check('constant_predictor_budget_exhaustion',lambda:assert_true(result.status=='Failure'))
    result=search(QueryOracle(E,X,p,0,lambda A,B:.5-A[0,0]),torch.zeros(1,dtype=torch.bool),s,max_updates=3)
    check('all_immutable_failure',lambda:assert_true(result.status=='Failure' and result.updates==0))
    result=search(QueryOracle(E,X,p,0,lambda A,B:.5-A[0,0]),mask,torch.zeros(1),max_updates=3)
    check('zero_scale_excluded',lambda:assert_true(result.status=='Failure' and result.updates==0))
    result=search(QueryOracle(E,X,p,0,lambda A,B:-1+A.sum()*0),mask,s,max_updates=3)
    check('non_alert_input_has_distinct_status',lambda:assert_true(result.status=='NotAlert'))
    result=search(QueryOracle(E,X,p,0,lambda A,B:A.sum()*0),mask,s,max_updates=0)
    check('threshold_tie_not_benign',lambda:assert_true(result.status=='Failure'))
    E,p,X=simple(2);score=lambda A,B:.5-A[0,1]/1e-10
    o=QueryOracle(E,X,p,0,score);d2,tr=restoration(o,torch.tensor([1.,1e-10]),torch.ones(2,dtype=torch.bool),epsilon=1e-8)
    assert torch.equal(d2,torch.tensor([0.,1e-10])) and float(o.score(d2))<0 and float(o.score(torch.zeros(2)))>=0
    check('sub_epsilon_validity_critical_change_not_silently_zeroed',lambda:assert_true(d2[1]!=0))
    E,p,X=simple(2);o=QueryOracle(E,X,p,0,lambda A,B:.5-A.sum())
    d2,tr=restoration(o,torch.ones(2),torch.ones(2,dtype=torch.bool),validator=lambda de:bool(de[0]==de[1]))
    check('coupled_validator_rechecked_during_restoration',lambda:assert_true(torch.equal(d2,torch.ones(2))))
    E,p,X=simple(3)
    def nonmonotone(A,B):
        z=A[0]/torch.tensor([1.,2.,3.]);v=0
        for vertex in [(1,1,1),(1,0,1),(0,0,1)]:
            term=torch.ones(())
            for j,bit in enumerate(vertex):term=term*(z[j] if bit else (1-z[j]))
            v=v+term
        return .5-v
    o=QueryOracle(E,X,p,0,nonmonotone);d2,tr=restoration(o,torch.tensor([1.,2.,3.]),torch.ones(3,dtype=torch.bool))
    later=d2.clone();later[0]=0
    assert torch.equal(d2,torch.tensor([1.,0.,3.])) and float(o.score(later))<0
    check('single_pass_not_final_coordinate_irreducible',lambda:assert_true(float(o.score(later))<0))
    raw['non_irreducibility']={'initial':[1,2,3],'final':d2.tolist(),'later_valid':later.tolist(),'trace':tr}
    E,p,X=simple(2)
    for Q,name in [(torch.tensor([[1.,2.],[0.,1.]]),'asymmetric_precision_rejected'),(torch.tensor([[1.,0.],[0.,-1.]]),'indefinite_precision_rejected')]:
        raised=False
        try:search(QueryOracle(E,X,p,0,lambda A,B:.5-A.sum()),torch.ones(2,dtype=torch.bool),torch.ones(2),precision=Q)
        except ValueError:raised=True
        check(name,lambda:assert_true(raised))
    # A full-rank coordinate transformation must preserve a Mahalanobis quadratic.
    S=torch.diag(torch.tensor([.3,2.,5.]));Cov=torch.tensor([[2.,.3,.1],[.3,1.,.2],[.1,.2,3.]])
    xi=torch.tensor([.2,-.4,.6]);de=S@xi;Covxi=torch.linalg.solve(S,Cov)@torch.linalg.inv(S)
    left=float(de@torch.linalg.solve(Cov,de));right=float(xi@torch.linalg.solve(Covxi,xi))
    check('full_rank_covariance_coordinate_transform',lambda:assert_true(abs(left-right)<1e-12))
    # Policy residual and immutability checks target separate interface contracts.
    E,idx,pol,g=fixture(8091);X=pol.rebuild(E);oracle=QueryOracle(E,X,pol,1,lambda A,B:A.sum()+B.sum())
    delta=torch.ones(E.shape[1]);old_edges=E.clone();old_nodes=X.clone();E2,N2=oracle.state(delta)
    check('computed_residual_matches_policy_state',lambda:assert_true(oracle.residual(delta)==0 and oracle.residual(delta,'frozen')>0))
    rows=torch.ones(E.shape[0],dtype=torch.bool);rows[1]=False
    check('unqueried_flows_and_original_inputs_unchanged',lambda:assert_true(torch.equal(E2[rows],E[rows]) and torch.equal(E,old_edges) and torch.equal(X,old_nodes)))
    class FreezeProbe(torch.nn.Module):
        def __init__(self):
            super().__init__();self.w=torch.nn.Parameter(torch.ones(1));self.drop=torch.nn.Dropout(.8)
        def forward(self,E,V):return .5-self.drop(E[0]*self.w).sum()
    E,p,X=simple();model=FreezeProbe();before=model.w.detach().clone();oracle=QueryOracle(E,X,p,0,model)
    rr=search(oracle,torch.ones(1,dtype=torch.bool),torch.ones(1),max_updates=10,lr=.1)
    check('module_eval_mode_and_parameters_frozen',lambda:assert_true(not model.training and not model.w.requires_grad and model.w.grad is None and torch.equal(model.w,before) and rr.status=='Valid'))
    # Explicitly test the endpoints of the common-population support bound.
    for F in [3,35,49,55]:
        for count in range(1,F+1):assert 1/count <= 1-count/F+1/F+1e-12
    records.append({'name':'reciprocal_support_bound_enumeration','status':'PASS','cases':142})
    return {'tests':records,'metrics':metrics,'raw':raw}


def assert_true(v):
    assert bool(v)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,choices=[1,2,3],default=3);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    result=execute(args.stage)
    result.update({'evidence_type':'constructed-data verification (not a network experiment)',
        'stage':args.stage,'groups_passed':len(result['tests']),'cases_passed':sum(x['cases'] for x in result['tests']),
        'seed_ranges':{'snapshot':'1000..1099','reference':'2000..2029','gradient':'3000..3023','restoration':'4000..4063'},
        'precision':'float64, CPU, one Torch thread','training_performed':False,'reference_detector':'fixed untrained two-head smooth attention test oracle and explicit affine fixtures',
        'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform(),'cuda_available':torch.cuda.is_available()}})
    import unittest                                                       # contract tests of the reference modules
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_*.py')
    tr=unittest.TextTestRunner(verbosity=0,stream=open('/dev/null','w')).run(suite)
    result.update({'contract_tests_run':tr.testsRun,'contract_tests_ok':tr.wasSuccessful(),
                   'fd_step':1e-6,'excluded_reference_max_abs_error':0.0,
                   'excluded_reference_check':'torch.equal (bitwise identical summaries)'})
    (args.out/'results.json').write_text(json.dumps(result,indent=2))
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    (args.out/'execution.json').write_text(json.dumps({'command':sys.argv,'elapsed_seconds':time.perf_counter()-start,'code_sha256':hashes},indent=2))
    print(json.dumps({k:result[k] for k in ['stage','groups_passed','cases_passed','metrics','environment']},indent=2))

if __name__=='__main__':main()
