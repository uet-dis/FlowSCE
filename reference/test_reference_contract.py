"""Contract tests of the reference implementation on constructed inputs.

    python -m unittest discover -s reference -p "test_*.py"
"""
from __future__ import annotations
import unittest
import numpy as np
import torch
from evaluation_contract import detector_metrics, query_summary, paired_cluster_bootstrap
from traffic_validator import validate_raw
from repeated_restoration import repeated_restoration
from flowsce_reference import QueryOracle, AffinePolicy, search

class ReferenceContracts(unittest.TestCase):
    def test_confusion_counts(self):
        r=detector_metrics([1,1,0,0],[.9,.2,.6,.1]);self.assertEqual([r[k] for k in ['TP','TN','FP','FN']],[1,1,1,1]);self.assertEqual(r['accuracy'],.5);self.assertEqual(r['macro_F1'],.5)
    def test_threshold_tie_is_alert(self):
        r=detector_metrics([1,0],[.5,.1]);self.assertEqual(r['TP'],1)
    def test_undefined_precision_is_na(self):
        self.assertIsNone(detector_metrics([1,0],[.1,.1])['precision'])
    def test_bad_probability_rejected(self):
        with self.assertRaises(ValueError):detector_metrics([1],[float('nan')])
    def test_no_population_rejected(self):
        with self.assertRaises(ValueError):detector_metrics([],[])
    def test_undefined_fpr_is_na(self):
        self.assertIsNone(detector_metrics([1,1],[.9,.1])['FPR'])
    @staticmethod
    def row(i=1,qr=-.1,qn=-.1):return dict(query_id=str(i),status='returned',F=2,delta=[.1,0.],nominal_score=qn,reconstructed_score=qr,admissible=True)
    def test_different_success_denominators(self):
        rows=[self.row(1),self.row(2,qr=.1),dict(query_id='3',status='failure')]
        r=query_summary(rows);self.assertEqual(r['CSR'],1/3);self.assertEqual(r['SVR'],.5);self.assertEqual(r['post_reconstruction_failure_all'],1/3)
    def test_no_nominal_success_is_na(self):
        self.assertIsNone(query_summary([dict(query_id='1',status='failure')])['SVR'])
    def test_tiny_critical_change_is_retained(self):
        row=self.row();row['delta']=[1e-10,0.];r=query_summary([row]);v=r['per_query_quality'][0];self.assertEqual(v['support_exact'],1);self.assertEqual(v['support_epsilon'],0)
    def test_domain_failure_not_success(self):
        row=self.row();row['admissible']=False;self.assertEqual(query_summary([row])['CSR'],0)
    def test_duplicate_query_rejected(self):
        with self.assertRaises(ValueError):query_summary([self.row(),self.row()])
    def test_not_alert_is_separate_population(self):
        with self.assertRaises(ValueError):query_summary([dict(query_id='1',status='NotAlert')])
    def test_missing_true_score_rejected(self):
        r=self.row();r.pop('reconstructed_score')
        with self.assertRaises(KeyError):query_summary([r])
    def test_paired_bootstrap_preserves_exact_offset(self):
        r=paired_cluster_bootstrap([1,2,3,4],[0,1,2,3],['a','a','b','b'],repetitions=200);self.assertEqual(r['ci95'],[1.,1.])
    def test_one_fixture_is_not_bootstrap_population(self):
        with self.assertRaises(ValueError):paired_cluster_bootstrap([1],[0],['single'],repetitions=200)
    def test_bootstrap_replay(self):
        a=paired_cluster_bootstrap([1,2,3],[0,0,0],['a','b','c'],repetitions=200);b=paired_cluster_bootstrap([1,2,3],[0,0,0],['a','b','c'],repetitions=200);self.assertEqual(a,b)
    @staticmethod
    def schema():
        return dict(evidence_type='constructed schema; not an extractor mapping',fields={
            'bytes':dict(unit='byte',mutable=True,lower=0,integer=True),
            'duration':dict(unit='s',mutable=True,lower=0),
            'bit_rate':dict(unit='bit/s',mutable=True,lower=0),
            'protocol':dict(unit='code',mutable=False,integer=True)},
            relations=[dict(kind='rate',rate='bit_rate',total='bytes',duration='duration',factor=8)])
    def test_valid_raw_identity(self):
        d=dict(bytes=100,duration=2,bit_rate=400,protocol=6);self.assertEqual(validate_raw(d,d,self.schema()),[])
    def test_negative_count_rejected(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6);d=dict(o,bytes=-1);self.assertTrue(validate_raw(d,o,self.schema()))
    def test_fractional_count_rejected(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6);d=dict(o,bytes=1.5);self.assertIn('bytes: noninteger',validate_raw(d,o,self.schema()))
    def test_incompatible_rate_rejected(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6);d=dict(o,bytes=50,bit_rate=500);self.assertIn('rate relation: bit_rate',validate_raw(d,o,self.schema()))
    def test_zero_duration_not_silently_imputed(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6);self.assertTrue(validate_raw(dict(o,duration=0),o,self.schema()))
    def test_immutable_raw_field_rejected(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6);self.assertIn('protocol: immutable',validate_raw(dict(o,protocol=17),o,self.schema()))
    def test_unknown_fields_not_guessed(self):
        o=dict(bytes=100,duration=2,bit_rate=400,protocol=6)
        with self.assertRaises(ValueError):validate_raw(dict(o,unknown=1),o,self.schema())
    def test_inactive_latent_not_used_to_reduce_covariance_penalty(self):
        from unittest.mock import patch
        torch.set_default_dtype(torch.float64)
        E=torch.zeros((1,2));p=AffinePolicy.snapshot(torch.tensor([[0],[1]]),2,2)
        o=QueryOracle(E,p.rebuild(E),p,0,lambda A,B:.5-A[0,0])
        saved=[];original_step=torch.optim.Adam.step
        def inspect_step(opt,*args,**kwargs):
            saved.append(opt.param_groups[0]['params'][0].grad.detach().clone())
            return original_step(opt,*args,**kwargs)
        with patch.object(torch.optim.Adam,'step',inspect_step):
            search(o,torch.tensor([True,False]),torch.ones(2),max_updates=2,lr=.1,lambda_prox=.1,precision=torch.tensor([[1.,.8],[.8,1.]]))
        self.assertEqual(len(saved),2)
        self.assertTrue(all(float(g[1])==0. for g in saved))
    def test_repeated_restoration_is_finite(self):
        torch.set_default_dtype(torch.float64);E=torch.zeros((1,4));p=AffinePolicy.snapshot(torch.tensor([[0],[1]]),2,4);X=p.rebuild(E)
        o=QueryOracle(E,X,p,0,lambda A,B: .5-A.sum());de,tr=repeated_restoration(o,torch.ones(4),torch.ones(4,dtype=torch.bool));self.assertEqual(int((de.abs()>1e-8).sum()),1);self.assertLessEqual(len(tr),10)
if __name__=='__main__':unittest.main(verbosity=2)
