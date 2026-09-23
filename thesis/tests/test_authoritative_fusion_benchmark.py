"""Scientific invariants for the new protocol, not historical score targets."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import authoritative_fusion_benchmark as b
from mad_etd.fusion import FusionAgent
from mad_etd.schemas import AgentEvidence,FeatureGroup,ReliabilityProfile
from mad_etd.ood import apply_ood_policy_to_evidence

class ProtocolTests(unittest.TestCase):
    def test_vectorized_yager_matches_existing_runtime(self):
        rng=np.random.default_rng(42);n=150
        p=rng.uniform(.001,.999,(n,3));app=rng.random((n,3))>.2;rel=rng.uniform(.65,1,3)
        shift=rng.choice([0.,.8,.96,.98,.995],(n,3));shift[:,2]=0
        ev=b.evidence(p,app,b.masses(p[:,2]),rel,shift)
        names=['StatsDetectorAgent','TemporalBehaviorAgent','TLSProtocolAgent'];fg=[FeatureGroup.STATS,FeatureGroup.SEQUENCE,FeatureGroup.TLS]
        for use_ood in (False,True):
            pp,pred,score,verdict=b.yager(ev,use_ood)
            for i in range(n):
                evidence=[]
                for j in range(3):
                    if not app[i,j]:continue
                    m=ev['mass'][i,j]
                    item=AgentEvidence(agent_name=names[j],feature_group=fg[j],benign_support=m[0],malicious_support=m[1],uncertainty=m[2],confidence=max(m[:2]),calibration_quality=1.,distribution_shift_score=shift[i,j],distribution_shift_level=('hard' if shift[i,j]>=.99 else 'warning' if shift[i,j]>=.95 else 'in_domain') if j<2 else 'off')
                    if use_ood:item=apply_ood_policy_to_evidence(item,'hybrid')
                    evidence.append(item)
                r=ReliabilityProfile(stats_reliability=rel[0],sequence_reliability=rel[1],tls_reliability=rel[2],payload_reliability=0,input_completeness=1,ood_suspected=bool(use_ood and np.any((shift[i,:2]>=.99)&app[i,:2])))
                res=FusionAgent().fuse(evidence,r,final=True)
                code={'benign':0,'malicious':1,'suspicious':2,'unknown':3}[res.verdict.value]
                self.assertEqual(int(verdict[i]),code,(i,use_ood))
                expected=res.malicious_support/max(res.benign_support+res.malicious_support,1e-15)
                self.assertAlmostEqual(float(pp[i]),float(expected),places=10)
    def test_unqueried_ood_does_not_change_prefix(self):
        p=np.array([[.1,.9,.8]]);app=np.ones((1,3),bool);ev=b.evidence(p,app,b.masses(p[:,2]),np.ones(3));used=np.array([[1,0,0]],bool)
        before=b.yager(ev,True,used);ev['ood'][0,1]=1.;ev['p'][0,1]=0.
        after=b.yager(ev,True,used)
        for a,c in zip(before,after):np.testing.assert_allclose(a,c)
    def test_missing_gating_weights_zero(self):
        torch=b.torch_module();net=torch.nn.Linear(6,3)
        x=np.array([[.1,.9,.5,1,0,0],[.1,.9,.5,0,0,0]],np.float32)
        p,w=b.net_predict(net,'gate',x)
        np.testing.assert_allclose(w[0],[1,0,0]);np.testing.assert_allclose(w[1],[0,0,0])
        self.assertAlmostEqual(float(p[0]),.1,places=6)
    def test_edl_finite_and_backpropagates(self):
        torch=b.torch_module();z=torch.tensor([[1.,2.],[-2.,3.]],requires_grad=True)
        a=torch.nn.functional.softplus(z)+1;loss=b.edl_loss(a,torch.tensor([0,1]),1.)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(z.grad).all())
    def test_mask_does_not_change_other_view(self):
        data=dict(stats=np.ones((2,11)),temporal=np.ones((2,30)),tls_features=np.ones((2,5)),app=np.ones((2,3),bool))
        s,t,tf,a=b.condition_arrays(data,'temporal_size_mask')
        np.testing.assert_equal(s,data['stats']);self.assertTrue(np.isnan(t[:,3:14]).all());np.testing.assert_equal(t[:,14:],1)
if __name__=='__main__':unittest.main()
