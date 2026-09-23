import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import authoritative_selective as a

class SelectiveTests(unittest.TestCase):
    def test_group_bootstrap_reranks_and_exact_aurc(self):
        group=np.array([0,1,0,1,0,1,0,1],np.int16)
        code=np.array([0,3,1,0,2,3,1,2],np.int8)
        weights=np.array([[1,1],[0,2],[2,0]],np.int16)
        ks=np.arange(1,9,dtype=np.int32);h=np.r_[0,np.cumsum(1/ks)]
        f,e,ar,c,t=a.weighted_curves(group,code,weights,ks,h)
        for b,w in enumerate(weights):
            expanded=np.repeat(np.arange(8),w[group]);codes=code[expanded]
            risks=[]
            for j,k in enumerate(ks):
                yy=codes[:k]//2;yp=codes[:k]%2;risks.append(np.mean(yy!=yp))
                self.assertAlmostEqual(f[b,j],f1_score(yy,yp,labels=[0,1],average='macro',zero_division=0))
                self.assertAlmostEqual(e[b,j],risks[-1]);self.assertEqual(c[b,j],expanded[k-1])
                self.assertEqual(t[b,j],np.sum(expanded[:k]==expanded[k-1]))
            self.assertAlmostEqual(ar[b],np.mean(risks))
        # Resampling changes acceptance, rather than weighting a fixed original prefix.
        self.assertNotEqual(c[0,0],c[1,0])
    def test_paired_boundary_copies(self):
        groups=np.array([0,1,0,1],np.int16);order=np.array([0,1,2,3]);order1=np.array([1,0,3,2]);rank=np.argsort(order1).astype(np.int32)
        weights=np.array([[1,1],[2,0],[0,2]],np.int16);flags=np.array([[1,0,0,0],[0,1,1,0],[0,1,1,1],[1,0,0,0]],np.int8)
        ks=np.array([1,3,4],np.int32);h=np.r_[0,np.cumsum(1/np.arange(1,5))]
        r0=a.weighted_curves(groups[order],np.zeros(4,np.int8),weights,ks,h)
        r1=a.weighted_curves(groups[order1],np.zeros(4,np.int8),weights,ks,h)
        got=a.paired_coverage_counts(order,rank,groups,flags,weights,r0[3],r0[4],r1[3],r1[4])
        for b,w in enumerate(weights):
            x0=[(i,j) for i in order for j in range(w[groups[i]])];x1=[(i,j) for i in order1 for j in range(w[groups[i]])]
            for q,k in enumerate(ks):
                joint=set(x0[:k])&set(x1[:k]);expected=np.r_[np.sum([flags[i] for i,j in joint],axis=0) if joint else np.zeros(4),len(joint)]
                np.testing.assert_allclose(got[b,q],expected)
    def test_native_rejects_last_and_unavailable_excluded(self):
        m='M';d={'index':np.array([3,2,1,0]),'clean__M__available':np.array([1,1,1,0],bool),'clean__M__verdict':np.array([3,0,2,0]),'clean__M__score':np.array([.99,.2,.9,1.])}
        np.testing.assert_array_equal(a.ordered(d,'clean',m),[1,2,0])

if __name__=='__main__':unittest.main()
