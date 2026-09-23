import sys,unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from natural_shift_audit import natural_bootstrap

class NaturalTests(unittest.TestCase):
    def test_unequal_group_sizes_and_empty_draw(self):
        group=np.array([0,1,0,1,1],np.int16);code=np.array([0,3,1,0,2],np.int8)
        weights=np.zeros((4,20),np.int16);weights[0,:2]=[1,1];weights[1,:2]=[0,2];weights[2,:2]=[2,0];weights[3,2]=2
        coverage=np.array([.000025,.25,.5,1]);h=np.r_[0,np.cumsum(1/np.arange(1,11))]
        f,e,a,n=natural_bootstrap(group,code,weights,coverage,h)
        for b in range(3):
            cc=np.repeat(code,weights[b,group]);yy=cc//2;pp=cc%2;risks=np.cumsum(yy!=pp)/np.arange(1,len(cc)+1)
            self.assertEqual(n[b],len(cc));self.assertAlmostEqual(a[b],risks.mean())
            for j,q in enumerate(coverage):
                k=max(1,round(q*len(cc)));self.assertAlmostEqual(e[b,j],risks[k-1]);self.assertAlmostEqual(f[b,j],f1_score(yy[:k],pp[:k],labels=[0,1],average='macro',zero_division=0))
        self.assertEqual(n[3],0);self.assertTrue(np.isnan(a[3]));self.assertTrue(np.isnan(f[3]).all())
if __name__=='__main__':unittest.main()
