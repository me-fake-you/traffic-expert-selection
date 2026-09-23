import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import frozen_fusion_fast as f

class FrozenFusionTests(unittest.TestCase):
    def test_probability_input_neutralizes_unavailable_views(self):
        p=np.array([[.1,.9,.99]]);app=np.array([[True,True,False]])
        np.testing.assert_allclose(f.probability_features(p,app),[[.9,.1,.1,.9,.5,.5]])
    def test_gating_masks_and_renormalizes(self):
        torch=f.torchlib();p=np.array([[.1,.9,.99],[.1,.9,.99]]);app=np.array([[1,0,1],[0,0,0]],bool)
        x=torch.tensor(f.gating_features(p,app));prob,w=f.gate_probability(torch.tensor([[0.,100.,0.],[0.,0.,0.]]),x)
        np.testing.assert_allclose(w.numpy(),[[.5,0,.5],[0,0,0]])
        np.testing.assert_allclose(prob.numpy(),[.545,.5],atol=1e-6)
    def test_network_parameter_counts(self):
        self.assertEqual(sum(x.numel() for x in f.make_net(f.CONFIG['mlp_dims']).parameters()),786)
        self.assertEqual(sum(x.numel() for x in f.make_net(f.CONFIG['gate_dims']).parameters()),211)
    def test_training_reader_excludes_test_and_base_artifacts(self):
        import inspect
        source=inspect.getsource(f.train_cell)
        for forbidden in ('test_clean.npz','labels_test.npz','features.npz','models.joblib','predict_proba'):
            self.assertNotIn(forbidden,source)
        self.assertIn('train_oof.npz',source);self.assertIn('validation.npz',source)
if __name__=='__main__':unittest.main()
