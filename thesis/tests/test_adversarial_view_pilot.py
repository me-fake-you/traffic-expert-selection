import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import adversarial_view_pilot as p

def example():
    full=np.array([100.,300.,800.,2000.]);directions=np.array([1,-1,1,-1])
    d=p.b.DetectorInput(stats=dict(packet_count=4,total_bytes=full.sum(),outbound_bytes=900.,inbound_bytes=2300.,mean_packet_length=full.mean(),packet_length_variance=full.var(),outbound_ratio=900/3200,duration=2.),sequence=p.b.SequenceFeatures(packet_lengths=[100,300],directions=[1,-1],iats=[.1],bursts=[],truncated=True,original_packet_count=4),tls={})
    return d,full,directions

def test_prefix_padding_matches_literal_full_moments():
    d,x,direction=example()
    for name,floors in [('stats_mild',[128,128]),('stats_strong',[1500,1500]),('stats_flip',[1500,128])]:
        got=p.transform(d,name);changed=x.copy();changed[:2]=np.maximum(changed[:2],floors)
        assert got.sequence==d.sequence and got.tls==d.tls
        np.testing.assert_allclose(got.stats['total_bytes'],changed.sum())
        np.testing.assert_allclose(got.stats['mean_packet_length'],changed.mean())
        np.testing.assert_allclose(got.stats['packet_length_variance'],changed.var())
        np.testing.assert_allclose(got.stats['outbound_bytes'],changed[direction==1].sum())
        np.testing.assert_allclose(got.stats['inbound_bytes'],changed[direction==-1].sum())

def test_temporal_isolation_and_missing_tls():
    d,_,_=example();got=p.transform(d,'temporal_flip')
    assert got.stats==d.stats and got.tls==d.tls
    assert got.sequence.iats==[2.] and got.sequence.directions==[-1,1]
    for name in ['tls_mild','tls_strong','tls_flip']:assert p.transform(d,name)==d

def test_cipher_negative_control():
    d,_,_=example();d.tls=dict(version='tls1.3',cipher_suite='original',handshake_complete=True)
    agent=p.b.TLSProtocolAgent(contract_native_fields=True)
    a=agent.analyze(d);c=agent.analyze(p.transform(d,'tls_mild'))
    assert (a.benign_support,a.malicious_support,a.uncertainty)==(c.benign_support,c.malicious_support,c.uncertainty)
