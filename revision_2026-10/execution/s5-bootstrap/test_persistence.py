import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
import copy,json,math,os
import numpy as np
import pytest
import run_bootstrap as runner
from verify_bootstrap import interval,extended_linear

HERE=Path(__file__).resolve().parent


class TinyCell:
    name='tiny';k=1
    def role_rng(self):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([91])))
    def draw_weights(self,rng):return {r:np.bincount(rng.integers(0,4,size=4),minlength=4) for r in runner.ROLES}
    def representative(self,k,weights,b):
        owner=np.repeat(np.arange(4),weights)
        rng=np.random.default_rng([23,b]);initial=rng.bit_generator.state
        index=rng.integers(0,3,size=len(owner))
        return 2.,len(owner),owner,index,{'seed':[23,b],'initial':initial,'final':rng.bit_generator.state}


def test_lock_contention_does_not_start_second_writer():
    with TemporaryDirectory(dir=HERE,prefix='test-lock-') as td:
        path=Path(td)/'lock'
        with runner.writer_lock(path):
            with pytest.raises(RuntimeError):
                with runner.writer_lock(path):pass
        with runner.writer_lock(path):pass


def test_atomic_draw_recovery_and_tamper_refusal():
    with TemporaryDirectory(dir=HERE,prefix='test-checkpoint-') as td:
        run=Path(td);cell=TinyCell();sig='s'*64
        folder,rng,prev,b=runner.restore_cell(run,cell,sig,2,False)
        before=copy.deepcopy(rng.bit_generator.state);weights=cell.draw_weights(rng)
        _,_,owner,index,reprec=cell.representative(0,weights['G1_cal'],0)
        arrays={**{'weights_'+r:v for r,v in weights.items()},'representative_source_0':owner,'representative_index_0':index}
        # Atomic draw exists but checkpoint still says0: supported crash window.
        recorded=runner.commit_draw(folder,cell,0,arrays,{'representative_rng':[reprec]},sig,prev,before,rng.bit_generator.state,.001)
        partial=folder/'.partial-preserve';partial.mkdir();(partial/'partial.txt').write_text('preserve')
        _,recovered,last,next_draw=runner.restore_cell(run,cell,sig,2,True)
        assert next_draw==1 and last==recorded and recovered.bit_generator.state==rng.bit_generator.state
        assert partial.exists()
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,sig,2,False)
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,'z'*64,2,True)
        meta=folder/'draw-00000/record.json';bad=json.loads(meta.read_text());bad['role_rng_after']['state']['state']+=1
        meta.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,sig,2,True)


def test_full_requires_review_and_no_pilot_interval():
    with pytest.raises(RuntimeError):runner.launch_gate(argparse.Namespace(mode='full',approval=None),'s'*64)
    assert interval([.8,.9])['status']=='UNAVAILABLE_INCOMPLETE_B'
    data=np.full(5000,.9);data[301]=np.nan
    assert interval(data)['interval'] is None
    assert interval(data)['missing_draws']==1
    assert extended_linear([1.,math.inf],.5)==math.inf
    assert extended_linear([math.inf,math.inf],.025)==math.inf
    assert extended_linear([1.,2.,3.],.5)==np.quantile([1.,2.,3.],.5,method='linear')
    inf=np.full(5000,.9);inf[4900:]=math.inf
    out=interval(inf)
    assert out['infinite_draws']==100 and out['status']=='AVAILABLE_EXTENDED_REAL'


def test_finite_extended_quantiles_match_numpy_linear():
    rng=np.random.default_rng(929)
    for n in [1,2,10,100,5000]:
        a=rng.normal(size=n)
        for p in [0.,.025,.5,.975,1.]:
            assert abs(extended_linear(a,p)-np.quantile(a,p,method='linear'))<2e-15


def test_checkpoint_lag_must_match_previous_prefix_and_negative_rejected():
    with TemporaryDirectory(dir=HERE,prefix='test-prefix-') as td:
        run=Path(td);cell=TinyCell();sig='p'*64
        folder,rng,prev,b=runner.restore_cell(run,cell,sig,2,False)
        checkpoint=folder/'checkpoint.json';correct=json.loads(checkpoint.read_text())
        bad=copy.deepcopy(correct);bad['next_draw']=-1
        checkpoint.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,sig,2,True)
        checkpoint.write_text(json.dumps(correct))
        before=copy.deepcopy(rng.bit_generator.state);weights=cell.draw_weights(rng)
        _,_,owner,index,reprec=cell.representative(0,weights['G1_cal'],0)
        arrays={**{'weights_'+r:v for r,v in weights.items()},'representative_source_0':owner,'representative_index_0':index}
        runner.commit_draw(folder,cell,0,arrays,{'representative_rng':[reprec]},sig,prev,before,rng.bit_generator.state,.001)
        # Checkpoint lags one atomic commit but its saved prior RNG/hash is wrong.
        bad=copy.deepcopy(correct);bad['role_rng_state']['state']['state']+=1
        checkpoint.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,sig,2,True)
        bad=copy.deepcopy(correct);bad['last_record_sha256']='0'*64
        checkpoint.write_text(json.dumps(bad))
        with pytest.raises(RuntimeError):runner.restore_cell(run,cell,sig,2,True)
        checkpoint.write_text(json.dumps(correct))
        _,_,_,n=runner.restore_cell(run,cell,sig,2,True)
        assert n==1
