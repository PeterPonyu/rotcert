"""Adversarial launch/replay fixtures; all writes stay in owned temporary dirs."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import copy,json,math,platform
import numpy as np
import pytest
import scipy
import run_bootstrap as runner
import verify_bootstrap as verifier

HERE=Path(__file__).resolve().parent


def make_freeze(root):
    (root/'tiny.py').write_text('value=1\n')
    data={'environment':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},
          'files':[{'path':'tiny.py','sha256':runner.sha(root/'tiny.py')}]}
    (root/'SOURCE-FREEZE.json').write_text(json.dumps(data))
    return data


def test_manifest_replacement_is_rejected_against_initial_identity(monkeypatch):
    with TemporaryDirectory(dir=HERE,prefix='test-freeze-') as td:
        root=Path(td);data=make_freeze(root);monkeypatch.setattr(runner,'HERE',root)
        signature=runner.check_freeze();assert runner.check_freeze(signature)==signature
        (root/'tiny.py').write_text('value=2\n')
        data['files'][0]['sha256']=runner.sha(root/'tiny.py')
        (root/'SOURCE-FREEZE.json').write_text(json.dumps(data))
        # Matching modified source and manifest is a distinct release, not the
        # old run, even if its internal file hashes agree with one another.
        assert runner.check_freeze()!=signature
        with pytest.raises(RuntimeError,match='original signature'):runner.check_freeze(signature)


def test_manifest_change_during_read_is_rejected(monkeypatch):
    with TemporaryDirectory(dir=HERE,prefix='test-manifest-read-') as td:
        root=Path(td);make_freeze(root);monkeypatch.setattr(runner,'HERE',root)
        signature=runner.check_freeze();original_sha=runner.sha
        def racing_sha(path):
            result=original_sha(path)
            file=root/'SOURCE-FREEZE.json';file.write_bytes(file.read_bytes()+b'\n')
            return result
        monkeypatch.setattr(runner,'sha',racing_sha)
        with pytest.raises(RuntimeError,match='while checking'):runner.check_freeze(signature)


def test_numeric_comparison_rejects_symmetric_na_and_infinity_failures():
    for actual,reference in [(math.nan,.25),(math.inf,.25),(-math.inf,.25),
                             (.25,math.nan),(math.inf,math.nan),(math.nan,math.inf),
                             (.25,math.inf),(-math.inf,math.inf)]:
        with pytest.raises(RuntimeError):verifier.compare_number(actual,reference,'fault')
    assert verifier.compare_number(math.nan,math.nan,'NA')==0
    assert verifier.compare_number(math.inf,math.inf,'all-space')==0
    assert verifier.compare_number(.25,.25,'finite')==0


def representative_fixture(b=11):
    m=np.array([0,2,3,1],dtype=np.int64)
    cell=SimpleNamespace(name='fixture',cell_id=7,k=1,roles={'G1_cal':[SimpleNamespace(m=m)]})
    weights={'G1_cal':np.array([1,2,0,1],dtype=np.int64)}
    owner=np.repeat(np.arange(4,dtype=np.int32),weights['G1_cal']);owner=owner[m[owner]>0]
    seed=[20260928,5,7,1,b,0]
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)));initial=copy.deepcopy(rng.bit_generator.state)
    index=rng.integers(0,m[owner],dtype=np.int64)
    state={'seed':seed,'initial':initial,'final':copy.deepcopy(rng.bit_generator.state)}
    return cell,weights,{'representative_rng':[state]}, {'representative_source_0':owner,'representative_index_0':index}


def test_representative_replay_uses_canonical_seed_and_copy_order():
    cell,weights,record,saved=representative_fixture()
    verifier.verify_representatives(cell,weights,record,saved,11)
    # Replay must cover an ordinary b=11, not just bounded heavy-oracle draws.
    with pytest.raises(RuntimeError):verifier.verify_representatives(cell,weights,record,saved,12)
    bad=copy.deepcopy(record);seed=[20260928,5,7,1,999,0]
    rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)));initial=copy.deepcopy(rng.bit_generator.state)
    altered=copy.deepcopy(saved);altered['representative_index_0']=rng.integers(0,cell.roles['G1_cal'][0].m[saved['representative_source_0']],dtype=np.int64)
    bad['representative_rng'][0]={'seed':seed,'initial':initial,'final':rng.bit_generator.state}
    with pytest.raises(RuntimeError,match='noncanonical'):verifier.verify_representatives(cell,weights,bad,altered,11)
    bad=copy.deepcopy(saved);bad['representative_source_0'][0]=0
    with pytest.raises(RuntimeError):verifier.verify_representatives(cell,weights,record,bad,11)
    bad=copy.deepcopy(saved);bad['representative_index_0']=bad['representative_index_0'].astype(float)
    with pytest.raises(RuntimeError):verifier.verify_representatives(cell,weights,record,bad,11)
    bad=copy.deepcopy(record);bad['representative_rng'][0]['final']['state']['state']+=1
    with pytest.raises(RuntimeError):verifier.verify_representatives(cell,weights,bad,saved,11)


def test_metadata_and_final_checkpoint_are_bound_to_run():
    cell=SimpleNamespace(name='fixture');sig='a'*64;state={'state':29}
    rec={'schema':'S5-bootstrap-draw-v1','signature':sig,'cell':'fixture','draw_id':0,'previous_record_sha256':None}
    verifier.verify_metadata(rec,cell,0,sig,None)
    for key,value in [('schema','unknown'),('signature','b'*64),('cell','other'),('draw_id',1),('draw_id',False),('previous_record_sha256','x')]:
        bad=copy.deepcopy(rec);bad[key]=value
        with pytest.raises(RuntimeError):verifier.verify_metadata(bad,cell,0,sig,None)
    cp={'signature':sig,'target':2,'next_draw':2,'last_record_sha256':'d','role_rng_state':state}
    verifier.verify_checkpoint(cp,2,sig,'d',state)
    for key,value in [('signature','b'*64),('target',5000),('target',2.0),('next_draw',-1),('next_draw',1),('last_record_sha256','z'),('role_rng_state',{})]:
        bad=copy.deepcopy(cp);bad[key]=value
        with pytest.raises(RuntimeError):verifier.verify_checkpoint(bad,2,sig,'d',state)
