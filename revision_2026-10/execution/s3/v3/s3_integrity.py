"""Independent role reconstruction, exact canonical RNG replay, fail-closed IO."""
import json
from pathlib import Path
import numpy as np
from s3_io import sha,write_json,load_result


def canonical_roles(cache):
    # Reconstruct original unstratified split without importing runner/split helper.
    universe=np.unique(cache['tp_scene'])
    names=[str(cache['scene_names'][i]) for i in universe]
    if len(names)!=len(set(names)) or len(names)<3:raise ValueError('source dictionary invalid')
    lookup=dict(zip(names,map(int,universe)));ordered=sorted(names)
    perm=np.random.Generator(np.random.PCG64(0)).permutation(len(names))
    shuffled=[ordered[i] for i in perm];n=len(shuffled)
    nc=max(1,round(.4*n));nd=max(1,round(.2*n))
    if nc+nd>=n:nc=max(1,n-2);nd=1
    return [np.array([lookup[x] for x in sorted(ids)],dtype=np.int64)
            for ids in [shuffled[:nc],shuffled[nc:nc+nd],shuffled[nc+nd:]]]


def expected_multiplicity(cache,roles,cell_index,draw):
    out=np.zeros((3,len(cache['scene_names'])),dtype=np.int64)
    for role,ids in enumerate(roles):
        if draw==-1:out[role,ids]=1
        elif draw>=0:
            rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260928,3,cell_index,draw,role])))
            selected=rng.choice(ids,size=len(ids),replace=True)
            np.add.at(out[role],selected,1)
        else:raise ValueError('draw outside scientific domain')
    return out


def role_record(cache,roles,cell,frozen):
    return dict(frozen_sha256=frozen,cell_index=cell['cell_index'],cache_sha256=cell['cache_sha256'],
                source_names=cache['scene_names'].tolist(),class_names=cache['class_names'].tolist(),
                role_source_indices=[r.tolist() for r in roles],original_split_seed=0,
                universe='TP_positive_sources',role_names=['calibration','design','evaluation'])


def bind_roles(directory,cache,cell,frozen):
    roles=canonical_roles(cache);expected=role_record(cache,roles,cell,frozen)
    p=Path(directory)/'ROLES.json';receipt=p.with_name('ROLES-SHA256.json')
    if p.exists()!=receipt.exists():raise RuntimeError('orphan ROLES evidence')
    if p.exists():
        record=json.loads(receipt.read_text())
        if record!={'sha256':sha(p),'frozen_sha256':frozen} or json.loads(p.read_text())!=expected:
            raise RuntimeError('ROLES mismatch against canonical source recomputation')
    else:
        write_json(p,expected,immutable=True)
        write_json(receipt,dict(sha256=sha(p),frozen_sha256=frozen),immutable=True)
    return roles


def verify_roles(directory,cache,cell,frozen):
    roles=canonical_roles(cache);p=Path(directory)/'ROLES.json';receipt=p.with_name('ROLES-SHA256.json')
    if not p.exists() or not receipt.exists():raise RuntimeError('missing role binding')
    if json.loads(receipt.read_text())!={'sha256':sha(p),'frozen_sha256':frozen}:raise RuntimeError('role receipt integrity')
    if json.loads(p.read_text())!=role_record(cache,roles,cell,frozen):raise RuntimeError('canonical roles differ')
    return roles


def validate_science(r,cache,roles,cell_index,draw):
    if not np.array_equal(r['multiplicity'],expected_multiplicity(cache,roles,cell_index,draw)):
        raise AssertionError('exact RNG/multiplicity replay failed')
    if not np.array_equal(r['values'].mean(axis=2),r['macro'],equal_nan=True):raise AssertionError('strict macro mismatch')
    if np.any(np.isfinite(r['selected_alpha'][:,1])&np.any(~np.isfinite(r['design_coverage'][:,1]),axis=1)):
        raise AssertionError('tuning with missing class')
    # Counts reconstructed independently for every class and role.
    support=np.zeros_like(r['support'])
    for role,m in enumerate(r['multiplicity']):
        for k in range(len(cache['class_names'])):
            s=cache['tp_scene'][cache['tp_cls']==k];cnt=np.bincount(s,minlength=len(m))
            eligible=cnt>0
            support[role,k]=m[eligible].sum(),np.count_nonzero(m[eligible]),m[s].sum()
    if not np.array_equal(support,r['support']):raise AssertionError('support counts mismatch')


def checkpoint_prefix(directory,B):
    """Return existing complete sequential draw IDs. Refuse holes or orphan files."""
    directory=Path(directory)
    if not directory.exists():return []
    if any(directory.glob('*.tmp-*')):raise RuntimeError('orphan temporary checkpoint')
    expected=['base']+[f'draw-{i:04d}' for i in range(B)]
    actual={p.stem for p in directory.glob('*.npz')}|{p.stem for p in directory.glob('*.json') if p.stem=='base' or p.stem.startswith('draw-')}
    if not actual<=set(expected):raise RuntimeError('unexpected draw checkpoint')
    present=[];gap=False
    for i,name in enumerate(expected):
        pp=directory/(name+'.npz');jj=directory/(name+'.json')
        if pp.exists()!=jj.exists():raise RuntimeError('orphan checkpoint '+name)
        if pp.exists():
            if gap:raise RuntimeError('checkpoint hole before '+name)
            present.append(i-1)
        else:gap=True
    return present


class InputGuard:
    def __init__(self,path,digest):
        self.path=Path(path);self.digest=digest
        before=self.stat()
        if sha(self.path)!=digest:raise RuntimeError('input hash mismatch')
        self.identity=self.stat()
        if before!=self.identity:raise RuntimeError('input changed during initial hash')
    def stat(self):
        s=self.path.stat();return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns
    def check(self,force_hash=False):
        current=self.stat()
        if current!=self.identity:
            actual=sha(self.path)
            raise RuntimeError('input stat changed; fail closed, current sha='+actual)
        if force_hash:
            actual=sha(self.path)
            if actual!=self.digest or self.stat()!=self.identity:raise RuntimeError('periodic input hash mismatch')
