import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
import numpy as np
from s3_scores import official_module,author_corners,author_margin
from s3_native import expand_batch,contains_batch,native_readout
from s3_geometry import gwd_halfwidth,endpoints,METRICS
from s3_io import ROOT,lock,gate,save_result,load_result,sha,write_json
from s3_integrity import InputGuard,checkpoint_prefix,canonical_roles,expected_multiplicity,bind_roles,verify_roles
from s3_pipeline import roles_from_cache,multiplicities
from contract_reference.geometry import critical_radius,angle_projection


class V2Tests(unittest.TestCase):
    def test_all_execution_scripts_parse_and_verifier_imports(self):
        for path in ROOT.glob('*.py'):
            compile(path.read_text(),str(path),'exec')
        import verify_s3
        self.assertTrue(callable(verify_s3.main))

    def test_gwd_raw_side_boundary_arbitrary_scales(self):
        pairs=[(10.,np.nextafter(10.,0.)),(123.456,123.455),(10.,2.),(1.,np.nextafter(1.,0.)),(10.,10.)]
        for scale in [2.**-500,2.**-50,1.,2.**50,2.**500]:
            for w,h in pairs:
                w*=scale;h*=scale;star=critical_radius(w,h)
                for q in [0.,np.nextafter(star,0.),star,np.nextafter(star,np.inf),star*.999,star*.25]:
                    raw=8*Fraction.from_float(q)**2 >= (Fraction.from_float(w)-Fraction.from_float(h))**2
                    half,full=gwd_halfwidth(np.array([w,h]),np.array([h,w]),np.array([q,q]))
                    self.assertTrue(np.all(full==raw),(w,h,q,raw,full))
                    np.testing.assert_array_equal(half,half[::-1])
                    self.assertTrue(np.all((half>=0)&(half<=np.pi/2)))
                    if raw:self.assertTrue(np.all(half==np.pi/2))
                    elif scale==1.:
                        ref=angle_projection(w,h,q)
                        self.assertLessEqual(abs(half[0]-ref.half_width_upper),2e-8)
        for w,h,q in [(10.,np.nextafter(10.,0.),6.280369834735101e-16),(123.456,123.455,.00035355339059496197)]:
            self.assertTrue(gwd_halfwidth([w],[h],[q])[1][0])

    def test_native_fixture_exact_official(self):
        official=official_module();gen=np.random.default_rng(20260928314)
        p=np.column_stack([gen.uniform(-2048,2048,(160,2)),gen.uniform(.1,1000,(160,2)),gen.uniform(-2*np.pi,2*np.pi,160)])
        p[:4]=[[0,0,10,4,0],[0,0,10,4,np.pi/2],[100,100,4,10,0],[1024,512,100,2,np.nextafter(np.pi/2,0)]]
        g=p.copy();g[:,:2]+=gen.uniform(-5,5,(160,2));g[:,4]+=.001
        pc=author_corners(p);gc=author_corners(g)
        op=np.array([official.convert_to_corners(x) for x in p]);og=np.array([official.convert_to_corners(x) for x in g])
        np.testing.assert_array_equal(pc,op);np.testing.assert_array_equal(gc,og)
        score=np.array([official.calculate_expansion_margin(a,b) for a,b in zip(op,og)])
        np.testing.assert_array_equal(author_margin(p,g),score)
        for q in [np.zeros(160),score,np.nextafter(score,np.inf),np.maximum(0,np.nextafter(score,-np.inf)),np.full(160,1.)]:
            expanded=expand_batch(pc,q);ref=np.array([official.expand_obb(x,float(v)) for x,v in zip(op,q)])
            np.testing.assert_array_equal(expanded,ref)
            np.testing.assert_array_equal(contains_batch(gc,expanded),[official.is_obb_inside_obb(b,a) for b,a in zip(og,ref)])
        with self.assertRaises(ValueError):expand_batch(pc,np.full(160,np.inf))
        event,area,radius=native_readout(p,g,np.full(160,np.inf))
        self.assertTrue(event.all() and np.isinf(area).all() and np.isinf(radius).all())

    def test_events_differ_not_epsilon_surrogate(self):
        p=np.tile([0.,0.,10.,4.,0.],(3,1));g=p.copy();g[:,0]=[.0005,1.0005,2.]
        q=np.array([0.,1.,2.]);score=author_margin(p,g);strict=score<=q
        np.testing.assert_array_equal(strict,[True,False,True])
        m,_=endpoints('expansion_margin',p,g,q[:,None],strict)
        np.testing.assert_array_equal(m[:,METRICS.index('score_coverage')],strict)
        np.testing.assert_array_equal(m[:,METRICS.index('eav_native_containment_coverage')],[1.,1.,1.])
        np.testing.assert_array_equal(m[:,METRICS.index('eav_native_only_rate')],[0.,1.,0.])
        for field in ['normalized_center_radius','full_angle_rate','joint_center_angle_coverage','containment_footprint_area_px2']:
            self.assertTrue(np.isnan(m[:,METRICS.index(field)]).all())
        self.assertTrue(np.isfinite(m[:,METRICS.index('eav_native_emitted_polygon_area_px2')]).all())

    def test_ownership_gate_hash_identity_orphans_and_holes(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            d=Path(d)
            with lock(d/'cell.lock'):
                with self.assertRaises(RuntimeError):
                    with lock(d/'cell.lock'):pass
                with lock(d/'other.lock'):pass
            f='f'*64;p=d/'PARENT-SCHEDULE.json'
            with self.assertRaises(RuntimeError):gate(f,1,p)
            receipt=dict(approve_full=True,independent_review_passed=True,frozen_sha256=f,B=5000,cells='all16',max_workers=6,max_cpu_seconds=1,max_wall_seconds=1)
            write_json(p,receipt)
            self.assertEqual(gate(f,6,p)['max_workers'],6)
            for key in ['max_cpu_seconds','max_wall_seconds']:
                for invalid in [True,False,float('inf'),float('nan'),-1,0]:
                    bad=dict(receipt);bad[key]=invalid
                    p.write_text(json.dumps(bad))
                    with self.assertRaises(RuntimeError):gate(f,1,p)
            write_json(p,receipt)
            for h,w in [('a'*64,1),(f,7)]:
                with self.assertRaises(RuntimeError):gate(h,w,p)
            x={'a':np.array([np.nan,np.inf,1.,0.]),'ids':np.array([1,2],dtype=np.int64)}
            save_result(d/'base.npz',x,f,3,-1)
            r=load_result(d/'base.npz',f,3,-1)
            for k in x:np.testing.assert_array_equal(x[k],r[k])
            for ci,draw in [(2,-1),(3,0)]:
                with self.assertRaises(RuntimeError):load_result(d/'base.npz',f,ci,draw)
            with self.assertRaises(RuntimeError):save_result(d/'base.npz',x,f,3,-1)
            self.assertEqual(checkpoint_prefix(d,2),[-1])
            save_result(d/'draw-0001.npz',x,f,3,1)
            with self.assertRaises(RuntimeError):checkpoint_prefix(d,2)
            (d/'draw-0001.json').unlink()
            with self.assertRaises(RuntimeError):checkpoint_prefix(d,2)
            (d/'base.npz').write_bytes(b'bad')
            with self.assertRaises(RuntimeError):load_result(d/'base.npz',f,3,-1)

    def test_exact_roles_rng_and_stat_guard(self):
        cells=Path(os.environ.get('ROTCERT_S3_CELLS',ROOT/'cells.json'))
        c=json.loads(cells.read_text())[-1]
        with np.load(c['cache_path'],allow_pickle=False) as z:cache={k:z[k] for k in z.files}
        roles=canonical_roles(cache)
        for x,y in zip(roles,roles_from_cache(cache)):np.testing.assert_array_equal(x,y)
        for d in [-1,0,1,4999]:np.testing.assert_array_equal(expected_multiplicity(cache,roles,15,d),multiplicities(cache,roles,15,d))
        with tempfile.TemporaryDirectory(dir=ROOT) as d:
            p=Path(d);bind_roles(p,cache,c,'f'*64);verify_roles(p,cache,c,'f'*64)
            doc=json.loads((p/'ROLES.json').read_text());doc['role_source_indices'][0][0]=999
            write_json(p/'ROLES.json',doc)
            write_json(p/'ROLES-SHA256.json',dict(sha256=sha(p/'ROLES.json'),frozen_sha256='f'*64))
            with self.assertRaises(RuntimeError):verify_roles(p,cache,c,'f'*64)
            inp=p/'input';inp.write_bytes(b'test');g=InputGuard(inp,sha(inp));g.check(force_hash=True)
            inp.write_bytes(b'test')
            with self.assertRaises(RuntimeError):g.check()

if __name__=='__main__':unittest.main(verbosity=2)
