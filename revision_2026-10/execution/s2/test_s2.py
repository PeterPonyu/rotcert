"""Scientific failure-risk tests, with independent Fraction/brute-force references."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import contextlib
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtr
from s2_core import (METHODS, METRICS, conditions, rng, generate_scenes, calibrate,
                    crc_thresholds, cp_rank, marginal_risks, replicate, population,
                    POIS_P, POIS_TAIL, ZMS, PS)
from s2_io import lock, save_chunk, load_chunks


def fraction_hcp(scene_scores):
    needed=Fraction(9,10)*(len(scene_scores)+1)
    candidates=sorted(set(itertools.chain.from_iterable(scene_scores)))
    for q in candidates:
        covered=sum((Fraction(sum(s<=q for s in scene),len(scene)) for scene in scene_scores),Fraction())
        if covered>=needed:return q
    return math.inf


def fraction_crc(scores,d):
    budget=Fraction(len(scores),10)-Fraction(9*d,10)
    for q in sorted(set([0]+list(scores))):
        if sum(s>q for s in scores)<=budget:return q
    return math.inf


class ScientificTests(unittest.TestCase):
    def test_unique_factorial_and_seed_streams(self):
        cc=conditions()
        self.assertEqual(len(cc),36)
        self.assertEqual(len({(c['ncal'],c['regime'],c['gamma'],c['rho']) for c in cc}),36)
        self.assertEqual([c['condition_id'] for c in cc],list(range(36)))
        self.assertEqual(sum(c['regime']=='constant8' for c in cc),9)
        self.assertTrue(all(c['gamma']==0 for c in cc if c['regime']=='constant8'))
        starts={tuple(rng(c['condition_id'],r,s).integers(0,2**32,size=4).tolist())
                for c in cc for r in [0,1] for s in [0,1,2]}
        self.assertEqual(len(starts),36*2*3)
        np.testing.assert_array_equal(rng(8,21,1).normal(size=10),rng(8,21,1).normal(size=10))

    def test_exact_hcp_fraction_reference_and_ties(self):
        gen=np.random.default_rng(5)
        for size in [1,8,9,10,50,200]:
            for _ in range(12):
                m=gen.choice([1,8,64],size=size)
                scores=gen.integers(0,12,size=int(m.sum())).astype(float)
                starts=np.r_[0,np.cumsum(m)[:-1]]
                scenes=[scores[a:a+n].tolist() for a,n in zip(starts,m)]
                _,q=calibrate(m,starts,scores,64)
                self.assertEqual(q[1],fraction_hcp(scenes))

    def test_crc_fraction_boundaries_and_no_float_slack(self):
        for n in range(0,121):
            scores=np.arange(1,n+1,dtype=float)
            ds=np.arange(0,25,dtype=np.int64)
            got=crc_thresholds(scores,ds)
            for d,q in zip(ds,got):
                self.assertEqual(q,fraction_crc(scores.tolist(),int(d)))
                if np.isfinite(q):
                    self.assertLessEqual(10*int(np.sum(scores>q)),n-9*int(d))
        # N=9D permits zero failures; N=9D-1 has no finite certificate.
        self.assertEqual(float(crc_thresholds(np.arange(1.,10.),np.array(1))),9)
        self.assertTrue(np.isinf(crc_thresholds(np.arange(1.,9.),np.array(1))))
        with self.assertRaises(ValueError):crc_thresholds(np.arange(8),np.array(1.0))
        with self.assertRaises(OverflowError):crc_thresholds(np.arange(8),np.array(2**62))
        for d in range(15):
            tied=[0.,0.,.5,.5,1.,1.,1.,2.,2.,2.]*10
            self.assertEqual(float(crc_thresholds(np.sort(tied),np.array(d))),fraction_crc(tied,d))

    def test_cp_rank_finite_boundary_and_strict_score_event(self):
        for n in range(1000):
            exact=math.ceil(Fraction(9,10)*(n+1))
            self.assertEqual(cp_rank(n),exact)
        self.assertEqual(cp_rank(8),9)
        self.assertEqual(cp_rank(9),9)
        scores=np.array([1.,1.,np.nextafter(1.,np.inf)])
        self.assertEqual(int(np.sum(scores>1)),1)
        self.assertEqual(int(np.sum(scores>=1)),3)

    def test_constant_count_exact_identity_not_same_threshold_claim(self):
        c=conditions()[0]
        for r in [0,1,3]:
            values,aux=replicate(c,r)
            np.testing.assert_array_equal(values[:,METRICS.index('scene_risk')],values[:,METRICS.index('test_object_ratio')])
            np.testing.assert_array_equal(values[:,METRICS.index('accounting_residual')],np.zeros(6))
        m=np.full(50,8);starts=np.arange(0,400,8);scores=np.arange(1.,401.)
        _,q=calibrate(m,starts,scores,8)
        self.assertEqual(q[0],361.)
        self.assertEqual(q[1],368.)
        self.assertNotEqual(q[0],q[1])

    def test_crc_D_order_dominated_M_only_when_D_le_M(self):
        scores=np.arange(1.,401.)
        ds=np.arange(0,101)
        q=crc_thresholds(scores,ds)
        self.assertTrue(np.all(q[1:]>=q[:-1]))
        qM=crc_thresholds(scores,np.array(64))
        self.assertTrue(np.all(q[:65]<=qM))
        # Small-M counterexample to an unconditional D better-than-M claim.
        self.assertGreater(float(crc_thresholds(scores,np.array(20))),float(crc_thresholds(scores,np.array(8))))

    def test_population_marginal_invariant_rho_by_independent_quadrature(self):
        # Independently integrate scene latent U; formula does not use generator.
        for gamma in (-.75,0,.75):
            for z in ZMS:
                mean=gamma*z
                for rho in (0,.5,.9):
                    for t in (-1.1,.25,1.7):
                        value,error=quad(lambda u: math.exp(-u*u/2)/math.sqrt(2*math.pi)*
                            ndtr((t-mean-math.sqrt(rho)*u)/math.sqrt(1-rho)),
                            -np.inf,np.inf,epsabs=1e-11,epsrel=1e-11)
                        self.assertLess(abs(value-float(ndtr(t-mean))),2e-10)
        self.assertAlmostEqual(float(PS@ZMS),0,places=14)
        self.assertAlmostEqual(float(PS@(ZMS*ZMS)),1,places=14)

    def test_generated_population_marginal_and_latent_correlation(self):
        base=conditions()[0].copy()
        for rho in (0,.5,.9):
            base['rho']=rho
            m,starts,s=generate_scenes(base,50000,np.random.default_rng(193+int(10*rho)))
            z=np.log(s).reshape(-1,8)
            self.assertLess(abs(z[:,0].mean()),.025)
            self.assertLess(abs(z[:,0].var()-1),.04)
            self.assertLess(abs(float(np.corrcoef(z[:,0],z[:,1])[0,1])-rho),.025)
            self.assertLess(abs(float(np.mean(z[:,0]<=.25))-float(ndtr(.25))),.012)

    def test_gamma0_removes_size_association_for_fixed_threshold(self):
        c=next(c for c in conditions() if c['regime']=='heterogeneous' and c['gamma']==0)
        p=marginal_risks(c,np.arange(.01,10,.01),np.ones(4))
        np.testing.assert_allclose(p[:4,0],p[:4,2],rtol=0,atol=2e-16)

    def test_known_denominator_not_finite_test_ratio(self):
        c=next(c for c in conditions() if c['regime']=='heterogeneous' and c['gamma']==-.75)
        values,aux=replicate(c,6)
        self.assertAlmostEqual(aux[2],5.9)
        ir=METRICS.index('object_expectation_estimator');jr=METRICS.index('test_object_ratio')
        np.testing.assert_allclose(values[:,ir],values[:,jr]*aux[1]/(2000*aux[2]),rtol=0,atol=5e-17)
        self.assertNotEqual(values[0,ir],values[0,jr])

    def test_population_analytic_matches_large_fresh_sample(self):
        # Tests conditional risk machinery with both D laws; large simulation is
        # only numerical QA, separate seed, excluded from experimental results.
        c=conditions()[3]
        m,st,sc=generate_scenes(c,50,np.random.default_rng(810))
        ss,q=calibrate(m,st,sc,64)
        expected=marginal_risks(c,ss,q)
        m,st,sc=generate_scenes(c,80000,np.random.default_rng(811))
        ds=[m,m+np.random.default_rng(812).poisson(5,len(m))]
        for a in range(6):
            thresholds=np.full(len(m),q[a]) if a<4 else crc_thresholds(ss,ds[a-4])
            fail=np.add.reduceat((sc>np.repeat(thresholds,m)).astype(int),st)
            self.assertLess(abs(float(np.mean(fail/m))-expected[a,0]),.009)
            self.assertLess(abs(float(fail.mean()/5.9)-expected[a,2]),.009)
        self.assertLess(POIS_TAIL,1e-40)
        self.assertAlmostEqual(float(POIS_P.sum()),1,places=14)

    def test_infinite_output_not_excluded_from_risk(self):
        # Every calibration count=1, no feasible CRC with D64.
        scores=np.arange(1.,11.)
        q=crc_thresholds(scores,np.array([1,64]))
        self.assertTrue(np.isinf(q[1]))
        self.assertEqual(int(np.sum(np.array([100.,1000.])>q[1])),0)
        # The theorem does not transfer to finite-only outputs (theory §4).
        types=[([1.],1),([0.]*9,9)]
        fail=0.;mass=0.;finite_fail=0.;finite_mass=0.
        for cal,mcal in types:
            for test,mtest in types:
                q=float(crc_thresholds(np.sort(cal),np.array(mtest)))
                nf=sum(s>q for s in test)
                fail+=nf/4;mass+=mtest/4
                if math.isfinite(q):finite_fail+=nf/4;finite_mass+=mtest/4
        self.assertAlmostEqual(fail/mass,.05)
        self.assertEqual(finite_fail/finite_mass,1.)

    def test_checkpoint_integrity_and_contiguous_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            directory=Path(tmp)/'00'
            v=np.zeros((3,6,20));v[0,0,0]=np.nan;a=np.ones((3,4))
            save_chunk(directory,0,v,a,'frozen')
            x,y,n=load_chunks(directory,'frozen',3)
            self.assertEqual(n,3);np.testing.assert_array_equal(x[0],v)
            with self.assertRaises(RuntimeError):save_chunk(directory,0,v,a,'frozen')
            with self.assertRaises(RuntimeError):load_chunks(directory,'different',3)
            path=next(directory.glob('*.npz'));path.write_bytes(path.read_bytes()+b'tamper')
            with self.assertRaises(RuntimeError):load_chunks(directory,'frozen',3)

    def test_duplicate_writer_lock(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            with lock(tmp):
                with self.assertRaises(RuntimeError):
                    with lock(tmp):pass

if __name__=='__main__':unittest.main(verbosity=2)
