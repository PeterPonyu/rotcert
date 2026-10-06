import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[k]='1'
from fractions import Fraction
import math
from pathlib import Path
import unittest
import numpy as np
from scipy.optimize import minimize_scalar
from s3_scores import (author_margin,author_corners,official_module,coordinate_arrays,WeightedMaxRank,
                       X,ALPHA_NUM,ALPHA_DEN,ARMS,obb_gwd)
from s3_calibration import quantile_grid,coverage_curve,choose_alpha,strict_macro
from s3_geometry import gwd_halfwidth,doubled_length,endpoints,METRICS
from s3_pipeline import multiplicities,roles_from_cache


def exact_hcp(v,sc,mult,alpha):
    cnt=np.bincount(sc,minlength=len(mult));K=sum(int(mult[i]) for i in range(len(cnt)) if cnt[i])
    target=(1-alpha)*(K+1)
    for q in sorted(set(v[mult[sc]>0])):
        mass=sum((Fraction(int(mult[s]),int(cnt[s])) for x,s in zip(v,sc) if x<=q),Fraction())
        if mass>=target:return q
    return np.inf


class S3Tests(unittest.TestCase):
    def test_fraction_hcp_vector_scene_and_ties(self):
        gen=np.random.default_rng(311)
        for n in [8,9,19,50]:
            counts=gen.integers(1,22,size=n);sc=np.repeat(np.arange(n),counts)
            mult=gen.integers(0,4,size=n);vals=gen.integers(0,20,size=(len(sc),3)).astype(float)
            ans=quantile_grid(vals,sc,mult,np.array([80,400,1200]),4000)
            for a,num in enumerate([80,400,1200]):
                for j in range(3):self.assertEqual(ans[a,j],exact_hcp(vals[:,j],sc,mult,Fraction(num,12000)))
        # n9 at alpha.1 exactly finite at max, n8 infinite; vector requires more scenes.
        for n,expected in [(8,np.inf),(9,9.)]:
            got=quantile_grid(np.arange(1,n+1)[:,None],np.arange(n),np.ones(n,dtype=int),[400],4000)
            self.assertEqual(got[0,0],expected)

    def test_legacy_replication_quantiles(self):
        v=np.array([[0.,1.],[2.,1.],[4.,8.],[10.,9.]])
        sc=np.array([0,0,1,2]);m=np.array([2,0,3]);rep=np.repeat(v,m[sc],axis=0)
        got=quantile_grid(v,sc,m,[400,1200],4000,'linear')
        np.testing.assert_array_equal(got,np.quantile(rep,[.9,.7],axis=0,method='linear'))

    def test_coverage_curve_bruteforce_and_class_missing(self):
        gen=np.random.default_rng(9);sc=np.repeat(np.arange(12),np.arange(1,13));m=gen.integers(0,4,size=12)
        v=gen.integers(0,10,(len(sc),3)).astype(float);q=np.sort(gen.integers(0,12,(9,3)),axis=0)[::-1].astype(float)
        q[0]=np.inf;got=coverage_curve(v,q,sc,m)
        ref=[]
        for qt in q:
            cover=np.all(v<=qt,axis=1);per=np.array([cover[sc==i].mean() for i in range(12)])
            ref.append(float(m@per/m.sum()))
        np.testing.assert_allclose(got,ref,atol=3e-16,rtol=0)
        self.assertIsNone(choose_alpha(np.array([[1.,.9],[np.nan,np.nan]])))
        self.assertTrue(np.isnan(strict_macro(np.array([.9,np.nan]))))
        self.assertEqual(choose_alpha(np.array([[1.,.9,.9]])),1)

    def test_maxrank_weighted_refit_and_strict_left_inverse(self):
        x=np.array([[1,0,2,4,.1],[1,3,5,4,.4],[5,7,8,9,.8]],float);w=np.array([2,1,3])
        model=WeightedMaxRank(x,w);ref=X.MaxRank.fit(np.repeat(x,w,axis=0))
        z=np.vstack([x,x+np.array([.001,0,0,0,0])]);score=model.score(z)[:,0]
        np.testing.assert_array_equal(score/model.total,ref.score(z))
        for q in range(model.total):
            bounds=model.invert(np.array([q]))[0]
            np.testing.assert_array_equal(np.all(z<=bounds,axis=1),score<=q)
        self.assertTrue(np.isinf(model.invert(np.array([model.total]))).all())
        # A changed design multiplicity must change fit, not reuse cached ranks.
        other=WeightedMaxRank(x,np.array([1,1,8]));self.assertFalse(np.array_equal(model.score(z)/model.total,other.score(z)/other.total))

    def test_author_corner_and_margin_parity(self):
        official=official_module();gen=np.random.default_rng(773)
        p=np.column_stack([gen.uniform(-100,1024,(100,2)),gen.uniform(2,200,(100,2)),gen.uniform(-np.pi,np.pi,100)])
        g=p+np.column_stack([gen.normal(0,12,(100,2)),gen.normal(0,.1,(100,2)),gen.normal(0,.01,100)])
        expected_c=np.array([official.convert_to_corners(b) for b in p])
        np.testing.assert_array_equal(author_corners(p),expected_c)
        ref=np.array([official.calculate_expansion_margin(official.convert_to_corners(a),official.convert_to_corners(b)) for a,b in zip(p,g)])
        got=author_margin(p,g)
        np.testing.assert_allclose(got,ref,rtol=0,atol=1e-12)
        # Author returns zero within1e-3 even if analytic positive.
        p=np.array([[0.,0.,10.,4.,0.]]);g=p.copy();g[:,0]=.0005
        self.assertEqual(author_margin(p,g)[0],0)
        self.assertGreater(X.expansion_margin(p,g)[0],0)

    def test_gwd_grid_free_shape_projection(self):
        for w in [4.,16.,64.,256.]:
            for ar in [1.,1.001,1.01,1.1,2.,5.,20.]:
                h=w/ar;qstar=(w-h)/(2*np.sqrt(2))
                fractions=[0,.01,.25,.5,.9,.999,1,1.001,2] if ar>1 else [0,1]
                for f in fractions:
                    q=qstar*f if ar>1 else f
                    hw,full=gwd_halfwidth(np.array([w]),np.array([h]),np.array([q]));hw=float(hw[0])
                    if full[0]:
                        self.assertEqual(hw,np.pi/2);continue
                    phi=hw
                    a,b=w*w/4,h*h/4
                    P=np.array([[a*np.cos(phi)**2+b*np.sin(phi)**2,np.sqrt(a*b)],
                                [np.sqrt(a*b),a*np.sin(phi)**2+b*np.cos(phi)**2]])
                    def neg(psi):
                        u=np.array([np.cos(psi),np.sin(psi)]);return -float(u@P@u)
                    optimum=minimize_scalar(neg,bounds=(0,np.pi/4),method='bounded',options={'xatol':1e-14})
                    squared=a+b+min(optimum.fun,neg(0),neg(np.pi/4))
                    self.assertLess(abs(squared-q*q)/max(w*w,1),2e-12)
        h,f=gwd_halfwidth(np.array([10.,10.]),np.array([10.,2.]),np.array([0.,0.]))
        np.testing.assert_array_equal(f,[True,False]);np.testing.assert_array_equal(h,[np.pi/2,0])

    def test_nearsquare_boundary_against_outward_reference(self):
        from contract_reference.geometry import angle_projection,critical_radius
        for w,h in [(10.,2.),(10.,10.),(1.,1.-2**-30),(256.,256./1.001)]:
            star=critical_radius(w,h)
            for q in [0.,star,np.nextafter(star,0),star*.999,star*.25]:
                ref=angle_projection(w,h,q)
                half,full=gwd_halfwidth(np.array([w]),np.array([h]),np.array([q]))
                self.assertEqual(bool(full[0]),ref.full)
                if not ref.full:
                    self.assertLess(abs(half[0]-ref.half_width_upper),2e-8)

    def test_doubled_arc_exact_against_dense_membership(self):
        a=np.linspace(-np.pi/2,np.pi/2,200000,endpoint=False)
        for t,cq,sq in [(.2,.1,.2),(1.55,.8,.6),(0,2,2),(.2,0,0),(.6,np.inf,.3)]:
            v=float(doubled_length(np.array([t]),np.array([cq]),np.array([sq]))[0])
            brute=np.pi*np.mean((abs(np.cos(2*a)-np.cos(2*t))<=cq)&(abs(np.sin(2*a)-np.sin(2*t))<=sq))
            self.assertLess(abs(v-brute),5e-5)

    def test_geometry_undefined_vs_full_and_free_shape_counterexample(self):
        p=np.array([[0,0,10,2,0]],float);g=p.copy();covered=np.array([True])
        for name in ['iou','kld','hellinger','bhattacharyya']:
            m,_=endpoints(name,p,g,np.array([[.2]]),covered)
            self.assertTrue(np.isnan(m[0,METRICS.index('normalized_center_radius')]))
            self.assertTrue(np.isnan(m[0,METRICS.index('full_angle_rate')]))
        m,_=endpoints('expansion_analytic',p,g,np.array([[0.]]),covered)
        self.assertEqual(m[0,METRICS.index('full_angle_rate')],1)
        # At q0 containment includes tiny rectangles at every angle, not radius0.
        self.assertGreater(m[0,METRICS.index('normalized_center_radius')],1)
        self.assertEqual(m[0,METRICS.index('containment_footprint_area_px2')],20)
        for name in ['gwd','wrapped-coord']:
            q=np.array([[np.inf]]) if name=='gwd' else np.full((1,5),np.inf)
            m,_=endpoints(name,p,g,q,covered)
            self.assertEqual(m[0,METRICS.index('full_angle_rate')],1)
            self.assertEqual(m[0,METRICS.index('finite_threshold_rate')],0)

    def test_role_bootstrap_never_crosses_original_ids(self):
        c={'scene_names':np.array([str(i) for i in range(40)]),'tp_scene':np.arange(40)}
        roles=roles_from_cache(c)
        self.assertEqual(sum(map(len,roles)),40)
        for draw in [0,1]:
            ms=multiplicities(c,roles,4,draw)
            for r,m in enumerate(ms):
                self.assertEqual(int(m.sum()),len(roles[r]))
                self.assertTrue(set(np.flatnonzero(m))<=set(roles[r]))
            self.assertTrue(np.all(np.sum(np.array(ms)>0,axis=0)<=1))
        self.assertEqual(len(set(a['score_id'] for a in ARMS)),14)

if __name__=='__main__':unittest.main(verbosity=2)
