import math
import unittest
from fractions import Fraction
import numpy as np
from core import hcp_sorted, pooled_sorted, roles_from_codes, original_roles, prepare, one_inner, strict_mean
from vendor.splits import three_way_scene_split


def exact_hcp(scores, sources, sizes, weights):
    target = Fraction(9*(int(weights[sizes > 0].sum())+1), 10)
    for q in np.unique(scores):
        counts = np.bincount(sources[scores <= q], minlength=len(sizes))
        mass = sum((Fraction(int(weights[i])*int(counts[i]),int(sizes[i]))
                    for i in range(len(sizes)) if sizes[i]),Fraction())
        if mass >= target: return float(q)
    return math.inf


class ScientificContracts(unittest.TestCase):
    def test_weighted_hcp_matches_rational_oracle(self):
        rng=np.random.default_rng(100)
        for _ in range(240):
            n=int(rng.integers(3,45)); sizes=rng.integers(0,11,n)
            sc=np.repeat(np.arange(n),sizes)
            s=rng.integers(0,17,len(sc))/7
            order=np.argsort(s,kind='stable');s,sc=s[order],sc[order]
            w=rng.integers(0,4,n)
            self.assertEqual(hcp_sorted(s,sc,sizes,w),exact_hcp(s,sc,sizes,w))

    def test_alpha_nine_scene_boundary(self):
        for n in [0,1,8,9,10,19,99]:
            s=np.arange(n,dtype=float);sc=np.arange(n);m=np.ones(n,dtype=int)
            self.assertEqual(hcp_sorted(s,sc,m,m),pooled_sorted(s,m))
        m=np.array([3]*9);sc=np.repeat(np.arange(9),3);s=np.arange(27.)
        self.assertEqual(hcp_sorted(s,sc,m,np.ones(9,dtype=int)),26.)

    def test_integer_pooled_equals_explicit_copies(self):
        rng=np.random.default_rng(23)
        for n in range(1,100):
            s=np.sort(rng.normal(size=n));w=rng.integers(0,5,n);expanded=np.repeat(s,w)
            rank=math.ceil(Fraction(9,10)*(len(expanded)+1))
            expected=expanded[rank-1] if rank<=len(expanded) else math.inf
            self.assertEqual(pooled_sorted(s,w),expected)

    def test_split_matches_frozen_reference(self):
        rng=np.random.default_rng(432)
        for n in [3,4,8,9,48,11738]:
            counts=rng.integers(0,4,n)
            if np.count_nonzero(counts)<3:counts[:3]=1
            names=[f's{i:07d}' for i in np.flatnonzero(counts)]
            for seed in [0,9,100003]:
                result=roles_from_codes(counts,seed)
                ref=three_way_scene_split(names,seed=seed)
                for role,key in [(1,'calibration'),(2,'matching'),(3,'eval')]:
                    self.assertEqual(np.flatnonzero(result==role).tolist(),[int(x[1:]) for x in ref[key]])
                self.assertTrue(np.all(result[counts==0]==0))

    def test_original_roles_use_names_not_codes(self):
        names=np.array(['Z','T','A','K','B','S'])
        universe=np.array([0,1,2,4,5]);role=original_roles(names,universe,0)
        ref=three_way_scene_split(names[universe],seed=0)
        for j,key in [(1,'calibration'),(2,'matching'),(3,'eval')]:
            self.assertEqual(sorted(names[role==j]),ref[key])

    def test_no_support_cannot_be_deleted(self):
        s=np.array([.2,.5]);sc=np.array([0,1]);cl=np.array([0,1]);p=np.tile([0.,0.,4.,2.,0.],(2,1))
        data=prepare(s,sc,cl,p,p,3,2)
        q,metrics,support=one_inner(data,np.ones(3,dtype=int),np.array([1,1,3]))
        self.assertTrue(np.isnan(metrics).all())
        self.assertTrue(np.isnan(strict_mean(metrics,axis=1)).all())
        self.assertTrue(np.isinf(q).all())

    def test_weighted_evaluation_and_empty_cal_is_full_set(self):
        s=np.array([1.,2.,5.,6.]);sc=np.array([0,0,1,2]);cl=np.zeros(4,dtype=int)
        p=np.tile([0.,0.,4.,2.,0.],(4,1));data=prepare(s,sc,cl,p,p,4,1)
        q,m,_=one_inner(data,np.array([1,2,3,1]),np.array([3,3,3,1]))
        self.assertTrue(np.isinf(q).all());self.assertTrue((m==1).all())
        q,m,_=one_inner(data,np.array([10,2,3,1]),np.array([1,3,3,2]))
        self.assertTrue((q==2).all());self.assertTrue((m==0).all())

if __name__=='__main__':unittest.main()
