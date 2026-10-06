"""Focused exactness tests against expanded-copy frozen reference functions."""
from fractions import Fraction
from pathlib import Path
import copy,json,math,sys
import numpy as np
import pytest
from bootstrap_core import RiskColumn,WeightedHCP,eb_exact_moments,diagnostic_halfwidth,CellData,Q,ROUTES,METRICS
from rotcert_contracts.bounds import empirical_bernstein
from rotcert_contracts.exact import hcp_threshold
from rotcert_contracts.geometry import angle_projection


def test_counted_eb_exactly_matches_expanded_copy_reference():
    rng=np.random.default_rng(20260928)
    for _ in range(80):
        den=rng.integers(0,150,80);num=np.asarray([rng.integers(d+1) for d in den])
        weights=rng.integers(0,5,80)
        expanded=[Fraction(int(a),int(b)) for a,b,c in zip(num,den,weights) if b for _ in range(c)]
        for eta in [Fraction(1,40),Fraction(1,800)]:
            actual=RiskColumn(num,den).bound(weights,eta);expected=empirical_bernstein(expanded,eta)
            assert actual==expected
    for den in [np.array([0]),np.array([1])]:
        actual=RiskColumn(np.array([0]),den).bound(np.array([1]),Fraction(1,40))
        assert not actual.supported and actual.upper==1


def test_counted_hcp_exactly_matches_expanded_groups():
    rng=np.random.default_rng(25)
    for _ in range(80):
        groups=[rng.integers(0,10,rng.integers(0,20)).astype(float) for _ in range(25)]
        m=np.asarray(list(map(len,groups)));scores=np.concatenate(groups)
        owners=np.repeat(np.arange(len(m)),m);weights=rng.integers(0,5,len(m))
        weighted=WeightedHCP(scores,owners,m).threshold(weights)
        reference=hcp_threshold([group for group,c in zip(groups,weights) for _ in range(c)],Fraction(1,10))
        assert weighted==reference


def test_geometry_diagnostic_and_infinity():
    w=np.array([4.,16.,64.,256.,4.,16.]);h=np.array([4.,15.99,12.,2.,2.,8.])
    for q in [0.,.001,.25,1.,8.,np.inf]:
        actual=diagnostic_halfwidth(w,h,q)
        reference=np.array([angle_projection(float(a),float(b),q).half_width_upper for a,b in zip(w,h)])
        np.testing.assert_allclose(actual,reference,rtol=0,atol=1e-12)


def test_fixed_sequence_does_not_skip_failed_candidate():
    weights=np.ones(500,dtype=np.int64)
    columns=[RiskColumn(np.ones(500,dtype=int),np.ones(500,dtype=int)) for _ in Q]
    columns[-1]=RiskColumn(np.zeros(500,dtype=int),np.ones(500,dtype=int))
    # Put an artificially passing earlier candidate behind a failure. It must
    # never be tested, even without loss monotonicity.
    columns[0]=columns[-1]
    chosen,trace=CellData.sequence(columns,weights,Fraction(1,40),Fraction(1,10))
    assert math.isinf(chosen) and len(trace)==2 and trace[-1][0]==len(Q)-2
