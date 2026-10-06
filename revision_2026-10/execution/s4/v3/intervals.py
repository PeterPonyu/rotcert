"""Fixed B5000 percentile intervals with explicit missing/infinite handling."""
from config import *
import math
from functools import lru_cache
import numpy as np
from scipy.stats import binom


def linear_extended(sorted_values,p):
 x=(len(sorted_values)-1)*p;i=math.floor(x);j=math.ceil(x);a=float(sorted_values[i]);b=float(sorted_values[j])
 if i==j or a==b:return a
 if a==-math.inf and b==math.inf:return math.nan
 if math.isinf(a):return a
 if math.isinf(b):return b
 return a+(x-i)*(b-a)



@lru_cache(maxsize=16)
def mc_ranks(N,p):
 return int(binom.ppf(.025,N,p)),int(binom.ppf(.975,N,p))+1

def interval(values,level=.95,expected_B=B,blocks=True):
 a=np.asarray(values,float)
 if a.ndim!=1 or len(a)!=expected_B:raise ValueError('Every predetermined bootstrap draw required')
 if not 0<level<1:raise ValueError('Interval level')
 missing=int(np.isnan(a).sum());report=dict(B=len(a),missing_draws=missing,positive_infinite_draws=int(np.isposinf(a).sum()),negative_infinite_draws=int(np.isneginf(a).sum()),level=level)
 if missing:return dict(report,status='UNAVAILABLE_REQUIRED_DRAW_NA',percentile=None,tail_MC=None,five_blocks1000=None)
 ordered=np.sort(a);tails=[(1-level)/2,1-(1-level)/2]
 ci=[linear_extended(ordered,p) for p in tails]
 if any(math.isnan(x) for x in ci):return dict(report,status='UNAVAILABLE_INDETERMINATE_INFINITY_INTERPOLATION',percentile=None,tail_MC=None,five_blocks1000=None)
 mc=[]
 for p in tails:
  l,u=mc_ranks(len(a),p)
  lo=-math.inf if l<=0 else float(ordered[l-1]);hi=math.inf if u>len(a) else float(ordered[u-1])
  mc.append(dict(tail_probability=p,point=linear_extended(ordered,p),binomial_order95_bracket=[lo,hi],ranks=[l,u],expected_tail_draws=len(a)*min(p,1-p),interpretation='Monte Carlo uncertainty of bootstrap quantile conditional on fitted bootstrap law'))
 block_result=None
 if blocks:
  if len(a)!=5000:raise ValueError('Prespecified five1000-block diagnostics require B5000')
  block_result=[interval(a[i*1000:(i+1)*1000],level,1000,False)['percentile'] for i in range(5)]
 return dict(report,status='APPROXIMATE_PERCENTILE',percentile=ci,tail_MC=mc,five_blocks1000=block_result,
             interpretation='Source bootstrap approximate confidence interval; not distribution-free or a simultaneous coverage theorem')


def coverage_level(tau,method,contrast):
 # Exactly16 primary family members; per-class, pooled, sensitivity and costs
 # are not silently added to this multiplicity family.
 return .996875 if tau==.5 and method==1 and contrast['shift'] in CORE_SHIFTS else .95


def summaries(draws,tau):
 if len(draws)!=B:raise ValueError('No partial-draw inferential results')
 out={};stack={k:np.stack([r[k] for r in draws]) for k in ['values','macro','contrasts','macro_contrasts','all_source']}
 for array,arr in stack.items():
  per={}
  for ix in np.ndindex(arr.shape[1:]):
   level=.95
   if array=='macro_contrasts' and ix[-1]==METRICS.index('scene_coverage'):level=coverage_level(tau,ix[1],CONTRASTS[ix[0]])
   per['/'.join(map(str,ix))]=interval(arr[(slice(None),)+ix],level)
  out[array]=per
 return dict(iou=tau,family='HCP class-macro scene coverage; .5 only, brightness/contrast only,16family across4models',arrays=out)


def contrast_numerical_status(report):
 """Exogenous readout rule; no change to B, alpha, draws or percentile CI.

A CI's lower or upper endpoint MC bracket touching zero makes its sign
numerically unresolved. Five blocks are all retained and never selected.
"""
 if report['percentile'] is None:
  return dict(status='unavailable',endpoint_MC_envelopes=None,endpoint_MC_widths=None,five_block_signs=None,
              sign_claim_allowed=False,reason='required draw/extended-real interval unavailable')
 brackets=[x['binomial_order95_bracket'] for x in report['tail_MC']]
 def sign(pair):return 'positive' if pair[0]>0 else 'negative' if pair[1]<0 else 'crosses_or_touches_zero'
 endpoint_uncertain=any(a<=0<=b for a,b in brackets)
 ci_sign=sign(report['percentile']);blocks=report['five_blocks1000']
 block_signs=[sign(p) if p is not None else 'unavailable' for p in blocks] if blocks else None
 stable=block_signs is not None and all(x==ci_sign for x in block_signs)
 status='numerically_unresolved' if endpoint_uncertain else 'positive' if ci_sign=='positive' else 'negative' if ci_sign=='negative' else 'interval_includes_zero'
 widths=[(0. if a==b else b-a) for a,b in brackets]
 return dict(status=status,percentile_sign=ci_sign,endpoint_MC_envelopes=brackets,endpoint_MC_widths=widths,
             five_block_signs=block_signs,five_block_sign_stable=stable,
             sign_claim_allowed=(not endpoint_uncertain and ci_sign in ('positive','negative') and stable),
             interpretation='Conditional numerical tail-resolution diagnostic; all5000 remain primary. No block selection, no achieved-power or exact familywise claim. A block sign change withholds a definitive sign claim even when the endpoint MC envelope is one-sided.')
