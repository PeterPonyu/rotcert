"""Fixed-role source bootstrap kernels. All imports are owned immutable copies."""
from __future__ import annotations
from dataclasses import dataclass
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import hashlib,json,math,sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'vendor'))
from rotcert_contracts.bounds import EBBound,decimal_up,float_up,PRECISION,tolerance_rank
from rotcert_contracts.geometry import angle_projection
from gwd import obb_gwd

Q=np.array([0.,1.,2.,4.,8.,16.,32.,64.,128.,256.,512.,math.inf])
ROLES=('G1_cal','G2_cal','eval')
ROUTES=('HCP','representative','EB','direct_beta20','direct_beta10',
        'joint_representative_beta20','joint_representative_beta10',
        'joint_EB_beta20','joint_EB_beta10')
METRICS=('q','TP_scene_risk','TP_scene_coverage','GT_scene_FNR','GT_scene_E2E_risk',
         'object_TP_risk','normalized_radius_scene_mean','angle_halfwidth_scene_mean_radians',
         'informative_angle_scene_fraction','infinite_fraction')
SUPPORT=('source_copies','unique_source_ids','TP_positive_copies','GT_positive_copies',
         'TP_instances','GT_instances','FP_instances')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=16)
def log_upper(eta):
    with localcontext() as ctx:
        ctx.prec=PRECISION;ctx.rounding=ROUND_CEILING
        return ctx.next_plus(ctx.ln(decimal_up(2/eta)))


def eb_exact_moments(n,total,total_sq,eta):
    """Algebraically/bitwise same bound as frozen expanded-copy EB function."""
    if n<2:return EBBound(1.,total if n else None,None,n,eta,False)
    mean=total/n;variance=(total_sq-total*total/n)/(n-1)
    if variance<0:raise ArithmeticError('negative exact variance')
    with localcontext() as ctx:
        ctx.prec=PRECISION;ctx.rounding=ROUND_CEILING
        logarithm=log_upper(eta)
        radicand=Decimal(2)*decimal_up(variance)*logarithm/Decimal(n)
        root=Decimal(0) if radicand==0 else ctx.next_plus(ctx.sqrt(radicand))
        upper=min(Decimal(1),decimal_up(mean)+root+Decimal(7)*logarithm/Decimal(3*(n-1)))
    return EBBound(min(1.,float_up(upper)),mean,variance,n,eta,True)


class RiskColumn:
    """Loss=count/denominator with exact rational histogram; denominator0 excluded."""
    def __init__(self,num,den):
        num=np.asarray(num,dtype=np.int64);den=np.asarray(den,dtype=np.int64)
        if np.any(num<0) or np.any(den<0) or np.any(num>den):raise ValueError('invalid bounded count loss')
        self.support=np.flatnonzero(den>0)
        nn,dd=num[self.support],den[self.support]
        if len(nn):
            gcd=np.gcd(nn,dd);pairs=np.column_stack([nn//gcd,dd//gcd])
            values,self.inverse=np.unique(pairs,axis=0,return_inverse=True)
        else:values=np.empty((0,2),dtype=np.int64);self.inverse=np.empty(0,dtype=int)
        self.denominator=math.lcm(*(int(d) for _,d in values)) if len(values) else 1
        self.coefficients=[int(a)*(self.denominator//int(b)) for a,b in values]
        self.squared=[a*a for a in self.coefficients]

    def moments(self,weights):
        weights=np.asarray(weights,dtype=np.int64)
        selected=weights[self.support]
        n=int(selected.sum())
        histogram=np.bincount(self.inverse,weights=selected,minlength=len(self.coefficients)).astype(np.int64)
        total=sum(int(c)*a for c,a in zip(histogram,self.coefficients) if c)
        square=sum(int(c)*a for c,a in zip(histogram,self.squared) if c)
        return n,Fraction(total,self.denominator),Fraction(square,self.denominator*self.denominator)

    def bound(self,weights,eta):return eb_exact_moments(*self.moments(weights),eta)


class WeightedHCP:
    """Exact weighted scene CDF on repeated original source copies; no object expansion."""
    def __init__(self,scores,owners,m):
        self.m=np.asarray(m,dtype=np.int64)
        positive=np.unique(self.m[self.m>0])
        self.by_size=[]
        for size in positive:
            mask=self.m[owners]==size
            order=np.argsort(scores[mask],kind='stable')
            self.by_size.append((int(size),scores[mask][order],owners[mask][order]))
        self.candidates=np.unique(scores)

    def threshold(self,weights):
        weights=np.asarray(weights,dtype=np.int64)
        K=int(weights[self.m>0].sum());target=Fraction(9,10)*(K+1)
        if K==0 or K<target:return math.inf
        # Precompute integer prefix copy mass once per draw. A zero-weight source
        # cannot make the exact monotone CDF cross at its unique score.
        partial=[(m,s,np.cumsum(weights[owner],dtype=np.int64)) for m,s,owner in self.by_size]
        def reaches(q):
            mass=Fraction(0)
            for m,s,prefix in partial:
                end=int(np.searchsorted(s,q,side='right'))
                if end:mass+=Fraction(int(prefix[end-1]),m)
            return mass>=target
        lo,hi=0,len(self.candidates)-1
        while lo<hi:
            mid=(lo+hi)//2
            if reaches(self.candidates[mid]):hi=mid
            else:lo=mid+1
        return float(self.candidates[lo])


def diagnostic_halfwidth(w,h,q):
    """Point-stage diagnostic with exact reference near critical branch/endpoints."""
    out=np.full(len(w),math.pi/2)
    if math.isinf(q):return out
    gap=(w-h)/(2*np.sqrt(2));inform=(w>h)&(q<gap)
    if q==0:out[inform]=0.;return out
    wi,hi=w[inform],h[inform];hh=hi/wi;qq=q/wi
    t=(1+hh*hh)/4;d=(1-hh)*(1+hh)/8
    out[inform]=.5*np.arcsin(np.clip((qq/d)*np.sqrt(np.maximum(t-qq*qq,0)),0,1))
    near=np.flatnonzero(np.abs(q-gap)<=64*np.finfo(float).eps*np.maximum.reduce([np.abs(gap),np.full(len(w),q),np.full(len(w),1e-300)]))
    for i in near:out[i]=angle_projection(float(w[i]),float(h[i]),float(q)).half_width_upper
    candidates=np.flatnonzero(inform)
    for i in candidates[[0,-1]] if len(candidates) else []:
        exact=angle_projection(float(w[i]),float(h[i]),float(q)).half_width_upper
        if abs(out[i]-exact)>1e-10:raise ArithmeticError('geometry diagnostic disagrees with exact reference')
    return out


@dataclass
class RoleClass:
    names:np.ndarray
    m:np.ndarray
    fn:np.ndarray
    fp:np.ndarray
    scores:np.ndarray
    owners:np.ndarray
    pred:np.ndarray
    offsets:np.ndarray

    @property
    def H(self):return self.m+self.fn

    def score_groups(self):
        return [self.scores[self.offsets[i]:self.offsets[i+1]] for i in range(len(self.m))]

    def support(self,weights):
        return np.asarray([int(weights.sum()),int(np.count_nonzero(weights)),int(weights[self.m>0].sum()),
                           int(weights[self.H>0].sum()),int(weights@self.m),int(weights@self.H),int(weights@self.fp)],dtype=np.int64)

    def exceedance(self,q):
        return np.bincount(self.owners,weights=self.scores>q,minlength=len(self.m)).astype(np.int64)

    def evaluate(self,weights,q):
        if q is None:return np.full(len(METRICS),np.nan)
        bad=self.exceedance(q);n_tp=int(weights[self.m>0].sum());n_gt=int(weights[self.H>0].sum())
        def ratio_average(num,den,total):
            pos=den>0
            return float(weights[pos]@(num[pos]/den[pos])/total) if total else math.nan
        risk=ratio_average(bad,self.m,n_tp);fnr=ratio_average(self.fn,self.H,n_gt)
        e2e=ratio_average(self.fn+bad,self.H,n_gt)
        object_den=int(weights@self.m)
        object_risk=float(weights@bad/object_den) if object_den else math.nan
        if not n_tp:radius=angle=inform=infinite=math.nan
        else:
            w=np.maximum(self.pred[:,2],self.pred[:,3]);h=np.minimum(self.pred[:,2],self.pred[:,3])
            objw=weights[self.owners]/self.m[self.owners]/n_tp
            half=diagnostic_halfwidth(w,h,q)
            radius=math.inf if math.isinf(q) else float(objw@(q/np.sqrt(w*h)))
            angle=float(objw@half);inform=float(objw@(half<math.pi/2));infinite=float(math.isinf(q))
        return np.asarray([q,risk,1-risk if not math.isnan(risk) else math.nan,fnr,e2e,object_risk,radius,angle,inform,infinite])


class CellData:
    def __init__(self,cell,root=HERE):
        self.cell=cell;self.name=cell['name'];self.cell_id=cell['cell_id']
        cache=root/'inputs/cache'/f'{self.name}.npz';roles_path=root/'inputs/roles'/f'{self.name}.json'
        if sha(cache)!=cell['cache_sha256'] or sha(roles_path)!=cell['roles_sha256']:raise ValueError('snapshot hash mismatch')
        with np.load(cache,allow_pickle=False) as z:c={key:z[key] for key in z.files}
        roles=json.loads(roles_path.read_text());names=c['scene_names'].astype(str);classes=c['class_names'].astype(str)
        if classes.tolist()!=cell['class_dictionary']:raise ValueError('class dictionary mismatch')
        if len(set(names))!=len(names):raise ValueError('source names not unique')
        lookup={name:i for i,name in enumerate(names)}
        if any(len(v)!=len(set(v)) for v in roles.values()):raise ValueError('duplicate role IDs')
        all_ids=[x for r in ROLES for x in roles[r]]
        if len(all_ids)!=len(names) or set(all_ids)!=set(names):raise ValueError('role partition invalid')
        if set(all_ids)&set(roles['train']+roles['design']):raise ValueError('role leakage')
        if not np.all(np.isfinite(c['tp_conf'])) or np.any(c['tp_conf']<.05):raise ValueError('confidence floor')
        if not np.all(np.isfinite(c['tp_iou'])) or np.any(c['tp_iou']<.5):raise ValueError('matching IoU')
        for key in ('tp_pred','tp_gt'):
            if not np.all(np.isfinite(c[key])) or np.any(c[key][:,2:4]<=0):raise ValueError('invalid OBB')
        scores=obb_gwd(c['tp_pred'],c['tp_gt'])
        if not np.all(np.isfinite(scores)) or np.any(scores<0):raise ValueError('invalid GWD score')
        self.classes=classes;self.role_names={r:np.asarray(sorted(roles[r])) for r in ROLES};self.roles={r:[] for r in ROLES}
        n=len(names)
        for k in range(len(classes)):
            mask=c['tp_cls']==k;tp_sc=c['tp_scene'][mask];s=scores[mask];pred=c['tp_pred'][mask]
            m=np.bincount(tp_sc,minlength=n).astype(np.int64)
            fn=np.bincount(c['fn_scene'][c['fn_cls']==k],minlength=n).astype(np.int64)
            fp=np.bincount(c['fp_scene'][c['fp_cls']==k],minlength=n).astype(np.int64)
            for role in ROLES:
                ids=np.asarray([lookup[x] for x in self.role_names[role]],dtype=np.int64)
                mapping=np.full(n,-1,dtype=np.int64);mapping[ids]=np.arange(len(ids))
                selected=mapping[tp_sc]>=0;owner=mapping[tp_sc[selected]]
                order=np.argsort(owner,kind='stable');owner=owner[order]
                offsets=np.r_[0,np.cumsum(m[ids])].astype(np.int64)
                self.roles[role].append(RoleClass(self.role_names[role],m[ids],fn[ids],fp[ids],s[selected][order],owner,pred[selected][order],offsets))
        self.k=len(classes);self.eta=Fraction(1,40*self.k);self.direct_eta=Fraction(1,20*self.k)
        self.hcp=[];self.g1_columns=[];self.g2_columns=[];self.direct_columns=[]
        for k in range(self.k):
            g1=self.roles['G1_cal'][k];g2=self.roles['G2_cal'][k]
            self.hcp.append(WeightedHCP(g1.scores,g1.owners,g1.m))
            self.g1_columns.append([RiskColumn(g1.exceedance(q),g1.m) for q in Q])
            self.g2_columns.append(RiskColumn(g2.fn,g2.H))
            self.direct_columns.append([RiskColumn(g2.fn+g2.exceedance(q),g2.H) for q in Q])

    def role_rng(self):return np.random.Generator(np.random.PCG64(np.random.SeedSequence([20260928,5,self.cell_id,0])))

    def draw_weights(self,rng):
        output={}
        for role in ROLES:
            n=len(self.role_names[role]);indices=rng.integers(0,n,size=n,dtype=np.int64)
            output[role]=np.bincount(indices,minlength=n).astype(np.int64)
        return output

    def representative(self,k,weights,b,point_seed=False):
        g=self.roles['G1_cal'][k]
        owner=np.repeat(np.arange(len(weights),dtype=np.int32),weights)
        owner=owner[g.m[owner]>0]
        seed=[20260928,5,k] if point_seed else [20260928,5,self.cell_id,1,int(b),k]
        rng=np.random.Generator(np.random.PCG64(np.random.SeedSequence(seed)))
        initial=rng.bit_generator.state
        if point_seed:
            index=np.asarray([rng.integers(int(g.m[i])) for i in owner],dtype=np.int64)
        else:index=rng.integers(0,g.m[owner],dtype=np.int64)
        values=g.scores[g.offsets[owner]+index]
        rank=tolerance_rank(len(owner),Fraction(1,10),self.eta)
        q=None if rank.rank is None else float(np.partition(values,rank.rank-1)[rank.rank-1])
        return q,rank.rank,owner,index,{'seed':seed,'initial':initial,'final':rng.bit_generator.state}

    @staticmethod
    def sequence(columns,weights,eta,limit):
        chosen=None;upper=[]
        for j in range(len(Q)-1,-1,-1):
            bound=columns[j].bound(weights,eta);upper.append((j,bound.upper))
            if not bound.passes(limit):break
            chosen=float(Q[j])
        return chosen,upper

    def compute(self,weights,b,point_seed=False):
        for role in ROLES:
            w=np.asarray(weights[role])
            if w.dtype.kind not in 'iu' or len(w)!=len(self.role_names[role]) or np.any(w<0) or int(w.sum())!=len(w):raise ValueError('invalid role multiplicity')
        metrics=np.full((self.k,len(ROUTES),len(METRICS)),np.nan)
        status=np.zeros((self.k,len(ROUTES)),dtype=np.int8)
        support=np.zeros((self.k,3,len(SUPPORT)),dtype=np.int64)
        g2_pass=np.zeros((self.k,2),dtype=np.int8);g2_upper=np.ones(self.k);raw_fnr=np.full(self.k,np.nan)
        rep_rank=np.full(self.k,-1,dtype=np.int64);arrays={};rng_records=[];traces=[]
        for k in range(self.k):
            for j,role in enumerate(ROLES):support[k,j]=self.roles[role][k].support(weights[role])
            hcp=self.hcp[k].threshold(weights['G1_cal'])
            rep,rank,owner,index,rng=self.representative(k,weights['G1_cal'],b,point_seed)
            rep_rank[k]=-1 if rank is None else rank
            arrays[f'representative_source_{k}']=owner;arrays[f'representative_index_{k}']=index;rng_records.append(rng)
            eb,eb_trace=self.sequence(self.g1_columns[k],weights['G1_cal'],self.eta,Fraction(1,10))
            recall=self.g2_columns[k].bound(weights['G2_cal'],self.eta);g2_upper[k]=recall.upper
            g2_pass[k]=[recall.passes(Fraction(1,5)),recall.passes(Fraction(1,10))]
            direct20,t20=self.sequence(self.direct_columns[k],weights['G2_cal'],self.direct_eta,Fraction(3,10))
            direct10,t10=self.sequence(self.direct_columns[k],weights['G2_cal'],self.direct_eta,Fraction(1,5))
            qs=[hcp,rep,eb,direct20,direct10,rep if g2_pass[k,0] else None,
                rep if g2_pass[k,1] else None,eb if g2_pass[k,0] else None,eb if g2_pass[k,1] else None]
            ev=self.roles['eval'][k];computed={}
            # Raw FNR has no dependency on a localization certificate.
            H=ev.H;pos=H>0;den=int(weights['eval'][pos].sum())
            raw_fnr[k]=float(weights['eval'][pos]@(ev.fn[pos]/H[pos])/den) if den else np.nan
            for r,q in enumerate(qs):
                if q is None:continue
                status[k,r]=2 if math.isinf(q) else 1
                if q not in computed:computed[q]=ev.evaluate(weights['eval'],q)
                metrics[k,r]=computed[q]
            traces.append({'EB':eb_trace,'direct_beta20':t20,'direct_beta10':t10})
        arrays.update(metrics=metrics,status=status,support=support,g2_pass=g2_pass,g2_upper=g2_upper,
                      representative_rank=rep_rank,raw_FNR=raw_fnr)
        # Do not skip unavailable classes. An infinity remains infinity.
        arrays['macro_metrics']=np.mean(metrics,axis=0)
        arrays['macro_FNR']=np.asarray(np.mean(raw_fnr))
        arrays['class_fraction_reported']=np.mean(status>0,axis=0)
        arrays['class_fraction_finite']=np.mean(status==1,axis=0)
        arrays['class_fraction_all_space']=np.mean(status==2,axis=0)
        arrays['class_fraction_G2']=np.mean(g2_pass,axis=0)
        return arrays,{'representative_rng':rng_records,'fixed_sequence_traces':traces,
                       'quantifier':'descriptive bootstrap refitting, not a new PAC theorem'}
