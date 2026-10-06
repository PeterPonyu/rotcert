"""Parent-owned, descriptive G2 resource planning from verified point results.
Does not issue certificates, resample, change original outputs, or start remote work.
"""
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext, ROUND_CEILING
from datetime import datetime, timezone
import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ[k]="1"
import json, hashlib, math, fcntl
import numpy as np
ROOT=Path(__file__).resolve().parent
E=ROOT.parent
MAX_N=10000000

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def upper(mu,var,n,eta):
    if n<2:return Decimal(1),None,None
    with localcontext() as c:
        c.prec=80;c.rounding=ROUND_CEILING
        def dec(f):return Decimal(f.numerator)/Decimal(f.denominator)
        ell=c.next_plus(c.ln(dec(2/eta)))
        a=Decimal(0) if var==0 else c.next_plus(c.sqrt(Decimal(2)*dec(var)*ell/Decimal(n)))
        b=Decimal(7)*ell/Decimal(3*(n-1))
        return min(Decimal(1),dec(mu)+a+b),a,b

def minimum_n(mu,var,eta,beta):
    if mu>=beta:return None,"OBSERVED_MEAN_AT_OR_ABOVE_TARGET"
    with localcontext() as c:
        c.prec=80;limit=Decimal(beta.numerator)/Decimal(beta.denominator)
        ok=lambda n:upper(mu,var,n,eta)[0]<=limit
        hi=2
        while not ok(hi) and hi<MAX_N:hi=min(MAX_N,hi*2)
        if not ok(hi):return None,"ABOVE_PLANNING_CAP"
        lo=1
        while hi-lo>1:
            mid=(lo+hi)//2
            if ok(mid):hi=mid
            else:lo=mid
        assert ok(hi) and (hi==2 or not ok(hi-1))
        return hi,"FIXED_MOMENT_MINIMUM_FOUND"

def rounded_up(d):
    f=float(d)
    return math.nextafter(f,math.inf) if Decimal.from_float(f)<d else f

def write_new(p,x):
    with p.open("x") as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False);f.write("\n")

def main():
    with (ROOT/"writer.lock").open("a+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (ROOT/"RESULTS.json").exists(),"Existing planning output; no overwrite"
        review=E/"review/s5/VERDICT.json"
        v=json.loads(review.read_text())
        assert v["verdict"]=="PASS_POINT_STAGE_WITH_REPORTING_LIMITS"
        inputs={str(review):sha(review),str(ROOT/"PROTOCOL.json"):sha(ROOT/"PROTOCOL.json"),str(Path(__file__)):sha(Path(__file__))}
        rows=[];checks=0;summaries=[]
        for path in sorted((E/"s5/results").glob("*.json")):
            p=json.loads(path.read_text());cell=p["cell"];inp=p["input"]
            inputs[str(path)]=sha(path)
            for key in ("cache","roles"):
                assert sha(inp[key])==inp[key+"_sha256"]
                inputs[inp[key]]=inp[key+"_sha256"]
            roles=json.loads(Path(inp["roles"]).read_text())
            with np.load(inp["cache"],allow_pickle=False) as z:data={k:z[k] for k in z.files}
            names=list(map(str,data["scene_names"]));lookup={s:i for i,s in enumerate(names)}
            assert len(lookup)==len(names)
            idx=np.array([lookup[s] for s in roles["G2_cal"]],dtype=int)
            assert len(set(idx))==len(idx)
            assert not set(roles["G2_cal"])&set(roles["eval"])
            classes=list(map(str,data["class_names"]))
            assert classes==inp["class_dictionary"]==[x["class"] for x in p["classes"]]
            start=len(rows)
            for k,name in enumerate(classes):
                tp=np.bincount(data["tp_scene"][data["tp_cls"]==k],minlength=len(names))
                fn=np.bincount(data["fn_scene"][data["fn_cls"]==k],minlength=len(names))
                good=[int(i) for i in idx if tp[i]+fn[i]>0]
                losses=[Fraction(int(fn[i]),int(tp[i]+fn[i])) for i in good]
                n=len(losses);mu=sum(losses,Fraction(0))/n if n else None
                var=sum(((y-mu)**2 for y in losses),Fraction(0))/(n-1) if n>=2 else None
                assert n==p["support"]["G2_cal"][k]["GT_positive_sources"]
                checks+=1
                for b in ("0.2","0.1"):
                    cert=p["classes"][k]["G2"][b];bound=cert["bound"]
                    eta=Fraction(cert["class_delta_exact"]);beta=Fraction(b)
                    assert eta==Fraction(1,40*len(classes))
                    assert bound["n"]==n
                    assert (mu is None and bound["mean"] is None) or float(mu)==bound["mean"]
                    assert (var is None and bound["variance_ddof1"] is None) or float(var)==bound["variance_ddof1"]
                    if n>=2:
                        u,a,c=upper(mu,var,n,eta)
                        assert rounded_up(u)==bound["upper"],(cell,name,b,rounded_up(u),bound["upper"])
                        assert cert["reported"]==(Fraction.from_float(bound["upper"])<=beta)
                        proposed,status=minimum_n(mu,var,eta,beta)
                        half,half_status=minimum_n(mu,var/2,eta,beta)
                    else:u=Decimal(1);a=c=None;proposed=half=None;status=half_status="UNSUPPORTED_MOMENTS"
                    checks+=6
                    rows.append({"cell":cell,"class":name,"beta":float(beta),"eta_exact":str(eta),
                        "n_current":n,"calibration_all_sources":len(idx),"calibration_TP_positive_count":sum(tp[i]>0 for i in good),
                        "calibration_zero_TP_count":sum(tp[i]==0 for i in good),"class_support_fraction":n/len(idx),
                        "mean_exact":None if mu is None else str(mu),"variance_exact_ddof1":None if var is None else str(var),
                        "mean":None if mu is None else float(mu),"variance_ddof1":None if var is None else float(var),
                        "original_upper":bound["upper"],"original_reported":cert["reported"],
                        "padding_sqrt":None if a is None else float(a),"padding_linear":None if c is None else float(c),
                        "planning_status":status,"minimum_n_in_fixed_moment_scenario":proposed,
                        "additional_class_bearing_units":None if proposed is None else max(0,proposed-n),
                        "sample_multiplier":None if proposed is None or n==0 else proposed/n,
                        "counterfactual_half_variance_n":half,"counterfactual_half_variance_status":half_status})
            rs=rows[start:]
            summaries.append({"cell":cell,"classes":len(classes),"risk_targets":{b:{
                "reported":sum(r["original_reported"] for r in rs if r["beta"]==float(b)),
                "observed_mean_at_or_above_target":sum(r["planning_status"]=="OBSERVED_MEAN_AT_OR_ABOVE_TARGET" for r in rs if r["beta"]==float(b)),
                "mean_below_but_not_reported":sum(r["mean"] is not None and r["mean"]<float(b) and not r["original_reported"] for r in rs if r["beta"]==float(b)),
                "finite_plug_in_additional_units_range":([min(vals),max(vals)] if (vals:=[r["additional_class_bearing_units"] for r in rs if r["beta"]==float(b) and not r["original_reported"] and r["additional_class_bearing_units"] is not None]) else None)
            } for b in ("0.2","0.1")}})
        assert len(rows)==192 and len(summaries)==6
        assert all(sha(k)==h for k,h in inputs.items()),"Input changed during read-only planning"
        # Additional arithmetic cross-check: regular float expression must agree away from its last ulp.
        for r in rows:
            if r["n_current"]>=2:
                ell=math.log(2/float(Fraction(r["eta_exact"])))
                approx=min(1.,r["mean"]+math.sqrt(2*r["variance_ddof1"]*ell/r["n_current"])+7*ell/(3*(r["n_current"]-1)))
                assert abs(approx-r["original_upper"])<=4e-15
                checks+=1
        result={"status":"DESCRIPTIVE_PLANNING_ARITHMETIC_VERIFIED_NOT_CERTIFICATION","utc":datetime.now(timezone.utc).isoformat(),
            "protocol_sha256":inputs[str(ROOT/"PROTOCOL.json")],"source_sha256":inputs[str(Path(__file__))],"input_sha256":inputs,
            "assertions":checks,"independent_second_reviewer":"PENDING","rows":rows,"summary":summaries,
            "interpretation":"Calibration-only plug-in EB padding scenario after inspecting point outcomes. Exact source moments verified. Not achieved power, new certificate, or guarantee that this many new scenes will pass. Do not sum overlapping class counts."}
        # numpy boolean sums are Python-cast through JSON default only for numeric scalars.
        def clean(x):
            if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
            if isinstance(x,list):return [clean(v) for v in x]
            if isinstance(x,np.generic):return x.item()
            return x
        write_new(ROOT/"RESULTS.json",clean(result))
        print(json.dumps(clean({"status":result["status"],"assertions":checks,"summary":summaries}),indent=2))

if __name__=="__main__":main()
