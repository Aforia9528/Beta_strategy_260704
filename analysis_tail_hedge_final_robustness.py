import json, contextlib, io
import numpy as np
import pandas as pd
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_tail_hedge_threshold_cadence as a

CANDS=[
 ('M30_0375','MONTHLY',30,.00375),('M30_050','MONTHLY',30,.005),('M30_0625','MONTHLY',30,.00625),('M30_075','MONTHLY',30,.0075),
 ('M25_050','MONTHLY',25,.005),('M35_050','MONTHLY',35,.005),('MALL_050','MONTHLY',999,.005),
 ('Q30_050','QUARTERLY',30,.005),('QALL_050','QUARTERLY',999,.005)]
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];LONG=['RY','AQ','WT']

def metret(r):
    r=np.asarray(r,float);eq=np.cumprod(1+r);n=len(r);ppy=252.;yrs=n/ppy;c=eq[-1]**(1/yrs)-1;v=np.std(r,ddof=1)*np.sqrt(ppy);sh=np.mean(r)*ppy/v if v>0 else np.nan;dd=np.min(eq/np.maximum.accumulate(eq)-1);cal=c/abs(dd) if dd<0 else np.nan
    return c,sh,dd,cal

def rolling5(base,cand):
    z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna();out=[]
    # quarterly-spaced 5y windows to limit overlap count but preserve many regimes
    starts=pd.date_range(z.index[0],z.index[-1]-pd.DateOffset(years=5),freq='QS')
    for st in starts:
        en=st+pd.DateOffset(years=5);x=z.loc[(z.index>=st)&(z.index<en)]
        if len(x)<900:continue
        mb=metret(x.b.values);mc=metret(x.c.values);out.append([mc[i]-mb[i] for i in range(4)])
    q=np.asarray(out)
    return {'N':len(q),'P_CAGR':float(np.mean(q[:,0]>0)),'P_Sharpe':float(np.mean(q[:,1]>0)),'P_MDD':float(np.mean(q[:,2]>0)),'P_Calmar':float(np.mean(q[:,3]>0)),'Med_dCAGR':float(np.median(q[:,0])),'Med_dSharpe':float(np.median(q[:,1])),'Med_dMDD':float(np.median(q[:,2])),'Med_dCalmar':float(np.median(q[:,3]))}

def bootstrap(base,cand,nboot=1200,years=10,seed=123):
    z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna().values;n=len(z);L=65;N=int(252*years);rng=np.random.default_rng(seed);d=[]
    for _ in range(nboot):
        parts=[];left=N
        while left>0:
            st=int(rng.integers(0,max(1,n-L)));blk=z[st:st+min(L,left)];parts.append(blk);left-=len(blk)
        x=np.vstack(parts)[:N];mb=metret(x[:,0]);mc=metret(x[:,1]);d.append([mc[i]-mb[i] for i in range(4)])
    q=np.asarray(d)
    return {'P_CAGR':float(np.mean(q[:,0]>0)),'P_Sharpe':float(np.mean(q[:,1]>0)),'P_MDD':float(np.mean(q[:,2]>0)),'P_Calmar':float(np.mean(q[:,3]>0)),'Med_dCAGR':float(np.median(q[:,0])),'Med_dSharpe':float(np.median(q[:,1])),'Med_dMDD':float(np.median(q[:,2])),'Med_dCalmar':float(np.median(q[:,3])),'P5_dCalmar':float(np.quantile(q[:,3],.05)),'P95_dCalmar':float(np.quantile(q[:,3],.95))}

Z={}
for name,cad,th,b in CANDS:
    Z[name]={s:a.overlay(s,b,th,cad) for s in SETS}

print('ROLLING5')
for name,_,_,_ in CANDS:
    vals=[]
    for s in SETS:
        x=rolling5(a.p.Z[s],Z[name][s]);vals.append(x);print('SET',name,s,json.dumps(x))
    keys=['P_CAGR','P_Sharpe','P_MDD','P_Calmar','Med_dCAGR','Med_dSharpe','Med_dMDD','Med_dCalmar']
    print('AVG',name,json.dumps({k:float(np.mean([x[k] for x in vals if x['N']>0])) for k in keys}))
print('\nBOOT10Y')
for name,_,_,_ in CANDS:
    vals=[]
    for j,s in enumerate(LONG):
        x=bootstrap(a.p.Z[s],Z[name][s],1200,10,123+j);vals.append(x);print('SET',name,s,json.dumps(x))
    keys=['P_CAGR','P_Sharpe','P_MDD','P_Calmar','Med_dCAGR','Med_dSharpe','Med_dMDD','Med_dCalmar','P5_dCalmar','P95_dCalmar']
    print('AVG',name,json.dumps({k:float(np.mean([x[k] for x in vals])) for k in keys}))
