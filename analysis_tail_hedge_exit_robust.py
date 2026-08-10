import json, contextlib, io, numpy as np, pandas as pd
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_exit_rules as e
RULES=['HOLD','X2','X3','VXN40','VXN50'];SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];LONG=['RY','AQ','WT']
def metret(r):
 r=np.asarray(r,float);eq=np.cumprod(1+r);n=len(r);yrs=n/252.;c=eq[-1]**(1/yrs)-1;v=np.std(r,ddof=1)*np.sqrt(252);sh=np.mean(r)*252/v;dd=np.min(eq/np.maximum.accumulate(eq)-1);cal=c/abs(dd) if dd<0 else np.nan;return c,sh,dd,cal
def roll(base,cand):
 z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna();arr=[];starts=pd.date_range(z.index[0],z.index[-1]-pd.DateOffset(years=5),freq='QS')
 for st in starts:
  x=z.loc[(z.index>=st)&(z.index<st+pd.DateOffset(years=5))]
  if len(x)<900:continue
  b=metret(x.b);c=metret(x.c);arr.append([c[i]-b[i] for i in range(4)])
 q=np.array(arr);return {'N':len(q),'P_CAGR':float((q[:,0]>0).mean()),'P_Sharpe':float((q[:,1]>0).mean()),'P_MDD':float((q[:,2]>0).mean()),'P_Calmar':float((q[:,3]>0).mean()),'Med_dCAGR':float(np.median(q[:,0])),'Med_dSharpe':float(np.median(q[:,1])),'Med_dMDD':float(np.median(q[:,2])),'Med_dCalmar':float(np.median(q[:,3]))}
def boot(base,cand,seed,nboot=1200):
 z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna().values;rng=np.random.default_rng(seed);N=2520;L=65;arr=[];n=len(z)
 for _ in range(nboot):
  parts=[];left=N
  while left>0:
   st=int(rng.integers(0,max(1,n-L)));blk=z[st:st+min(L,left)];parts.append(blk);left-=len(blk)
  x=np.vstack(parts)[:N];b=metret(x[:,0]);c=metret(x[:,1]);arr.append([c[i]-b[i] for i in range(4)])
 q=np.array(arr);return {'P_CAGR':float((q[:,0]>0).mean()),'P_Sharpe':float((q[:,1]>0).mean()),'P_MDD':float((q[:,2]>0).mean()),'P_Calmar':float((q[:,3]>0).mean()),'Med_dCAGR':float(np.median(q[:,0])),'Med_dSharpe':float(np.median(q[:,1])),'Med_dMDD':float(np.median(q[:,2])),'Med_dCalmar':float(np.median(q[:,3])),'P5_dCalmar':float(np.quantile(q[:,3],.05)),'P95_dCalmar':float(np.quantile(q[:,3],.95))}
Z={r:{s:e.overlay_exit(s,r) for s in SETS} for r in RULES}
print('ROLL')
for r in RULES:
 xs=[roll(e.a.p.Z[s],Z[r][s]) for s in SETS];print('AVG',r,json.dumps({k:float(np.mean([x[k] for x in xs])) for k in ['P_CAGR','P_Sharpe','P_MDD','P_Calmar','Med_dCAGR','Med_dSharpe','Med_dMDD','Med_dCalmar']}))
print('BOOT')
for r in RULES:
 xs=[boot(e.a.p.Z[s],Z[r][s],321+i) for i,s in enumerate(LONG)];print('AVG',r,json.dumps({k:float(np.mean([x[k] for x in xs])) for k in ['P_CAGR','P_Sharpe','P_MDD','P_Calmar','Med_dCAGR','Med_dSharpe','Med_dMDD','Med_dCalmar','P5_dCalmar','P95_dCalmar']}))
