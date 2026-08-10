import json, contextlib, io, numpy as np, pandas as pd
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_partial_exit as a
RULES=['HOLD','X3_HALF','X3_FULL'];SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];LONG=['RY','AQ','WT']
def met(r):
 r=np.asarray(r,float);eq=np.cumprod(1+r);yrs=len(r)/252.;c=eq[-1]**(1/yrs)-1;v=np.std(r,ddof=1)*np.sqrt(252);sh=np.mean(r)*252/v;dd=np.min(eq/np.maximum.accumulate(eq)-1);return c,sh,dd,c/abs(dd)
def roll(base,cand):
 z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna();d=[]
 for st in pd.date_range(z.index[0],z.index[-1]-pd.DateOffset(years=5),freq='QS'):
  x=z.loc[(z.index>=st)&(z.index<st+pd.DateOffset(years=5))]
  if len(x)<900:continue
  b=met(x.b);c=met(x.c);d.append([c[i]-b[i] for i in range(4)])
 q=np.array(d);return [float((q[:,i]>0).mean()) for i in range(4)]+[float(np.median(q[:,i])) for i in range(4)]
def boot(base,cand,seed):
 z=pd.concat({'b':base.ret,'c':cand.ret},axis=1).dropna().values;rng=np.random.default_rng(seed);d=[];L=65;N=2520;n=len(z)
 for _ in range(1200):
  ps=[];left=N
  while left>0:
   st=int(rng.integers(0,max(1,n-L)));blk=z[st:st+min(L,left)];ps.append(blk);left-=len(blk)
  x=np.vstack(ps)[:N];b=met(x[:,0]);c=met(x[:,1]);d.append([c[i]-b[i] for i in range(4)])
 q=np.array(d);return [float((q[:,i]>0).mean()) for i in range(4)]+[float(np.median(q[:,i])) for i in range(4)]+[float(np.quantile(q[:,3],.05)),float(np.quantile(q[:,3],.95))]
Z={r:{s:a.overlay(s,r) for s in SETS} for r in RULES}
for r in RULES:
 xs=[roll(a.a.p.Z[s],Z[r][s]) for s in SETS];print('ROLL',r,json.dumps(np.mean(xs,axis=0).tolist()))
for r in RULES:
 xs=[boot(a.a.p.Z[s],Z[r][s],777+i) for i,s in enumerate(LONG)];print('BOOT',r,json.dumps(np.mean(xs,axis=0).tolist()))
print('ORDER: P_CAGR,P_SHARPE,P_MDD,P_CALMAR,MED_DCAGR,MED_DSHARPE,MED_DMDD,MED_DCALMAR,(BOOT P5,P95 CALMAR)')
