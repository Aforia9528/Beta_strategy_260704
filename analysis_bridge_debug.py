import json, contextlib, io
import pandas as pd, numpy as np
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as prod

def mret(r): return (1+r).resample('ME').prod()-1

def met(r):
 r=r.dropna();yrs=(r.index[-1]-r.index[0]).days/365.25;eq=(1+r).cumprod();return {'start':str(r.index[0]),'end':str(r.index[-1]),'n':len(r),'yrs':yrs,'terminal':float(eq.iloc[-1]),'cagr':float(eq.iloc[-1]**(1/yrs)-1),'arith_ann_252':float(r.mean()*252),'vol_252':float(r.std()*np.sqrt(252))}
for s in ['RY','AQ','WT','DB']:
 z=prod.Z[s]
 mr=mret(z.ret)
 print(s,'PROD_METRIC',json.dumps(prod.metric(z)))
 print(s,'DAILY_RECALC',json.dumps(met(z.ret)))
 print(s,'MONTHLY_RECALC',json.dumps(met(mr)))
 mr_cut=mr.loc[mr.index<=pd.Timestamp('2026-04-30')]
 print(s,'MONTHLY_CUT',json.dumps(met(mr_cut)))
 print(s,'NAV_LAST',float(z.nav.iloc[-1]),'PROD_RET_TERMINAL',float((1+z.ret).prod()),'MONTHLY_TERMINAL',float((1+mr).prod()))
