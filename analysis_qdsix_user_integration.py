import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf

buf=io.StringIO()
with contextlib.redirect_stdout(buf):import analysis_production_summary_dca as p

END=pd.Timestamp('2026-06-30')
def dl(t,start='2014-01-01'):
 x=yf.download(t,start=start,end='2026-07-02',auto_adjust=True,progress=False,threads=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
qds=dl('QDSIX','2020-06-01');fx=p.raw['FX'];sgov=p.raw['SGOV'];qld=p.raw['QLD'];gold=p.raw['GOLD_H'];dbmf=p.raw['DBMF']

def total_px_usd_to_krw(px):
 idx=px.index.union(fx.index).sort_values();return px.reindex(idx).ffill()*fx.reindex(idx).ffill()
qds_k=total_px_usd_to_krw(qds);db_k=total_px_usd_to_krw(dbmf);cash_k=total_px_usd_to_krw(sgov)

def make_ret(series,idx):return series.reindex(idx).ffill().pct_change()
idx=pd.DatetimeIndex(sorted(set(qld.index)|set(gold.index)|set(qds_k.index)|set(db_k.index)|set(cash_k.index)))
idx=idx[(idx>=pd.Timestamp('2020-07-01'))&(idx<=END)]
R=pd.DataFrame({'q':make_ret(qld,idx),'gold':make_ret(gold,idx),'db':make_ret(db_k,idx),'qds':make_ret(qds_k,idx),'cash':make_ret(cash_k,idx)},index=idx).dropna()
Q=p.Q.reindex(R.index,method='ffill').dropna();R=R.loc[Q.index]

def sim(qds_frac):
 # Defensive bucket: qds_frac to QDSIX, residual split 50/50 Gold(H) and DBMF.
 q=Q.values;hb=np.minimum(.70,np.maximum(0,1-q));cash=np.maximum(0,1-q-hb);res=1-qds_frac
 W=np.c_[q,.5*res*hb,.5*res*hb,qds_frac*hb,cash]
 A=R[['q','gold','db','qds','cash']].values
 wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,dt in enumerate(R.index):
  wt=W[i];fee=0.
  if np.max(np.abs(wc-wt))>.075:
   to=float(np.sum(np.abs(wt-wc)));fee=.0007*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);ret=(1-fee)*(1+gr)-1;nav*=1+ret;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,ret,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');z.attrs={'turn':turn,'nt':nt};return z

def metric(z):
 d=z.ret;eq=z.nav;yrs=(d.index[-1]-d.index[0]).days/365.25;c=float(eq.iloc[-1]**(1/yrs)-1);mdd=float((eq/eq.cummax()-1).min());m=(1+d).resample('ME').prod()-1;bil=(1+p.raw['BIL'].pct_change()).resample('ME').prod()-1;bil=bil.reindex(m.index).ffill().bfill();ex=m-bil;vol=float(m.std()*np.sqrt(12));sh=float(ex.mean()*12/(ex.std()*np.sqrt(12)));return {'Years':yrs,'CAGR':c,'VolMonthlyAnn':vol,'SharpeExcess':sh,'MDDDaily':mdd,'Calmar':c/abs(mdd),'TradesYr':z.attrs['nt']/yrs,'TurnYr':z.attrs['turn']/yrs}
for f in [0,.25,.5,.75,1.0]:print('HEDGE_QDS_FRAC',f,json.dumps(metric(sim(f))))

# Top-level monthly blending of exact production DB_ACT and QDSIX, no re-levering.
u=(1+p.Z['DB_ACT'].ret).resample('ME').prod()-1;q=qds.pct_change().resample('ME').apply(lambda x:(1+x).prod()-1);Z=pd.concat({'u':u,'q':q},axis=1).dropna();Z=Z.loc[Z.index<=END]
def mmetric(r):
 yrs=(r.index[-1]-r.index[0]).days/365.25;eq=(1+r).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);vol=float(r.std()*np.sqrt(12));bil=(1+p.raw['BIL'].pct_change()).resample('ME').prod()-1;rf=bil.reindex(r.index).ffill().bfill();ex=r-rf;sh=float(ex.mean()*12/(ex.std()*np.sqrt(12)));mdd=float((eq/eq.cummax()-1).min());return {'CAGR':c,'Vol':vol,'SharpeExcess':sh,'MDDmonthly':mdd,'Calmar':c/abs(mdd)}
for f in [0,.1,.2,.3,.4,.5]:
 r=(1-f)*Z.u+f*Z.q;print('TOPLEVEL_QDS_FRAC',f,json.dumps(mmetric(r)))
print('TOPLEVEL_CORR',float(Z.u.corr(Z.q)))
