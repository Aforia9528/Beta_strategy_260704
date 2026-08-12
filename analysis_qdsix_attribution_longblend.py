import io, json, contextlib
import numpy as np
import pandas as pd
import yfinance as yf

END='2026-07-01'; W=np.array([.13,.13,.13,.13,.33,.13,.02]); AQR=['ADAIX','QMNIX','QGMIX','QMHIX','AQRIX','QSPIX','BIL']
def dl(t,start='2008-01-01'):
 x=yf.download(t,start=start,end=END,auto_adjust=True,progress=False,threads=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
def mr(t):return dl(t).resample('ME').last().pct_change().dropna()
R={t:mr(t) for t in AQR}; H=pd.concat(R,axis=1).dropna(); H.columns=AQR
pseudo=pd.Series(H.values@W,index=H.index,name='qds_pseudo')
rf=H.BIL

def met(r):
 r=pd.Series(r).dropna(); yrs=(r.index[-1]-r.index[0]).days/365.25;eq=(1+r).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);v=float(r.std(ddof=1)*np.sqrt(12));rr=rf.reindex(r.index).ffill().bfill();ex=r-rr;sh=float(ex.mean()*12/(ex.std(ddof=1)*np.sqrt(12)));dd=float((eq/eq.cummax()-1).min());return {'Years':yrs,'CAGR':c,'Vol':v,'SharpeExcess':sh,'MDDmonthly':dd,'Calmar':c/abs(dd),'N':len(r)}

def attrib(Z,label):
 # Cash has ~0 excess return but is included in covariance.
 rr=rf.reindex(Z.index).ffill().bfill(); excess=Z.copy()
 for c in AQR: excess[c]=Z[c]-rr
 mu=excess.mean().values*12; cov=Z.cov().values*12
 pret=float(W@mu); var=float(W@cov@W); vol=np.sqrt(var)
 mrc=cov@W; rc=W*mrc/var
 retc=W*mu
 div_ratio=float((W*np.sqrt(np.diag(cov))).sum()/vol)
 print('ATTRIB',label,json.dumps({'PortfolioExcessArithmetic':pret,'PortfolioVol':vol,'SharpeApprox':pret/vol,'DiversificationRatio':div_ratio,'ReturnContribution':dict(zip(AQR,map(float,retc))),'VarianceContributionPct':dict(zip(AQR,map(float,rc)))}))

attrib(H.loc['2014-11-01':'2019-12-31'],'PRE2020')
attrib(H.loc['2020-07-01':'2026-06-30'],'POST2020')
attrib(H,'FULL')

buf=io.StringIO()
with contextlib.redirect_stdout(buf):import analysis_production_summary_dca as p
# AQ_ACT provides a production-style path throughout 2014 onward.
u=(1+p.Z['AQ_ACT'].ret).resample('ME').prod()-1
Z=pd.concat({'user':u,'alt':pseudo},axis=1).dropna();Z=Z.loc[(Z.index>=pd.Timestamp('2014-11-30'))&(Z.index<=pd.Timestamp('2026-06-30'))]

def blend_table(data,label):
 print('BASE_CORR',label,float(data.user.corr(data.alt)),'USER',json.dumps(met(data.user)),'ALT',json.dumps(met(data.alt)))
 for f in [0,.05,.10,.15,.20,.25,.30,.40,.50]:
  r=(1-f)*data.user+f*data.alt
  print('BLEND',label,f,json.dumps(met(r)))
 # risk-scale each blend to baseline monthly vol using BIL as collateral/financing approximation:
 basev=data.user.std()*np.sqrt(12)
 for f in [.10,.20,.30,.40,.50]:
  raw=(1-f)*data.user+f*data.alt; lev=float(basev/(raw.std()*np.sqrt(12))); rr=rf.reindex(raw.index).ffill().bfill(); scaled=rr+lev*(raw-rr)
  print('RISK_SCALED',label,f,json.dumps({'LeverageOnBlend':lev,**met(scaled)}))
blend_table(Z,'FULL_2014_2026')
blend_table(Z.loc[:'2019-12-31'],'PRE2020_OOSLIKE')
blend_table(Z.loc['2020-07-01':],'POST2020')

# rolling 36m and 60m comparison for pseudo alt, user and 20% blend; report worst/median Sharpe and CAGR.
def roll_stats(data,n):
 out={}
 for name,r in {'user':data.user,'alt':data.alt,'blend20':.8*data.user+.2*data.alt}.items():
  vals=[]
  for i in range(n,len(r)+1):
   z=r.iloc[i-n:i];m=met(z);vals.append((m['CAGR'],m['SharpeExcess'],m['MDDmonthly']))
  a=np.array(vals);out[name]={'CAGR_p10':float(np.quantile(a[:,0],.1)),'CAGR_med':float(np.median(a[:,0])),'Sharpe_p10':float(np.quantile(a[:,1],.1)),'Sharpe_med':float(np.median(a[:,1])),'MDD_p10':float(np.quantile(a[:,2],.1))}
 print('ROLL',n,json.dumps(out))
roll_stats(Z,36);roll_stats(Z,60)
