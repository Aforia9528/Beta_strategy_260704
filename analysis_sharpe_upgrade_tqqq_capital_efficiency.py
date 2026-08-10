import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

def dl(t):
 x=yf.download(t,start='2009-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
TQQQ=dl('TQQQ');FX=p.raw['FX'];SETS=['RY','AQ','WT','DB']

def sim(R,k=.6666667,unhedged=False,mode='FREE_TO_HEDGE'):
 idx=R.index;qo=p.Q.reindex(idx,method='ffill').ffill();qt=(k*qo).clip(0,1);hb0=np.minimum(p.H_CAP,np.maximum(0,1-qo.values));cash0=np.maximum(0,1-qo.values-hb0)
 if mode=='FREE_TO_HEDGE':
  hb=hb0+(qo.values-qt.values);cash=cash0
 elif mode=='STANDARD_HCAP':
  hb=np.minimum(p.H_CAP,np.maximum(0,1-qt.values));cash=np.maximum(0,1-qt.values-hb)
 elif mode=='FREE_TO_CASH':
  hb=hb0;cash=np.maximum(0,1-qt.values-hb)
 else:raise ValueError
 W=np.c_[qt.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,dt in enumerate(idx):
  wt=W[i];cc=0.
  if np.max(np.abs(wc-wt))>p.TRADE_BAND:
   to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');z.attrs.update(years=(idx[-1]-idx[0]).days/365.25,turn=turn,ntrade=nt);return z

def metret(z):
 r=z.ret;yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs;eq=(1+r).cumprod();c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;dd=(eq/eq.cummax()-1).min();return {'CAGR':float(c),'Vol':float(v),'Sharpe':float(sh),'MDD':float(dd),'Calmar':float(c/abs(dd)),'Years':yrs}
def compare(label,s,z):
 b=p.Z[s];idx=z.index.intersection(b.index);z=z.loc[idx];b=b.loc[idx];m=metret(z);bm=metret(b);x={'Label':label,'Set':s,'Years':m['Years'],**m,'dCAGR':m['CAGR']-bm['CAGR'],'dVol':m['Vol']-bm['Vol'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']};print('ROW',json.dumps(x));return x
rows=[]
for uh in [False,True]:
 for k in [.60,.625,.6666667,.70,.75]:
  for mode in ['FREE_TO_HEDGE','STANDARD_HCAP','FREE_TO_CASH']:
   for s in SETS:
    R=p.SETS[s].copy();idx=R.index;px=TQQQ.reindex(idx).ffill();
    if uh:px=px*FX.reindex(idx).ffill()
    R['q']=px.pct_change();R=R.dropna();z=sim(R,k,uh,mode);rows.append(compare(f'TQQQ_{"U" if uh else "H"}_K{int(round(k*1000))}_{mode}',s,z))
df=pd.DataFrame(rows);print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dVol=('dVol','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST')
rob=df[df.Set.isin(['RY','AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(30).iterrows():print('ROB',json.dumps(r.to_dict()))
