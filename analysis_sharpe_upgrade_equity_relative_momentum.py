import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
QQQ=dl('QQQ');SSO=dl('SSO');QLD=p.raw['QLD'];SPY=p.raw['SPY'];SETS=['RY','AQ','WT','DB']

def build_sleeve(lb,winner_weight):
 idx=QLD.index.intersection(SSO.index);q=QQQ.reindex(idx).ffill();s=SPY.reindex(idx).ffill();qld=QLD.reindex(idx).ffill();sso=SSO.reindex(idx).ffill();qm=q.pct_change(lb);sm=s.pct_change(lb)
 # determine winner only at month boundaries using prior close information
 months=idx.to_period('M');first=np.r_[True,months[1:]!=months[:-1]];wq=np.zeros(len(idx));cur=.5
 for i in range(len(idx)):
  if first[i] and i>0:
   j=i-1
   if pd.notna(qm.iloc[j]) and pd.notna(sm.iloc[j]):cur=winner_weight if qm.iloc[j]>=sm.iloc[j] else 1-winner_weight
  wq[i]=cur
 r_q=qld.pct_change().fillna(0);r_s=sso.pct_change().fillna(0);r=wq*r_q.values+(1-wq)*r_s.values
 return pd.Series(r,index=idx),pd.Series(wq,index=idx)

def make_signal(r):
 idx=r.index;vol=r.rolling(16).std()*np.sqrt(252);te=(.20/vol).clip(.20,1.0);cur=0.;a=[]
 for z in te:
  if pd.isna(z):a.append(cur);continue
  if z<cur:cur=float(z)
  elif z-cur>.15:cur=float(z)
  a.append(cur)
 sp=SPY.reindex(idx).ffill();ma=sp.rolling(200).mean();g=np.array(a)*np.where((sp<ma).values,.5,1.);last=g[0];out=[]
 for z in g:
  if abs(z-last)>.05:last=float(z)
  out.append(last)
 return pd.Series(out,index=idx+pd.Timedelta(days=1))

def sim(R,Q):
 q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(p.H_CAP,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values;wc=W[0].copy();nav=1.;rows=[]
 for i,dt in enumerate(R.index):
  wt=W[i];cc=0.
  if np.max(np.abs(wc-wt))>p.TRADE_BAND:
   to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;wc=wt.copy()
  net=(1-cc)*(1+float(wc@A[i]))-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');return z

def met(z):
 r=z.ret;yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs;eq=(1+r).cumprod();c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;dd=(eq/eq.cummax()-1).min();return {'Years':yrs,'CAGR':float(c),'Vol':float(v),'Sharpe':float(sh),'MDD':float(dd),'Calmar':float(c/abs(dd))}
def compare(label,s,z):
 b=p.Z[s];idx=z.index.intersection(b.index);m=met(z.loc[idx]);bm=met(b.loc[idx]);x={'Label':label,'Set':s,**m,'dCAGR':m['CAGR']-bm['CAGR'],'dVol':m['Vol']-bm['Vol'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']};print('ROW',json.dumps(x));return x
rows=[]
for lb in [63,126,252]:
 for ww in [1.0,.75]:
  r,w=build_sleeve(lb,ww);Q=make_signal(r);print('SELECTION',lb,ww,'avgQQQ',float(w.mean()),'switches',int((w.diff().abs()>0).sum()))
  for s in SETS:
   R=p.SETS[s].copy();R['q']=r.reindex(R.index);R=R.dropna();rows.append(compare(f'RELMOM_L{lb}_W{int(ww*100)}',s,sim(R,Q)))
df=pd.DataFrame(rows);print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dVol=('dVol','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST')
rob=df[df.Set.isin(['RY','AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).iterrows():print('ROB',json.dumps(r.to_dict()))
