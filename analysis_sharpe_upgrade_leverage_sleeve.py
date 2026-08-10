import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

def dl(t):
 x=yf.download(t,start='2008-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
TQQQ=dl('TQQQ');UPRO=dl('UPRO');FX=p.raw['FX'];SPY=p.raw['SPY']

def signal(px):
 r=px.pct_change();vol=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/vol).values,.20,1.);cur=0.;a=[]
 for z in te:
  if np.isnan(z):a.append(cur);continue
  if z<cur:cur=z
  elif z-cur>.15:cur=z
  a.append(cur)
 ma=SPY.reindex(px.index).ffill().rolling(200).mean();sp=SPY.reindex(px.index).ffill();g=np.array(a)*np.where((sp<ma).values,.5,1.);last=g[0];o=[]
 for z in g:
  if abs(z-last)>p.SIGNAL_DEADBAND:last=z
  o.append(last)
 return pd.Series(o,index=px.index+pd.Timedelta(days=1))

def sim(R,Q,hcap=.70):
 q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(hcap,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,dt in enumerate(R.index):
  wt=W[i];cc=0.
  if np.max(np.abs(wc-wt))>p.TRADE_BAND:
   to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25;z.attrs.update(turn=turn,ntrade=nt,years=yrs);return z

def compare(label,s,px,Q,hcap,unhedged=False):
 R=p.SETS[s].copy();idx=R.index;pp=px.reindex(idx).ffill();
 if unhedged:pp=pp*FX.reindex(idx).ffill()
 R['q']=pp.pct_change();R=R.dropna();bR=p.SETS[s].reindex(R.index).dropna();idx=R.index.intersection(bR.index);R=R.loc[idx];bR=bR.loc[idx];m=p.metric(sim(R,Q,hcap));b=p.metric(p.sim(bR));x={'Label':label,'Set':s,'Start':m['Start'],'Years':m['Years'],'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'dCAGR':m['CAGR']-b['CAGR'],'dSharpe':m['Sharpe']-b['Sharpe'],'dMDD':m['MDD']-b['MDD'],'dCalmar':m['Calmar']-b['Calmar']};print('ROW',json.dumps(x));return x
rows=[]
for name,px in [('TQQQ',TQQQ),('UPRO',UPRO)]:
 Q=signal(px)
 for uh in [False,True]:
  for h in [.70,.80,.90,1.0]:
   for s in ['RY','AQ','WT','DB']:
    rows.append(compare(f'{name}_{"U" if uh else "H"}_H{int(h*100)}',s,px,Q,h,uh))
df=pd.DataFrame(rows);print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST')
rob=df[df.Set.isin(['AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).iterrows():print('ROB',json.dumps(r.to_dict()))
