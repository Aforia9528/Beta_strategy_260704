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
AGG=dl('AGG')

def sim_dyn(R,hcap=.70,weight_func=None):
 q=p.Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(hcap,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb)
 if weight_func is None:
  gw=np.full(len(R),.5);mw=np.full(len(R),.5);rw=np.zeros(len(R))
 else: gw,mw,rw=weight_func(R)
 W=np.c_[q.values,gw*hb,mw*hb,rw*hb,cash];A=R[['q','gold','mf','rsbt','cash']].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,dt in enumerate(R.index):
  wt=W[i];cc=0.
  if np.max(np.abs(wc-wt))>p.TRADE_BAND:
   to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25;z.attrs.update(turn=turn,ntrade=nt,years=yrs);return z

def makeR(s):
 R=p.SETS[s].copy();idx=R.index;mf_usd_t,cash_t={'RY':('RYMTX','BIL'),'AQ':('AQMIX','BIL'),'WT':('WTMF','BIL'),'DB':('DBMF','SGOV')}[s]
 mf=p.raw[mf_usd_t].reindex(idx).ffill();cash=p.raw[cash_t].reindex(idx).ffill();agg=AGG.reindex(idx).ffill();fx=p.raw['FX'].reindex(idx).ffill()
 # synthetic RSBT: broad US bonds + managed-futures excess return over collateral - 1.02% fee, then KRW exposure
 rs_usd=agg.pct_change()+(mf.pct_change()-cash.pct_change())-.0102/252
 rs_price=(1+rs_usd.fillna(0)).cumprod()*fx
 R['rsbt']=rs_price.pct_change();return R.dropna()
RS={s:makeR(s) for s in ['RY','AQ','WT','DB']}

def cmp(label,s,z):
 idx=z.index;bR=p.SETS[s].reindex(idx).dropna();idx=idx.intersection(bR.index);m=p.metric(z.loc[idx] if False else z); # z already starts common
 b=p.metric(p.sim(bR));x={'Label':label,'Set':s,'Start':m['Start'],'Years':m['Years'],'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'dCAGR':m['CAGR']-b['CAGR'],'dSharpe':m['Sharpe']-b['Sharpe'],'dMDD':m['MDD']-b['MDD'],'dCalmar':m['Calmar']-b['Calmar']};print('ROW',json.dumps(x));return x
rows=[]
# HCAP sweep, baseline hedge split
for h in [.70,.80,.90,1.0]:
 for s,R in RS.items():
  def fbase(R):return np.full(len(R),.5),np.full(len(R),.5),np.zeros(len(R))
  rows.append(cmp(f'HCAP_{int(h*100)}',s,sim_dyn(R,h,fbase)))
# Replace part of MF half-sleeve with synthetic RSBT, preserving gold 50%.
# repl x => weights Gold=.5, MF=.5*(1-x), RSBT=.5*x. Gross economic defensive exposure increases via RSBT internal bond stack.
for x in [.25,.50,.75,1.0]:
 for h in [.70,.80,.90,1.0]:
  for s,R in RS.items():
   def f(R,x=x):return np.full(len(R),.5),np.full(len(R),.5*(1-x)),np.full(len(R),.5*x)
   rows.append(cmp(f'RSBT_REPL_{int(x*100)}_H{int(h*100)}',s,sim_dyn(R,h,f)))
# Inverse-vol Gold/MF weights. clamp gold to 25%-75% of hedge bucket.
for win in [63,126,252]:
 for h in [.70,.80,.90]:
  for s,R in RS.items():
   def f(R,win=win):
    vg=R.gold.rolling(win).std().shift(1);vm=R.mf.rolling(win).std().shift(1);wg=(1/vg)/((1/vg)+(1/vm));wg=wg.clip(.25,.75).fillna(.5);return wg.values,1-wg.values,np.zeros(len(R))
   rows.append(cmp(f'INVOL_{win}_H{int(h*100)}',s,sim_dyn(R,h,f)))
# Absolute trend filters on hedge assets; filtered portion goes to cash (implemented by reducing hb weights; residual lands nowhere so must add to portfolio cash return exposure).
# Here represent filtered share using rsbt column overwritten as cash-return proxy, so total hedge allocation sums to 1.
for ma in [150,200,250]:
 for mode in ['GOLD','BOTH']:
  for s,R0 in RS.items():
   R=R0.copy();R['rsbt']=R['cash']
   pxg=(1+R.gold.fillna(0)).cumprod();pxm=(1+R.mf.fillna(0)).cumprod();sg=(pxg>pxg.rolling(ma).mean()).shift(1).fillna(True);sm=(pxm>pxm.rolling(ma).mean()).shift(1).fillna(True)
   def f(R,sg=sg,sm=sm,mode=mode):
    g=.5*sg.astype(float);m=.5*(sm.astype(float) if mode=='BOTH' else 1.0);c=1-g-m;return g.values,np.asarray(m),np.asarray(c)
   rows.append(cmp(f'TREND_{mode}_{ma}',s,sim_dyn(R,.70,f)))

df=pd.DataFrame(rows);print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST_LONG')
rob=df[df.Set.isin(['RY','AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(30).iterrows():print('ROB',json.dumps(r.to_dict()))
