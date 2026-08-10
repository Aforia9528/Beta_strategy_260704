import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

EXTRA=['MNA','QAI','PHDG','TAIL','SWAN']
def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
raw={k:dl(k) for k in EXTRA};fx=p.raw['FX'];spy=p.raw['SPY'];qld=p.raw['QLD']

def sim_custom(R,Q,hw):
 q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(p.H_CAP,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);cols=['q']+list(hw)+['cash'];W=np.zeros((len(R),len(cols)));W[:,0]=q.values
 for j,(k,w) in enumerate(hw.items(),1):W[:,j]=w*hb
 W[:,-1]=cash;A=R[cols].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,dt in enumerate(R.index):
  wt=W[i];cc=0.
  if np.max(np.abs(wc-wt))>p.TRADE_BAND:
   to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25;z.attrs.update(turn=turn,ntrade=nt,years=yrs);return z

def Q_from_return(r):
 idx=r.index;vol=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/vol).values,.20,1.);cur=0.;a=[]
 for z in te:
  if np.isnan(z):a.append(cur);continue
  if z<cur:cur=z
  elif z-cur>.15:cur=z
  a.append(cur)
 sp=spy.reindex(idx).ffill();ma=sp.rolling(200).mean();g=np.array(a)*np.where((sp<ma).values,.5,1.);last=g[0];o=[]
 for z in g:
  if abs(z-last)>p.SIGNAL_DEADBAND:last=z
  o.append(last)
 return pd.Series(o,index=idx+pd.Timedelta(days=1))

def compare(label,s,R,Q,hw):
 R=R.dropna();bR=p.SETS[s].reindex(R.index).dropna();idx=R.index.intersection(bR.index);R=R.loc[idx];bR=bR.loc[idx];c=p.metric(sim_custom(R,Q,hw));b=p.metric(p.sim(bR));x={'Label':label,'Set':s,'Start':c['Start'],'Years':c['Years'],'CAGR':c['CAGR'],'Sharpe':c['Sharpe'],'MDD':c['MDD'],'Calmar':c['Calmar'],'dCAGR':c['CAGR']-b['CAGR'],'dSharpe':c['Sharpe']-b['Sharpe'],'dMDD':c['MDD']-b['MDD'],'dCalmar':c['Calmar']-b['Calmar']};print('ROW',json.dumps(x));return x
rows=[]
# RSST-style proxy: SPY total return + managed-futures excess return over USD T-bill, less 99bp annual fee.
# For each proxy set use its underlying USD MF fund and BIL/SGOV before FX conversion.
MAP={'RY':('RYMTX','BIL'),'AQ':('AQMIX','BIL'),'WT':('WTMF','BIL'),'DB':('DBMF','SGOV')}
for s,(mf,cash) in MAP.items():
 idx=p.SETS[s].index;spyr=spy.reindex(idx).ffill().pct_change();mfr=p.raw[mf].reindex(idx).ffill().pct_change();cr=p.raw[cash].reindex(idx).ffill().pct_change();stack_usd=spyr+(mfr-cr)-.0099/252
 # actual US-listed RSST-like exposure includes USD/KRW; structural hedged version isolates strategy.
 fxr=fx.reindex(idx).ffill().pct_change();stack_u=(1+stack_usd)*(1+fxr)-1
 qr=p.SETS[s]['q']
 for mode,stack in [('H',stack_usd),('U',stack_u)]:
  for w in [.10,.20,.30,.40,.50]:
   blend=(1-w)*qr+w*stack;Qb=Q_from_return(blend);R=p.SETS[s].copy();R['q']=blend;rows.append(compare(f'STACK_{mode}_{int(w*100)}',s,R,Qb,{'gold':.5,'mf':.5}))
# Long-live alternative ETF diversifiers, actual unhedged KRW exposure, replacing a slice of hedge bucket.
for alt in EXTRA:
 for w in [.10,.20,.30]:
  for s in ['RY','AQ','WT','DB']:
   R=p.SETS[s].copy();idx=R.index;px=raw[alt].reindex(idx).ffill()*fx.reindex(idx).ffill();R['alt']=px.pct_change();rows.append(compare(f'ALT_{alt}_{int(w*100)}',s,R,p.Q,{'gold':(1-w)/2,'mf':(1-w)/2,'alt':w}))
# MNA as third sleeve with weights optimized coarsely, not just equal residual.
for gw,mw,aw in [(0.45,.45,.10),(.40,.40,.20),(.35,.45,.20),(.45,.35,.20),(.35,.35,.30),(.30,.50,.20),(.50,.30,.20)]:
 for s in ['RY','AQ','WT','DB']:
  R=p.SETS[s].copy();idx=R.index;px=raw['MNA'].reindex(idx).ffill()*fx.reindex(idx).ffill();R['alt']=px.pct_change();rows.append(compare(f'MNA_G{int(gw*100)}_M{int(mw*100)}_A{int(aw*100)}',s,R,p.Q,{'gold':gw,'mf':mw,'alt':aw}))
df=pd.DataFrame(rows)
print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST_LONG')
rob=df[df.Set.isin(['RY','AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(30).iterrows():print('ROB',json.dumps(r.to_dict()))
