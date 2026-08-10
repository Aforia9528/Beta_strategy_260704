import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

EXTRA=['SSO','TLT','IEF','DBC','BTAL','CAOS','KMLM','CTA']
def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(k) for k in EXTRA}; fx=p.raw['FX']; spy=p.raw['SPY']; qld=p.raw['QLD']

def metrics(z): return p.metric(z)
def sim_custom(R,Q,hedge_weights):
    q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(p.H_CAP,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb)
    cols=['q']+list(hedge_weights)+['cash']; W=np.zeros((len(R),len(cols)));W[:,0]=q.values
    for j,(k,w) in enumerate(hedge_weights.items(),1): W[:,j]=w*hb
    W[:,-1]=cash
    A=R[cols].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
    for i,dt in enumerate(R.index):
        wt=W[i];cc=0.
        if np.max(np.abs(wc-wt))>p.TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
        gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25;z.attrs.update(turn=turn,ntrade=nt,years=yrs);return z

def Q_from_return(r):
    idx=r.index;px=(1+r.fillna(0)).cumprod();vol=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/vol).values,.20,1.);cur=0.;a=[]
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

def base_frame(setname,start=None):
    R=p.SETS[setname].copy()
    if start is not None:R=R.loc[R.index>=pd.Timestamp(start)]
    return R

def alt_series(ticker,idx,unhedged=True):
    s=raw[ticker].reindex(idx).ffill()
    if unhedged:s=s*fx.reindex(idx).ffill()
    return s.pct_change()

def trend_bond(ticker,idx,ma=200,unhedged=True):
    usd=raw[ticker].reindex(idx).ffill(); f=fx.reindex(idx).ffill(); cash=p.raw['BIL'].reindex(idx).ffill()*f
    risky=usd*f if unhedged else usd
    sig=(usd>usd.rolling(ma).mean()).shift(1).fillna(False)
    rr=risky.pct_change();cr=cash.pct_change();return rr.where(sig,cr)

def compare(label,setname,Rcand,Qcand,hw,baseQ=None):
    Rcand=Rcand.dropna();q0=p.Q if baseQ is None else baseQ
    bR=p.SETS[setname].reindex(Rcand.index).dropna();common=Rcand.index.intersection(bR.index);Rcand=Rcand.loc[common];bR=bR.loc[common]
    c=metrics(sim_custom(Rcand,Qcand,hw));b=metrics(p.sim(bR))
    out={'Label':label,'Set':setname,'Start':c['Start'],'Years':c['Years'],'CAGR':c['CAGR'],'Sharpe':c['Sharpe'],'MDD':c['MDD'],'Calmar':c['Calmar'],'dCAGR':c['CAGR']-b['CAGR'],'dSharpe':c['Sharpe']-b['Sharpe'],'dMDD':c['MDD']-b['MDD'],'dCalmar':c['Calmar']-b['Calmar']};print('ROW',json.dumps(out));return out

rows=[]
# 1) Equity sleeve blends QLD/SSO, recompute vol signal on the blended 2x sleeve.
idx=qld.index.intersection(raw['SSO'].index);qr=qld.reindex(idx).pct_change();sr=raw['SSO'].reindex(idx).pct_change()
for a in [.75,.50,.25,0.0]:
    br=a*qr+(1-a)*sr;Qb=Q_from_return(br);lbl=f'EQ_Q{int(a*100)}_S{int((1-a)*100)}'
    for s in ['RY','AQ','WT','DB']:
        R=p.SETS[s].copy();R['q']=br.reindex(R.index);rows.append(compare(lbl,s,R,Qb,{'gold':.5,'mf':.5}))
# 2) Alternative hedge assets. Evaluate US-listed actual unhedged ETF exposure.
for alt in ['TLT','IEF','DBC','BTAL','CAOS']:
  for w in [.10,.20,.30]:
    lbl=f'HEDGE_{alt}_{int(w*100)}'
    for s in ['RY','AQ','WT','DB']:
      R=p.SETS[s].copy();R[alt]=alt_series(alt,R.index,True);rows.append(compare(lbl,s,R,p.Q,{'gold':(1-w)/2,'mf':(1-w)/2,alt:w}))
# 3) Trend-filtered TLT/IEF, 150/200/250d MA, 10-30% of hedge bucket.
for alt in ['TLT','IEF']:
  for ma in [150,200,250]:
    for w in [.10,.20,.30]:
      lbl=f'TREND_{alt}_{ma}_{int(w*100)}'
      for s in ['RY','AQ','WT','DB']:
        R=p.SETS[s].copy();name='tb';R[name]=trend_bond(alt,R.index,ma,True);rows.append(compare(lbl,s,R,p.Q,{'gold':(1-w)/2,'mf':(1-w)/2,name:w}))
# 4) BTAL only when SPY gate is ON, otherwise cash for that sub-sleeve.
for w in [.10,.20,.30]:
  lbl=f'GATE_BTAL_{int(w*100)}'
  for s in ['RY','AQ','WT','DB']:
    R=p.SETS[s].copy();idx=R.index;bt=alt_series('BTAL',idx,True);cr=R['cash'];sp=spy.reindex(idx).ffill();on=(sp<sp.rolling(200).mean()).shift(1).fillna(False);R['bt']=bt.where(on,cr);rows.append(compare(lbl,s,R,p.Q,{'gold':(1-w)/2,'mf':(1-w)/2,'bt':w}))
# 5) actual MF alternatives over common live history; replace mf leg, keep FX conversion.
for mf in ['KMLM','CTA']:
  for s in ['DB']:
    R=p.SETS[s].copy();idx=R.index;R['mf']=alt_series(mf,idx,True);rows.append(compare(f'MF_{mf}',s,R,p.Q,{'gold':.5,'mf':.5}))
# 6) 50/50 DBMF + KMLM / CTA ensemble in mf leg.
for mf in ['KMLM','CTA']:
  s='DB';R=p.SETS[s].copy();idx=R.index;other=alt_series(mf,idx,True);R['mf']=.5*R['mf']+.5*other;rows.append(compare(f'MF_DBMF50_{mf}50',s,R,p.Q,{'gold':.5,'mf':.5}))

df=pd.DataFrame(rows)
print('\nAVG_BY_LABEL')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),Sharpe=('Sharpe','mean'),CAGR=('CAGR','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nTOP_ROBUST')
# require at least 3 long-set observations and rank by worst dSharpe then avg
rob=df[df.Set.isin(['RY','AQ','WT'])].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(30).iterrows():print('ROB',json.dumps(r.to_dict()))
