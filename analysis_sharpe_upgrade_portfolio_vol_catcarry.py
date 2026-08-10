import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p

EXTRA=['SHRIX','DBV','QUAL']
def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(k) for k in EXTRA};fx=p.raw['FX']
SETS=['RY','AQ','WT','DB']; LONG=['RY','AQ','WT']

def metric(z):
    r=z.ret;yrs=z.attrs['years'];ppy=len(r)/yrs;eq=z.nav;c=float(eq.iloc[-1]**(1/yrs)-1);v=float(r.std()*np.sqrt(ppy));sh=float(r.mean()*ppy/v);m=float((eq/eq.cummax()-1).min());cal=c/abs(m);neg=r[r<0].std()*np.sqrt(ppy);so=float(r.mean()*ppy/neg)
    return {'Start':str(z.index[0].date()),'End':str(z.index[-1].date()),'Years':yrs,'CAGR':c,'Vol':v,'Sharpe':sh,'Sortino':so,'MDD':m,'Calmar':cal,'TradesYr':z.attrs.get('ntrade',0)/yrs,'TurnYr':z.attrs.get('turn',0)/yrs}

def portfolio_vol_overlay(base,cash,win,target,floor=.25,band=.05,cost_mult=2.0):
    idx=base.index.intersection(cash.index);br=base.ret.reindex(idx);cr=cash.reindex(idx).fillna(0);vol=br.rolling(win).std()*np.sqrt(252);desired=(target/vol).clip(lower=floor,upper=1).shift(1).fillna(1)
    w=float(desired.iloc[0]);nav=1.;rows=[];turn=0.;nt=0
    for i,dt in enumerate(idx):
        wt=float(desired.iloc[i]);cc=0.
        if abs(w-wt)>band:
            d=abs(wt-w);cc=p.COST*cost_mult*d;turn+=cost_mult*d;nt+=1;w=wt
        ret=(1-cc)*(1+w*float(br.iloc[i])+(1-w)*float(cr.iloc[i]))-1;nav*=1+ret
        # drift weight after returns
        wb=w*(1+float(br.iloc[i]));cb=(1-w)*(1+float(cr.iloc[i]));den=wb+cb
        if den>0:w=wb/den
        rows.append((dt,ret,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs.update(years=yrs,turn=turn,ntrade=nt);return z

def sim_custom(R,weights,hcap=.70):
    q=p.Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(hcap,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);cols=['q']+list(weights)+['cash'];W=np.zeros((len(R),len(cols)));W[:,0]=q.values
    for j,(k,w) in enumerate(weights.items(),1):W[:,j]=w*hb
    W[:,-1]=cash;A=R[cols].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
    for i,dt in enumerate(R.index):
        wt=W[i];cc=0.
        if np.max(np.abs(wc-wt))>p.TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc)));cc=p.COST*to;turn+=to;nt+=1;wc=wt.copy()
        gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25;z.attrs.update(years=yrs,turn=turn,ntrade=nt);return z

def compare(label,s,z,base=None):
    if base is None: base=p.Z[s]
    idx=z.index.intersection(base.index);z=z.loc[idx];base=base.loc[idx].copy();base.attrs['years']=(idx[-1]-idx[0]).days/365.25;base.attrs['turn']=0;base.attrs['ntrade']=0
    m=metric(z);b=metric(base);x={'Label':label,'Set':s,'Start':m['Start'],'Years':m['Years'],'CAGR':m['CAGR'],'Vol':m['Vol'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'dCAGR':m['CAGR']-b['CAGR'],'dVol':m['Vol']-b['Vol'],'dSharpe':m['Sharpe']-b['Sharpe'],'dMDD':m['MDD']-b['MDD'],'dCalmar':m['Calmar']-b['Calmar'],'TradesYr':m['TradesYr'],'TurnYr':m['TurnYr']};print('ROW',json.dumps(x));return x
rows=[]
# 1. Portfolio-level de-risk-only vol overlay.
for win in [16,32,63]:
  for target in [.125,.15,.175,.20,.225]:
    for floor in [.25,.50]:
      for s in SETS:
        z=portfolio_vol_overlay(p.Z[s],p.SETS[s]['cash'],win,target,floor,.05,2.0);rows.append(compare(f'PVOL_W{win}_T{int(target*1000)}_F{int(floor*100)}',s,z))
# wider overlay trade band to see if turnover is the issue
for win,target,floor in [(16,.15,.25),(32,.15,.25),(63,.15,.25),(32,.175,.25),(63,.175,.25)]:
  for band in [.075,.10]:
    for s in SETS:
      z=portfolio_vol_overlay(p.Z[s],p.SETS[s]['cash'],win,target,floor,band,2.0);rows.append(compare(f'PVOLB_W{win}_T{int(target*1000)}_F{int(floor*100)}_B{int(band*1000)}',s,z))
# 2. Reinsurance, currency carry, quality as third hedge sleeve, unhedged USD/KRW exposure.
for alt in EXTRA:
  print('DATA',alt,str(raw[alt].index.min()),str(raw[alt].index.max()),len(raw[alt]))
  for w in [.05,.10,.15,.20,.30]:
    for s in SETS:
      R=p.SETS[s].copy();idx=R.index;px=raw[alt].reindex(idx).ffill()*fx.reindex(idx).ffill();R['alt']=px.pct_change();R=R.dropna();z=sim_custom(R,{'gold':(1-w)/2,'mf':(1-w)/2,'alt':w});rows.append(compare(f'ALT_{alt}_{int(w*100)}',s,z))
# 3. Reinsurance replaces Gold only or MF only, for interpretation.
for w in [.10,.20,.30]:
  for mode in ['REPL_GOLD','REPL_MF']:
    for s in SETS:
      R=p.SETS[s].copy();idx=R.index;px=raw['SHRIX'].reindex(idx).ffill()*fx.reindex(idx).ffill();R['alt']=px.pct_change();R=R.dropna()
      if mode=='REPL_GOLD': ww={'gold':.5-w,'mf':.5,'alt':w}
      else: ww={'gold':.5,'mf':.5-w,'alt':w}
      z=sim_custom(R,ww);rows.append(compare(f'SHRIX_{mode}_{int(w*100)}',s,z))

df=pd.DataFrame(rows)
print('\nAVG_BY_LABEL')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dVol=('dVol','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Vol=('Vol','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean'),TradesYr=('TradesYr','mean'),TurnYr=('TurnYr','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST_LONG')
rob=df[df.Set.isin(LONG)].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dVol=('dVol','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(50).iterrows():print('ROB',json.dumps(r.to_dict()))
# monthly autocorrelation diagnostics for alternative NAV series (raw USD and KRW irrelevant for smoothing)
print('\nAUTOCORR_MONTHLY')
for alt in EXTRA:
    m=raw[alt].resample('ME').last().pct_change().dropna();print(alt,json.dumps({'start':str(m.index.min()),'n':len(m),'ac1':float(m.autocorr(1)),'vol_ann':float(m.std()*np.sqrt(12)),'ret_ann_arith':float(m.mean()*12),'mdd':float(((1+m).cumprod()/((1+m).cumprod().cummax())-1).min())}))
