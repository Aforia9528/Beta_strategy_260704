import json, contextlib, io
import numpy as np
import pandas as pd
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_sharpe_upgrade_portfolio_vol_catcarry as a
p=a.p
SETS=['RY','AQ','WT','DB'];LONG=['RY','AQ','WT']

def metric_ret(z):
    r=z.ret.dropna();yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs;eq=(1+r).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);v=float(r.std()*np.sqrt(ppy));sh=float(r.mean()*ppy/v);m=float((eq/eq.cummax()-1).min());cal=c/abs(m)
    return {'Start':str(r.index[0].date()),'Years':yrs,'CAGR':c,'Vol':v,'Sharpe':sh,'MDD':m,'Calmar':cal,'TradesYr':z.attrs.get('ntrade',0)/yrs,'TurnYr':z.attrs.get('turn',0)/yrs}
def compare(label,s,z):
    base=p.Z[s];idx=z.index.intersection(base.index);z=z.loc[idx];b0=base.loc[idx].copy();m=metric_ret(z);b=metric_ret(b0);x={'Label':label,'Set':s,'Start':m['Start'],'Years':m['Years'],'CAGR':m['CAGR'],'Vol':m['Vol'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'dCAGR':m['CAGR']-b['CAGR'],'dVol':m['Vol']-b['Vol'],'dSharpe':m['Sharpe']-b['Sharpe'],'dMDD':m['MDD']-b['MDD'],'dCalmar':m['Calmar']-b['Calmar'],'TradesYr':m['TradesYr'],'TurnYr':m['TurnYr']};print('ROW',json.dumps(x));return x
rows=[]
# Re-run selected best portfolio-vol candidates plus no-op-ish targets for correct CAGR/Calmar.
for win,target,floor,band in [(63,.125,.5,.05),(63,.15,.25,.10),(63,.15,.25,.05),(63,.175,.25,.10),(63,.20,.25,.05),(63,.225,.25,.05),(32,.15,.25,.10)]:
    for s in SETS:
        z=a.portfolio_vol_overlay(p.Z[s],p.SETS[s]['cash'],win,target,floor,band,2.0);rows.append(compare(f'PVOL_W{win}_T{int(target*1000)}_F{int(floor*100)}_B{int(band*1000)}',s,z))
# Re-run alternative sleeves with correct common-period metrics.
for alt in a.EXTRA:
  print('DATA',alt,str(a.raw[alt].index.min()),str(a.raw[alt].index.max()),len(a.raw[alt]))
  for w in [.05,.10,.15,.20,.30]:
    for s in SETS:
      R=p.SETS[s].copy();idx=R.index;px=a.raw[alt].reindex(idx).ffill()*a.fx.reindex(idx).ffill();R['alt']=px.pct_change();R=R.dropna();z=a.sim_custom(R,{'gold':(1-w)/2,'mf':(1-w)/2,'alt':w});rows.append(compare(f'ALT_{alt}_{int(w*100)}',s,z))
for w in [.10,.20,.30]:
  for mode in ['REPL_GOLD','REPL_MF']:
    for s in SETS:
      R=p.SETS[s].copy();idx=R.index;px=a.raw['SHRIX'].reindex(idx).ffill()*a.fx.reindex(idx).ffill();R['alt']=px.pct_change();R=R.dropna();ww={'gold':.5-w,'mf':.5,'alt':w} if mode=='REPL_GOLD' else {'gold':.5,'mf':.5-w,'alt':w};z=a.sim_custom(R,ww);rows.append(compare(f'SHRIX_{mode}_{int(w*100)}',s,z))
df=pd.DataFrame(rows)
print('\nAVG')
ag=df.groupby('Label').agg(N=('Set','count'),Years=('Years','mean'),dCAGR=('dCAGR','mean'),dVol=('dVol','mean'),dSharpe=('dSharpe','mean'),dMDD=('dMDD','mean'),dCalmar=('dCalmar','mean'),CAGR=('CAGR','mean'),Vol=('Vol','mean'),Sharpe=('Sharpe','mean'),MDD=('MDD','mean'),Calmar=('Calmar','mean')).reset_index()
for _,r in ag.sort_values(['dSharpe','dCalmar'],ascending=False).iterrows():print('AVG',json.dumps(r.to_dict()))
print('\nROBUST')
rob=df[df.Set.isin(LONG)].groupby('Label').agg(min_dSharpe=('dSharpe','min'),avg_dSharpe=('dSharpe','mean'),min_dCalmar=('dCalmar','min'),avg_dCalmar=('dCalmar','mean'),avg_dCAGR=('dCAGR','mean'),avg_dMDD=('dMDD','mean')).reset_index()
for _,r in rob.sort_values(['min_dSharpe','avg_dSharpe'],ascending=False).head(30).iterrows():print('ROB',json.dumps(r.to_dict()))
print('\nAUTOCORR_MONTHLY')
for alt in a.EXTRA:
 m=a.raw[alt].resample('ME').last().pct_change().dropna();eq=(1+m).cumprod();print(alt,json.dumps({'n':len(m),'ac1':float(m.autocorr(1)),'vol_ann':float(m.std()*np.sqrt(12)),'arith_ret_ann':float(m.mean()*12),'mdd':float((eq/eq.cummax()-1).min())}))
