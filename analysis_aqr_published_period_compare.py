import contextlib, io, json
import numpy as np
import pandas as pd
import yfinance as yf
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p
START=pd.Timestamp('2021-12-16'); END=pd.Timestamp('2026-06-30')
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']
def dl(t):
 x=yf.download(t,start='2021-11-01',end='2026-07-05',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
IRX=dl('^IRX')/100.0

def monthly(daily):
 r=daily.loc[(daily.index>=START)&(daily.index<=END)].dropna();return (1+r).resample('ME').prod()-1

def rf_month(index):
 d=IRX.reindex(pd.date_range('2021-11-01','2026-06-30',freq='D')).ffill();y=d.resample('ME').mean();m=(1+y)**(1/12)-1;return m.reindex(index).ffill().bfill()
def stats(m):
 m=m.dropna();rf=rf_month(m.index);ex=m-rf;yrs=(m.index[-1]-m.index[0]).days/365.25;eq=(1+m).cumprod();c=eq.iloc[-1]**(1/yrs)-1;vol=m.std(ddof=1)*np.sqrt(12);sh0=m.mean()*12/vol;she=ex.mean()*12/(ex.std(ddof=1)*np.sqrt(12));dd=(eq/eq.cummax()-1).min();return {'NMonths':len(m),'Start':str(m.index[0].date()),'End':str(m.index[-1].date()),'CAGR':float(c),'AnnArithmetic':float(m.mean()*12),'Vol':float(vol),'Sharpe0':float(sh0),'SharpeExcess':float(she),'AvgRFAnnApprox':float(rf.mean()*12),'MDD':float(dd),'Calmar':float(c/abs(dd))}
rows=[]
for s in SETS:
 st=stats(monthly(p.Z[s].ret));st['Set']=s;rows.append(st);print('STRAT',json.dumps(st))
df=pd.DataFrame(rows)
for grp,names in [('CORE4',['RY','AQ','WT','DB']),('ACTUAL2',['AQ_ACT','DB_ACT'])]:
 z=df[df.Set.isin(names)];print('AVG',grp,json.dumps({k:float(z[k].mean()) for k in ['CAGR','AnnArithmetic','Vol','Sharpe0','SharpeExcess','AvgRFAnnApprox','MDD','Calmar']}))
print('AQR_PUBLISHED',json.dumps({'Inception':'2021-12-16','AsOf':'2026-06-30','PublishedSharpe':1.64,'PublishedVol':.1245,'CurrentStrategyChanged':'2024-08-19'}))
