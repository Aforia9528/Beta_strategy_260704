import json, math, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
# suppress the imported summary script's prints; use its locked production baseline paths
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_production_summary_dca as p

B=[0,.005,.0075,.01,.02,.03,.05]
def dl(t,adjust=True):
    x=yf.download(t,start='2006-01-01',auto_adjust=adjust,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
QQQ=dl('QQQ',False);VXN=dl('^VXN')/100.;IRX=dl('^IRX')/100.
def put(s,k,t,r,sg,q=.005):
    if t<=1e-8:return max(k-s,0)
    sq=math.sqrt(t);d1=(math.log(s/k)+(r-q+.5*sg*sg)*t)/(sg*sq);d2=d1-sg*sq
    return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)
def sv(s,k1,k2,t,r,sg):return max(0,put(s,k1,t,r,sg)-put(s,k2,t,r,sg))
def overlay(base,b,load=1.25):
    if b==0:return base.copy()
    idx=base.index;S=QQQ.reindex(idx).ffill().bfill();V=VXN.reindex(idx).ffill().bfill().clip(.08,1.5);RF=IRX.reindex(idx).ffill().bfill().clip(-.01,.10)
    bc=1.;tr=[];prev=1.;rows=[];mons=idx.to_period('M');first=np.r_[True,mons[1:]!=mons[:-1]]
    for i,dt in enumerate(idx):
        bc*=1+float(base.ret.iloc[i]);s=float(S.iloc[i]);r=float(RF.iloc[i]);sg=float(.75*V.iloc[i]+.25*.20);live=[];ov=0.
        for x in tr:
            if dt>=x['e']:bc+=x['q']*(max(x['k1']-s,0)-max(x['k2']-s,0))
            else:
                tau=max((x['e']-dt).days/365.25,1/365.25);ov+=x['q']*sv(s,x['k1'],x['k2'],tau,r,sg);live.append(x)
        tr=live;nav=bc+ov
        if first[i]:
            spend=b/12*nav;e=dt+pd.DateOffset(months=3);k1=.90*s;k2=.70*s;tau=(e-dt).days/365.25;fair=sv(s,k1,k2,tau,r,sg)
            if fair>1e-12 and bc>spend:
                q=spend/(fair*load);bc-=spend;ov+=q*fair;tr.append({'e':e,'k1':k1,'k2':k2,'q':q});nav=bc+ov
        rows.append((dt,nav/prev-1,nav));prev=nav
    return pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')
def m(z):
    yrs=(z.index[-1]-z.index[0]).days/365.25;r=z.ret;ppy=len(r)/yrs;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;dd=(eq/eq.cummax()-1).min();cal=c/abs(dd);w20=((1+r).rolling(20).apply(np.prod,raw=True)-1).min()
    return {'CAGR':float(c),'Sharpe':float(sh),'MDD':float(dd),'Calmar':float(cal),'Worst20D':float(w20)}
sets=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']
for s in sets:
    base=p.Z[s];mb=m(base);print('\nSET',s,'BASE',json.dumps(mb))
    for b in B[1:]:
        mm=m(overlay(base,b));print(json.dumps({'Budget':b,**mm,'dCAGR':mm['CAGR']-mb['CAGR'],'dSharpe':mm['Sharpe']-mb['Sharpe'],'dMDD':mm['MDD']-mb['MDD'],'dCalmar':mm['Calmar']-mb['Calmar'],'dWorst20D':mm['Worst20D']-mb['Worst20D']}))
