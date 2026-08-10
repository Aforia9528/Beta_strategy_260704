import json, math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

H_CAP=.70; SIGNAL_DEADBAND=.05; TRADE_BAND=.075; COST=.0007
BUDGETS=[0,.005,.0075,.01,.02,.03,.05]
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','BIL':'BIL','RYMTX':'RYMTX','QQQ':'QQQ','VXN':'^VXN','IRX':'^IRX'}
def dl(t,adjust=True):
    x=yf.download(t,start='2006-01-01',auto_adjust=adjust,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v,k!='QQQ') for k,v in T.items()}
sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()
r=sig.q.pct_change();vol=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/vol).values,.20,1.);cur=0.;a=[]
for z in te:
    if np.isnan(z):a.append(cur);continue
    if z<cur:cur=z
    elif z-cur>.15:cur=z
    a.append(cur)
ma=sig.spy.rolling(200).mean();g=np.array(a)*np.where((sig.spy<ma).values,.5,1.);last=g[0];o=[]
for z in g:
    if abs(z-last)>.05:last=z
    o.append(last)
Q=pd.Series(o,index=sig.index+pd.Timedelta(days=1))
keys=['QLD','FX','GLD','RYMTX','BIL'];idx0=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])));idx0=idx0[idx0>=pd.Timestamp('2007-05-31')]
ff=lambda k:raw[k].reindex(idx0).ffill();fx=ff('FX');P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GLD'),'mf':ff('RYMTX')*fx,'cash':ff('BIL')*fx},index=idx0).dropna();R=P.pct_change().dropna()
q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(.70,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values
wc=W[0].copy();nav=1.;rows=[]
for i,dt in enumerate(R.index):
    wt=W[i];cc=0
    if np.max(np.abs(wc-wt))>.075:
        to=float(np.sum(np.abs(wt-wc)));cc=.0007*to;wc=wt.copy()
    gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net;end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
BASE=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');idx=BASE.index
S=raw['QQQ'].reindex(idx).ffill().bfill();V=(raw['VXN'].reindex(idx).ffill().bfill()/100).clip(.08,1.5);RF=(raw['IRX'].reindex(idx).ffill().bfill()/100).clip(-.01,.10)
def put(s,k,t,r,sg,q=.005):
    if t<=1e-8:return max(k-s,0)
    sq=math.sqrt(t);d1=(math.log(s/k)+(r-q+.5*sg*sg)*t)/(sg*sq);d2=d1-sg*sq
    return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)
def sv(s,k1,k2,t,r,sg):return max(0,put(s,k1,t,r,sg)-put(s,k2,t,r,sg))
def alpha(months):return {1:1.0,3:.75,6:.55}[months]
def sigma(v,months):
    a=alpha(months);return max(.08,a*v+(1-a)*.20)
def overlay(b,months,k1m,k2m,load=1.25):
    if b==0:return BASE
    bc=1.;tr=[];prev=1.;rows=[];mons=idx.to_period('M');first=np.r_[True,mons[1:]!=mons[:-1]]
    for i,dt in enumerate(idx):
        bc*=1+float(BASE.ret.iloc[i]);s=float(S.iloc[i]);r=float(RF.iloc[i]);sg=sigma(float(V.iloc[i]),months);live=[];ov=0.
        for x in tr:
            if dt>=x['e']:
                bc+=x['qty']*(max(x['k1']-s,0)-max(x['k2']-s,0))
            else:
                tau=max((x['e']-dt).days/365.25,1/365.25);val=x['qty']*sv(s,x['k1'],x['k2'],tau,r,sg);ov+=val;live.append(x)
        tr=live;nav=bc+ov
        if first[i]:
            spend=(b/12)*nav;e=dt+pd.DateOffset(months=months);k1=k1m*s;k2=k2m*s;tau=(e-dt).days/365.25;fair=sv(s,k1,k2,tau,r,sg)
            if fair>1e-12 and bc>spend:
                qty=spend/(fair*load);bc-=spend;ov+=qty*fair;tr.append({'e':e,'k1':k1,'k2':k2,'qty':qty});nav=bc+ov
        rows.append((dt,nav/prev-1,nav));prev=nav
    return pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')
def metrics(z):
    yrs=(z.index[-1]-z.index[0]).days/365.25;r=z.ret;ppy=len(r)/yrs;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;m=(eq/eq.cummax()-1).min();cal=c/abs(m);w20=((1+r).rolling(20).apply(np.prod,raw=True)-1).min()
    return {'CAGR':float(c),'Sharpe':float(sh),'MDD':float(m),'Calmar':float(cal),'Worst20D':float(w20)}
mb=metrics(BASE);print('BASE',json.dumps(mb))
structures=[('1M_95_80',1,.95,.80),('3M_95_75',3,.95,.75),('3M_90_70',3,.90,.70),('6M_90_70',6,.90,.70)]
for name,mo,k1,k2 in structures:
    print('\nSTRUCT',name,'LOAD 1.25')
    for b in BUDGETS:
        m=metrics(overlay(b,mo,k1,k2,1.25));print(json.dumps({'Budget':b,**m,'dCAGR':m['CAGR']-mb['CAGR'],'dSharpe':m['Sharpe']-mb['Sharpe'],'dMDD':m['MDD']-mb['MDD'],'dCalmar':m['Calmar']-mb['Calmar'],'dWorst20D':m['Worst20D']-mb['Worst20D']}))
