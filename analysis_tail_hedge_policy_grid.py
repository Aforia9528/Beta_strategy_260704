import json, math, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_production_summary_dca as p

BUDGETS=[.0025,.00375,.005,.00625,.0075,.00875,.01]
MODES=['ALWAYS','QPROP','QGE50','QGE60','VXN30','Q50_VXN30','CHEAP_TIER']
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']

def dl(t,adjust=True):
    x=yf.download(t,start='2006-01-01',auto_adjust=adjust,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
QQQ=dl('QQQ',False);VXN=dl('^VXN')/100.;IRX=dl('^IRX')/100.

def put(s,k,t,r,sg,q=.005):
    if t<=1e-8:return max(k-s,0.0)
    sq=math.sqrt(t);d1=(math.log(s/k)+(r-q+.5*sg*sg)*t)/(sg*sq);d2=d1-sg*sq
    return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)
def sv(s,k1,k2,t,r,sg):return max(0.0,put(s,k1,t,r,sg)-put(s,k2,t,r,sg))

def mult(mode,q,v):
    if mode=='ALWAYS':return 1.0
    if mode=='QPROP':return min(1.5,max(0.0,q/.60))
    if mode=='QGE50':return 1.0 if q>=.50 else 0.0
    if mode=='QGE60':return 1.0 if q>=.60 else 0.0
    if mode=='VXN30':return 1.0 if v<=.30 else 0.0
    if mode=='Q50_VXN30':return 1.0 if (q>=.50 and v<=.30) else 0.0
    if mode=='CHEAP_TIER':
        if v<.20:return 1.50
        if v<.30:return 1.00
        return .50
    raise ValueError(mode)

def overlay(base,budget,mode,load=1.25):
    idx=base.index;S=QQQ.reindex(idx).ffill().bfill();V=VXN.reindex(idx).ffill().bfill().clip(.08,1.5);RF=IRX.reindex(idx).ffill().bfill().clip(-.01,.10)
    Q=p.Q.reindex(idx,method='ffill').ffill().bfill().clip(0,1)
    bc=1.;tr=[];prev=1.;rows=[];mons=idx.to_period('M');first=np.r_[True,mons[1:]!=mons[:-1]];sum_spend_frac=0.;buys=0
    for i,dt in enumerate(idx):
        bc*=1+float(base.ret.iloc[i]);s=float(S.iloc[i]);v=float(V.iloc[i]);r=float(RF.iloc[i]);q=float(Q.iloc[i]);sg=float(.75*v+.25*.20);live=[];ov=0.
        for x in tr:
            if dt>=x['e']:
                bc+=x['qty']*(max(x['k1']-s,0)-max(x['k2']-s,0))
            else:
                tau=max((x['e']-dt).days/365.25,1/365.25);ov+=x['qty']*sv(s,x['k1'],x['k2'],tau,r,sg);live.append(x)
        tr=live;nav=bc+ov
        if first[i]:
            mu=mult(mode,q,v);spend=budget/12*mu*nav
            if spend>0:
                e=dt+pd.DateOffset(months=3);k1=.90*s;k2=.70*s;tau=(e-dt).days/365.25;fair=sv(s,k1,k2,tau,r,sg)
                if fair>1e-12 and bc>spend:
                    qty=spend/(fair*load);bc-=spend;ov+=qty*fair;tr.append({'e':e,'k1':k1,'k2':k2,'qty':qty});nav=bc+ov;sum_spend_frac+=spend/max(nav,1e-12);buys+=1
        rows.append((dt,nav/prev-1,nav));prev=nav
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25
    z.attrs['spendYr']=sum_spend_frac/yrs;z.attrs['buysYr']=buys/yrs;return z

def max_underwater(eq):
    peak=eq.iloc[0];start=None;mx=0
    for dt,x in eq.items():
        if x>=peak:
            peak=x
            if start is not None:mx=max(mx,(dt-start).days);start=None
        elif start is None:start=dt
    if start is not None:mx=max(mx,(eq.index[-1]-start).days)
    return int(mx)

def metric(z):
    yrs=(z.index[-1]-z.index[0]).days/365.25;r=z.ret;ppy=len(r)/yrs;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;vol=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/vol;dd=(eq/eq.cummax()-1).min();cal=c/abs(dd)
    w5=((1+r).rolling(5).apply(np.prod,raw=True)-1).min();w20=((1+r).rolling(20).apply(np.prod,raw=True)-1).min();q01=r.quantile(.01);cvar=float(r[r<=q01].mean())
    return {'CAGR':float(c),'Vol':float(vol),'Sharpe':float(sh),'MDD':float(dd),'Calmar':float(cal),'Worst5D':float(w5),'Worst20D':float(w20),'CVaR1D_1pct':cvar,'UnderwaterDays':max_underwater(eq),'SpendYr':float(z.attrs.get('spendYr',0)),'BuysYr':float(z.attrs.get('buysYr',0))}

baseM={s:metric(p.Z[s]) for s in SETS}
rows=[]
for mode in MODES:
    for b in BUDGETS:
        vals=[]
        for s in SETS:
            z=overlay(p.Z[s],b,mode);m=metric(z);mb=baseM[s]
            rec={'Set':s,'Mode':mode,'Budget':b,**m,'dCAGR':m['CAGR']-mb['CAGR'],'dSharpe':m['Sharpe']-mb['Sharpe'],'dMDD':m['MDD']-mb['MDD'],'dCalmar':m['Calmar']-mb['Calmar'],'dWorst20D':m['Worst20D']-mb['Worst20D'],'dCVaR':m['CVaR1D_1pct']-mb['CVaR1D_1pct']}
            rows.append(rec);vals.append(rec)
        av={k:float(np.mean([v[k] for v in vals])) for k in ['CAGR','Sharpe','MDD','Calmar','Worst20D','CVaR1D_1pct','SpendYr','BuysYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR']}
        print('AVG',json.dumps({'Mode':mode,'Budget':b,**av}))

# Efficiency table: improvement per 1% actual annual premium spent.
df=pd.DataFrame(rows)
print('\nTOP_BY_CALMAR')
ag=df.groupby(['Mode','Budget']).mean(numeric_only=True).reset_index()
ag['CalmarEff']=ag.dCalmar/(ag.SpendYr+1e-12);ag['MDDEff']=ag.dMDD/(ag.SpendYr+1e-12);ag['Tail20Eff']=ag.dWorst20D/(ag.SpendYr+1e-12)
for _,r in ag.sort_values(['dCalmar','dMDD'],ascending=False).head(15).iterrows():print(json.dumps(r[['Mode','Budget','SpendYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR','CalmarEff','MDDEff']].to_dict()))
print('\nTOP_EFFICIENCY_MIN_SPEND_0.002')
for _,r in ag[ag.SpendYr>=.002].sort_values('CalmarEff',ascending=False).head(15).iterrows():print(json.dumps(r[['Mode','Budget','SpendYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR','CalmarEff','MDDEff']].to_dict()))
