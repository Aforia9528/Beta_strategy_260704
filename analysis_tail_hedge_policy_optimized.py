import json, math, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_production_summary_dca as p
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']
BUDGETS=[.0025,.00375,.005,.00625,.0075,.00875,.01]
MODES=['ALWAYS','QPROP','QGE50','QGE60','VXN30','Q50_VXN30','CHEAP_TIER']
def dl(t,adjust=True):
    x=yf.download(t,start='2006-01-01',auto_adjust=adjust,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
QQQ=dl('QQQ',False);VXN=dl('^VXN')/100.;IRX=dl('^IRX')/100.
def put(s,k,t,r,sg,q=.005):
    if t<=1e-8:return max(k-s,0.)
    sq=math.sqrt(t);d1=(math.log(s/k)+(r-q+.5*sg*sg)*t)/(sg*sq);d2=d1-sg*sq
    return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)
def spread(s,k1,k2,t,r,sg):return max(0.,put(s,k1,t,r,sg)-put(s,k2,t,r,sg))
def mu(mode,q,v):
    if mode=='ALWAYS':return 1.
    if mode=='QPROP':return min(1.5,max(0.,q/.60))
    if mode=='QGE50':return float(q>=.50)
    if mode=='QGE60':return float(q>=.60)
    if mode=='VXN30':return float(v<=.30)
    if mode=='Q50_VXN30':return float(q>=.50 and v<=.30)
    if mode=='CHEAP_TIER':return 1.5 if v<.20 else (1. if v<.30 else .5)

def prep(base):
    idx=base.index;n=len(idx);S=QQQ.reindex(idx).ffill().bfill();V=VXN.reindex(idx).ffill().bfill().clip(.08,1.5);RF=IRX.reindex(idx).ffill().bfill().clip(-.01,.10);Q=p.Q.reindex(idx,method='ffill').ffill().bfill().clip(0,1)
    months=idx.to_period('M');starts=np.where(np.r_[True,months[1:]!=months[:-1]])[0]
    tranches={}
    for j in starts:
        dt=idx[j];s0=float(S.iloc[j]);k1=.90*s0;k2=.70*s0;expiry=dt+pd.DateOffset(months=3);end=int(np.searchsorted(idx.values,np.datetime64(expiry),side='left'));end=min(end,n-1)
        vals=[]
        for i in range(j,end+1):
            s=float(S.iloc[i]);v=float(V.iloc[i]);r=float(RF.iloc[i]);sg=.75*v+.25*.20
            if i==end or idx[i]>=expiry: val=max(k1-s,0)-max(k2-s,0)
            else: val=spread(s,k1,k2,max((expiry-idx[i]).days/365.25,1/365.25),r,sg)
            vals.append(val)
        tranches[j]=(end,np.asarray(vals,float))
    return idx,S,V,Q,starts,tranches
PREP={s:prep(p.Z[s]) for s in SETS}
def overlay(s,budget,mode,load=1.25):
    base=p.Z[s];idx,S,V,Q,starts,tranches=PREP[s];n=len(idx);startset=set(starts.tolist());active=[];bc=1.;prev=1.;rows=[];spendfrac=0.;buys=0
    for i in range(n):
        bc*=1+float(base.ret.iloc[i]);ov=0.;newactive=[]
        for x in active:
            j,end,qty=x;rel=i-j
            if i>=end:
                bc+=qty*tranches[j][1][end-j]
            else:
                ov+=qty*tranches[j][1][rel];newactive.append(x)
        active=newactive;nav=bc+ov
        if i in startset:
            m=mu(mode,float(Q.iloc[i]),float(V.iloc[i]));spend=budget/12*m*nav
            if spend>0:
                end,vals=tranches[i];fair0=float(vals[0])
                if fair0>1e-12 and bc>spend:
                    qty=spend/(fair0*load);bc-=spend;ov+=qty*fair0;active.append((i,end,qty));nav=bc+ov;spendfrac+=spend/max(nav,1e-12);buys+=1
        rows.append((idx[i],nav/prev-1,nav));prev=nav
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs['SpendYr']=spendfrac/yrs;z.attrs['BuysYr']=buys/yrs;return z
def metric(z):
    r=z.ret;yrs=(z.index[-1]-z.index[0]).days/365.25;ppy=len(r)/yrs;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;vol=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/vol;dd=(eq/eq.cummax()-1).min();cal=c/abs(dd);w20=((1+r).rolling(20).apply(np.prod,raw=True)-1).min();q1=r.quantile(.01);cv=r[r<=q1].mean()
    return dict(CAGR=float(c),Sharpe=float(sh),MDD=float(dd),Calmar=float(cal),Worst20D=float(w20),CVaR=float(cv),SpendYr=float(z.attrs.get('SpendYr',0)),BuysYr=float(z.attrs.get('BuysYr',0)))
baseM={s:metric(p.Z[s]) for s in SETS};out=[]
for mode in MODES:
  for b in BUDGETS:
    rr=[]
    for s in SETS:
      m=metric(overlay(s,b,mode));mb=baseM[s];x={'Set':s,'Mode':mode,'Budget':b,**m,'dCAGR':m['CAGR']-mb['CAGR'],'dSharpe':m['Sharpe']-mb['Sharpe'],'dMDD':m['MDD']-mb['MDD'],'dCalmar':m['Calmar']-mb['Calmar'],'dWorst20D':m['Worst20D']-mb['Worst20D'],'dCVaR':m['CVaR']-mb['CVaR']};out.append(x);rr.append(x)
    av={k:float(np.mean([x[k] for x in rr])) for k in ['CAGR','Sharpe','MDD','Calmar','Worst20D','CVaR','SpendYr','BuysYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR']}
    print('AVG',json.dumps({'Mode':mode,'Budget':b,**av}))
df=pd.DataFrame(out);ag=df.groupby(['Mode','Budget']).mean(numeric_only=True).reset_index();ag['CalmarEff']=ag.dCalmar/(ag.SpendYr+1e-12);ag['MDDEff']=ag.dMDD/(ag.SpendYr+1e-12)
print('\nTOP_CALMAR');
for _,r in ag.sort_values('dCalmar',ascending=False).head(20).iterrows():print(json.dumps(r[['Mode','Budget','SpendYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR','CalmarEff','MDDEff']].to_dict()))
print('\nTOP_EFF');
for _,r in ag[ag.SpendYr>=.0015].sort_values('CalmarEff',ascending=False).head(20).iterrows():print(json.dumps(r[['Mode','Budget','SpendYr','dCAGR','dSharpe','dMDD','dCalmar','dWorst20D','dCVaR','CalmarEff','MDDEff']].to_dict()))
# set detail for shortlisted low-complexity policies
print('\nDETAIL_SHORTLIST')
for mode,b in [('ALWAYS',.005),('ALWAYS',.0075),('QPROP',.005),('QPROP',.0075),('QGE50',.0075),('VXN30',.0075),('Q50_VXN30',.01),('CHEAP_TIER',.005)]:
  for s in SETS:
    row=df[(df.Mode==mode)&(df.Budget==b)&(df.Set==s)].iloc[0]
    print(json.dumps(row[['Set','Mode','Budget','SpendYr','CAGR','Sharpe','MDD','Calmar','Worst20D','CVaR','dCAGR','dSharpe','dMDD','dCalmar']].to_dict()))
