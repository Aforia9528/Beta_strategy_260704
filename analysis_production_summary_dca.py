import json, math
import numpy as np
import pandas as pd
import yfinance as yf

H_CAP=.70
SIGNAL_DEADBAND=.05
TRADE_BAND=.075
COST=.0007
MONTHLY=2_000_000.0
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL','SGOV':'SGOV','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX','DBMF':'DBMF'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()

def qsignal():
    r=sig.q.pct_change();vol=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/vol).values,.20,1.)
    cur=0.;a=[]
    for z in te:
        if np.isnan(z):a.append(cur);continue
        if z<cur:cur=z
        elif z-cur>.15:cur=z
        a.append(cur)
    ma=sig.spy.rolling(200).mean();g=np.array(a)*np.where((sig.spy<ma).values,.5,1.)
    last=g[0];o=[]
    for z in g:
        if abs(z-last)>SIGNAL_DEADBAND:last=z
        o.append(last)
    return pd.Series(o,index=sig.index+pd.Timedelta(days=1))
Q=qsignal()

def retframe(start,mf,cash,actual_gold=False):
    keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual_gold else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(start)]
    ff=lambda k:raw[k].reindex(idx).ffill();fx=ff('FX')
    P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GOLD_H') if actual_gold else ff('GLD'),'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna()
    return P.pct_change().dropna()
SETS={'RY':retframe('2007-05-31','RYMTX','BIL',False),'AQ':retframe('2010-10-04','AQMIX','BIL',False),'WT':retframe('2011-01-06','WTMF','BIL',False),'DB':retframe('2020-06-02','DBMF','SGOV',False),'AQ_ACT':retframe('2010-10-04','AQMIX','BIL',True),'DB_ACT':retframe('2020-06-02','DBMF','SGOV',True)}
CORE=['RY','AQ','WT','DB'];ACT=['AQ_ACT','DB_ACT']

def sim(R):
    q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(H_CAP,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb)
    W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values
    wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
    for i,dt in enumerate(R.index):
        wt=W[i];cc=0.
        if np.max(np.abs(wc-wt))>TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc)));cc=COST*to;turn+=to;nt+=1;wc=wt.copy()
        gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net
        end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25
    z.attrs.update(turn=turn,ntrade=nt,years=yrs);return z

def metric(z):
    r=z.ret;yrs=z.attrs['years'];ppy=len(r)/yrs;eq=z.nav
    c=float(eq.iloc[-1]**(1/yrs)-1);v=float(r.std()*np.sqrt(ppy));sh=float(r.mean()*ppy/v);m=float((eq/eq.cummax()-1).min());cal=c/abs(m)
    neg=r[r<0].std()*np.sqrt(ppy);so=float(r.mean()*ppy/neg)
    return {'Start':str(z.index[0].date()),'End':str(z.index[-1].date()),'Years':yrs,'CAGR':c,'Sharpe':sh,'Sortino':so,'MDD':m,'Calmar':cal,'TradesYr':z.attrs['ntrade']/yrs,'TurnYr':z.attrs['turn']/yrs,'FeeApproxYr':COST*z.attrs['turn']/yrs}

def xnpv(rate,dates,cfs):
    d0=dates[0]
    return sum(cf/((1+rate)**(((d-d0).days)/365.25)) for d,cf in zip(dates,cfs))
def xirr(dates,cfs):
    lo=-0.9999;hi=10.0
    flo=xnpv(lo,dates,cfs);fhi=xnpv(hi,dates,cfs)
    # expand hi if needed
    while flo*fhi>0 and hi<1e6:
        hi*=2;fhi=xnpv(hi,dates,cfs)
    if flo*fhi>0:return float('nan')
    for _ in range(250):
        mid=(lo+hi)/2;fm=xnpv(mid,dates,cfs)
        if abs(fm)<1e-8:break
        if flo*fm<=0:hi=mid;fhi=fm
        else:lo=mid;flo=fm
    return (lo+hi)/2

def dca(z,monthly=MONTHLY,beginning=True):
    # monthly contribution on the first trading day of each calendar month.
    r=z.ret.copy();months=r.index.to_period('M');first=np.r_[True,months[1:]!=months[:-1]]
    bal=0.;principal=0.;cf_dates=[];cfs=[];contrib_count=0
    for i,(dt,ret) in enumerate(r.items()):
        if first[i] and beginning:
            bal+=monthly;principal+=monthly;cf_dates.append(dt);cfs.append(-monthly);contrib_count+=1
        bal*=1+ret
        if first[i] and not beginning:
            bal+=monthly;principal+=monthly;cf_dates.append(dt);cfs.append(-monthly);contrib_count+=1
    cf_dates.append(r.index[-1]);cfs.append(bal)
    irr=xirr(cf_dates,cfs)
    return {'Contributions':contrib_count,'Principal':principal,'EndingValue':bal,'Profit':bal-principal,'MultipleOnPrincipal':bal/principal,'XIRR':irr}

Z={s:sim(R) for s,R in SETS.items()};M={s:metric(z) for s,z in Z.items()}
print('METRICS')
for s in Z:print(s,json.dumps(M[s]))
print('CORE_AVG',json.dumps({k:float(np.mean([M[s][k] for s in CORE])) for k in ['CAGR','Sharpe','Sortino','MDD','Calmar','TradesYr','TurnYr','FeeApproxYr']}))
print('ACTUAL_AVG',json.dumps({k:float(np.mean([M[s][k] for s in ACT])) for k in ['CAGR','Sharpe','Sortino','MDD','Calmar','TradesYr','TurnYr','FeeApproxYr']}))
print('\nDCA_MONTHLY_2M')
for s in ['RY','AQ','AQ_ACT','DB','DB_ACT']:
    print(s,json.dumps(dca(Z[s])))

# Constant-return future value using the observed core-average CAGR and selected individual CAGR,
# monthly contributions at beginning of month. Effective monthly rate derived from annual CAGR.
def fv_monthly(annual,years,pmt=MONTHLY):
    rm=(1+annual)**(1/12)-1;n=int(round(years*12))
    return pmt*(1+rm)*(((1+rm)**n-1)/rm) if abs(rm)>1e-12 else pmt*n
print('\nFV_EQUIVALENT')
for label,annual in [('CORE_AVG_CAGR',np.mean([M[s]['CAGR'] for s in CORE])),('RY_CAGR',M['RY']['CAGR']),('AQ_ACT_CAGR',M['AQ_ACT']['CAGR'])]:
    print(label,json.dumps({'annual':float(annual),**{f'Y{y}':float(fv_monthly(annual,y)) for y in [10,20,30,40]},'principal':{f'Y{y}':MONTHLY*12*y for y in [10,20,30,40]}}))
