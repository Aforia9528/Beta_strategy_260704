import json, math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm

# Locked production strategy
H_CAP=.70
SIGNAL_DEADBAND=.05
TRADE_BAND=.075
COST=.0007
BUDGETS=[0.0,.005,.0075,.01,.02,.03,.05]

T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','BIL':'BIL','RYMTX':'RYMTX','QQQ':'QQQ','VXN':'^VXN','IRX':'^IRX'}

def dl(t, adjust=True):
    x=yf.download(t,start='2006-01-01',auto_adjust=adjust,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)

raw={k:dl(v, adjust=(k!='QQQ')) for k,v in T.items()}
sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()

def qsignal():
    r=sig.q.pct_change(); vol=r.rolling(16).std()*np.sqrt(252); te=np.clip((.20/vol).values,.20,1.)
    cur=0.; a=[]
    for z in te:
        if np.isnan(z): a.append(cur); continue
        if z<cur: cur=z
        elif z-cur>.15: cur=z
        a.append(cur)
    ma=sig.spy.rolling(200).mean(); g=np.array(a)*np.where((sig.spy<ma).values,.5,1.)
    last=g[0]; o=[]
    for z in g:
        if abs(z-last)>SIGNAL_DEADBAND: last=z
        o.append(last)
    return pd.Series(o,index=sig.index+pd.Timedelta(days=1))
Q=qsignal()

def retframe():
    keys=['QLD','FX','GLD','RYMTX','BIL']
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp('2007-05-31')]
    ff=lambda k:raw[k].reindex(idx).ffill(); fx=ff('FX')
    P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GLD'),'mf':ff('RYMTX')*fx,'cash':ff('BIL')*fx},index=idx).dropna()
    return P.pct_change().dropna()
R=retframe()

def baseline(R):
    q=Q.reindex(R.index,method='ffill').dropna(); R=R.loc[q.index]
    hb=np.minimum(H_CAP,np.maximum(0,1-q.values)); cash=np.maximum(0,1-q.values-hb)
    W=np.c_[q.values,.5*hb,.5*hb,cash]; A=R[['q','gold','mf','cash']].values
    wc=W[0].copy(); nav=1.; rows=[]; turn=0.; nt=0
    for i,dt in enumerate(R.index):
        wt=W[i]; cc=0.
        if np.max(np.abs(wc-wt))>TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc))); cc=COST*to; turn+=to; nt+=1; wc=wt.copy()
        gr=float(wc@A[i]); net=(1-cc)*(1+gr)-1; nav*=1+net
        end=wc*(1+A[i]); wc=end/end.sum(); rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')
    return z
BASE=baseline(R)

# Option inputs.  VXN is 30d NDX implied vol. For 3m options we test both full VXN
# and a damped term-vol proxy. QQQ is used only as a price-scaled NDX proxy for moneyness.
idx=BASE.index
S=raw['QQQ'].reindex(idx).ffill().bfill()
VXN=(raw['VXN'].reindex(idx).ffill().bfill()/100.0).clip(.08,1.50)
IRX=(raw['IRX'].reindex(idx).ffill().bfill()/100.0).clip(-.01,.10)

def bs_put(s,k,t,r,sigma,q=.005):
    if t<=1e-8: return max(k-s,0.0)
    sigma=max(float(sigma),1e-6); sq=math.sqrt(t)
    d1=(math.log(s/k)+(r-q+.5*sigma*sigma)*t)/(sigma*sq); d2=d1-sigma*sq
    return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)

def spread_value(s,k1,k2,t,r,sigma):
    return max(0.0,bs_put(s,k1,t,r,sigma)-bs_put(s,k2,t,r,sigma))

def vol_for(vxn, mode):
    if mode=='FULL': return float(vxn)
    return float(.75*vxn+.25*.20)  # conservative 3m term damping during short vol spikes

# Each month after close, spend annual_budget/12 of NAV on a 3m 90%-70% put spread.
# The purchase is self-financed by reducing the baseline sleeve. load>1 means the buyer pays
# above model fair value (slippage/skew/model-error stress); mark-to-market remains fair value.
def overlay(annual_budget, mode='DAMP', load=1.0):
    if annual_budget==0:
        return BASE.copy()
    base_cap=1.0; tr=[]; nav_prev=1.0; rows=[]
    months=idx.to_period('M'); first=np.r_[True,months[1:]!=months[:-1]]
    for i,dt in enumerate(idx):
        # baseline sleeve earns baseline strategy return
        base_cap*=1+float(BASE.ret.iloc[i])
        s=float(S.iloc[i]); v=vol_for(float(VXN.iloc[i]),mode); r=float(IRX.iloc[i])
        # revalue / settle existing tranches
        live=[]; opt_val=0.0
        for x in tr:
            if dt>=x['expiry']:
                payoff=max(x['k1']-s,0.0)-max(x['k2']-s,0.0)
                base_cap += x['qty']*payoff
            else:
                tau=max((x['expiry']-dt).days/365.25,1/365.25)
                val=x['qty']*spread_value(s,x['k1'],x['k2'],tau,r,v)
                opt_val+=val; live.append(x)
        tr=live
        nav=base_cap+opt_val
        # buy new monthly tranche after close; it begins taking P&L next trading day
        if first[i]:
            spend=(annual_budget/12.0)*nav
            expiry=dt+pd.DateOffset(months=3)
            k1=.90*s; k2=.70*s; tau=max((expiry-dt).days/365.25,1/365.25)
            fair=spread_value(s,k1,k2,tau,r,v)
            paid_per_unit=fair*load
            if fair>1e-12 and spend>0 and base_cap>spend:
                qty=spend/paid_per_unit
                base_cap-=spend
                fair_val=qty*fair
                # immediate model/slippage loss when load>1
                opt_val+=fair_val
                tr.append({'expiry':expiry,'k1':k1,'k2':k2,'qty':qty})
                nav=base_cap+opt_val
        ret=nav/nav_prev-1; rows.append((dt,ret,nav)); nav_prev=nav
    return pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')

def max_underwater_days(eq):
    peak=eq.iloc[0]; start=None; mx=0
    for dt,x in eq.items():
        if x>=peak:
            peak=x
            if start is not None:
                mx=max(mx,(dt-start).days); start=None
        else:
            if start is None:start=dt
    if start is not None:mx=max(mx,(eq.index[-1]-start).days)
    return int(mx)

def metric(z):
    r=z.ret; yrs=(z.index[-1]-z.index[0]).days/365.25; ppy=len(r)/yrs; eq=z.nav
    c=float(eq.iloc[-1]**(1/yrs)-1); vol=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/vol)
    m=float((eq/eq.cummax()-1).min()); cal=c/abs(m); neg=r[r<0].std()*np.sqrt(ppy); so=float(r.mean()*ppy/neg)
    w1=float(r.min())
    w5=float(((1+r).rolling(5).apply(np.prod,raw=True)-1).min())
    w20=float(((1+r).rolling(20).apply(np.prod,raw=True)-1).min())
    return {'CAGR':c,'Vol':vol,'Sharpe':sh,'Sortino':so,'MDD':m,'Calmar':cal,'Worst1D':w1,'Worst5D':w5,'Worst20D':w20,'MaxUnderwaterDays':max_underwater_days(eq)}

def yearret(z,y):
    a=z.loc[z.index.year==y,'ret']
    return float((1+a).prod()-1) if len(a) else None

MODELS=[('FULL',1.0),('DAMP',1.0),('DAMP',1.25),('DAMP',1.50)]
print('PERIOD',str(idx[0].date()),str(idx[-1].date()),(idx[-1]-idx[0]).days/365.25)
print('STRUCTURE monthly 3m 90/70 QQQ put spread; VXN implied vol; self-financed; budgets are annual premium spend')
print('BASE',json.dumps(metric(BASE)))
for mode,load in MODELS:
    print('\nMODEL',mode,'LOAD',load)
    for b in BUDGETS:
        z=overlay(b,mode,load); m=metric(z)
        out={'Budget':b,**m,'Y2008':yearret(z,2008),'Y2020':yearret(z,2020),'Y2022':yearret(z,2022)}
        print(json.dumps(out))

# Marginal comparison vs baseline for the most decision-relevant damped model.
print('\nDELTA_VS_BASE_DAMP_LOAD1')
mb=metric(BASE)
for b in BUDGETS[1:]:
    m=metric(overlay(b,'DAMP',1.0))
    print(json.dumps({'Budget':b,**{k:m[k]-mb[k] for k in ['CAGR','Sharpe','MDD','Calmar','Worst1D','Worst5D','Worst20D']}}))

print('\nDELTA_VS_BASE_CONSERVATIVE_LOAD125')
for b in BUDGETS[1:]:
    m=metric(overlay(b,'DAMP',1.25))
    print(json.dumps({'Budget':b,**{k:m[k]-mb[k] for k in ['CAGR','Sharpe','MDD','Calmar','Worst1D','Worst5D','Worst20D']}}))
