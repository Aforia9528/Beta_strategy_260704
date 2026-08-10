import json, numpy as np, pandas as pd, yfinance as yf

# Recompute H_CAP / signal-deadband candidates under 7bp per traded notional,
# while sweeping the actual rebalance BAND.
TARGET=.20; WIN=16; FLOOR=.20; CAP=1.; INC=.15; GATE_MULT=.5
COST=.0007
HGRID=[.60,.65,.675,.70,.725,.75,.775,.80,.825]
DGRID=[0,.0125,.025,.0375,.05,.075,.10]
BANDS=[0,.025,.05,.075,.10,.125,.15]
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL','SGOV':'SGOV','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX','DBMF':'DBMF'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()

def qsignal(dead):
    r=sig.q.pct_change(); vol=r.rolling(WIN).std()*np.sqrt(252)
    te=np.clip((TARGET/vol).values,FLOOR,CAP)
    cur=0.; a=[]
    for z in te:
        if np.isnan(z): a.append(cur); continue
        if z<cur: cur=z
        elif z-cur>INC: cur=z
        a.append(cur)
    ma=sig.spy.rolling(200).mean()
    g=np.array(a)*np.where((sig.spy<ma).values,GATE_MULT,1.)
    last=g[0]; out=[]
    for z in g:
        if abs(z-last)>dead: last=z
        out.append(last)
    return pd.Series(out,index=sig.index+pd.Timedelta(days=1))
Q={float(d):qsignal(float(d)) for d in DGRID}

def retframe(start,mf,cash,actual_gold=False):
    keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual_gold else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp(start)]
    ff=lambda k:raw[k].reindex(idx).ffill(); fx=ff('FX')
    P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GOLD_H') if actual_gold else ff('GLD'),'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna()
    return P.pct_change().dropna()
SETS={'RY':retframe('2007-05-31','RYMTX','BIL',False),'AQ':retframe('2010-10-04','AQMIX','BIL',False),'WT':retframe('2011-01-06','WTMF','BIL',False),'DB':retframe('2020-06-02','DBMF','SGOV',False),'AQ_ACT':retframe('2010-10-04','AQMIX','BIL',True),'DB_ACT':retframe('2020-06-02','DBMF','SGOV',True)}
CORE=['RY','AQ','WT','DB']; ACT=['AQ_ACT','DB_ACT']

def simulate(R,h,d,band,cost=COST):
    q=Q[float(d)].reindex(R.index,method='ffill').dropna(); R=R.loc[q.index]
    hb=np.minimum(h,np.maximum(0,1-q.values)); cash=np.maximum(0,1-q.values-hb)
    W=np.c_[q.values,.5*hb,.5*hb,cash]; A=R[['q','gold','mf','cash']].values
    wc=W[0].copy(); nav=1.; rows=[]; turn=0.; ntrade=0; paid=0.
    for i,dt in enumerate(R.index):
        wt=W[i]; cc=0.
        if np.max(np.abs(wc-wt))>band:
            to=float(np.sum(np.abs(wt-wc))); cc=cost*to; turn+=to; paid+=cc; ntrade+=1; wc=wt.copy()
        gr=float(wc@A[i]); net=(1-cc)*(1+gr)-1; nav*=1+net
        end=wc*(1+A[i]); wc=end/end.sum(); rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')
    yrs=(z.index[-1]-z.index[0]).days/365.25
    z.attrs.update(turn=turn,ntrade=ntrade,years=yrs,paid=paid)
    return z

def metric(z):
    r=z.ret; yrs=z.attrs['years']; ppy=len(r)/yrs; eq=z.nav
    c=float(eq.iloc[-1]**(1/yrs)-1); vol=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/vol); m=float((eq/eq.cummax()-1).min())
    return {'CAGR':c,'Sharpe':sh,'MDD':m,'Calmar':c/abs(m),'TradesYr':z.attrs['ntrade']/yrs,'TurnYr':z.attrs['turn']/yrs,'FeeApproxYr':COST*z.attrs['turn']/yrs}

# Full candidate sweep by band on core proxies.
print('BAND_SWEEP_CORE')
ALL={}
for band in BANDS:
    rows=[]
    for h in HGRID:
        for d in DGRID:
            ms=[metric(simulate(SETS[s],h,d,band)) for s in CORE]
            rows.append({'H':h,'D':d,'Band':band,**{f'avg_{k}':float(np.mean([m[k] for m in ms])) for k in ms[0]}})
    df=pd.DataFrame(rows)
    # balanced score: average percentile ranks of Sharpe and Calmar, with MDD as tiebreaker
    df['rS']=df.avg_Sharpe.rank(pct=True); df['rC']=df.avg_Calmar.rank(pct=True); df['rM']=df.avg_MDD.rank(pct=True)
    df['score']=.45*df.rS+.45*df.rC+.10*df.rM
    ALL[band]=df
    print('BAND',band)
    for col in ['score','avg_Sharpe','avg_Calmar','avg_CAGR']:
        best=df.loc[df[col].idxmax()]
        print(col,json.dumps({k:float(best[k]) for k in ['H','D','Band','avg_CAGR','avg_Sharpe','avg_MDD','avg_Calmar','avg_TradesYr','avg_TurnYr','avg_FeeApproxYr','score']}))

# Production candidates under all bands, including original and recommended.
CANDS=[(.6,.05),(.7,.05),(.725,.05),(.7,.025),(.725,.025),(.8,.05)]
print('\nPRODUCTION_CANDIDATES')
for h,d in CANDS:
    print('CAND',h,d)
    for band in BANDS:
        ms=[metric(simulate(SETS[s],h,d,band)) for s in CORE]
        print(json.dumps({'Band':band,**{f'avg_{k}':float(np.mean([m[k] for m in ms])) for k in ms[0]}}))

# Exact fee effect: compare zero cost vs 7bp for recommended and alternative at each band.
print('\nFEE_EFFECT')
for h,d in [(.7,.05),(.725,.05),(.7,.025)]:
    print('CAND',h,d)
    for band in BANDS:
        deltas=[]
        for s in CORE:
            a=metric(simulate(SETS[s],h,d,band,cost=0.0)); b=metric(simulate(SETS[s],h,d,band,cost=COST))
            deltas.append({'CAGR':b['CAGR']-a['CAGR'],'Sharpe':b['Sharpe']-a['Sharpe'],'MDD':b['MDD']-a['MDD'],'Calmar':b['Calmar']-a['Calmar']})
        print(json.dumps({'Band':band,**{f'd_{k}':float(np.mean([x[k] for x in deltas])) for k in deltas[0]}}))

# Actual gold subset rank check.
print('\nACTUAL_GOLD')
for band in BANDS:
    rows=[]
    for h,d in CANDS:
        ms=[metric(simulate(SETS[s],h,d,band)) for s in ACT]
        rows.append({'H':h,'D':d,'Band':band,**{f'avg_{k}':float(np.mean([m[k] for m in ms])) for k in ms[0]}})
    df=pd.DataFrame(rows); df['score']=.5*df.avg_Sharpe.rank(pct=True)+.5*df.avg_Calmar.rank(pct=True)
    print('BAND',band,json.dumps(df.sort_values('score',ascending=False).head(3).to_dict('records')))
