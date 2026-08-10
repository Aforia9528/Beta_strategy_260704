import numpy as np
import pandas as pd
import yfinance as yf

# ===== exact live-strategy config =====
TARGET, WIN, FLOOR, CAP, INC = 0.20, 16, 0.20, 1.0, 0.15
DEADBAND = 0.05
GOLD_FRAC, DBMF_FRAC, H_CAP = 0.50, 0.50, 0.60
BAND = 0.05
GATE_MA, GATE_MULT = 200, 0.5
TCOST = 0.001  # conservative: 10 bp on each traded notional

T = {
    'QLD':'QLD', 'DBMF':'DBMF', 'SGOV':'SGOV', 'SPY':'SPY', 'FX':'KRW=X',
    'GLD':'GLD', 'GOLD_H':'132030.KS', 'ACE_SPOT':'411060.KS', 'TIGER_SPOT':'0072R0.KS'
}


def dl(t, start='2009-01-01'):
    x = yf.download(t, start=start, auto_adjust=True, progress=False)
    if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)

raw = {k: dl(v) for k,v in T.items()}

# ----- strategy target weights from actual US signal dates -----
us = pd.concat({'QLD':raw['QLD'], 'SPY':raw['SPY']}, axis=1).dropna()
rq = us.QLD.pct_change()
vol = rq.rolling(WIN).std() * np.sqrt(252)
te = np.clip((TARGET / vol).values, FLOOR, CAP)
cur = 0.0; asym=[]
for x in te:
    if np.isnan(x): asym.append(cur); continue
    if x < cur: cur = x
    elif x-cur > INC: cur = x
    asym.append(cur)
asym = np.array(asym)
ma = us.SPY.rolling(GATE_MA).mean()
gate = (us.SPY < ma).values
gated = asym * np.where(gate, GATE_MULT, 1.0)
last = gated[0]; w=[]
for g in gated:
    if abs(g-last) > DEADBAND: last = g
    w.append(last)
weights=[]
for q in w:
    hb=min(H_CAP,max(0.0,1-q))
    weights.append((q,GOLD_FRAC*hb,DBMF_FRAC*hb,max(0.0,1-q-hb)))
target = pd.DataFrame(weights,index=us.index,columns=['q','gold','dbmf','cash'])
# Signal from US close becomes tradable next calendar day in Korea / next US session.
target.index = target.index + pd.Timedelta(days=1)


def make_returns(gold_kind, start):
    # Global valuation calendar = union of relevant market dates.
    keys=['QLD','DBMF','SGOV','FX','GOLD_H']
    if gold_kind=='ACE': keys.append('ACE_SPOT')
    elif gold_kind=='TIGER': keys.append('TIGER_SPOT')
    elif gold_kind=='GLD_KRW': keys.append('GLD')
    idx = pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])))
    idx = idx[idx >= pd.Timestamp(start)]

    def ff(s): return s.reindex(idx).ffill()
    fx=ff(raw['FX'])
    # USD-listed assets translated to KRW terminal wealth.
    q=ff(raw['QLD'])*fx
    d=ff(raw['DBMF'])*fx
    c=ff(raw['SGOV'])*fx
    h=ff(raw['GOLD_H'])
    if gold_kind=='ACE': g=ff(raw['ACE_SPOT'])
    elif gold_kind=='TIGER': g=ff(raw['TIGER_SPOT'])
    else: g=ff(raw['GLD'])*fx
    prices=pd.DataFrame({'q':q,'gold':g,'dbmf':d,'cash':c,'gold_h':h},index=idx).dropna()
    return prices.pct_change().dropna(), prices


def sim(gold_kind, start, use_h=False, band=True, cost=TCOST):
    R,P = make_returns(gold_kind,start)
    # For H scenario replace gold-return column only, everything else identical.
    if use_h: R=R.copy(); R['gold']=R['gold_h']
    R=R[['q','gold','dbmf','cash']]
    tar=target.reindex(R.index,method='ffill').dropna()
    R=R.loc[tar.index]
    tar=tar.loc[R.index]
    nav=1.0
    wcur=tar.iloc[0].values.astype(float)
    out=[]; turns=[]
    for dt in R.index:
        wt=tar.loc[dt].values.astype(float)
        turnover=0.0
        if (not band) or np.max(np.abs(wcur-wt)) > BAND:
            turnover=float(np.sum(np.abs(wt-wcur)))
            nav *= max(0.0, 1 - cost*turnover)
            wcur=wt.copy()
        rr=R.loc[dt].values.astype(float)
        pr=float(np.dot(wcur,rr))
        nav *= (1+pr)
        # drift weights after returns
        gross=wcur*(1+rr)
        wcur=gross/gross.sum()
        out.append((dt,pr,nav)); turns.append(turnover)
    z=pd.DataFrame(out,columns=['date','ret','nav']).set_index('date')
    z['turnover']=turns
    return z,R,tar


def metrics(z):
    r=z.ret.dropna(); nav=z.nav.loc[r.index]
    yrs=(r.index[-1]-r.index[0]).days/365.25
    ppy=len(r)/yrs
    cagr=float(nav.iloc[-1]**(1/yrs)-1)
    vol=float(r.std()*np.sqrt(ppy))
    sharpe=float((r.mean()*ppy)/vol) if vol else np.nan
    neg=r[r<0].std()*np.sqrt(ppy)
    sortino=float((r.mean()*ppy)/neg) if neg and np.isfinite(neg) else np.nan
    mdd=float((nav/nav.cummax()-1).min())
    calmar=float(cagr/abs(mdd)) if mdd<0 else np.nan
    return {'start':str(r.index[0].date()),'end':str(r.index[-1].date()),'N':len(r),
            'CAGR':cagr,'Vol':vol,'Sharpe0':sharpe,'Sortino0':sortino,'MDD':mdd,'Calmar':calmar,
            'End':float(nav.iloc[-1]),'TurnoverGross':float(z.turnover.sum())}


def pair_report(label, spot_kind, start):
    zs,Rs,Tar = sim(spot_kind,start,use_h=False,band=True,cost=TCOST)
    zh,Rh,_ = sim(spot_kind,start,use_h=True,band=True,cost=TCOST)
    common=zs.index.intersection(zh.index); zs=zs.loc[common]; zh=zh.loc[common]
    print('\n===',label,'===')
    print('SPOT',metrics(zs))
    print('H',metrics(zh))
    yrs=(common[-1]-common[0]).days/365.25
    rel=float((zs.nav.iloc[-1]/zh.nav.iloc[-1])**(1/yrs)-1)
    print('ANNUALIZED_REL_SPOT_OVER_H',rel)
    # rolling 1y winner share using 252 global observations
    rr=pd.DataFrame({'s':zs.ret,'h':zh.ret}).dropna()
    rs=(1+rr.s).rolling(252).apply(np.prod,raw=True)-1
    rh=(1+rr.h).rolling(252).apply(np.prod,raw=True)-1
    m=(rs.notna()&rh.notna())
    print('ROLL252_SPOT_WIN_SHARE',float((rs[m]>rh[m]).mean()),'windows',int(m.sum()))
    # calendar-year returns
    yr=(1+rr).groupby(rr.index.year).prod()-1
    print('YEARLY_RETURNS')
    for y,row in yr.iterrows(): print(int(y),float(row.s),float(row.h),float(row.s-row.h))
    # gold sleeve diversification correlations in same valuation calendar
    _,Rbase,_=sim(spot_kind,start,use_h=False,band=False,cost=0)
    _,Rhbase,_=sim(spot_kind,start,use_h=True,band=False,cost=0)
    idx=Rbase.index.intersection(Rhbase.index)
    print('CORR_SPOT_Q',float(Rbase.loc[idx,'gold'].corr(Rbase.loc[idx,'q'])))
    print('CORR_H_Q',float(Rhbase.loc[idx,'gold'].corr(Rhbase.loc[idx,'q'])))
    print('CORR_SPOT_DBMF',float(Rbase.loc[idx,'gold'].corr(Rbase.loc[idx,'dbmf'])))
    print('CORR_H_DBMF',float(Rhbase.loc[idx,'gold'].corr(Rhbase.loc[idx,'dbmf'])))
    # Worst 5% QLD-KRW days: average gold return
    q=Rbase.loc[idx,'q']; cut=q.quantile(0.05); bad=q<=cut
    print('WORST5_Q_CUTOFF',float(cut),'SPOT_GOLD_AVG',float(Rbase.loc[idx,'gold'][bad].mean()),'H_GOLD_AVG',float(Rhbase.loc[idx,'gold'][bad].mean()),'N',int(bad.sum()))
    # no-band / no-cost sensitivity
    zs0,_,_=sim(spot_kind,start,use_h=False,band=False,cost=0)
    zh0,_,_=sim(spot_kind,start,use_h=True,band=False,cost=0)
    print('SENS_DAILY_TARGET_SPOT',metrics(zs0))
    print('SENS_DAILY_TARGET_H',metrics(zh0))

# 1) longest real same-index proxy available for TIGER: ACE KRX Gold Spot
pair_report('EXACT_INVESTABLE_ACE_PROXY_2021', 'ACE', '2021-12-16')
# 2) actual TIGER since inception
pair_report('ACTUAL_TIGER_2025', 'TIGER', '2025-06-25')
# 3) structural longer proxy: GLD in KRW vs actual KODEX H, with actual DBMF+SGOV
pair_report('EXTENDED_GLD_KRW_PROXY_2020', 'GLD_KRW', '2020-10-01')
