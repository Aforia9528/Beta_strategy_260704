import numpy as np
import pandas as pd
import yfinance as yf

TIGER='0072R0.KS'
KODEX='132030.KS'
ACE='411060.KS'
USD_KRW='KRW=X'
GLD='GLD'


def dl(t,start='2020-01-01'):
    x=yf.download(t,start=start,auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna()

series={t:dl(t) for t in [TIGER,KODEX,ACE,USD_KRW,GLD]}
px=pd.concat(series,axis=1).sort_index()


def stats(r, periods=252):
    r=r.dropna(); eq=(1+r).cumprod(); years=len(r)/periods
    return {
        'start':str(r.index[0].date()),'end':str(r.index[-1].date()),'N':len(r),
        'CAGR':float(eq.iloc[-1]**(1/years)-1),
        'Vol':float(r.std()*np.sqrt(periods)),
        'MDD':float((eq/eq.cummax()-1).min()),
        'End':float(eq.iloc[-1])
    }

# Exact actual-product common period, daily KRW returns
exact=px[[TIGER,KODEX]].dropna()
print('EXACT_COMMON')
for c in exact.columns:
    print(c,stats(exact[c].pct_change().dropna()))
print('EXACT_CORR',float(exact.pct_change().dropna().corr().iloc[0,1]))

# Weekly correlation only (close-time/holiday noise lower); metrics annualized correctly at 52.
wk=exact.resample('W-FRI').last().dropna()
print('EXACT_WEEKLY')
for c in wk.columns:
    print(c,stats(wk[c].pct_change().dropna(),52))
print('EXACT_WEEKLY_CORR',float(wk.pct_change().dropna().corr().iloc[0,1]))

# Extended proxy: ACE tracks same KRX Gold Spot Index as TIGER, from 2021-12-15.
proxy=px[[ACE,KODEX]].dropna()
print('PROXY_COMMON')
for c in proxy.columns:
    print(c,stats(proxy[c].pct_change().dropna()))
print('PROXY_CORR',float(proxy.pct_change().dropna().corr().iloc[0,1]))

# Factor decomposition: GLD USD (spot-like) vs GLD translated to KRW vs KODEX H.
base=px[[GLD,USD_KRW,KODEX]].dropna()
r_gld_usd=base[GLD].pct_change()
r_fx=base[USD_KRW].pct_change()
r_gld_krw=(1+r_gld_usd)*(1+r_fx)-1
r_kodex=base[KODEX].pct_change()
print('FACTOR_COMMON')
print('GLD_USD',stats(r_gld_usd.dropna()))
print('GLD_KRW',stats(r_gld_krw.dropna()))
print('KODEX_H',stats(r_kodex.dropna()))

# Same decomposition over TIGER's actual history.
base2=px[[TIGER,GLD,USD_KRW,KODEX]].dropna()
r2_usd=base2[GLD].pct_change(); r2_fx=base2[USD_KRW].pct_change(); r2_krw=(1+r2_usd)*(1+r2_fx)-1
print('FACTOR_TIGER_WINDOW')
print('TIGER',stats(base2[TIGER].pct_change().dropna()))
print('GLD_USD',stats(r2_usd.dropna()))
print('GLD_KRW',stats(r2_krw.dropna()))
print('KODEX_H',stats(base2[KODEX].pct_change().dropna()))
