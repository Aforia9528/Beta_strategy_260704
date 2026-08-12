import contextlib, io, json
import numpy as np
import pandas as pd
import yfinance as yf

buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_production_summary_dca as p

START=pd.Timestamp('2024-08-19')
END=pd.Timestamp('2026-06-30')
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']

# 3m T-bill yield for a monthly excess-return Sharpe comparable in spirit to AQR's published Sharpe.
def dl(t):
    x=yf.download(t,start='2024-07-01',end='2026-07-05',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
IRX=dl('^IRX')/100.0

# Also pull the AQR fund share class that currently publishes 1.64 Sharpe, as a sanity check only.
try:
    QNZRX=dl('QNZRX')
except Exception:
    QNZRX=pd.Series(dtype=float)

def period_monthly_returns(daily):
    r=daily.loc[(daily.index>=START)&(daily.index<=END)].dropna()
    return (1+r).resample('ME').prod()-1

def monthly_rf(index):
    # Use month-average quoted annual T-bill yield converted to effective monthly return.
    y=IRX.reindex(pd.date_range('2024-07-01','2026-06-30',freq='D')).ffill().resample('ME').mean()
    m=(1+y)**(1/12)-1
    return m.reindex(index).ffill().bfill()

def stats_from_monthly(m):
    m=m.dropna();rf=monthly_rf(m.index);ex=m-rf
    yrs=(m.index[-1]-m.index[0]).days/365.25
    eq=(1+m).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1)
    vol=float(m.std(ddof=1)*np.sqrt(12))
    sharpe0=float(m.mean()*12/vol)
    sh_ex=float(ex.mean()*12/(ex.std(ddof=1)*np.sqrt(12)))
    sh_ex_totalvol=float(ex.mean()*12/vol)
    mdd=float((eq/eq.cummax()-1).min())
    return {'NMonths':len(m),'Start':str(m.index[0].date()),'End':str(m.index[-1].date()),'CAGR':c,'AnnArithmetic':float(m.mean()*12),'Vol':vol,'Sharpe0':sharpe0,'SharpeExcess':sh_ex,'SharpeExcess_TotalVolDenom':sh_ex_totalvol,'AvgRFAnnApprox':float(rf.mean()*12),'MDD':mdd,'Calmar':c/abs(mdd) if mdd<0 else np.nan}

rows=[]
for s in SETS:
    m=period_monthly_returns(p.Z[s].ret)
    st=stats_from_monthly(m)
    st['Set']=s
    rows.append(st)
    print('STRAT',json.dumps(st))

# Core averages, and actual-implementation emphasis.
df=pd.DataFrame(rows)
for grp,names in [('CORE4',['RY','AQ','WT','DB']),('ACTUAL2',['AQ_ACT','DB_ACT'])]:
    z=df[df.Set.isin(names)]
    print('AVG',grp,json.dumps({k:float(z[k].mean()) for k in ['CAGR','AnnArithmetic','Vol','Sharpe0','SharpeExcess','SharpeExcess_TotalVolDenom','AvgRFAnnApprox','MDD','Calmar']}))

# AQR share-class sanity check from yfinance adjusted NAV if available.
if len(QNZRX):
    q=QNZRX.loc[(QNZRX.index>=START)&(QNZRX.index<=END)]
    qm=q.resample('ME').last().pct_change().dropna()
    print('AQR_YF_SANITY',json.dumps(stats_from_monthly(qm)))

# Implied AQR excess-rate from official published stats: annualized return 25.27%, vol 12.45%, Sharpe 1.64.
print('AQR_OFFICIAL',json.dumps({'PeriodStart':'2024-08-19','AsOf':'2026-06-30','AnnReturn':.2527,'Vol':.1245,'Sharpe':1.64,'ImpliedExcessReturn':1.64*.1245,'ImpliedRfFromArithmeticApprox':.2527-1.64*.1245}))
