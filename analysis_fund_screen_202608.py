import json, math
import numpy as np
import pandas as pd
import yfinance as yf

END='2026-07-01'
TICKERS={
# mutual funds / alt funds
'QNZNX':'AQR Trend Total Return N','QDSIX':'AQR Diversifying Strategies I','QMNIX':'AQR Equity Market Neutral I','QSPIX':'AQR Style Premia Alternative I','QRPIX':'AQR Alternative Risk Premia I','AQMIX':'AQR Managed Futures I','AQRIX':'AQR Multi-Asset I','QLEIX':'AQR Long-Short Equity I','ADAIX':'AQR Diversified Arbitrage I','AUEIX':'AQR Large Cap Defensive I','BLNDX':'Standpoint Multi-Asset Inst','REMIX':'Standpoint Multi-Asset Investor','ASFYX':'AlphaSimplex Managed Futures Y','MBXIX':'Catalyst Millburn Hedge Strategy I','ABYIX':'Abbey Capital Futures Strategy I',
# US ETFs / exchange-traded implementations
'DBMF':'iMGP DBi Managed Futures ETF','KMLM':'KFA Mount Lucas Managed Futures ETF','WTMF':'WisdomTree Managed Futures Strategy ETF','QAI':'IQ Hedge Multi-Strategy Tracker ETF','BTAL':'AGF US Market Neutral Anti-Beta ETF','PHDG':'Invesco S&P 500 Downside Hedged ETF','RPAR':'RPAR Risk Parity ETF','SWAN':'Amplify BlackSwan Growth & Treasury Core ETF','TAIL':'Cambria Tail Risk ETF','CAOS':'Alpha Architect Tail Risk ETF','CTA':'Simplify Managed Futures Strategy ETF','RSST':'Return Stacked US Stocks & Managed Futures ETF','RSBT':'Return Stacked Bonds & Managed Futures ETF','RSSB':'Return Stacked Global Stocks & Bonds ETF','GDE':'WisdomTree Efficient Gold Plus Equity Strategy Fund'}

def dl(t):
    try:
        x=yf.download(t,start='2006-01-01',end=END,auto_adjust=True,progress=False,threads=False)
        if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
        return x['Close'].dropna().astype(float)
    except Exception:
        return pd.Series(dtype=float)

irx=dl('^IRX')/100.0

def monthly_rf(idx):
    if irx.empty:return pd.Series(0,index=idx)
    d=irx.reindex(pd.date_range(irx.index.min(),pd.Timestamp(END),freq='D')).ffill()
    m=(1+d.resample('ME').mean())**(1/12)-1
    return m.reindex(idx).ffill().bfill()

def metrics(px,start=None):
    if start is not None:px=px.loc[px.index>=pd.Timestamp(start)]
    m=px.resample('ME').last().pct_change().dropna()
    if len(m)<12:return None
    rf=monthly_rf(m.index);ex=m-rf
    yrs=(m.index[-1]-m.index[0]).days/365.25
    eq=(1+m).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);vol=float(m.std(ddof=1)*np.sqrt(12));sh0=float(m.mean()*12/vol);she=float(ex.mean()*12/(ex.std(ddof=1)*np.sqrt(12)));dd=float((eq/eq.cummax()-1).min());cal=float(c/abs(dd)) if dd<0 else np.nan
    return {'Years':yrs,'Start':str(m.index[0].date()),'End':str(m.index[-1].date()),'CAGR':c,'Vol':vol,'Sharpe0':sh0,'SharpeExcess':she,'MDD':dd,'Calmar':cal,'NMonths':len(m)}
rows=[]
for t,n in TICKERS.items():
    px=dl(t)
    if px.empty:
        print('FAIL',t,n);continue
    allm=metrics(px)
    m5=metrics(px,start=str((pd.Timestamp('2026-06-30')-pd.DateOffset(years=5)).date()))
    m10=metrics(px,start=str((pd.Timestamp('2026-06-30')-pd.DateOffset(years=10)).date()))
    rec={'Ticker':t,'Name':n,**allm}
    for prefix,m in [('Y5',m5),('Y10',m10)]:
        if m:
            for k in ['CAGR','Vol','SharpeExcess','MDD','Calmar']:rec[prefix+'_'+k]=m[k]
    rows.append(rec);print('ROW',json.dumps(rec))
df=pd.DataFrame(rows)
print('\nRANK_ALL_MIN5Y')
z=df[df.Years>=5].copy();z['Score']=z.SharpeExcess+0.25*z.Calmar
for _,r in z.sort_values(['SharpeExcess','Calmar'],ascending=False).iterrows():print('RANK',json.dumps(r.to_dict()))
print('\nRANK_10Y_ACTUAL')
z=df[df.Years>=9.5].copy()
for _,r in z.sort_values(['SharpeExcess','Calmar'],ascending=False).iterrows():print('RANK10',json.dumps(r.to_dict()))
df.to_csv('analysis_fund_screen_202608_result.csv',index=False)
