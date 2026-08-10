import io, json, math, re, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import requests
import yfinance as yf

START='1985-01-01'
H_CAP=.70
SIGNAL_DEADBAND=.05
TRADE_BAND=.075
COST=.0007
TARGET=.20
WIN=16
FLOOR=.20
CAP=1.00
INC=.15
GATE_MULT=.50
AQR_URL='https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Time-Series-Momentum-Factors-Monthly.xlsx'
STOOQ_GOLD='https://stooq.com/q/d/l/?s=xauusd&i=d'

def ydl(t,start=START):
    x=yf.download(t,start=start,auto_adjust=True,progress=False,threads=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)

def fred(series):
    u=f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    x=pd.read_csv(u)
    x.columns=['Date',series]
    x['Date']=pd.to_datetime(x.Date)
    x[series]=pd.to_numeric(x[series],errors='coerce')
    return x.set_index('Date')[series].dropna().astype(float)

def metrics_ret(r):
    r=pd.Series(r).dropna()
    if len(r)<3:return None
    yrs=(r.index[-1]-r.index[0]).days/365.25
    ppy=len(r)/yrs
    eq=(1+r).cumprod()
    c=float(eq.iloc[-1]**(1/yrs)-1)
    v=float(r.std()*np.sqrt(ppy))
    sh=float(r.mean()*ppy/v) if v>0 else np.nan
    dd=eq/eq.cummax()-1
    m=float(dd.min())
    neg=r[r<0].std()*np.sqrt(ppy)
    so=float(r.mean()*ppy/neg) if neg and neg>0 else np.nan
    return {'Start':str(r.index[0].date()),'End':str(r.index[-1].date()),'Years':yrs,'CAGR':c,'Vol':v,'Sharpe0':sh,'Sortino0':so,'MDD':m,'Calmar':c/abs(m) if m<0 else np.nan,'Terminal':float(eq.iloc[-1])}

def submetric(r,a,b):
    z=r.loc[(r.index>=pd.Timestamp(a))&(r.index<=pd.Timestamp(b))]
    return metrics_ret(z) if len(z)>2 else None

def rolling_stats(r):
    m=(1+r).resample('ME').prod()-1
    out={}
    for y in [3,5,10]:
        n=12*y
        vals=[]
        for i in range(n,len(m)+1):
            z=m.iloc[i-n:i]
            yrs=(z.index[-1]-z.index[0]).days/365.25
            vals.append(float((1+z).prod()**(1/yrs)-1))
        out[f'WorstRolling{y}yCAGR']=float(np.min(vals)) if vals else np.nan
        out[f'MedianRolling{y}yCAGR']=float(np.median(vals)) if vals else np.nan
    return out

print('DOWNLOAD_MARKET_DATA')
ndx=ydl('^NDX')
xndx=ydl('^XNDX')
gspc=ydl('^GSPC')
spy=ydl('SPY','1993-01-01')
qld=ydl('QLD','2006-01-01')
proxies={k:ydl(k,'2006-01-01') for k in ['RYMTX','AQMIX','WTMF','DBMF']}
rf=fred('DTB3')
fx=fred('DEXKOUS')
print('DATA_STARTS',json.dumps({'NDX':str(ndx.index.min().date()),'XNDX':str(xndx.index.min().date()),'GSPC':str(gspc.index.min().date()),'SPY':str(spy.index.min().date()),'QLD':str(qld.index.min().date()),'DTB3':str(rf.index.min().date()),'DEXKOUS':str(fx.index.min().date())}))

# Build Nasdaq-100 total-return history. If XNDX starts later than NDX, estimate the total-return uplift vs price index on overlap and backfill NDX.
common=xndx.index.intersection(ndx.index)
rtr=xndx.pct_change().reindex(common);rpx=ndx.pct_change().reindex(common)
tr_uplift=float((rtr-rpx).dropna().mean()*252) if len(common)>100 else .006
print('XNDX_UPLIFT_ANNUAL_EST',tr_uplift)
idx=ndx.index.union(xndx.index).sort_values(); ndx_ff=ndx.reindex(idx).ffill(); x_ff=xndx.reindex(idx)
ndx_r=ndx_ff.pct_change(); synth_tr_r=ndx_r + tr_uplift/252
x_r=x_ff.pct_change(); total_r=x_r.combine_first(synth_tr_r)
total_px=(1+total_r.fillna(0)).cumprod(); total_px=total_px.loc[total_px.index>=pd.Timestamp(START)]

# Daily risk-free approximation from T-bill yield. This is a financing-state proxy, not exact bill total return.
rf_daily=(1+rf.reindex(total_px.index).ffill()/100.0)**(1/252)-1

# Calibrate constant non-rate drag so synthetic daily 2x matches live QLD as closely as possible.
idxc=qld.index.intersection(total_px.index)
base_2x=2*total_px.pct_change().reindex(idxc)-rf_daily.reindex(idxc)
qret=qld.pct_change().reindex(idxc)
res=(qret-base_2x).dropna()
drag_ann=float(-res.mean()*252)
byyear=(res.groupby(res.index.year).mean()*252).to_dict()
print('QLD_CALIBRATION',json.dumps({'constant_drag_ann':drag_ann,'annual_residual_by_year':{str(k):float(v) for k,v in byyear.items()},'resid_ann_std':float(pd.Series(list(byyear.values())).std())}))

def synth_2x(extra_drag=0.0):
    rr=2*total_px.pct_change()-rf_daily-(drag_ann+extra_drag)/252
    return rr.dropna()

syn=synth_2x(0)
cal=pd.concat({'actual':qret,'synthetic':syn.reindex(idxc)},axis=1).dropna()
calm=cal.resample('ME').apply(lambda x:(1+x).prod()-1)
print('QLD_CALIBRATION_QUALITY',json.dumps({'daily_corr':float(cal.actual.corr(cal.synthetic)),'daily_tracking_error_ann':float((cal.actual-cal.synthetic).std()*np.sqrt(252)),'monthly_corr':float(calm.actual.corr(calm.synthetic)),'actual':metrics_ret(cal.actual),'synthetic':metrics_ret(cal.synthetic)}))

# Gate proxy validation: GSPC vs SPY 200DMA over SPY live history.
gidx=spy.index.intersection(gspc.index)
spyg=spy.reindex(gidx);gpx=gspc.reindex(gidx)
g1=(spyg<spyg.rolling(200).mean());g2=(gpx<gpx.rolling(200).mean())
valid=g1.notna()&g2.notna();print('GATE_PROXY',json.dumps({'agreement':float((g1[valid]==g2[valid]).mean()),'mismatch_days':int((g1[valid]!=g2[valid]).sum()),'n':int(valid.sum())}))

# Gold spot from Stooq.
rg=requests.get(STOOQ_GOLD,headers={'User-Agent':'Mozilla/5.0'},timeout=45);rg.raise_for_status()
gold_df=pd.read_csv(io.StringIO(rg.text));gold_df.columns=[c.capitalize() for c in gold_df.columns];gold_df['Date']=pd.to_datetime(gold_df.Date);gold=gold_df.set_index('Date')['Close'].astype(float).sort_index();gold=gold.loc[gold.index>=pd.Timestamp(START)]
print('GOLD_DATA',json.dumps({'start':str(gold.index.min().date()),'end':str(gold.index.max().date()),'n':len(gold)}))

# Download and robustly parse AQR TSMOM workbook.
ra=requests.get(AQR_URL,headers={'User-Agent':'Mozilla/5.0'},timeout=60);ra.raise_for_status(); bio=io.BytesIO(ra.content); xl=pd.ExcelFile(bio)
print('AQR_SHEETS',json.dumps(xl.sheet_names))

def parse_aqr():
    candidates=[]
    for sh in xl.sheet_names:
        raw=pd.read_excel(xl,sh,header=None)
        # Try each row as header, looking for a Date-like column and enough numeric content.
        for h in range(min(25,len(raw))):
            cols=[str(x).strip() for x in raw.iloc[h].tolist()]
            dfi=raw.iloc[h+1:].copy();dfi.columns=cols
            datecol=None
            for c in dfi.columns:
                if 'date' in str(c).lower():datecol=c;break
            if datecol is None:continue
            dd=pd.to_datetime(dfi[datecol],errors='coerce')
            ok=dd.notna()
            if ok.sum()<100:continue
            nums={}
            for c in dfi.columns:
                if c==datecol:continue
                z=pd.to_numeric(dfi.loc[ok,c],errors='coerce')
                if z.notna().sum()>100: nums[c]=z
            if not nums:continue
            tmp=pd.DataFrame(nums,index=dd[ok]);tmp=tmp[~tmp.index.duplicated()].sort_index()
            candidates.append((sh,h,tmp))
    if not candidates:raise RuntimeError('Could not parse AQR workbook')
    # prefer longest data frame with a combined/TSMOM column if present
    candidates.sort(key=lambda x:len(x[2]),reverse=True)
    sh,h,df=candidates[0]
    cols=list(df.columns)
    comb=[c for c in cols if ('tsmom' in str(c).lower() or 'time series' in str(c).lower() or 'all'==str(c).strip().lower())]
    if comb: s=df[comb[0]].copy(); chosen=comb[0]
    else:
        # Avoid non-return columns if present; average the 4-ish return factor columns.
        retcols=[c for c in cols if not any(k in str(c).lower() for k in ['vol','sharpe','count','year'])]
        s=df[retcols].mean(axis=1);chosen='AVERAGE:'+','.join(map(str,retcols))
    s=pd.to_numeric(s,errors='coerce').dropna()
    # AQR spreadsheets usually express returns as decimals; detect percentages conservatively.
    scale=100.0 if s.abs().quantile(.95)>1.0 else 1.0
    s=s/scale
    return sh,h,chosen,s
ash,ah,acol,tsmom=parse_aqr();print('AQR_PARSE',json.dumps({'sheet':ash,'header_row':ah,'column':acol,'start':str(tsmom.index.min().date()),'end':str(tsmom.index.max().date()),'n':len(tsmom),'ann_mean':float(tsmom.mean()*12),'ann_vol':float(tsmom.std()*np.sqrt(12))}))

# Build daily q target from synthetic 2x and GSPC gate; signals act next trading day.
def qsignal(eqret):
    idx=eqret.index
    vol=eqret.rolling(WIN).std()*np.sqrt(252)
    te=(TARGET/vol).clip(FLOOR,CAP)
    cur=0.;asym=[]
    for z in te:
        if pd.isna(z):asym.append(cur);continue
        if z<cur:cur=float(z)
        elif z-cur>INC:cur=float(z)
        asym.append(cur)
    sp=gspc.reindex(idx).ffill();ma=sp.rolling(200).mean();g=np.array(asym)*np.where((sp<ma).values,GATE_MULT,1.0)
    last=float(g[0]) if np.isfinite(g[0]) else 0.;out=[]
    for z in g:
        if not np.isfinite(z):out.append(last);continue
        if abs(z-last)>SIGNAL_DEADBAND:last=float(z)
        out.append(last)
    return pd.Series(out,index=idx).shift(1).fillna(0)
Q=qsignal(syn)

# Exact daily equity-control diagnostic: q sleeve + hedged T-bill cash, preserving daily risk updates.
def daily_control(eqret,Q):
    idx=eqret.index.intersection(Q.index);r=eqret.reindex(idx);q=Q.reindex(idx);cr=rf_daily.reindex(idx).fillna(0)
    ret=q*r+(1-q)*cr
    return ret.dropna()
control=daily_control(syn,Q)
print('DAILY_CONTROL_FULL',json.dumps({**metrics_ret(control),**rolling_stats(control)}))
for name,a,b in [('PRE_DOTCOM','1990-01-01','1999-12-31'),('DOTCOM','2000-01-01','2002-12-31'),('LOST_DECADE','2000-01-01','2009-12-31'),('GFC','2007-01-01','2009-06-30'),('POST_GFC','2010-01-01','2019-12-31'),('COVID','2020-01-01','2020-12-31'),('INFLATION_2022','2022-01-01','2022-12-31')]:
    print('DAILY_CONTROL_REGIME',name,json.dumps(submetric(control,a,b)))

# Monthly building blocks. Gold is hedged USD gold return; cash and MF are USD/KRW exposed to preserve production convention.
def month_ret_from_daily(r):return (1+r).resample('ME').prod()-1
qm=month_ret_from_daily(syn)
gm=gold.resample('ME').last().pct_change()
fxm=fx.resample('ME').last().pct_change()
# Approximate monthly T-bill total return from month-average annual rate.
rfm=(1+rf.resample('ME').mean()/100.0)**(1/12)-1

# Live managed-futures proxy vol scaling vs AQR TSMOM excess factor.
scales={}
for k,s in proxies.items():
    mr=s.resample('ME').last().pct_change(); ix=mr.index.intersection(tsmom.index).intersection(rfm.index)
    if len(ix)>24:
        ex=(mr.reindex(ix)-rfm.reindex(ix)).dropna();tt=tsmom.reindex(ex.index).dropna();ex=ex.reindex(tt.index)
        scales[k]=float(ex.std()/tt.std()) if tt.std()>0 else np.nan
valid_sc=[v for v in scales.values() if np.isfinite(v) and .2<v<3]
med_scale=float(np.median(valid_sc)) if valid_sc else 1.0
print('MF_SCALE',json.dumps({'per_proxy':scales,'median':med_scale}))

# Month-start targets use only information available at prior month-end.
q_month=Q.resample('ME').last().shift(1)

def monthly_strategy(extra_financing=0.0,gold_drag=.0,mf_scale=None,mf_fee=.01,with_gate=True,hcap=H_CAP):
    er=synth_2x(extra_financing)
    qsig=qsignal(er)
    if not with_gate:
        # recompute same vol/asym/deadband without gate
        vol=er.rolling(WIN).std()*np.sqrt(252);te=(TARGET/vol).clip(FLOOR,CAP);cur=0.;aa=[]
        for z in te:
            if pd.isna(z):aa.append(cur);continue
            if z<cur:cur=float(z)
            elif z-cur>INC:cur=float(z)
            aa.append(cur)
        last=aa[0];oo=[]
        for z in aa:
            if abs(z-last)>SIGNAL_DEADBAND:last=z
            oo.append(last)
        qsig=pd.Series(oo,index=er.index).shift(1).fillna(0)
    qt=qsig.resample('ME').last().shift(1)
    qmr=month_ret_from_daily(er)
    gret=gm-gold_drag/12
    sc=med_scale if mf_scale is None else mf_scale
    # AQR TSMOM is excess return; add T-bill collateral, subtract implementation fee, then apply USD/KRW FX.
    mf_usd=rfm+sc*tsmom-mf_fee/12
    mf_krw=(1+mf_usd)*(1+fxm)-1
    cash_krw=(1+rfm)*(1+fxm)-1
    R=pd.concat({'q':qmr,'gold':gret,'mf':mf_krw,'cash':cash_krw,'target':qt},axis=1).dropna()
    wc=None;rets=[];turn=0.;ntrade=0
    for dt,row in R.iterrows():
        q=float(np.clip(row.target,0,1));hb=min(hcap,max(0,1-q));cash=max(0,1-q-hb);wt=np.array([q,.5*hb,.5*hb,cash],float)
        if wc is None:wc=wt.copy()
        if np.max(np.abs(wc-wt))>TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc)));turn+=to;ntrade+=1;cost=COST*to;wc=wt.copy()
        else:cost=0.
        a=np.array([row.q,row.gold,row.mf,row.cash],float);gross=float(wc@a);net=(1-cost)*(1+gross)-1;rets.append((dt,net))
        end=wc*(1+a);wc=end/end.sum()
    out=pd.Series(dict(rets)).sort_index();out.attrs={'turn':turn,'ntrade':ntrade}
    return out

variants={
 'BASE':dict(extra_financing=0,gold_drag=0,mf_scale=med_scale,mf_fee=.01,with_gate=True,hcap=.70),
 'NO_GATE':dict(extra_financing=0,gold_drag=0,mf_scale=med_scale,mf_fee=.01,with_gate=False,hcap=.70),
 'FIN_PLUS_100BP':dict(extra_financing=.01,gold_drag=0,mf_scale=med_scale,mf_fee=.01,with_gate=True,hcap=.70),
 'GOLD_DRAG_70BP':dict(extra_financing=0,gold_drag=.007,mf_scale=med_scale,mf_fee=.01,with_gate=True,hcap=.70),
 'MF_SCALE_75':dict(extra_financing=0,gold_drag=0,mf_scale=.75*med_scale,mf_fee=.01,with_gate=True,hcap=.70),
 'MF_SCALE_125':dict(extra_financing=0,gold_drag=0,mf_scale=1.25*med_scale,mf_fee=.01,with_gate=True,hcap=.70),
 'MF_FEE_150BP':dict(extra_financing=0,gold_drag=0,mf_scale=med_scale,mf_fee=.015,with_gate=True,hcap=.70),
 'H60':dict(extra_financing=0,gold_drag=0,mf_scale=med_scale,mf_fee=.01,with_gate=True,hcap=.60),
 'H80':dict(extra_financing=0,gold_drag=0,mf_scale=med_scale,mf_fee=.01,with_gate=True,hcap=.80),
}
OUT={}
for name,kw in variants.items():
    z=monthly_strategy(**kw);OUT[name]=z
    mm=metrics_ret(z);mm.update(rolling_stats(z));print('FULL30',name,json.dumps(mm))
    for rg,a,b in [('PRE_DOTCOM','1990-01-01','1999-12-31'),('DOTCOM','2000-01-01','2002-12-31'),('LOST_DECADE','2000-01-01','2009-12-31'),('GFC','2007-01-01','2009-06-30'),('POST_GFC','2010-01-01','2019-12-31'),('COVID','2020-01-01','2020-12-31'),('INFLATION_2022','2022-01-01','2022-12-31'),('RECENT','2020-01-01','2026-12-31')]:
        sm=submetric(z,a,b)
        if sm: print('REGIME',name,rg,json.dumps(sm))

# Benchmarks at monthly frequency over same available window.
bench_2x=month_ret_from_daily(syn);bench_1x=month_ret_from_daily(total_px.pct_change())
for nm,z in [('SYNTH_2X_BUYHOLD',bench_2x),('NASDAQ100_TR_1X',bench_1x)]:
    ix=z.index.intersection(OUT['BASE'].index);z=z.reindex(ix).dropna();print('BENCH',nm,json.dumps({**metrics_ret(z),**rolling_stats(z)}))
    for rg,a,b in [('DOTCOM','2000-01-01','2002-12-31'),('LOST_DECADE','2000-01-01','2009-12-31'),('GFC','2007-01-01','2009-06-30')]: print('BENCH_REGIME',nm,rg,json.dumps(submetric(z,a,b)))

# 10y block bootstrap on BASE monthly returns: 12-month blocks, 10k paths. Report terminal/CAGR and max DD distribution.
rng=np.random.default_rng(260811);base=OUT['BASE'].dropna().values;block=12;horizon=120;npaths=10000
cag=[];mdds=[]
for _ in range(npaths):
    arr=[]
    while len(arr)<horizon:
        j=int(rng.integers(0,max(1,len(base)-block+1)));arr.extend(base[j:j+block])
    r=np.array(arr[:horizon]);eq=np.cumprod(1+r);cag.append(eq[-1]**(1/10)-1);mdds.append(np.min(eq/np.maximum.accumulate(eq)-1))
print('BOOTSTRAP10Y',json.dumps({'N':npaths,'CAGR_p05':float(np.quantile(cag,.05)),'CAGR_p25':float(np.quantile(cag,.25)),'CAGR_med':float(np.median(cag)),'CAGR_p75':float(np.quantile(cag,.75)),'CAGR_p95':float(np.quantile(cag,.95)),'MDD_p05_worst':float(np.quantile(mdds,.05)),'MDD_med':float(np.median(mdds)),'MDD_p95_best':float(np.quantile(mdds,.95)),'P_CAGR_lt_0':float(np.mean(np.array(cag)<0)),'P_MDD_below_30':float(np.mean(np.array(mdds)<-.30))}))

# compact summary CSV committed by workflow
rows=[]
for name,z in OUT.items():
    m=metrics_ret(z);m.update(rolling_stats(z));m['Variant']=name;rows.append(m)
pd.DataFrame(rows).to_csv('analysis_30y_synthetic_dotcom_summary.csv',index=False)
