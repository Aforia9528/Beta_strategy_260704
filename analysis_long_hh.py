import numpy as np, pandas as pd, yfinance as yf

TARGET,WIN,FLOOR,CAP,INC=0.20,16,0.20,1.0,0.15
DEADBAND=0.05; H_CAP=0.60; BAND=0.05; GATE_MA=200; GATE_MULT=0.5; TCOST=0.001
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL','SGOV':'SGOV','DBMF':'DBMF','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
print('DATA_STARTS',{k:str(v.index[0].date()) for k,v in raw.items()})

# signal path from QLD + SPY; independent of FX implementation
us=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()
rq=us.q.pct_change(); vol=rq.rolling(WIN).std()*np.sqrt(252); te=np.clip((TARGET/vol).values,FLOOR,CAP)
cur=0.; asym=[]
for x in te:
    if np.isnan(x): asym.append(cur); continue
    if x<cur: cur=x
    elif x-cur>INC: cur=x
    asym.append(cur)
ma=us.spy.rolling(GATE_MA).mean(); gated=np.array(asym)*np.where((us.spy<ma).values,GATE_MULT,1.0)
last=gated[0]; qw=[]
for g in gated:
    if abs(g-last)>DEADBAND:last=g
    qw.append(last)
qsignal=pd.Series(qw,index=us.index+pd.Timedelta(days=1),name='q')


def target_df(idx,gfrac=.5):
    q=qsignal.reindex(idx,method='ffill').dropna(); out=[]
    for x in q:
        hb=min(H_CAP,max(0,1-x)); out.append((x,gfrac*hb,(1-gfrac)*hb,max(0,1-x-hb)))
    return pd.DataFrame(out,index=q.index,columns=['q','gold','mf','cash'])

def price_frame(start,mfkey,gold_actual=False,qhedged=True,ghedged=True,cashkey='BIL'):
    keys=['QLD','FX','GLD',mfkey,cashkey]
    if gold_actual: keys+=['GOLD_H']
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp(start)]
    def ff(k): return raw[k].reindex(idx).ffill()
    fx=ff('FX')
    q=ff('QLD') if qhedged else ff('QLD')*fx
    if gold_actual: gh=ff('GOLD_H'); gu=ff('GLD')*fx; g=gh if ghedged else gu
    else: g=ff('GLD') if ghedged else ff('GLD')*fx
    mf=ff(mfkey)*fx; cash=ff(cashkey)*fx
    P=pd.DataFrame({'q':q,'gold':g,'mf':mf,'cash':cash},index=idx).dropna()
    return P

def sim(start,mfkey,gold_actual=False,qhedged=True,ghedged=True,gfrac=.5,cashkey='BIL',weekly=False):
    P=price_frame(start,mfkey,gold_actual,qhedged,ghedged,cashkey)
    if weekly: P=P.resample('W-FRI').last().dropna()
    R=P.pct_change().dropna(); W=target_df(R.index,gfrac); R=R.loc[W.index]
    nav=1.; wc=W.iloc[0].values.astype(float); rows=[]; turnover=0.
    for dt in R.index:
        wt=W.loc[dt].values.astype(float)
        if np.max(np.abs(wc-wt))>BAND:
            to=float(np.sum(np.abs(wt-wc))); nav*=max(0,1-TCOST*to); wc=wt.copy(); turnover+=to
        rr=R.loc[dt].values.astype(float); pr=float(wc@rr); nav*=1+pr
        gross=wc*(1+rr); wc=gross/gross.sum(); rows.append((dt,pr,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date'); z.attrs['turnover']=turnover; return z

def met(z):
    r=z.ret; yrs=(z.index[-1]-z.index[0]).days/365.25; ppy=len(r)/yrs
    c=float(z.nav.iloc[-1]**(1/yrs)-1); v=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/v)
    neg=r[r<0].std()*np.sqrt(ppy); so=float(r.mean()*ppy/neg); m=float((z.nav/z.nav.cummax()-1).min()); cal=float(c/abs(m))
    return {'start':str(z.index[0].date()),'end':str(z.index[-1].date()),'yrs':yrs,'CAGR':c,'Vol':v,'Sharpe0':sh,'Sortino0':so,'MDD':m,'Calmar':cal,'End':float(z.nav.iloc[-1]),'turn':z.attrs.get('turnover',0)}
def rolling5(z):
    # calendar 5y windows every ~quarter (63 obs for daily-union-ish not fixed); use dates exactly
    vals=[]
    starts=z.index[::63]
    for s in starts:
        e=s+pd.DateOffset(years=5); zz=z[(z.index>=s)&(z.index<=e)]
        if len(zz)<500: continue
        mm=met(zz); vals.append((mm['CAGR'],mm['Sharpe0'],mm['MDD'],mm['Calmar']))
    a=np.array(vals)
    if len(a)==0:return {}
    return {'n':len(a),'CAGR_med':float(np.median(a[:,0])),'CAGR_min':float(np.min(a[:,0])),'Sharpe_med':float(np.median(a[:,1])),'Sharpe_min':float(np.min(a[:,1])),'MDD_med':float(np.median(a[:,2])),'MDD_worst':float(np.min(a[:,2])),'Calmar_med':float(np.median(a[:,3])),'Calmar_min':float(np.min(a[:,3]))}

def report(label,start,mfkey,gold_actual=False,cashkey='BIL'):
    print('\n===',label,'===')
    for qh,gh,nm in [(1,1,'HH'),(1,0,'HU'),(0,1,'UH'),(0,0,'UU')]:
        z=sim(start,mfkey,gold_actual,bool(qh),bool(gh),.5,cashkey,False); print(nm,met(z)); print(nm+'_ROLL5',rolling5(z))
    print('GFRAC_SWEEP_HH')
    for gf in [0,.25,.5,.75,1.0]:
        z=sim(start,mfkey,gold_actual,True,True,gf,cashkey,False); print(gf,met(z))
    # weekly robustness for HH only
    print('HH_WEEKLY',met(sim(start,mfkey,gold_actual,True,True,.5,cashkey,True)))

# actual-ish: actual Korean hedged gold, systematic MF with consistent process from 2010
report('AQMIX_ACTUAL_GOLDH_2010','2010-01-05','AQMIX',True,'BIL')
# alternate managed futures implementation
report('WTMF_ACTUAL_GOLDH_2011','2011-01-05','WTMF',True,'BIL')
# near-20y structural proxy. RYMTX has managed-futures history from 2007 but methodology changed pre-2013.
report('RYMTX_IDEAL_GOLDH_2007','2007-05-25','RYMTX',False,'BIL')
# actual DBMF sanity window with actual SGOV where available
report('DBMF_ACTUAL_GOLDH_2020','2020-05-26','DBMF',True,'SGOV')
