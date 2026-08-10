import itertools, json, math
import numpy as np
import pandas as pd
import yfinance as yf

# Baseline live strategy
BASE = dict(target=0.20, win=16, floor=0.20, cap=1.0, inc=0.15,
            deadband=0.05, hcap=0.60, band=0.05, gate_ma=200, gate_mult=0.5)
TCOST=0.001
GRID=np.round(np.arange(0,1.0001,0.05),2)
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL',
   'SGOV':'SGOV','DBMF':'DBMF','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
print('DATA_STARTS',json.dumps({k:str(v.index[0].date()) for k,v in raw.items()}))

def qsignal(params):
    us=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()
    rq=us.q.pct_change(); vol=rq.rolling(params['win']).std()*np.sqrt(252)
    te=np.clip((params['target']/vol).values,params['floor'],params['cap'])
    cur=0.; asym=[]
    for x in te:
        if np.isnan(x): asym.append(cur); continue
        if x<cur: cur=x
        elif x-cur>params['inc']: cur=x
        asym.append(cur)
    ma=us.spy.rolling(params['gate_ma']).mean(); gated=np.array(asym)*np.where((us.spy<ma).values,params['gate_mult'],1.0)
    last=gated[0]; out=[]
    for g in gated:
        if abs(g-last)>params['deadband']: last=g
        out.append(last)
    return pd.Series(out,index=us.index+pd.Timedelta(days=1),name='q')

def base_returns(start,mfkey,gold_actual=True,cashkey='BIL'):
    keys=['QLD','FX','GLD',mfkey,cashkey] + (['GOLD_H'] if gold_actual else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp(start)]
    def ff(k):return raw[k].reindex(idx).ffill()
    fx=ff('FX')
    # HH implementation: QLD USD return as 2x Nasdaq hedged proxy; gold H actual if available;
    # managed futures and T-bill remain USD/KRW exposed as requested.
    q=ff('QLD')
    gold=ff('GOLD_H') if gold_actual else ff('GLD')
    mf=ff(mfkey)*fx; cash=ff(cashkey)*fx
    P=pd.DataFrame({'q':q,'gold':gold,'mf':mf,'cash':cash},index=idx).dropna()
    return P.pct_change().dropna()

def target_weights(idx,gfrac,params):
    qs=qsignal(params).reindex(idx,method='ffill').dropna(); out=[]
    for q in qs:
        hb=min(params['hcap'],max(0,1-q)); out.append((q,gfrac*hb,(1-gfrac)*hb,max(0,1-q-hb)))
    return pd.DataFrame(out,index=qs.index,columns=['q','gold','mf','cash'])

def sim_from_returns(R,gfrac,params=BASE,cost=TCOST):
    W=target_weights(R.index,gfrac,params); R=R.loc[W.index]
    nav=1.; wc=W.iloc[0].values.astype(float); rows=[]
    for dt in R.index:
        wt=W.loc[dt].values.astype(float)
        if np.max(np.abs(wc-wt))>params['band']:
            nav*=max(0,1-cost*np.sum(np.abs(wt-wc))); wc=wt.copy()
        rr=R.loc[dt,['q','gold','mf','cash']].values.astype(float)
        pr=float(wc@rr); nav*=1+pr
        gross=wc*(1+rr); wc=gross/gross.sum(); rows.append((dt,pr,nav))
    return pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')

def met_returns(r, ppy=None):
    r=pd.Series(r).dropna()
    if len(r)<2:return dict(CAGR=np.nan,Vol=np.nan,Sharpe=np.nan,Sortino=np.nan,MDD=np.nan,Calmar=np.nan)
    if ppy is None:
        yrs=(r.index[-1]-r.index[0]).days/365.25; ppy=len(r)/yrs
    else: yrs=len(r)/ppy
    eq=(1+r).cumprod(); c=float(eq.iloc[-1]**(1/yrs)-1); v=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/v) if v else np.nan
    neg=r[r<0].std()*np.sqrt(ppy); so=float(r.mean()*ppy/neg) if neg and np.isfinite(neg) else np.nan
    m=float((eq/eq.cummax()-1).min()); cal=float(c/abs(m)) if m<0 else np.nan
    return dict(CAGR=c,Vol=v,Sharpe=sh,Sortino=so,MDD=m,Calmar=cal)

def met(z):return met_returns(z.ret)

def sweep(R,params=BASE):
    out={}
    for gf in GRID: out[float(gf)]=met(sim_from_returns(R,float(gf),params))
    return out

def bests(sw):
    fs=list(sw)
    return {'Sharpe':max(fs,key=lambda f:sw[f]['Sharpe']), 'Calmar':max(fs,key=lambda f:sw[f]['Calmar']),
            'Sortino':max(fs,key=lambda f:sw[f]['Sortino']), 'CAGR':max(fs,key=lambda f:sw[f]['CAGR']),
            'MinVol':min(fs,key=lambda f:sw[f]['Vol']), 'MinMDD':max(fs,key=lambda f:sw[f]['MDD'])}

def plateau(sw,key,tol=.98):
    vals={f:sw[f][key] for f in sw}; b=max(vals.values()); ok=[f for f,v in vals.items() if v>=tol*b]
    return [min(ok),max(ok)] if ok else []

def weekly_joint(Z):
    d=pd.concat({f:z.ret for f,z in Z.items()},axis=1).dropna()
    return (1+d).resample('W-FRI').prod()-1

def block_bootstrap_pair(W,a=.70,b=.50,block=13,horizon=520,n=3000,seed=7):
    rng=np.random.default_rng(seed); arr=W[[a,b]].values; N=len(arr); starts=np.arange(0,max(1,N-block+1))
    wins={'wealth':0,'sharpe':0,'calmar':0,'mdd':0}; diffs=[]
    for _ in range(n):
        chunks=[]; need=horizon
        while need>0:
            s=int(rng.choice(starts)); ch=arr[s:s+block]; chunks.append(ch); need-=len(ch)
        x=np.vstack(chunks)[:horizon]
        ma=met_returns(pd.Series(x[:,0]),52); mb=met_returns(pd.Series(x[:,1]),52)
        wins['wealth']+=ma['CAGR']>mb['CAGR']; wins['sharpe']+=ma['Sharpe']>mb['Sharpe']; wins['calmar']+=ma['Calmar']>mb['Calmar']; wins['mdd']+=ma['MDD']>mb['MDD']
        diffs.append((ma['CAGR']-mb['CAGR'],ma['Sharpe']-mb['Sharpe'],ma['Calmar']-mb['Calmar'],ma['MDD']-mb['MDD']))
    A=np.array(diffs)
    return {'block_weeks':block,'horizon_years':horizon/52,'n':n,
            'P70gt50_CAGR':wins['wealth']/n,'P70gt50_Sharpe':wins['sharpe']/n,'P70gt50_Calmar':wins['calmar']/n,'P70better_MDD':wins['mdd']/n,
            'dCAGR_p05_med_p95':[float(x) for x in np.quantile(A[:,0],[.05,.5,.95])],
            'dSharpe_p05_med_p95':[float(x) for x in np.quantile(A[:,1],[.05,.5,.95])],
            'dCalmar_p05_med_p95':[float(x) for x in np.quantile(A[:,2],[.05,.5,.95])]}

def walkforward(W,train_years=5,start_year=2016,end_year=2026):
    chosenS=[]; chosenC=[]; oosS=[]; oosC=[]; fixed50=[]; fixed70=[]
    for y in range(start_year,end_year+1):
        tr=W[(W.index>=f'{y-train_years}-01-01')&(W.index<f'{y}-01-01')]
        te=W[(W.index>=f'{y}-01-01')&(W.index<f'{y+1}-01-01')]
        if len(tr)<150 or len(te)<10:continue
        ms={f:met_returns(tr[f],52) for f in W.columns}
        fs=max(ms,key=lambda f:ms[f]['Sharpe']); fc=max(ms,key=lambda f:ms[f]['Calmar'])
        chosenS.append((y,float(fs))); chosenC.append((y,float(fc)))
        oosS.append(te[fs]); oosC.append(te[fc]); fixed50.append(te[.50]); fixed70.append(te[.70])
    def join(xs):return pd.concat(xs) if xs else pd.Series(dtype=float)
    return {'chosen_sharpe':chosenS,'chosen_calmar':chosenC,
            'OOS_sharpe_selector':met_returns(join(oosS),52),'OOS_calmar_selector':met_returns(join(oosC),52),
            'OOS_fixed50':met_returns(join(fixed50),52),'OOS_fixed70':met_returns(join(fixed70),52)}

def pbo(W,segments=8):
    # CSCV-style overfitting diagnostic on weekly returns. Objective: Sharpe.
    n=len(W); edges=np.linspace(0,n,segments+1,dtype=int); blocks=[np.arange(edges[i],edges[i+1]) for i in range(segments)]
    logs=[]; selected=[]
    for comb in itertools.combinations(range(segments),segments//2):
        ins=np.concatenate([blocks[i] for i in comb]); outs=np.concatenate([blocks[i] for i in range(segments) if i not in comb])
        mis={f:met_returns(W.iloc[ins][f],52)['Sharpe'] for f in W.columns}; best=max(mis,key=mis.get); selected.append(float(best))
        mos={f:met_returns(W.iloc[outs][f],52)['Sharpe'] for f in W.columns}; ordered=sorted(mos,key=mos.get)
        rank=(ordered.index(best)+1)/(len(ordered)+1); logs.append(math.log(rank/(1-rank)))
    return {'splits':len(logs),'PBO':float(np.mean(np.array(logs)<0)), 'median_logit':float(np.median(logs)),
            'selected_frac_median':float(np.median(selected)), 'selected_frac_q25_q75':[float(x) for x in np.quantile(selected,[.25,.75])]}

def regimes(Z):
    periods={'2010_14':('2010-01-01','2015-01-01'),'2015_19':('2015-01-01','2020-01-01'),'2020_22':('2020-01-01','2023-01-01'),'2023_26':('2023-01-01','2027-01-01'),
             'gold_bear_2013_15':('2013-01-01','2016-01-01'),'covid_2020':('2020-02-01','2020-06-01'),'inflation_2022':('2022-01-01','2023-01-01')}
    out={}
    for nm,(a,b) in periods.items():
        out[nm]={}
        for f,z in Z.items():
            rr=z.ret[(z.index>=a)&(z.index<b)]
            if len(rr)>5:out[nm][float(f)]=met_returns(rr)
    return out

def mean_tests(R):
    out={}
    cases={'ORIGINAL':R.copy()}
    # Equalize gold and MF arithmetic daily means: keeps their entire covariance/volatility/path shape, removes return-forecast advantage.
    tgt=(R.gold.mean()+R.mf.mean())/2
    E=R.copy(); E['gold']+=tgt-R.gold.mean(); E['mf']+=tgt-R.mf.mean(); cases['EQUAL_MEAN']=E
    # Covariance-only: set both diversifiers to cash mean.
    C=R.copy(); C['gold']+=R.cash.mean()-R.gold.mean(); C['mf']+=R.cash.mean()-R.mf.mean(); cases['BOTH_TO_CASH_MEAN']=C
    # Swap their historical mean advantages while preserving shocks/volatility/correlation.
    S=R.copy(); mg,mm=R.gold.mean(),R.mf.mean(); S['gold']+=mm-mg; S['mf']+=mg-mm; cases['SWAP_MEANS']=S
    for nm,X in cases.items():
        sw=sweep(X); out[nm]={'means_ann_approx':{'gold':float(X.gold.mean()*252),'mf':float(X.mf.mean()*252),'cash':float(X.cash.mean()*252)},
                              'best':bests(sw),'plateau_sharpe_98':plateau(sw,'Sharpe'), 'plateau_calmar_98':plateau(sw,'Calmar'),
                              'm50':sw[.5],'m70':sw[.7],'m80':sw[.8]}
    return out

def parameter_perturb(R):
    picksS=[]; picksC=[]
    for target in [0.15,0.20,0.25]:
      for win in [10,16,32]:
       for ma in [150,200,250]:
        for inc in [0.10,0.15,0.20]:
         p=BASE.copy(); p.update(target=target,win=win,gate_ma=ma,inc=inc)
         sw={}
         for gf in np.round(np.arange(.4,.91,.05),2): sw[float(gf)]=met(sim_from_returns(R,float(gf),p))
         picksS.append(max(sw,key=lambda f:sw[f]['Sharpe'])); picksC.append(max(sw,key=lambda f:sw[f]['Calmar']))
    def summ(a):return {'n':len(a),'median':float(np.median(a)),'q10_q90':[float(x) for x in np.quantile(a,[.1,.9])], 'freq':{str(f):int(a.count(f)) for f in sorted(set(a))}}
    return {'Sharpe_opt':summ(picksS),'Calmar_opt':summ(picksC)}

# Core actual-H study (AQMIX), plus alternative managed-futures implementations.
sets=[('AQMIX','2010-10-04',True,'BIL'),('WTMF','2011-01-06',True,'BIL'),('RYMTX','2007-05-31',False,'BIL'),('DBMF','2020-06-02',True,'SGOV')]
for mf,start,gactual,cash in sets:
    R=base_returns(start,mf,gactual,cash); sw=sweep(R); Z={f:sim_from_returns(R,float(f)) for f in GRID}; W=weekly_joint(Z)
    print('\nPROXY',mf,'START',start)
    print('FULL_BEST',json.dumps(bests(sw)))
    print('PLATEAU',json.dumps({'Sharpe98':plateau(sw,'Sharpe'),'Calmar98':plateau(sw,'Calmar')}))
    print('KEY',json.dumps({str(f):sw[f] for f in [.5,.6,.7,.75,.8,.9]}))
    if mf=='AQMIX':
        print('MEAN_TESTS',json.dumps(mean_tests(R)))
        print('PARAM_PERTURB',json.dumps(parameter_perturb(R)))
        print('WALKFORWARD',json.dumps(walkforward(W)))
        print('PBO',json.dumps(pbo(W,8)))
        print('BOOT_4W_70v50',json.dumps(block_bootstrap_pair(W,.70,.50,4,520,3000,11)))
        print('BOOT_13W_70v50',json.dumps(block_bootstrap_pair(W,.70,.50,13,520,3000,12)))
        print('BOOT_13W_70v80',json.dumps(block_bootstrap_pair(W,.70,.80,13,520,3000,13)))
        rg=regimes({f:Z[f] for f in [.5,.7,.8]}); print('REGIMES',json.dumps(rg))
