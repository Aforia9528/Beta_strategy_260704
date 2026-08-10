import json, math
import numpy as np
import pandas as pd
import yfinance as yf

# Fixed production candidate from prior study
H_CAP=.70
SIGNAL_DEADBAND=.05
COST=.0007
BANDS=np.round(np.arange(.05,.1001,.005),3)
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL','SGOV':'SGOV','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX','DBMF':'DBMF'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()

def qsignal():
    r=sig.q.pct_change(); vol=r.rolling(16).std()*np.sqrt(252)
    te=np.clip((.20/vol).values,.20,1.)
    cur=0.; a=[]
    for z in te:
        if np.isnan(z): a.append(cur); continue
        if z<cur: cur=z
        elif z-cur>.15: cur=z
        a.append(cur)
    ma=sig.spy.rolling(200).mean(); g=np.array(a)*np.where((sig.spy<ma).values,.5,1.)
    last=g[0]; out=[]
    for z in g:
        if abs(z-last)>SIGNAL_DEADBAND: last=z
        out.append(last)
    return pd.Series(out,index=sig.index+pd.Timedelta(days=1))
Q=qsignal()

def retframe(start,mf,cash,actual_gold=False):
    keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual_gold else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(start)]
    ff=lambda k:raw[k].reindex(idx).ffill(); fx=ff('FX')
    P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GOLD_H') if actual_gold else ff('GLD'),'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna()
    return P.pct_change().dropna()
SETS={'RY':retframe('2007-05-31','RYMTX','BIL',False),'AQ':retframe('2010-10-04','AQMIX','BIL',False),'WT':retframe('2011-01-06','WTMF','BIL',False),'DB':retframe('2020-06-02','DBMF','SGOV',False),'AQ_ACT':retframe('2010-10-04','AQMIX','BIL',True),'DB_ACT':retframe('2020-06-02','DBMF','SGOV',True)}
CORE=['RY','AQ','WT','DB']; ACT=['AQ_ACT','DB_ACT']

def sim(R,band,cost=COST):
    q=Q.reindex(R.index,method='ffill').dropna();R=R.loc[q.index]
    hb=np.minimum(H_CAP,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb)
    W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','gold','mf','cash']].values
    wc=W[0].copy();nav=1.;rows=[];turn=0.;ntrade=0
    for i,dt in enumerate(R.index):
        wt=W[i];cc=0.
        if np.max(np.abs(wc-wt))>band:
            to=float(np.sum(np.abs(wt-wc)));cc=cost*to;turn+=to;ntrade+=1;wc=wt.copy()
        gr=float(wc@A[i]);net=(1-cc)*(1+gr)-1;nav*=1+net
        end=wc*(1+A[i]);wc=end/end.sum();rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(z.index[-1]-z.index[0]).days/365.25
    z.attrs.update(turn=turn,ntrade=ntrade,years=yrs);return z

def metrics_ret(r,ppy=None):
    r=pd.Series(r).dropna()
    if len(r)<2:return None
    if ppy is None:
        yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs
    else:yrs=len(r)/ppy
    eq=(1+r).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);v=float(r.std()*np.sqrt(ppy));sh=float(r.mean()*ppy/v);m=float((eq/eq.cummax()-1).min());cal=float(c/abs(m));ui=float(np.sqrt(np.mean(np.square(eq/eq.cummax()-1))))
    return {'CAGR':c,'Vol':v,'Sharpe':sh,'MDD':m,'Calmar':cal,'Ulcer':ui}

def metric(z):
    m=metrics_ret(z.ret);m.update({'TradesYr':z.attrs['ntrade']/z.attrs['years'],'TurnYr':z.attrs['turn']/z.attrs['years'],'FeeApproxYr':COST*z.attrs['turn']/z.attrs['years']});return m

Z={s:{float(b):sim(R,float(b)) for b in BANDS} for s,R in SETS.items()}
M={s:{b:metric(z) for b,z in Z[s].items()} for s in Z}

print('FULL_CORE')
for b in BANDS:
    vals=[M[s][float(b)] for s in CORE]
    print(float(b),json.dumps({k:float(np.mean([v[k] for v in vals])) for k in vals[0]}))
print('\nFULL_ACTUAL')
for b in BANDS:
    vals=[M[s][float(b)] for s in ACT]
    print(float(b),json.dumps({k:float(np.mean([v[k] for v in vals])) for k in vals[0]}))

# Plateau definition: within 0.005 Sharpe and 0.015 Calmar of best core averages, and MDD no worse than 1pp from best.
coreavg={float(b):{k:float(np.mean([M[s][float(b)][k] for s in CORE])) for k in M[CORE[0]][float(b)]} for b in BANDS}
mxS=max(v['Sharpe'] for v in coreavg.values());mxC=max(v['Calmar'] for v in coreavg.values());bestM=max(v['MDD'] for v in coreavg.values())
plat=[b for b,v in coreavg.items() if v['Sharpe']>=mxS-.005 and v['Calmar']>=mxC-.015 and v['MDD']>=bestM-.01]
print('\nPLATEAU',json.dumps({'bands':plat,'range':[min(plat),max(plat)] if plat else None,'bestSharpe':max(coreavg,key=lambda b:coreavg[b]['Sharpe']),'bestCalmar':max(coreavg,key=lambda b:coreavg[b]['Calmar']),'bestCAGR':max(coreavg,key=lambda b:coreavg[b]['CAGR'])}))

# Rolling 5y windows every quarter, compare median/worst across windows.
def rolling5(z):
    rows=[]
    for start in z.index[::63]:
        end=start+pd.DateOffset(years=5);q=z[(z.index>=start)&(z.index<=end)]
        if (q.index[-1]-q.index[0]).days<365*4.8:continue
        m=metrics_ret(q.ret);rows.append(m)
    a=pd.DataFrame(rows)
    return {'n':len(a),'CAGR_med':float(a.CAGR.median()),'CAGR_p10':float(a.CAGR.quantile(.1)),'Sharpe_med':float(a.Sharpe.median()),'Sharpe_p10':float(a.Sharpe.quantile(.1)),'MDD_med':float(a.MDD.median()),'MDD_worst':float(a.MDD.min()),'Calmar_med':float(a.Calmar.median()),'Calmar_p10':float(a.Calmar.quantile(.1))}
print('\nROLL5_RY')
for b in BANDS:print(float(b),json.dumps(rolling5(Z['RY'][float(b)])))
print('\nROLL5_AQACT')
for b in BANDS:print(float(b),json.dumps(rolling5(Z['AQ_ACT'][float(b)])))

# 5y train -> next year. Pick band based on train Sharpe/Calmar; also report each fixed band over same OOS years.
def walkforward(s,start_year):
    Zs=Z[s];choices={'Sharpe':[],'Calmar':[]};parts={'Sharpe':[],'Calmar':[]};fixed={b:[] for b in BANDS}
    for y in range(start_year,2027):
        tr0=f'{y-5}-01-01';tr1=f'{y}-01-01';te1=f'{y+1}-01-01'
        train={b:metrics_ret(z[(z.index>=tr0)&(z.index<tr1)].ret) for b,z in Zs.items()}
        if any(v is None for v in train.values()):continue
        te={b:z[(z.index>=tr1)&(z.index<te1)].ret for b,z in Zs.items()}
        if min(len(x) for x in te.values())<20:continue
        for obj in ['Sharpe','Calmar']:
            bk=max(train,key=lambda b:train[b][obj]);choices[obj].append([y,bk]);parts[obj].append(te[bk])
        for b in BANDS:fixed[b].append(te[float(b)])
    out={'choices':choices}
    for obj in ['Sharpe','Calmar']:out['dynamic_'+obj]=metrics_ret(pd.concat(parts[obj]).sort_index())
    out['fixed']={float(b):metrics_ret(pd.concat(fixed[b]).sort_index()) for b in BANDS}
    return out
print('\nWALK_RY',json.dumps(walkforward('RY',2013)))
print('\nWALK_AQACT',json.dumps(walkforward('AQ_ACT',2016)))

# leave-one-proxy-out selection transfer on core proxies
print('\nLOPO')
for held in CORE:
    tr=[s for s in CORE if s!=held]
    for obj in ['Sharpe','Calmar']:
        bk=max(BANDS,key=lambda b:np.mean([M[s][float(b)][obj] for s in tr]));o=M[held][float(bk)];base=M[held][.075]
        print(held,obj,float(bk),json.dumps({'selected':o,'vs75':{k:o[k]-base[k] for k in ['CAGR','Sharpe','MDD','Calmar']}}))

# paired weekly 13-week block bootstrap, 10y horizon, every band vs 7.5%; longest RY and actual gold AQ.
def bm(arr):
    eq=np.cumprod(1+arr);yrs=len(arr)/52;c=eq[-1]**(1/yrs)-1;v=np.std(arr,ddof=1)*np.sqrt(52);sh=np.mean(arr)*52/v;m=np.min(eq/np.maximum.accumulate(eq)-1);return np.array([c,sh,m,c/abs(m)])
def bootstrap(s,b,ref=.075,n=4000,block=13,years=10,seed=11):
    a=(1+Z[s][float(b)].ret).resample('W-FRI').prod()-1;rr=(1+Z[s][float(ref)].ret).resample('W-FRI').prod()-1;W=pd.concat({'a':a,'r':rr},axis=1).dropna().values
    rng=np.random.default_rng(seed+int(b*10000));N=len(W);L=52*years;starts=np.arange(N-block+1);win=np.zeros(4);ds=[]
    for _ in range(n):
        ids=[]
        while len(ids)<L:
            st=int(rng.choice(starts));ids.extend(range(st,st+block))
        Y=W[np.array(ids[:L])];d=bm(Y[:,0])-bm(Y[:,1]);win+=d>0;ds.append(d)
    D=np.array(ds)
    return {'Pbetter':{'CAGR':float(win[0]/n),'Sharpe':float(win[1]/n),'MDD':float(win[2]/n),'Calmar':float(win[3]/n)},'medianDiff':[float(x) for x in np.median(D,axis=0)],'p05p95':[[float(x) for x in np.quantile(D[:,j],[.05,.95])] for j in range(4)]}
print('\nBOOT_RY_VS75')
for b in BANDS:
    if abs(b-.075)<1e-9:continue
    print(float(b),json.dumps(bootstrap('RY',float(b))))
print('\nBOOT_AQACT_VS75')
for b in BANDS:
    if abs(b-.075)<1e-9:continue
    print(float(b),json.dumps(bootstrap('AQ_ACT',float(b),seed=31)))

# exact fee delta for each band: 0 cost vs 7bp on core avg
print('\nFEE_DRAG_CORE')
for b in BANDS:
    ds=[]
    for s in CORE:
        a=metric(sim(SETS[s],float(b),cost=0.0));c=metric(sim(SETS[s],float(b),cost=COST));ds.append({k:c[k]-a[k] for k in ['CAGR','Sharpe','MDD','Calmar']})
    print(float(b),json.dumps({k:float(np.mean([x[k] for x in ds])) for k in ds[0]}))
