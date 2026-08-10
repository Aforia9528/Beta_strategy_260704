import json, math
import numpy as np
import pandas as pd
import yfinance as yf

# Intended current architecture for long-horizon comparison:
# Nasdaq 2x(H) proxy = QLD USD return (daily-reset/financing embedded, FX removed)
# Gold(H) long proxy = GLD USD for structural tests; actual 132030.KS used in validation
# Managed futures & cash = USD products translated to KRW (unhedged), per user's intended structure.
BASE=dict(target=.20,win=16,floor=.20,cap=1.,inc=.15,dead=.05,hcap=.60,band=.05,
          gate_ma=200,gate_mult=.50,gate_mode='SPY',vol_mode='ROLL16',asym_mode='LOCKED')
TCOST=.001
T={'QLD':'QLD','SPY':'SPY','QQQ':'QQQ','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS',
   'BIL':'BIL','SGOV':'SGOV','DBMF':'DBMF','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
print('STARTS',json.dumps({k:str(v.index[0].date()) for k,v in raw.items()}))

sigpx=pd.concat({'q':raw['QLD'],'spy':raw['SPY'],'qqq':raw['QQQ']},axis=1).dropna().ffill()

def ewma_vol(r,lam=.94):
    x=r.fillna(0).values; out=np.full(len(x),np.nan); var=np.nan
    for i,z in enumerate(x):
        if i==0: var=z*z
        else: var=lam*var+(1-lam)*z*z
        if i>=20: out[i]=math.sqrt(max(var,0))*math.sqrt(252)
    return pd.Series(out,index=r.index)

def qsignal(p):
    r=sigpx.q.pct_change()
    v16=r.rolling(p.get('win',16)).std()*np.sqrt(252)
    if p['vol_mode']=='ROLL16': vol=v16
    elif p['vol_mode']=='EWMA94': vol=ewma_vol(r,.94)
    elif p['vol_mode']=='MAX16_63': vol=pd.concat([v16,r.rolling(63).std()*np.sqrt(252)],axis=1).max(axis=1)
    else: raise ValueError(p['vol_mode'])
    te=np.clip((p['target']/vol).values,p['floor'],p['cap'])
    cur=0.; asym=[]
    for x in te:
        if np.isnan(x): asym.append(cur); continue
        if p['asym_mode']=='LOCKED':
            if x<cur: cur=x
            elif x-cur>p['inc']: cur=x
        elif p['asym_mode']=='STEPUP':
            if x<cur: cur=x
            elif x>cur: cur=min(x,cur+p['inc'])
        asym.append(cur)
    a=np.array(asym)
    spyma=sigpx.spy.rolling(p['gate_ma']).mean(); qqqma=sigpx.qqq.rolling(p['gate_ma']).mean()
    if p['gate_mode']=='SPY': scale=np.where((sigpx.spy<spyma).values,p['gate_mult'],1.)
    elif p['gate_mode']=='QQQ': scale=np.where((sigpx.qqq<qqqma).values,p['gate_mult'],1.)
    elif p['gate_mode']=='DUAL_SOFT':
        bs=(sigpx.spy<spyma).values.astype(int); bq=(sigpx.qqq<qqqma).values.astype(int)
        # no fitted parameter: 1.0 / 0.75 / 0.50 for 0/1/2 broken trends
        scale=1.-.25*(bs+bq)
    elif p['gate_mode']=='BOTH_ONLY': scale=np.where(((sigpx.spy<spyma)&(sigpx.qqq<qqqma)).values,p['gate_mult'],1.)
    else: raise ValueError(p['gate_mode'])
    gated=a*scale
    last=gated[0]; final=[]
    for g in gated:
        if abs(g-last)>p['dead']: last=g
        final.append(last)
    return pd.Series(final,index=sigpx.index+pd.Timedelta(days=1),name='q')

def retframe(start,mf='AQMIX',cash='BIL',actual_gold=False):
    keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual_gold else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp(start)]
    def ff(k): return raw[k].reindex(idx).ffill()
    fx=ff('FX')
    g=ff('GOLD_H') if actual_gold else ff('GLD')
    P=pd.DataFrame({'q':ff('QLD'),'gold':g,'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna()
    return P.pct_change().dropna()

def simulate(R,p,gfrac=.50):
    qs=qsignal(p).reindex(R.index,method='ffill').dropna(); R=R.loc[qs.index]
    h=np.minimum(p['hcap'],np.maximum(0,1-qs.values))
    W=np.column_stack([qs.values,gfrac*h,(1-gfrac)*h,np.maximum(0,1-qs.values-h)])
    A=R[['q','gold','mf','cash']].values
    wc=W[0].copy(); nav=1.; rows=[]; turnover=0.; ntrade=0
    for i,dt in enumerate(R.index):
        wt=W[i]; cost=0.
        if np.max(np.abs(wc-wt))>p['band']:
            to=float(np.sum(np.abs(wt-wc))); cost=TCOST*to; turnover+=to; ntrade+=1; wc=wt.copy()
        gross=float(wc@A[i]); net=(1-cost)*(1+gross)-1; nav*=1+net
        end=wc*(1+A[i]); wc=end/end.sum(); rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date'); z.attrs['turn']=turnover; z.attrs['ntrade']=ntrade
    return z

def metrics(z):
    r=z.ret; yrs=(z.index[-1]-z.index[0]).days/365.25; ppy=len(r)/yrs; eq=z.nav
    c=float(eq.iloc[-1]**(1/yrs)-1); vol=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/vol)
    neg=r[r<0].std()*np.sqrt(ppy); so=float(r.mean()*ppy/neg); m=float((eq/eq.cummax()-1).min())
    return {'CAGR':c,'Vol':vol,'Sharpe':sh,'Sortino':so,'MDD':m,'Calmar':c/abs(m),'Turn':z.attrs['turn'],'TradesYr':z.attrs['ntrade']/yrs}

def subset_metrics(z,a,b):
    q=z[(z.index>=a)&(z.index<b)]
    if len(q)<20:return None
    # rebase to avoid pre-window nav
    r=q.ret; eq=(1+r).cumprod(); qq=q.copy();qq['nav']=eq;qq.attrs['turn']=0.;qq.attrs['ntrade']=0
    return metrics(qq)

VAR={
'BASE':{},
'QQQ_GATE':{'gate_mode':'QQQ'},
'DUAL_SOFT':{'gate_mode':'DUAL_SOFT'},
'BOTH_ONLY':{'gate_mode':'BOTH_ONLY'},
'EWMA94':{'vol_mode':'EWMA94'},
'MAX16_63':{'vol_mode':'MAX16_63'},
'STEPUP':{'asym_mode':'STEPUP'},
'HCAP70':{'hcap':.70},
'HCAP80':{'hcap':.80},
'BAND075':{'band':.075},
'BAND10':{'band':.10},
}
def cfg(ch):
    p=BASE.copy();p.update(ch);return p

SETS=[('AQMIX','2010-10-04','AQMIX','BIL',False),('WTMF','2011-01-06','WTMF','BIL',False),
      ('RYMTX','2007-05-31','RYMTX','BIL',False),('DBMF','2020-06-02','DBMF','SGOV',False),
      ('AQMIX_ACTUAL_GOLD','2010-10-04','AQMIX','BIL',True),('DBMF_ACTUAL_GOLD','2020-06-02','DBMF','SGOV',True)]
allres={}
for label,start,mf,cash,actualg in SETS:
    R=retframe(start,mf,cash,actualg); allres[label]={}
    print('\nSET',label)
    for name,ch in VAR.items():
        z=simulate(R,cfg(ch),.50); m=metrics(z); allres[label][name]=m; print(name,json.dumps(m))

# Rank candidates across core proxies: reward Sharpe, Calmar, CAGR; penalize Vol, |MDD|, turnover.
CORE=['AQMIX','WTMF','RYMTX','DBMF']
print('\nRANKS_CORE')
for name in VAR:
    ranks=[]
    for key in ['Sharpe','Calmar','CAGR']:
        vals={n:np.mean([allres[s][n][key] for s in CORE]) for n in VAR}; order=sorted(vals,key=vals.get,reverse=True); ranks.append(order.index(name)+1)
    for key in ['Vol','Turn']:
        vals={n:np.mean([allres[s][n][key] for s in CORE]) for n in VAR}; order=sorted(vals,key=vals.get); ranks.append(order.index(name)+1)
    vals={n:np.mean([abs(allres[s][n]['MDD']) for s in CORE]) for n in VAR};order=sorted(vals,key=vals.get);ranks.append(order.index(name)+1)
    print(name,'avg_rank',float(np.mean(ranks)),'ranks',ranks)

# Stress/regime comparison on AQMIX structural proxy
R=retframe('2010-10-04','AQMIX','BIL',False)
periods={'2011_EURO':('2011-04','2011-12'),'GOLD_BEAR':('2013-01','2016-01'),'COVID':('2020-02','2020-06'),
         'INFLATION22':('2022-01','2023-01'),'POST22':('2023-01','2027-01')}
print('\nREGIMES')
for name,ch in VAR.items():
    z=simulate(R,cfg(ch),.50); print(name,json.dumps({k:subset_metrics(z,a,b) for k,(a,b) in periods.items()}))

# Only predeclared combinations, not exhaustive optimization.
COMB={
'DUAL_EWMA':{'gate_mode':'DUAL_SOFT','vol_mode':'EWMA94'},
'DUAL_BAND075':{'gate_mode':'DUAL_SOFT','band':.075},
'EWMA_BAND075':{'vol_mode':'EWMA94','band':.075},
'DUAL_EWMA_BAND075':{'gate_mode':'DUAL_SOFT','vol_mode':'EWMA94','band':.075},
}
print('\nCOMBOS')
for label,start,mf,cash,actualg in SETS:
    R=retframe(start,mf,cash,actualg); print('SET',label)
    b=metrics(simulate(R,BASE,.50)); print('BASE',json.dumps(b))
    for n,ch in COMB.items():print(n,json.dumps(metrics(simulate(R,cfg(ch),.50))))
