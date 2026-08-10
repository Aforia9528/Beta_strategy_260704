import json, math, itertools
import numpy as np
import pandas as pd
import yfinance as yf

# Deep two-parameter study. Everything else is fixed.
# Primary implementation: trade BAND=7.5% (from prior challenger test), cost=10bp per gross turnover.
# Sensitivity repeats finalists with BAND=5% and cost=0/10/25/50bp.
FIX=dict(target=.20,win=16,floor=.20,cap=1.,inc=.15,gate_ma=200,gate_mult=.50,
         gold_frac=.50,trade_band=.075,cost=.001)
HGRID=np.round(np.arange(.40,1.0001,.025),4)
DGRID=np.round(np.arange(0,.1501,.0125),4)
BASE_H=.60; BASE_D=.05
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS',
   'BIL':'BIL','SGOV':'SGOV','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX','DBMF':'DBMF'}

def dl(t):
    x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
    if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
    return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
print('DATA_STARTS',json.dumps({k:str(v.index[0].date()) for k,v in raw.items()}))

sig=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()

def qsignal(dead):
    r=sig.q.pct_change(); vol=r.rolling(FIX['win']).std()*np.sqrt(252)
    te=np.clip((FIX['target']/vol).values,FIX['floor'],FIX['cap'])
    cur=0.; a=[]
    for z in te:
        if np.isnan(z): a.append(cur); continue
        if z<cur: cur=z
        elif z-cur>FIX['inc']: cur=z
        a.append(cur)
    ma=sig.spy.rolling(FIX['gate_ma']).mean()
    g=np.array(a)*np.where((sig.spy<ma).values,FIX['gate_mult'],1.)
    last=g[0]; out=[]
    for z in g:
        if abs(z-last)>dead: last=z
        out.append(last)
    return pd.Series(out,index=sig.index+pd.Timedelta(days=1),name='q')
QS={float(d):qsignal(float(d)) for d in DGRID}

def retframe(start,mf,cash,actual_gold=False):
    keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual_gold else [])
    idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(start)]
    ff=lambda k:raw[k].reindex(idx).ffill(); fx=ff('FX')
    P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GOLD_H') if actual_gold else ff('GLD'),
                    'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna()
    return P.pct_change().dropna()

def simulate(R,h,dead,band=None,cost=None,keep_weights=False):
    if band is None: band=FIX['trade_band']
    if cost is None: cost=FIX['cost']
    q=QS[float(dead)].reindex(R.index,method='ffill').dropna(); R=R.loc[q.index]
    hb=np.minimum(h,np.maximum(0,1-q.values)); cash=np.maximum(0,1-q.values-hb)
    W=np.c_[q.values,.5*hb,.5*hb,cash]; A=R[['q','gold','mf','cash']].values
    wc=W[0].copy(); nav=1.; rows=[]; turn=0.; nt=0
    for i,dt in enumerate(R.index):
        wt=W[i]; cc=0.
        if np.max(np.abs(wc-wt))>band:
            to=float(np.sum(np.abs(wt-wc))); cc=cost*to; turn+=to; nt+=1; wc=wt.copy()
        gr=float(wc@A[i]); net=(1-cc)*(1+gr)-1; nav*=1+net
        end=wc*(1+A[i]); wc=end/end.sum(); rows.append((dt,net,nav))
    z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date')
    yrs=(z.index[-1]-z.index[0]).days/365.25
    z.attrs.update(turn=turn,ntrade=nt,years=yrs)
    if keep_weights:
        z.attrs['avg_q']=float(np.mean(q.values));z.attrs['avg_hb']=float(np.mean(hb));z.attrs['avg_cash']=float(np.mean(cash));z.attrs['bind_frac']=float(np.mean((1-q.values)>h+1e-12))
        updates=float(np.count_nonzero(np.diff(q.values)));z.attrs['signal_updates_yr']=updates/yrs
    return z

def metric(z):
    r=z.ret; yrs=z.attrs.get('years',(z.index[-1]-z.index[0]).days/365.25); ppy=len(r)/yrs
    eq=z.nav; c=float(eq.iloc[-1]**(1/yrs)-1); vol=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/vol)
    neg=r[r<0].std()*np.sqrt(ppy); so=float(r.mean()*ppy/neg); m=float((eq/eq.cummax()-1).min())
    return {'CAGR':c,'Vol':vol,'Sharpe':sh,'Sortino':so,'MDD':m,'Calmar':c/abs(m),
            'TradesYr':z.attrs['ntrade']/yrs,'TurnYr':z.attrs['turn']/yrs}

def metric_returns(r,ppy=252):
    r=pd.Series(r).dropna(); eq=(1+r).cumprod(); yrs=len(r)/ppy
    c=float(eq.iloc[-1]**(1/yrs)-1); v=float(r.std()*np.sqrt(ppy)); sh=float(r.mean()*ppy/v); m=float((eq/eq.cummax()-1).min())
    return {'CAGR':c,'Sharpe':sh,'MDD':m,'Calmar':c/abs(m)}

SETS=[('RY','2007-05-31','RYMTX','BIL',False),('AQ','2010-10-04','AQMIX','BIL',False),
      ('WT','2011-01-06','WTMF','BIL',False),('DB','2020-06-02','DBMF','SGOV',False),
      ('AQ_ACT','2010-10-04','AQMIX','BIL',True),('DB_ACT','2020-06-02','DBMF','SGOV',True)]
RF={lab:retframe(st,mf,ca,act) for lab,st,mf,ca,act in SETS}
CORE=['RY','AQ','WT','DB']; ACT=['AQ_ACT','DB_ACT']

# Primary surface
RES={s:{} for s in RF}; KEEP={'RY':{},'AQ_ACT':{}}
for s,R in RF.items():
    for h in HGRID:
        for d in DGRID:
            key=(float(h),float(d)); z=simulate(R,*key); RES[s][key]=metric(z)
            if s in KEEP: KEEP[s][key]=z.ret.copy()
base=(BASE_H,BASE_D)

def aggregate(group):
    rows=[]
    for key in RES[group[0]]:
        x={'H':key[0],'D':key[1]}
        for met in ['CAGR','Sharpe','MDD','Calmar','TradesYr','TurnYr']:
            vals=[RES[s][key][met] for s in group]; b=[RES[s][base][met] for s in group]
            x['avg_'+met]=float(np.mean(vals));x['d_'+met]=float(np.mean(np.array(vals)-np.array(b)))
        x['winsS']=int(sum(RES[s][key]['Sharpe']>RES[s][base]['Sharpe'] for s in group))
        x['winsC']=int(sum(RES[s][key]['Calmar']>RES[s][base]['Calmar'] for s in group))
        x['winsG']=int(sum(RES[s][key]['CAGR']>RES[s][base]['CAGR'] for s in group))
        x['winsM']=int(sum(RES[s][key]['MDD']>RES[s][base]['MDD'] for s in group))
        x['min_dS']=float(min(RES[s][key]['Sharpe']-RES[s][base]['Sharpe'] for s in group))
        x['min_dC']=float(min(RES[s][key]['Calmar']-RES[s][base]['Calmar'] for s in group))
        rows.append(x)
    df=pd.DataFrame(rows)
    # rank stability, lower better
    rankcols=[]
    for s in group:
        rs=pd.Series({k:RES[s][k]['Sharpe'] for k in RES[s]}).rank(ascending=False,method='average')
        rc=pd.Series({k:RES[s][k]['Calmar'] for k in RES[s]}).rank(ascending=False,method='average')
        rankcols.append((rs,rc))
    df['avgRank']=0.
    for i,row in df.iterrows():
        k=(row.H,row.D);df.at[i,'avgRank']=float(np.mean([z.loc[k] for pair in rankcols for z in pair]))
    return df
AGC=aggregate(CORE); AGA=aggregate(ACT)

def rowkey(r):return (float(r.H),float(r.D))
def show_top(df,name,col,asc=False,n=12):
    z=df.sort_values(col,ascending=asc).head(n)
    print(name,json.dumps(z[['H','D','avg_CAGR','avg_Sharpe','avg_MDD','avg_Calmar','d_CAGR','d_Sharpe','d_MDD','d_Calmar','winsS','winsC','min_dS','min_dC','avgRank']].to_dict('records')))
print('\nBASE_CORE',json.dumps({s:RES[s][base] for s in CORE}))
print('BASE_ACTUAL',json.dumps({s:RES[s][base] for s in ACT}))
show_top(AGC,'TOP_AVG_SHARPE','avg_Sharpe')
show_top(AGC,'TOP_AVG_CALMAR','avg_Calmar')
show_top(AGC,'TOP_AVG_RANK','avgRank',True)
show_top(AGC,'TOP_WORSTCASE_SHARPE','min_dS')
show_top(AGC,'TOP_WORSTCASE_CALMAR','min_dC')
show_top(AGA,'TOP_ACTUAL_RANK','avgRank',True)

# Broad plateau: economically near best on both avg Sharpe and avg Calmar.
mxs=AGC.avg_Sharpe.max();mxc=AGC.avg_Calmar.max();plat=AGC[(AGC.avg_Sharpe>=mxs-.01)&(AGC.avg_Calmar>=mxc-.02)]
print('PLATEAU_CORE',json.dumps({'n':len(plat),'H_range':[float(plat.H.min()),float(plat.H.max())] if len(plat) else None,
 'D_range':[float(plat.D.min()),float(plat.D.max())] if len(plat) else None,
 'points':plat.sort_values('avgRank').head(30)[['H','D','avg_Sharpe','avg_Calmar','avgRank']].to_dict('records')}))

# Candidate set from distinct selection rules
rules=[AGC.loc[AGC.avg_Sharpe.idxmax()],AGC.loc[AGC.avg_Calmar.idxmax()],AGC.loc[AGC.avgRank.idxmin()],
       AGC.loc[AGC.min_dS.idxmax()],AGC.loc[AGC.min_dC.idxmax()],AGA.loc[AGA.avgRank.idxmin()]]
FIN=[base]
for r in rules:
    k=rowkey(r)
    if k not in FIN:FIN.append(k)
print('FINALISTS',json.dumps(FIN))

# Allocation mechanics
print('\nMECHANICS')
for k in FIN:
    z=simulate(RF['AQ_ACT'],*k,keep_weights=True)
    print(k,json.dumps({x:z.attrs[x] for x in ['avg_q','avg_hb','avg_cash','bind_frac','signal_updates_yr','ntrade','turn']}))

# Leave-one-proxy-out: select on other 3 proxies, evaluate held-out.
print('\nLEAVE_ONE_PROXY_OUT')
for held in CORE:
    train=[s for s in CORE if s!=held]
    for objective in ['Sharpe','Calmar']:
        bestk=max(RES[train[0]],key=lambda k:np.mean([RES[s][k][objective] for s in train]))
        out=RES[held][bestk];b=RES[held][base]
        print(held,objective,bestk,json.dumps({'out':out,'base':b,'dSharpe':out['Sharpe']-b['Sharpe'],'dCalmar':out['Calmar']-b['Calmar'],'dCAGR':out['CAGR']-b['CAGR']}))

# Early/late rank transfer on longest proxy and actual-gold proxy.
def submetric(r,a,b):
    q=r[(r.index>=a)&(r.index<b)]
    return metric_returns(q,252) if len(q)>100 else None
print('\nEARLY_LATE')
for s,cut in [('RY','2017-01-01'),('AQ_ACT','2018-01-01')]:
    early={k:submetric(KEEP[s][k],'2007-01-01' if s=='RY' else '2010-01-01',cut) for k in KEEP[s]}
    late={k:submetric(KEEP[s][k],cut,'2027-01-01') for k in KEEP[s]}
    for obj in ['Sharpe','Calmar']:
        bk=max(early,key=lambda k:early[k][obj]); print(s,obj,'early_best',bk,json.dumps({'early':early[bk],'late':late[bk],'late_base':late[base]}))
    # Spearman rank correlation early vs late for Sharpe/Calmar
    for obj in ['Sharpe','Calmar']:
        a=pd.Series({k:v[obj] for k,v in early.items()});b=pd.Series({k:v[obj] for k,v in late.items()})
        print(s,'rankcorr',obj,float(a.rank().corr(b.rank())))

# LOYO influence test for finalists
print('\nLOYO_FINALISTS')
for s in ['RY','AQ_ACT']:
    years=sorted(set(KEEP[s][base].index.year));print('SET',s)
    for k in FIN:
        ws=wc=0;ds=[]
        for y in years:
            a=metric_returns(KEEP[s][k][KEEP[s][k].index.year!=y],252);b=metric_returns(KEEP[s][base][KEEP[s][base].index.year!=y],252)
            ws+=a['Sharpe']>b['Sharpe'];wc+=a['Calmar']>b['Calmar'];ds.append([a['CAGR']-b['CAGR'],a['Sharpe']-b['Sharpe'],a['Calmar']-b['Calmar']])
        ar=np.array(ds);print(k,json.dumps({'n':len(years),'SharpeWin':int(ws),'CalmarWin':int(wc),'med_dCAGR':float(np.median(ar[:,0])),'med_dSharpe':float(np.median(ar[:,1])),'med_dCalmar':float(np.median(ar[:,2]))}))

# 5y rolling train -> next-year selected parameter on RY. Shows whether optimizing these two params is stable/useful.
def concat_metric(parts):
    return metric_returns(pd.concat(parts).sort_index(),252)
print('\nWALK_FORWARD_RY')
for obj in ['Sharpe','Calmar']:
    choices=[];oos=[];baseparts=[]
    for y in range(2013,2027):
        a=f'{y-5}-01-01';b=f'{y}-01-01';c=f'{y+1}-01-01'
        train={k:submetric(KEEP['RY'][k],a,b) for k in KEEP['RY']}
        if any(v is None for v in train.values()):continue
        bk=max(train,key=lambda k:train[k][obj]);te=KEEP['RY'][bk][(KEEP['RY'][bk].index>=b)&(KEEP['RY'][bk].index<c)];bb=KEEP['RY'][base][(KEEP['RY'][base].index>=b)&(KEEP['RY'][base].index<c)]
        if len(te)>20:choices.append([y,bk[0],bk[1]]);oos.append(te);baseparts.append(bb)
    print(obj,json.dumps({'choices':choices,'dynamic_oos':concat_metric(oos),'base_oos':concat_metric(baseparts)}))

# CSCV-style PBO for Sharpe selection on full and coarse grids, longest weekly history.
def weekly_matrix(keys):
    return pd.concat({k:(1+KEEP['RY'][k]).resample('W-FRI').prod()-1 for k in keys},axis=1).dropna()
def pbo(keys,kblocks=8):
    W=weekly_matrix(keys);n=len(W);ed=np.linspace(0,n,kblocks+1,dtype=int);blocks=[np.arange(ed[i],ed[i+1]) for i in range(kblocks)];logits=[];sel=[]
    for co in itertools.combinations(range(kblocks),kblocks//2):
        I=np.concatenate([blocks[i] for i in co]);O=np.concatenate([blocks[i] for i in range(kblocks) if i not in co]);tr=W.iloc[I];te=W.iloc[O]
        shtr=tr.mean()*52/(tr.std()*np.sqrt(52));best=shtr.idxmax();shte=te.mean()*52/(te.std()*np.sqrt(52));rank=shte.rank(ascending=True)[best]/(len(shte)+1);rank=float(np.clip(rank,1e-6,1-1e-6));logits.append(math.log(rank/(1-rank)));sel.append(best)
    H=[x[0] for x in sel];D=[x[1] for x in sel]
    return {'splits':len(logits),'PBO':float(np.mean(np.array(logits)<0)),'selected_H_med':float(np.median(H)),'selected_D_med':float(np.median(D)),'H_q25q75':[float(x) for x in np.quantile(H,[.25,.75])],'D_q25q75':[float(x) for x in np.quantile(D,[.25,.75])]}
allkeys=list(KEEP['RY'].keys());coarse=[k for k in allkeys if k[0] in [.5,.6,.7,.8,.9,1.] and k[1] in [.025,.05,.075,.10,.125,.15]]
print('\nPBO',json.dumps({'full':pbo(allkeys),'coarse':pbo(coarse)}))

# Paired 13-week block bootstrap, 10-year horizon, finalist vs baseline.
def bm(arr,ppy=52):
    eq=np.cumprod(1+arr);yrs=len(arr)/ppy;c=eq[-1]**(1/yrs)-1;v=np.std(arr,ddof=1)*np.sqrt(ppy);sh=np.mean(arr)*ppy/v;m=np.min(eq/np.maximum.accumulate(eq)-1);return c,sh,m,c/abs(m)
def boot(s,k,n=3000,block=13,years=10,seed=7):
    W=pd.concat({'a':(1+KEEP[s][k]).resample('W-FRI').prod()-1,'b':(1+KEEP[s][base]).resample('W-FRI').prod()-1},axis=1).dropna().values
    rng=np.random.default_rng(seed+int(k[0]*1000)+int(k[1]*10000));N=len(W);L=52*years;starts=np.arange(N-block+1);wins=np.zeros(4);diff=[]
    for _ in range(n):
        ids=[]
        while len(ids)<L:
            st=int(rng.choice(starts));ids.extend(range(st,st+block))
        Y=W[np.array(ids[:L])];a=np.array(bm(Y[:,0]));b=np.array(bm(Y[:,1]));z=a-b;wins+=z>0;diff.append(z)
    d=np.array(diff);return {'Pbetter':{'CAGR':float(wins[0]/n),'Sharpe':float(wins[1]/n),'MDD':float(wins[2]/n),'Calmar':float(wins[3]/n)},'medianDiff':[float(x) for x in np.median(d,axis=0)],'p05p95':[[float(x) for x in np.quantile(d[:,j],[.05,.95])] for j in range(4)]}
print('\nBOOTSTRAP')
for s in ['RY','AQ_ACT']:
    for k in FIN[1:]:print(s,k,json.dumps(boot(s,k)))

# Cost and trade-band sensitivity for finalists on CORE.
print('\nSENSITIVITY_FINALISTS')
for band in [.05,.075]:
    for cost in [0.,.001,.0025,.005]:
        print('BAND_COST',band,cost)
        for k in FIN:
            vals=[];bvals=[]
            for s in CORE:
                m=metric(simulate(RF[s],*k,band=band,cost=cost));bb=metric(simulate(RF[s],*base,band=band,cost=cost));vals.append(m);bvals.append(bb)
            print(k,json.dumps({x:float(np.mean([v[x]-b[x] for v,b in zip(vals,bvals)])) for x in ['CAGR','Sharpe','MDD','Calmar','TradesYr','TurnYr']}))
