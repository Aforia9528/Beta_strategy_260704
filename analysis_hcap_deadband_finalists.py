import json,numpy as np,pandas as pd,yfinance as yf
T={'q':'QLD','spy':'SPY','fx':'KRW=X','g':'GLD','gh':'132030.KS','bil':'BIL','sgov':'SGOV','aq':'AQMIX','wt':'WTMF','ry':'RYMTX','db':'DBMF'}
def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x.Close.dropna().astype(float)
x={k:dl(v) for k,v in T.items()};sig=pd.concat({'q':x['q'],'spy':x['spy']},axis=1).dropna()
def qs(dead):
 r=sig.q.pct_change();v=r.rolling(16).std()*np.sqrt(252);te=np.clip((.2/v).values,.2,1.);cur=0;a=[]
 for z in te:
  if np.isnan(z):a.append(cur);continue
  if z<cur:cur=z
  elif z-cur>.15:cur=z
  a.append(cur)
 ma=sig.spy.rolling(200).mean();g=np.array(a)*np.where((sig.spy<ma).values,.5,1);last=g[0];o=[]
 for z in g:
  if abs(z-last)>dead:last=z
  o.append(last)
 return pd.Series(o,index=sig.index+pd.Timedelta(days=1))
Q={d:qs(d) for d in [.025,.05]}
def rf(st,mf,ca,act=False):
 keys=['q','fx','g',mf,ca]+(['gh'] if act else []);idx=pd.DatetimeIndex(sorted(set().union(*[set(x[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(st)];ff=lambda k:x[k].reindex(idx).ffill();fx=ff('fx');P=pd.DataFrame({'q':ff('q'),'g':ff('gh') if act else ff('g'),'m':ff(mf)*fx,'c':ff(ca)*fx},index=idx).dropna();return P.pct_change().dropna()
def sim(R,h,d,band=.075,cost=.001):
 q=Q[d].reindex(R.index,method='ffill').dropna();R=R.loc[q.index];hb=np.minimum(h,np.maximum(0,1-q.values));cash=np.maximum(0,1-q.values-hb);W=np.c_[q.values,.5*hb,.5*hb,cash];A=R[['q','g','m','c']].values;wc=W[0].copy();nav=1;rows=[]
 for i,dt in enumerate(R.index):
  wt=W[i];cc=0
  if np.max(np.abs(wc-wt))>band:
   cc=cost*np.sum(np.abs(wt-wc));wc=wt.copy()
  gr=wc@A[i];net=(1-cc)*(1+gr)-1;nav*=1+net;en=wc*(1+A[i]);wc=en/en.sum();rows.append((dt,net,nav,q.iloc[i],hb[i],cash[i]))
 return pd.DataFrame(rows,columns=['d','r','nav','q','hb','cash']).set_index('d')
def metret(r,ppy=252):
 r=pd.Series(r).dropna();eq=(1+r).cumprod();yrs=len(r)/ppy;c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;m=(eq/eq.cummax()-1).min();ui=float(np.sqrt(np.mean(np.square(eq/eq.cummax()-1))));
 roll21=(1+r).rolling(21).apply(np.prod,raw=True)-1;roll63=(1+r).rolling(63).apply(np.prod,raw=True)-1;roll252=(1+r).rolling(252).apply(np.prod,raw=True)-1
 return {'CAGR':float(c),'Sharpe':float(sh),'MDD':float(m),'Calmar':float(c/abs(m)),'Ulcer':ui,'Worst1m':float(roll21.min()),'Worst3m':float(roll63.min()),'Worst12m':float(roll252.min())}
def sub(z,a,b):return metret(z.loc[(z.index>=a)&(z.index<b),'r'])
SETS={'RY':rf('2007-05-31','ry','bil',False),'AQ':rf('2010-10-04','aq','bil',False),'WT':rf('2011-01-06','wt','bil',False),'DB':rf('2020-06-02','db','sgov',False),'AQ_ACT':rf('2010-10-04','aq','bil',True),'DB_ACT':rf('2020-06-02','db','sgov',True)}
C=[(.6,.05),(.7,.025),(.7,.05),(.725,.025),(.725,.05),(.75,.025),(.75,.05),(.8,.05)]
Z={s:{k:sim(R,*k) for k in C} for s,R in SETS.items()}
print('FULL')
for s in Z:
 print('SET',s)
 for k in C:print(k,json.dumps(metret(Z[s][k].r)))
REG={'GFC':('2007-07','2010-01'),'EURO11':('2011-04','2012-01'),'GOLD_BEAR':('2013-01','2016-01'),'Q4_2018':('2018-09','2019-01'),'COVID':('2020-02','2020-07'),'INFL22':('2022-01','2023-01'),'RECENT':('2023-01','2027-01')}
print('\nREGIMES_RY')
for n,(a,b) in REG.items():
 print(n)
 for k in C:print(k,json.dumps(sub(Z['RY'][k],a,b)))
# allocation averages during regimes on AQ/RY where available
print('\nALLOC_REGIMES')
for n,(a,b) in REG.items():
 print(n)
 for k in [(.6,.05),(.7,.025),(.7,.05),(.725,.025),(.725,.05),(.75,.05),(.8,.05)]:
  z=Z['RY'][k];q=z[(z.index>=a)&(z.index<b)];
  if len(q):print(k,json.dumps({'q':float(q.q.mean()),'hb':float(q.hb.mean()),'cash':float(q.cash.mean())}))
# direct isolated effects averaged across four core proxies
print('\nISOLATED_EFFECTS')
pairs=[('HCAP60to70_D5',(.7,.05),(.6,.05)),('HCAP70to725_D5',(.725,.05),(.7,.05)),('HCAP725to75_D5',(.75,.05),(.725,.05)),('HCAP75to80_D5',(.8,.05),(.75,.05)),('D5to25_H70',(.7,.025),(.7,.05)),('D5to25_H725',(.725,.025),(.725,.05)),('D5to25_H75',(.75,.025),(.75,.05))]
for name,a,b in pairs:
 ds=[]
 for s in ['RY','AQ','WT','DB']:
  A=metret(Z[s][a].r);B=metret(Z[s][b].r);ds.append({m:A[m]-B[m] for m in ['CAGR','Sharpe','MDD','Calmar','Ulcer','Worst12m']})
 print(name,json.dumps({m:float(np.mean([d[m] for d in ds])) for m in ds[0]}))
# calendar-year consistency versus baseline
print('\nYEAR_WINS')
for s in ['RY','AQ_ACT']:
 b=Z[s][(.6,.05)];years=sorted(set(b.index.year));print('SET',s)
 for k in C[1:]:
  wr=ws=wc=wm=0;n=0
  for y in years:
   aa=Z[s][k][Z[s][k].index.year==y].r;bb=b[b.index.year==y].r
   if len(aa)<20 or len(bb)<20:continue
   A=metret(aa);B=metret(bb);n+=1;wr+=A['CAGR']>B['CAGR'];ws+=A['Sharpe']>B['Sharpe'];wc+=A['Calmar']>B['Calmar'];wm+=A['MDD']>B['MDD']
  print(k,json.dumps({'n':n,'CAGR':wr,'Sharpe':ws,'Calmar':wc,'MDD':wm}))
# signal relation D2.5 vs D5
q25=Q[.025].reindex(Z['RY'][(.6,.05)].index,method='ffill');q5=Q[.05].reindex(q25.index,method='ffill');dif=q25-q5
print('\nDEADBAND_SIGNAL',json.dumps({'avg_q25':float(q25.mean()),'avg_q5':float(q5.mean()),'avg_diff':float(dif.mean()),'mean_abs_diff':float(dif.abs().mean()),'q25_lower_frac':float((dif<0).mean()),'q25_higher_frac':float((dif>0).mean()),'p05p95':[float(v) for v in np.quantile(dif,[.05,.95])]}))
