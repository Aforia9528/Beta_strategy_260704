import numpy as np,pandas as pd,yfinance as yf,json
P=dict(target=.20,win=16,floor=.20,cap=1.,inc=.15,dead=.05,hcap=.60,band=.05,ma=200,gm=.5); COST=.001
T={'q':'QLD','spy':'SPY','fx':'KRW=X','g':'132030.KS','mf':'AQMIX','cash':'BIL'}
def dl(t):
 x=yf.download(t,start='2010-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x.Close.dropna()
x={k:dl(v) for k,v in T.items()};us=pd.concat({'q':x['q'],'spy':x['spy']},axis=1).dropna();v=us.q.pct_change().rolling(P['win']).std()*np.sqrt(252);te=np.clip((P['target']/v).values,P['floor'],P['cap']);cur=0;a=[]
for z in te:
 if np.isnan(z):a.append(cur);continue
 if z<cur:cur=z
 elif z-cur>P['inc']:cur=z
 a.append(cur)
ma=us.spy.rolling(P['ma']).mean();gg=np.array(a)*np.where((us.spy<ma).values,P['gm'],1);last=gg[0];qq=[]
for z in gg:
 if abs(z-last)>P['dead']:last=z
 qq.append(last)
qs=pd.Series(qq,index=us.index+pd.Timedelta(days=1));keys=['q','fx','g','mf','cash'];idx=pd.DatetimeIndex(sorted(set().union(*[set(x[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp('2010-10-04')];ff=lambda k:x[k].reindex(idx).ffill();fx=ff('fx');prices=pd.DataFrame({'q':ff('q'),'g':ff('g'),'mf':ff('mf')*fx,'cash':ff('cash')*fx},index=idx).dropna();R=prices.pct_change().dropna()
def sim(gf):
 q=qs.reindex(R.index,method='ffill').dropna();A=R.loc[q.index,['q','g','mf','cash']].values;h=np.minimum(P['hcap'],np.maximum(0,1-q.values));W=np.c_[q.values,gf*h,(1-gf)*h,np.maximum(0,1-q.values-h)];wc=W[0].copy();out=[]
 for i in range(len(q)):
  wt=W[i];cf=1.
  if np.max(np.abs(wc-wt))>P['band']:
   cf=max(0,1-COST*np.sum(np.abs(wt-wc)));wc=wt.copy()
  grossret=float(wc@A[i]);net=cf*(1+grossret)-1;out.append(net);g=wc*(1+A[i]);wc=g/g.sum()
 return pd.Series(out,index=q.index)
def met(r,ppy=None):
 if ppy is None: yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs
 else:yrs=len(r)/ppy
 eq=(1+r).cumprod();c=eq.iloc[-1]**(1/yrs)-1;vol=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/vol;neg=r[r<0].std()*np.sqrt(ppy);m=(eq/eq.cummax()-1).min();return {'CAGR':float(c),'Vol':float(vol),'Sharpe':float(sh),'Sortino':float(r.mean()*ppy/neg),'MDD':float(m),'Calmar':float(c/abs(m))}
F=[.5,.6,.65,.7,.75,.8];D={f:sim(f) for f in F};W=(1+pd.concat(D,axis=1)).resample('W-FRI').prod()-1
print('FULL',json.dumps({str(f):met(D[f]) for f in F}))
def oos():
 out={f:[] for f in F}
 for y in range(2016,2027):
  te=W[(W.index>=f'{y}-01-01')&(W.index<f'{y+1}-01-01')]
  if len(te)>10:
   for f in F:out[f].append(te[f])
 return {str(f):met(pd.concat(out[f]),52) for f in F}
print('OOS_FIXED_2016_26',json.dumps(oos()))
def boot(a,b,block=13,n=10000,years=10,seed=1):
 rng=np.random.default_rng(seed);X=W[[a,b]].dropna().values;N=len(X);L=52*years;starts=np.arange(N-block+1);win=np.zeros(4);d=[]
 for _ in range(n):
  ids=[]
  while len(ids)<L:
   s=int(rng.choice(starts));ids.extend(range(s,s+block))
  Y=X[np.array(ids[:L])];A=met(pd.Series(Y[:,0]),52);B=met(pd.Series(Y[:,1]),52);z=np.array([A['CAGR']-B['CAGR'],A['Sharpe']-B['Sharpe'],A['Calmar']-B['Calmar'],A['MDD']-B['MDD']]);win+=z>0;d.append(z)
 d=np.array(d);return {'P':{'CAGR':float(win[0]/n),'Sharpe':float(win[1]/n),'Calmar':float(win[2]/n),'MDD':float(win[3]/n)},'median':[float(v) for v in np.median(d,0)],'p05p95':[[float(v) for v in np.quantile(d[:,j],[.05,.95])] for j in range(4)]}
for a,b,s in [(.6,.5,31),(.65,.5,32),(.7,.5,33),(.6,.7,34),(.65,.7,35),(.7,.8,36)]:print('BOOT',a,b,json.dumps(boot(a,b,13,10000,10,s)))
# leave-one-calendar-year-out full-period metrics: how often each fixed ratio wins Sharpe/Calmar
Ys=sorted(set(W.index.year));winsS={f:0 for f in F};winsC={f:0 for f in F};bestS=[];bestC=[]
for y in Ys:
 Z=W[W.index.year!=y];M={f:met(Z[f],52) for f in F};s=max(F,key=lambda f:M[f]['Sharpe']);c=max(F,key=lambda f:M[f]['Calmar']);winsS[s]+=1;winsC[c]+=1;bestS.append([int(y),s]);bestC.append([int(y),c])
print('LOYO',json.dumps({'winsSharpe':winsS,'winsCalmar':winsC,'bestSharpe':bestS,'bestCalmar':bestC}))
