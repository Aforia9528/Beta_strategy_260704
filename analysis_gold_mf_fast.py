import itertools,json,math
import numpy as np,pandas as pd,yfinance as yf
BASE=dict(target=.20,win=16,floor=.20,cap=1.,inc=.15,deadband=.05,hcap=.60,band=.05,gate_ma=200,gate_mult=.5)
TCOST=.001; GRID=np.round(np.arange(0,1.001,.05),2)
T={'QLD':'QLD','SPY':'SPY','FX':'KRW=X','GLD':'GLD','GOLD_H':'132030.KS','BIL':'BIL','SGOV':'SGOV','DBMF':'DBMF','AQMIX':'AQMIX','WTMF':'WTMF','RYMTX':'RYMTX'}
def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
us=pd.concat({'q':raw['QLD'],'spy':raw['SPY']},axis=1).dropna()
def qs(p):
 vol=us.q.pct_change().rolling(p['win']).std()*np.sqrt(252); te=np.clip((p['target']/vol).values,p['floor'],p['cap']); cur=0.; a=[]
 for x in te:
  if np.isnan(x):a.append(cur);continue
  if x<cur:cur=x
  elif x-cur>p['inc']:cur=x
  a.append(cur)
 ma=us.spy.rolling(p['gate_ma']).mean(); gg=np.array(a)*np.where((us.spy<ma).values,p['gate_mult'],1.); last=gg[0]; z=[]
 for g in gg:
  if abs(g-last)>p['deadband']:last=g
  z.append(last)
 return pd.Series(z,index=us.index+pd.Timedelta(days=1))
BASEQS=qs(BASE)
def returns(start,mf,actual=True,cash='BIL'):
 keys=['QLD','FX','GLD',mf,cash]+(['GOLD_H'] if actual else []); idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(start)]
 ff=lambda k:raw[k].reindex(idx).ffill(); fx=ff('FX'); P=pd.DataFrame({'q':ff('QLD'),'gold':ff('GOLD_H') if actual else ff('GLD'),'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna();return P.pct_change().dropna()
def targets(idx,gf,p=BASE,qseries=None):
 q=(qseries if qseries is not None else qs(p)).reindex(idx,method='ffill').dropna(); h=np.minimum(p['hcap'],np.maximum(0,1-q.values)); return q.index,np.column_stack([q.values,gf*h,(1-gf)*h,np.maximum(0,1-q.values-h)])
def sim(R,gf,p=BASE,qseries=None,exact=True):
 idx,W=targets(R.index,gf,p,qseries); A=R.loc[idx,['q','gold','mf','cash']].values
 if not exact:
  rr=(W*A).sum(1); return pd.Series(rr,index=idx)
 nav=1.;wc=W[0].copy();ret=[]
 for i in range(len(idx)):
  wt=W[i]
  if np.max(np.abs(wc-wt))>p['band']:
   nav*=max(0,1-TCOST*np.sum(np.abs(wt-wc)));wc=wt.copy()
  r=float(wc@A[i]);ret.append(r);gross=wc*(1+A[i]);wc=gross/gross.sum()
 return pd.Series(ret,index=idx)
def met(r,ppy=None):
 r=pd.Series(r).dropna();
 if ppy is None: yrs=(r.index[-1]-r.index[0]).days/365.25;ppy=len(r)/yrs
 else:yrs=len(r)/ppy
 eq=(1+r).cumprod();c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;neg=r[r<0].std()*np.sqrt(ppy);so=r.mean()*ppy/neg;m=(eq/eq.cummax()-1).min();return {'CAGR':float(c),'Vol':float(v),'Sharpe':float(sh),'Sortino':float(so),'MDD':float(m),'Calmar':float(c/abs(m))}
def allstrats(R,exact=True,p=BASE,qseries=BASEQS):return {float(f):sim(R,float(f),p,qseries,exact) for f in GRID}
def table(Z,ppy=None):return {f:met(r,ppy) for f,r in Z.items()}
def best(M,key):return float(max(M,key=lambda f:M[f][key]))
def plateau(M,key,rel=.98):
 b=max(x[key] for x in M.values()); a=[f for f,x in M.items() if x[key]>=b*rel];return [float(min(a)),float(max(a))]
def weekly(Z):return (1+pd.concat(Z,axis=1)).resample('W-FRI').prod()-1
def mean_tests(R):
 cases={'original':R.copy()};avg=(R.gold.mean()+R.mf.mean())/2;E=R.copy();E.gold+=avg-R.gold.mean();E.mf+=avg-R.mf.mean();cases['equal_mean']=E
 C=R.copy();C.gold+=R.cash.mean()-R.gold.mean();C.mf+=R.cash.mean()-R.mf.mean();cases['both_cash_mean']=C
 S=R.copy();mg,mm=R.gold.mean(),R.mf.mean();S.gold+=mm-mg;S.mf+=mg-mm;cases['swap_means']=S;out={}
 for n,X in cases.items():
  M=table(allstrats(X,True));out[n]={'ann_means':{'gold':float(X.gold.mean()*252),'mf':float(X.mf.mean()*252),'cash':float(X.cash.mean()*252)},'bestS':best(M,'Sharpe'),'bestC':best(M,'Calmar'),'S98':plateau(M,'Sharpe'),'C98':plateau(M,'Calmar'),'m50':M[.5],'m70':M[.7],'m80':M[.8]}
 return out
def walk(W):
 os=[];oc=[];f50=[];f70=[];cs=[];cc=[]
 for y in range(2016,2027):
  tr=W[(W.index>=f'{y-5}-01-01')&(W.index<f'{y}-01-01')];te=W[(W.index>=f'{y}-01-01')&(W.index<f'{y+1}-01-01')]
  if len(tr)<150 or len(te)<10:continue
  M={f:met(tr[f],52) for f in W.columns};s=best(M,'Sharpe');c=best(M,'Calmar');cs.append([y,s]);cc.append([y,c]);os.append(te[s]);oc.append(te[c]);f50.append(te[.5]);f70.append(te[.7])
 J=lambda a:pd.concat(a)
 return {'choicesS':cs,'choicesC':cc,'dynamicS':met(J(os),52),'dynamicC':met(J(oc),52),'fixed50':met(J(f50),52),'fixed70':met(J(f70),52)}
def pbo(W,k=8):
 n=len(W);ed=np.linspace(0,n,k+1,dtype=int);bl=[np.arange(ed[i],ed[i+1]) for i in range(k)];lg=[];sel=[]
 for co in itertools.combinations(range(k),k//2):
  I=np.concatenate([bl[i] for i in co]);O=np.concatenate([bl[i] for i in range(k) if i not in co]);Mi={f:met(W.iloc[I][f],52)['Sharpe'] for f in W};b=max(Mi,key=Mi.get);Mo={f:met(W.iloc[O][f],52)['Sharpe'] for f in W};ordr=sorted(Mo,key=Mo.get);rank=(ordr.index(b)+1)/(len(ordr)+1);lg.append(math.log(rank/(1-rank)));sel.append(float(b))
 return {'splits':len(lg),'PBO':float(np.mean(np.array(lg)<0)),'selected_med':float(np.median(sel)),'selected_q25q75':[float(x) for x in np.quantile(sel,[.25,.75])]}
def boot(W,a,b,block,n=5000,years=10,seed=1):
 rng=np.random.default_rng(seed);X=W[[a,b]].dropna().values;N=len(X);L=52*years;st=np.arange(N-block+1);win=np.zeros(4);D=[]
 for _ in range(n):
  ids=[]
  while len(ids)<L:
   s=int(rng.choice(st));ids.extend(range(s,s+block))
  Y=X[np.array(ids[:L])];A=met(pd.Series(Y[:,0]),52);B=met(pd.Series(Y[:,1]),52);vals=[A['CAGR']-B['CAGR'],A['Sharpe']-B['Sharpe'],A['Calmar']-B['Calmar'],A['MDD']-B['MDD']];D.append(vals);win+=np.array(vals)>0
 D=np.array(D);return {'a':a,'b':b,'block':block,'n':n,'P_a_better':{'CAGR':float(win[0]/n),'Sharpe':float(win[1]/n),'Calmar':float(win[2]/n),'MDD':float(win[3]/n)},'d_median':[float(x) for x in np.median(D,axis=0)],'d_p05p95':[[float(x) for x in np.quantile(D[:,j],[.05,.95])] for j in range(4)]}
def perturb(R):
 s=[];c=[]
 # broad 81 parameter neighborhoods; vectorized target return approximation intentionally used only for stability check
 for ta in [.15,.20,.25]:
  for wi in [10,16,32]:
   for ma in [150,200,250]:
    for inc in [.10,.15,.20]:
     p=BASE.copy();p.update(target=ta,win=wi,gate_ma=ma,inc=inc);q=qs(p);M={}
     for f in np.round(np.arange(.4,.901,.05),2):M[float(f)]=met(sim(R,float(f),p,q,False))
     s.append(best(M,'Sharpe'));c.append(best(M,'Calmar'))
 def sm(x):return {'n':len(x),'median':float(np.median(x)),'q10q90':[float(v) for v in np.quantile(x,[.1,.9])],'share_60_80':float(np.mean((np.array(x)>=.6)&(np.array(x)<=.8)))}
 return {'Sharpe':sm(s),'Calmar':sm(c)}
def regimes(Z):
 ps={'2010_14':('2010','2015'),'2015_19':('2015','2020'),'2020_22':('2020','2023'),'2023_26':('2023','2027'),'gold_bear_2013_15':('2013','2016'),'covid':('2020-02','2020-06'),'inflation22':('2022','2023')};o={}
 for n,(a,b) in ps.items():o[n]={str(f):met(r[(r.index>=a)&(r.index<b)]) for f,r in Z.items() if ((r.index>=a)&(r.index<b)).sum()>5}
 return o
sets=[('AQMIX','2010-10-04',True,'BIL'),('WTMF','2011-01-06',True,'BIL'),('RYMTX','2007-05-31',False,'BIL'),('DBMF','2020-06-02',True,'SGOV')]
for mf,st,act,ca in sets:
 R=returns(st,mf,act,ca);Z=allstrats(R,True);M=table(Z);print('PROXY',mf);print('BEST',json.dumps({'S':best(M,'Sharpe'),'C':best(M,'Calmar'),'So':best(M,'Sortino'),'CAGR':best(M,'CAGR'),'S98':plateau(M,'Sharpe'),'C98':plateau(M,'Calmar')}));print('KEY',json.dumps({str(f):M[f] for f in [.5,.6,.7,.75,.8,.9]}))
 if mf=='AQMIX':
  W=weekly(Z);print('MEAN',json.dumps(mean_tests(R)));print('PERTURB',json.dumps(perturb(R)));print('WALK',json.dumps(walk(W)));print('PBO',json.dumps(pbo(W)));print('BOOT4_70_50',json.dumps(boot(W,.7,.5,4,5000,10,21)));print('BOOT13_70_50',json.dumps(boot(W,.7,.5,13,5000,10,22)));print('BOOT13_70_80',json.dumps(boot(W,.7,.8,13,5000,10,23)));print('REG',json.dumps(regimes({f:Z[f] for f in [.5,.7,.8]})))
