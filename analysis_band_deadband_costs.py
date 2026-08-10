import json,numpy as np,pandas as pd,yfinance as yf
T={'q':'QLD','spy':'SPY','fx':'KRW=X','g':'GLD','gh':'132030.KS','bil':'BIL','sgov':'SGOV','aq':'AQMIX','wt':'WTMF','ry':'RYMTX','db':'DBMF'}
def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x.Close.dropna().astype(float)
x={k:dl(v) for k,v in T.items()};sig=pd.concat({'q':x.q if hasattr(x,'q') else x['q'],'spy':x['spy']},axis=1).dropna() if False else pd.concat({'q':x['q'],'spy':x['spy']},axis=1).dropna()
def qs(dead):
 r=sig.q.pct_change();v=r.rolling(16).std()*np.sqrt(252);te=np.clip((.20/v).values,.20,1.);cur=0.;a=[]
 for z in te:
  if np.isnan(z):a.append(cur);continue
  if z<cur:cur=z
  elif z-cur>.15:cur=z
  a.append(cur)
 ma=sig.spy.rolling(200).mean();gg=np.array(a)*np.where((sig.spy<ma).values,.5,1.);last=gg[0];o=[]
 for z in gg:
  if abs(z-last)>dead:last=z
  o.append(last)
 return pd.Series(o,index=sig.index+pd.Timedelta(days=1))
def rf(st,mf,ca,act=False):
 keys=['q','fx','g',mf,ca]+(['gh'] if act else []);idx=pd.DatetimeIndex(sorted(set().union(*[set(x[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(st)];ff=lambda k:x[k].reindex(idx).ffill();fx=ff('fx');P=pd.DataFrame({'q':ff('q'),'g':ff('gh') if act else ff('g'),'m':ff(mf)*fx,'c':ff(ca)*fx},index=idx).dropna();return P.pct_change().dropna()
def sim(R,dead,band,cost):
 q=qs(dead).reindex(R.index,method='ffill').dropna();R=R.loc[q.index];h=np.minimum(.6,np.maximum(0,1-q.values));W=np.c_[q.values,.5*h,.5*h,np.maximum(0,1-q.values-h)];A=R[['q','g','m','c']].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,d in enumerate(R.index):
  wt=W[i];cc=0
  if np.max(np.abs(wc-wt))>band:
   to=np.sum(np.abs(wt-wc));cc=cost*to;turn+=to;nt+=1;wc=wt.copy()
  gr=wc@A[i];net=(1-cc)*(1+gr)-1;nav*=1+net;en=wc*(1+A[i]);wc=en/en.sum();rows.append((d,net,nav))
 z=pd.DataFrame(rows,columns=['d','r','nav']).set_index('d');yrs=(z.index[-1]-z.index[0]).days/365.25;ppy=len(z)/yrs;r=z.r;c=z.nav.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);m=(z.nav/z.nav.cummax()-1).min();return {'CAGR':float(c),'Sharpe':float(r.mean()*ppy/v),'MDD':float(m),'Calmar':float(c/abs(m)),'TradesYr':float(nt/yrs),'TurnYr':float(turn/yrs)}
SETS=[('AQ','2010-10-04','aq','bil',False),('WT','2011-01-06','wt','bil',False),('RY','2007-05-31','ry','bil',False),('DB','2020-06-02','db','sgov',False),('AQ_ACT','2010-10-04','aq','bil',True),('DB_ACT','2020-06-02','db','sgov',True)]
for cost in [.001,.0025,.005]:
 print('\nCOST',cost)
 RES={}
 for lab,st,mf,ca,act in SETS:
  R=rf(st,mf,ca,act);RES[lab]={};print('SET',lab)
  for de in [.05,.075,.10]:
   for ba in [.05,.075]:
    n=f'D{de:.3f}_B{ba:.3f}';m=sim(R,de,ba,cost);RES[lab][n]=m;print(n,json.dumps(m))
 core=['AQ','WT','RY','DB'];base='D0.050_B0.050';print('AGG_CORE')
 for n in RES['AQ']:
  print(n,json.dumps({k:float(np.mean([RES[s][n][k]-RES[s][base][k] for s in core])) for k in ['CAGR','Sharpe','MDD','Calmar','TradesYr','TurnYr']}))
