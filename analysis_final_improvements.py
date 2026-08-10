import json, numpy as np, pandas as pd, yfinance as yf
BASE=dict(target=.20,win=16,floor=.20,cap=1.,inc=.15,dead=.05,hcap=.60,band=.05,ma=200,gm=.50)
TCOST=.001
T={'q':'QLD','spy':'SPY','fx':'KRW=X','gld':'GLD','gh':'132030.KS','bil':'BIL','sgov':'SGOV','aq':'AQMIX','wt':'WTMF','ry':'RYMTX','db':'DBMF'}
def dl(t):
 x=yf.download(t,start='2006-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x.Close.dropna().astype(float)
x={k:dl(v) for k,v in T.items()};sig=pd.concat({'q':x['q'],'spy':x['spy']},axis=1).dropna()
def qs(p,nogate=False):
 r=sig.q.pct_change();v=r.rolling(p['win']).std()*np.sqrt(252);te=np.clip((p['target']/v).values,p['floor'],p['cap']);cur=0.;a=[]
 for z in te:
  if np.isnan(z):a.append(cur);continue
  if z<cur:cur=z
  elif z-cur>p['inc']:cur=z
  a.append(cur)
 ma=sig.spy.rolling(p['ma']).mean();scale=np.ones(len(sig)) if nogate else np.where((sig.spy<ma).values,p['gm'],1.);gg=np.array(a)*scale;last=gg[0];out=[]
 for z in gg:
  if abs(z-last)>p['dead']:last=z
  out.append(last)
 return pd.Series(out,index=sig.index+pd.Timedelta(days=1))
def Rframe(start,mf,cash,actual=False):
 keys=['q','fx','gld',mf,cash]+(['gh'] if actual else []);idx=pd.DatetimeIndex(sorted(set().union(*[set(x[k].index) for k in keys])));idx=idx[idx>=pd.Timestamp(start)]
 ff=lambda k:x[k].reindex(idx).ffill();fx=ff('fx');P=pd.DataFrame({'q':ff('q'),'gold':ff('gh') if actual else ff('gld'),'mf':ff(mf)*fx,'cash':ff(cash)*fx},index=idx).dropna();return P.pct_change().dropna()
def sim(R,p,nogate=False):
 q=qs(p,nogate).reindex(R.index,method='ffill').dropna();R=R.loc[q.index];h=np.minimum(p['hcap'],np.maximum(0,1-q.values));W=np.c_[q.values,.5*h,.5*h,np.maximum(0,1-q.values-h)];A=R[['q','gold','mf','cash']].values;wc=W[0].copy();nav=1.;rows=[];turn=0.;nt=0
 for i,d in enumerate(R.index):
  wt=W[i];cost=0.
  if np.max(np.abs(wc-wt))>p['band']:
   to=np.sum(np.abs(wt-wc));cost=TCOST*to;turn+=to;nt+=1;wc=wt.copy()
  gr=float(wc@A[i]);net=(1-cost)*(1+gr)-1;nav*=1+net;en=wc*(1+A[i]);wc=en/en.sum();rows.append((d,net,nav))
 z=pd.DataFrame(rows,columns=['d','r','nav']).set_index('d');z.attrs['turn']=turn;z.attrs['nt']=nt;return z
def met(z):
 yrs=(z.index[-1]-z.index[0]).days/365.25;ppy=len(z)/yrs;r=z.r;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;neg=r[r<0].std()*np.sqrt(ppy);m=(eq/eq.cummax()-1).min();return {'CAGR':float(c),'Vol':float(v),'Sharpe':float(sh),'Sortino':float(r.mean()*ppy/neg),'MDD':float(m),'Calmar':float(c/abs(m)),'TradesYr':float(z.attrs['nt']/yrs),'TurnYr':float(z.attrs['turn']/yrs)}
def cfg(**kw):p=BASE.copy();p.update(kw);return p
SETS=[('AQ','2010-10-04','aq','bil',False),('WT','2011-01-06','wt','bil',False),('RY','2007-05-31','ry','bil',False),('DB','2020-06-02','db','sgov',False),('AQ_ACT','2010-10-04','aq','bil',True),('DB_ACT','2020-06-02','db','sgov',True)]
CANDS={'BASE':(cfg(),False)}
for b in [.06,.07,.075,.08,.09,.10]:CANDS[f'BAND_{b:.3f}']=(cfg(band=b),False)
for h in [.65,.70,.75,.80]:CANDS[f'HCAP_{h:.2f}']=(cfg(hcap=h),False)
for h in [.65,.70,.75]:
 for b in [.07,.075,.08]:CANDS[f'H{h:.2f}_B{b:.3f}']=(cfg(hcap=h,band=b),False)
CANDS['NO_GATE']=(cfg(),True);CANDS['GATE075']=(cfg(gm=.75),False);CANDS['GATE025']=(cfg(gm=.25),False)
RES={}
for lab,st,mf,ca,act in SETS:
 R=Rframe(st,mf,ca,act);RES[lab]={};print('\nSET',lab)
 for n,(p,ng) in CANDS.items():
  m=met(sim(R,p,ng));RES[lab][n]=m;print(n,json.dumps(m))
for group,ss in [('CORE',['AQ','WT','RY','DB']),('ACTUALG',['AQ_ACT','DB_ACT'])]:
 print('\nAGG',group)
 for n in CANDS:
  ds={k:float(np.mean([RES[s][n][k]-RES[s]['BASE'][k] for s in ss])) for k in ['CAGR','Vol','Sharpe','MDD','Calmar','TradesYr','TurnYr']};print(n,json.dumps(ds))
FIN=['BASE','BAND_0.070','BAND_0.075','BAND_0.080','HCAP_0.70','H0.70_B0.075','NO_GATE']
for label,actual in [('LOYO_STRUCT',False),('LOYO_ACTUALG',True)]:
 R=Rframe('2010-10-04','aq','bil',actual);print('\n',label);full={n:sim(R,CANDS[n][0],CANDS[n][1]) for n in FIN};years=sorted(set(full['BASE'].index.year))
 for n in FIN:
  winsS=winsC=0;ds=[]
  for y in years:
   def sm(z):
    q=z[z.index.year!=y]; rr=q.r; eq=(1+rr).cumprod(); yrs=(q.index[-1]-q.index[0]).days/365.25;ppy=len(q)/yrs;c=eq.iloc[-1]**(1/yrs)-1;v=rr.std()*np.sqrt(ppy);m=(eq/eq.cummax()-1).min();return c,rr.mean()*ppy/v,c/abs(m)
   a=sm(full[n]);b=sm(full['BASE']);winsS+=int(a[1]>b[1]);winsC+=int(a[2]>b[2]);ds.append([a[i]-b[i] for i in range(3)])
  arr=np.array(ds);print(n,json.dumps({'n':int(len(years)),'SharpeWin':int(winsS),'CalmarWin':int(winsC),'median_dCAGR':float(np.median(arr[:,0])),'median_dSharpe':float(np.median(arr[:,1])),'median_dCalmar':float(np.median(arr[:,2]))}))
