import json, math, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_production_summary_dca as p
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];B=[.00375,.005,.00625,.0075];TH=[20,25,30,35,40,999]
def dl(t,adj=True):
 x=yf.download(t,start='2006-01-01',auto_adjust=adj,progress=False);x.columns=x.columns.get_level_values(0) if isinstance(x.columns,pd.MultiIndex) else x.columns;return x['Close'].dropna().astype(float)
QQQ=dl('QQQ',False);VXN=dl('^VXN')/100.;IRX=dl('^IRX')/100.
def put(s,k,t,r,sg,q=.005):
 if t<=1e-8:return max(k-s,0.)
 sq=math.sqrt(t);d1=(math.log(s/k)+(r-q+.5*sg*sg)*t)/(sg*sq);d2=d1-sg*sq;return k*math.exp(-r*t)*norm.cdf(-d2)-s*math.exp(-q*t)*norm.cdf(-d1)
def spr(s,k1,k2,t,r,sg):return max(0.,put(s,k1,t,r,sg)-put(s,k2,t,r,sg))
def prep(base):
 idx=base.index;n=len(idx);S=QQQ.reindex(idx).ffill().bfill();V=VXN.reindex(idx).ffill().bfill().clip(.08,1.5);RF=IRX.reindex(idx).ffill().bfill().clip(-.01,.10);months=idx.to_period('M');starts=np.where(np.r_[True,months[1:]!=months[:-1]])[0];tr={}
 for j in starts:
  dt=idx[j];s0=float(S.iloc[j]);k1=.9*s0;k2=.7*s0;e=dt+pd.DateOffset(months=3);end=min(int(np.searchsorted(idx.values,np.datetime64(e),side='left')),n-1);vals=[]
  for i in range(j,end+1):
   s=float(S.iloc[i]);v=float(V.iloc[i]);r=float(RF.iloc[i]);sg=.75*v+.25*.20;val=(max(k1-s,0)-max(k2-s,0)) if (i==end or idx[i]>=e) else spr(s,k1,k2,max((e-idx[i]).days/365.25,1/365.25),r,sg);vals.append(val)
  tr[j]=(end,np.asarray(vals))
 return idx,V,starts,tr
PRE={s:prep(p.Z[s]) for s in SETS}
def overlay(s,b,th,cadence='MONTHLY',load=1.25):
 base=p.Z[s];idx,V,starts,tr=PRE[s];startset=set(starts.tolist());bc=1.;prev=1.;active=[];rows=[];sp=0.;nb=0;last_buy=None
 for i,dt in enumerate(idx):
  bc*=1+float(base.ret.iloc[i]);ov=0.;live=[]
  for j,end,q in active:
   if i>=end:bc+=q*tr[j][1][end-j]
   else:ov+=q*tr[j][1][i-j];live.append((j,end,q))
  active=live;nav=bc+ov
  eligible=(i in startset and float(V.iloc[i])<=th/100.)
  if eligible and cadence=='QUARTERLY': eligible=(last_buy is None or (dt-last_buy).days>=80)
  if eligible:
   spend=(b/12 if cadence=='MONTHLY' else b/4)*nav;end,vals=tr[i];f=float(vals[0])
   if f>1e-12 and bc>spend:
    q=spend/(f*load);bc-=spend;ov+=q*f;active.append((i,end,q));nav=bc+ov;sp+=spend/max(nav,1e-12);nb+=1;last_buy=dt
  rows.append((dt,nav/prev-1,nav));prev=nav
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs.update(SpendYr=sp/yrs,BuysYr=nb/yrs);return z
def met(z):
 r=z.ret;yrs=(z.index[-1]-z.index[0]).days/365.25;ppy=len(r)/yrs;eq=z.nav;c=eq.iloc[-1]**(1/yrs)-1;v=r.std()*np.sqrt(ppy);sh=r.mean()*ppy/v;dd=(eq/eq.cummax()-1).min();cal=c/abs(dd);w20=((1+r).rolling(20).apply(np.prod,raw=True)-1).min();return dict(CAGR=float(c),Sharpe=float(sh),MDD=float(dd),Calmar=float(cal),Worst20D=float(w20),SpendYr=z.attrs.get('SpendYr',0),BuysYr=z.attrs.get('BuysYr',0))
def yret(z,y):
 r=z.loc[z.index.year==y,'ret'];return float((1+r).prod()-1) if len(r) else None
BM={s:met(p.Z[s]) for s in SETS};rows=[]
for cad in ['MONTHLY','QUARTERLY']:
 for th in TH:
  for b in B:
   vals=[]
   for s in SETS:
    z=overlay(s,b,th,cad);m=met(z);bm=BM[s];x={'Set':s,'Cadence':cad,'Threshold':th,'Budget':b,**m,'dCAGR':m['CAGR']-bm['CAGR'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']};rows.append(x);vals.append(x)
   av={k:float(np.mean([x[k] for x in vals])) for k in ['CAGR','Sharpe','MDD','Calmar','Worst20D','SpendYr','BuysYr','dCAGR','dSharpe','dMDD','dCalmar']};print('AVG',json.dumps({'Cadence':cad,'Threshold':th,'Budget':b,**av}))
df=pd.DataFrame(rows);ag=df.groupby(['Cadence','Threshold','Budget']).mean(numeric_only=True).reset_index();print('\nTOP')
for _,r in ag.sort_values('dCalmar',ascending=False).head(20).iterrows():print(json.dumps(r[['Cadence','Threshold','Budget','SpendYr','BuysYr','dCAGR','dSharpe','dMDD','dCalmar']].to_dict()))
print('\nCRISIS_RY')
for th,b,cad in [(999,.005,'MONTHLY'),(25,.005,'MONTHLY'),(30,.005,'MONTHLY'),(35,.005,'MONTHLY'),(30,.0075,'MONTHLY'),(30,.005,'QUARTERLY')]:
 z=overlay('RY',b,th,cad);print(json.dumps({'Threshold':th,'Budget':b,'Cadence':cad,'SpendYr':z.attrs['SpendYr'],'Y2008':yret(z,2008),'Y2020':yret(z,2020),'Y2022':yret(z,2022),**met(z)}))
