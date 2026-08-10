import numpy as np, pandas as pd, yfinance as yf
TARGET,WIN,FLOOR,CAP,INC=0.20,16,0.20,1.0,0.15
DEADBAND=0.05; GOLD_FRAC=DBMF_FRAC=0.5; H_CAP=0.60; BAND=0.05; GATE_MA=200; GATE_MULT=0.5; TCOST=0.001
T={'QLD':'QLD','QLD_H':'409820.KS','DBMF':'DBMF','SGOV':'SGOV','SPY':'SPY','FX':'KRW=X','GOLD_H':'132030.KS','ACE':'411060.KS','TIGER':'0072R0.KS'}
def dl(t):
 x=yf.download(t,start='2021-01-01',auto_adjust=True,progress=False)
 if isinstance(x.columns,pd.MultiIndex): x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
raw={k:dl(v) for k,v in T.items()}
us=pd.concat({'QLD':raw['QLD'],'SPY':raw['SPY']},axis=1).dropna(); rq=us.QLD.pct_change(); vol=rq.rolling(WIN).std()*np.sqrt(252); te=np.clip((TARGET/vol).values,FLOOR,CAP)
cur=0.; a=[]
for x in te:
 if np.isnan(x): a.append(cur); continue
 if x<cur: cur=x
 elif x-cur>INC: cur=x
 a.append(cur)
ma=us.SPY.rolling(GATE_MA).mean(); gated=np.array(a)*np.where((us.SPY<ma).values,GATE_MULT,1.0); last=gated[0]; w=[]
for g in gated:
 if abs(g-last)>DEADBAND:last=g
 w.append(last)
ww=[]
for q in w:
 hb=min(H_CAP,max(0,1-q)); ww.append((q,.5*hb,.5*hb,max(0,1-q-hb)))
tar=pd.DataFrame(ww,index=us.index+pd.Timedelta(days=1),columns=['q','gold','dbmf','cash'])

def returns(spot,start,qhedged):
 keys=['QLD','QLD_H','DBMF','SGOV','FX','GOLD_H',spot]
 idx=pd.DatetimeIndex(sorted(set().union(*[set(raw[k].index) for k in keys]))); idx=idx[idx>=pd.Timestamp(start)]
 def ff(k):return raw[k].reindex(idx).ffill()
 fx=ff('FX'); q=ff('QLD_H') if qhedged else ff('QLD')*fx
 P=pd.DataFrame({'q':q,'spot':ff(spot),'h':ff('GOLD_H'),'dbmf':ff('DBMF')*fx,'cash':ff('SGOV')*fx},index=idx).dropna()
 return P.pct_change().dropna()

def sim(spot,start,qhedged,use_h):
 R=returns(spot,start,qhedged); R['gold']=R['h'] if use_h else R['spot']; R=R[['q','gold','dbmf','cash']]
 W=tar.reindex(R.index,method='ffill').dropna(); R=R.loc[W.index]
 nav=1.; wc=W.iloc[0].values; out=[]
 for dt in R.index:
  wt=W.loc[dt].values
  if np.max(np.abs(wc-wt))>BAND:
   nav*=1-TCOST*np.sum(np.abs(wt-wc)); wc=wt.copy()
  rr=R.loc[dt].values; pr=float(wc@rr); nav*=1+pr; gross=wc*(1+rr); wc=gross/gross.sum(); out.append((dt,pr,nav))
 return pd.DataFrame(out,columns=['d','r','nav']).set_index('d')
def met(z):
 yrs=(z.index[-1]-z.index[0]).days/365.25; ppy=len(z)/yrs; r=z.r; c=z.nav.iloc[-1]**(1/yrs)-1; v=r.std()*np.sqrt(ppy); m=(z.nav/z.nav.cummax()-1).min(); sh=r.mean()*ppy/v
 return c,v,sh,m,c/abs(m),z.nav.iloc[-1]
def rep(label,spot,start):
 print('\n',label)
 for qh in [False,True]:
  s=sim(spot,start,qh,False); h=sim(spot,start,qh,True); print('QH',qh,'SPOT',met(s),'H',met(h))
rep('ACE_2021','ACE','2021-12-16')
rep('TIGER_2025','TIGER','2025-06-25')
