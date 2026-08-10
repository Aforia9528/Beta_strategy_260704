import json, contextlib, io, numpy as np
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_threshold_cadence as a
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];LOAD=1.25
def overlay(s,multiple):
 base=a.p.Z[s];idx,V,starts,tr=a.PRE[s];startset=set(starts.tolist());bc=1.;prev=1.;active=[];rows=[];last=None;sp=0.
 for i,dt in enumerate(idx):
  bc*=1+float(base.ret.iloc[i]);ov=0.;live=[]
  for j,end,q,f,done in active:
   cur=float(tr[j][1][min(i-j,end-j)])
   if i>=end:bc+=q*cur;continue
   if not done and cur>=multiple*(LOAD*f):
    sell=.5*q;bc+=sell*cur;q-=sell;done=True
   ov+=q*cur;live.append((j,end,q,f,done))
  active=live;nav=bc+ov
  if i in startset and float(V.iloc[i])<=.30 and (last is None or (dt-last).days>=80):
   spend=.005/4*nav;end,vals=tr[i];f=float(vals[0])
   if f>1e-12 and bc>spend:
    q=spend/(LOAD*f);bc-=spend;ov+=q*f;active.append((i,end,q,f,False));nav=bc+ov;sp+=spend/max(nav,1e-12);last=dt
  rows.append((dt,nav/prev-1,nav));prev=nav
 z=a.pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs['SpendYr']=sp/yrs;return z
for mult in [2.,2.5,3.]:
 vals=[]
 for s in SETS:
  z=overlay(s,mult);m=a.met(z);bm=a.BM[s];vals.append({'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'dCAGR':m['CAGR']-bm['CAGR'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']})
 print('AVG',mult,json.dumps({k:float(np.mean([x[k] for x in vals])) for k in vals[0]}))
