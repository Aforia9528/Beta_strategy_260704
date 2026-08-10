import json, contextlib, io, numpy as np, pandas as pd
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_threshold_cadence as a
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']

def overlay_phase(s,b,th,offset,load=1.25):
 base=a.p.Z[s];idx,V,starts,tr=a.PRE[s];allowed_starts=set(starts[offset:].tolist());bc=1.;prev=1.;active=[];rows=[];sp=0.;nb=0;last=None
 for i,dt in enumerate(idx):
  bc*=1+float(base.ret.iloc[i]);ov=0.;live=[]
  for j,end,q in active:
   if i>=end:bc+=q*tr[j][1][end-j]
   else:ov+=q*tr[j][1][i-j];live.append((j,end,q))
  active=live;nav=bc+ov
  eligible=(i in allowed_starts and float(V.iloc[i])<=th/100. and (last is None or (dt-last).days>=80))
  if eligible:
   spend=b/4*nav;end,vals=tr[i];f=float(vals[0])
   if f>1e-12 and bc>spend:
    q=spend/(f*load);bc-=spend;ov+=q*f;active.append((i,end,q));nav=bc+ov;sp+=spend/max(nav,1e-12);nb+=1;last=dt
  rows.append((dt,nav/prev-1,nav));prev=nav
 z=pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs.update(SpendYr=sp/yrs,BuysYr=nb/yrs);return z
for th in [30,32.5]:
 for off in [0,1,2]:
  vals=[]
  for s in SETS:
   z=overlay_phase(s,.005,th,off);m=a.met(z);bm=a.BM[s];vals.append({'dCAGR':m['CAGR']-bm['CAGR'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar'],'SpendYr':m['SpendYr'],'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar']})
  print('AVG',json.dumps({'Threshold':th,'OffsetMonths':off,**{k:float(np.mean([x[k] for x in vals])) for k in vals[0]}}))
