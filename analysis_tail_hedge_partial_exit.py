import json, contextlib, io, numpy as np
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_threshold_cadence as a
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT'];RULES=['HOLD','X2_HALF','X3_HALF','VXN40_HALF','X3_FULL']
def overlay(s,rule,load=1.25):
 base=a.p.Z[s];idx,V,starts,tr=a.PRE[s];startset=set(starts.tolist());bc=1.;prev=1.;active=[];rows=[];sp=0.;last=None
 for i,dt in enumerate(idx):
  bc*=1+float(base.ret.iloc[i]);ov=0.;live=[]
  for j,end,q,f,done in active:
   cur=float(tr[j][1][min(i-j,end-j)])
   if i>=end:bc+=q*cur;continue
   trigger=(rule=='X2_HALF' and cur>=2*f) or (rule in ['X3_HALF','X3_FULL'] and cur>=3*f) or (rule=='VXN40_HALF' and float(V.iloc[i])>=.40 and cur>=1.25*f)
   if trigger and not done:
    frac=1.0 if rule=='X3_FULL' else .5;sell=q*frac;bc+=sell*cur;q-=sell;done=True
   if q>1e-12:ov+=q*cur;live.append((j,end,q,f,done))
  active=live;nav=bc+ov
  eligible=(i in startset and float(V.iloc[i])<=.30 and (last is None or (dt-last).days>=80))
  if eligible:
   spend=.005/4*nav;end,vals=tr[i];f=float(vals[0])
   if f>1e-12 and bc>spend:
    q=spend/(f*load);bc-=spend;ov+=q*f;active.append((i,end,q,f,False));nav=bc+ov;sp+=spend/max(nav,1e-12);last=dt
  rows.append((dt,nav/prev-1,nav));prev=nav
 z=a.pd.DataFrame(rows,columns=['date','ret','nav']).set_index('date');yrs=(idx[-1]-idx[0]).days/365.25;z.attrs['SpendYr']=sp/yrs;return z
for rule in RULES:
 vals=[]
 for s in SETS:
  z=overlay(s,rule);m=a.met(z);bm=a.BM[s];x={'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'SpendYr':m['SpendYr'],'dCAGR':m['CAGR']-bm['CAGR'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']};vals.append(x);print('SET',rule,s,json.dumps(x))
 print('AVG',rule,json.dumps({k:float(np.mean([x[k] for x in vals])) for k in vals[0]}))
