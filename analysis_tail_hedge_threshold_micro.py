import json, contextlib, io, numpy as np
buf=io.StringIO()
with contextlib.redirect_stdout(buf): import analysis_tail_hedge_threshold_cadence as a
SETS=['RY','AQ','WT','DB','AQ_ACT','DB_ACT']
for cad in ['MONTHLY','QUARTERLY']:
 for th in [27.5,30,32.5]:
  vals=[]
  for s in SETS:
   z=a.overlay(s,.005,th,cad);m=a.met(z);bm=a.BM[s];x={'Set':s,'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'SpendYr':m['SpendYr'],'dCAGR':m['CAGR']-bm['CAGR'],'dSharpe':m['Sharpe']-bm['Sharpe'],'dMDD':m['MDD']-bm['MDD'],'dCalmar':m['Calmar']-bm['Calmar']};vals.append(x);print('SET',cad,th,json.dumps(x))
  print('AVG',cad,th,json.dumps({k:float(np.mean([x[k] for x in vals])) for k in ['CAGR','Sharpe','MDD','Calmar','SpendYr','dCAGR','dSharpe','dMDD','dCalmar']}))
