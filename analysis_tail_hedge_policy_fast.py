# Fast independent policy screen: longest history + actual Gold(H) long + actual DBMF/Gold(H)
import json, contextlib, io
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_tail_hedge_policy_grid as a
for s in ['RY','AQ_ACT','DB_ACT']:
    mb=a.baseM[s];print('\nSET',s,'BASE',json.dumps(mb))
    for mode in ['ALWAYS','QPROP','QGE50','VXN30','Q50_VXN30','CHEAP_TIER']:
        for b in [.0025,.005,.0075,.01]:
            z=a.overlay(a.p.Z[s],b,mode);m=a.metric(z)
            print(json.dumps({'Mode':mode,'Budget':b,'SpendYr':m['SpendYr'],'CAGR':m['CAGR'],'Sharpe':m['Sharpe'],'MDD':m['MDD'],'Calmar':m['Calmar'],'Worst20D':m['Worst20D'],'CVaR':m['CVaR1D_1pct'],'dCAGR':m['CAGR']-mb['CAGR'],'dSharpe':m['Sharpe']-mb['Sharpe'],'dMDD':m['MDD']-mb['MDD'],'dCalmar':m['Calmar']-mb['Calmar'],'dWorst20D':m['Worst20D']-mb['Worst20D']}))
