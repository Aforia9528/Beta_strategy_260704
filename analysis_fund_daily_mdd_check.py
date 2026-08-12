import json
import numpy as np
import pandas as pd
import yfinance as yf
T=['QNZNX','QDSIX','QLEIX','ADAIX','AUEIX','BLNDX','REMIX','DBMF','QAI','PHDG','GDE','RSST','RSSB']
END='2026-07-01'
def dl(t):
 x=yf.download(t,start='2006-01-01',end=END,auto_adjust=True,progress=False,threads=False)
 if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
 return x['Close'].dropna().astype(float)
for t in T:
 try:px=dl(t)
 except:continue
 if len(px)<20:continue
 r=px.pct_change().dropna();yrs=(px.index[-1]-px.index[0]).days/365.25;eq=px/px.iloc[0];c=float(eq.iloc[-1]**(1/yrs)-1);dd=eq/eq.cummax()-1;m=float(dd.min());tr=dd.idxmin();pk=eq.loc[:tr].idxmax();aft=eq.loc[tr:];rec=aft[aft>=eq.loc[pk]];recdate=str(rec.index[0].date()) if len(rec) else None
 print(json.dumps({'Ticker':t,'Years':yrs,'CAGR_daily':c,'MDD_daily':m,'Calmar_daily':c/abs(m),'Peak':str(pk.date()),'Trough':str(tr.date()),'Recovery':recdate}))
