import json, contextlib, io
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

END='2026-07-01'
AQR=['ADAIX','QMNIX','QGMIX','QMHIX','AQRIX','QSPIX']
STRAT_W=np.array([.13,.13,.13,.13,.33,.13,.02])
CURR_W=np.array([.13,.17,.12,.13,.30,.13,.02])
ALL=AQR+['BIL','QDSIX']
ETF_UNIVERSE=['MNA','QAI','BTAL','PHDG','DBMF','KMLM','CTA','FMF','GMOM','RPAR','UPAR','QIS','GDE','RSST','RSSB']

def dl(t):
    try:
        x=yf.download(t,start='2008-01-01',end=END,auto_adjust=True,progress=False,threads=False)
        if isinstance(x.columns,pd.MultiIndex):x.columns=x.columns.get_level_values(0)
        return x['Close'].dropna().astype(float)
    except Exception:
        return pd.Series(dtype=float)

def mret(px):return px.resample('ME').last().pct_change().dropna()
def dret(px):return px.pct_change().dropna()
PX={t:dl(t) for t in ALL+ETF_UNIVERSE}
MR={t:mret(PX[t]) for t in PX if len(PX[t])>10}

# Risk-free proxy for excess Sharpe.
def met(r):
    r=pd.Series(r).dropna()
    if len(r)<12:return None
    yrs=(r.index[-1]-r.index[0]).days/365.25;eq=(1+r).cumprod();c=float(eq.iloc[-1]**(1/yrs)-1);vol=float(r.std(ddof=1)*np.sqrt(12));rf=MR['BIL'].reindex(r.index).ffill().bfill();ex=r-rf;sh=float(ex.mean()*12/(ex.std(ddof=1)*np.sqrt(12)));mdd=float((eq/eq.cummax()-1).min());return {'Years':yrs,'CAGR':c,'Vol':vol,'SharpeExcess':sh,'MDD_monthly':mdd,'Calmar_monthly':c/abs(mdd) if mdd<0 else np.nan,'N':len(r)}

def daily_mdd_from_px(px):
    eq=px/px.iloc[0];dd=eq/eq.cummax()-1;return float(dd.min())

def monthly_rebalanced(data,weights):
    return (data*weights).sum(axis=1)

# Exact fund-of-funds decomposition, common QDSIX live history.
D=pd.concat({t:MR[t] for t in AQR+['BIL','QDSIX']},axis=1).dropna()
D=D.loc[D.index>=pd.Timestamp('2020-07-31')]
static=monthly_rebalanced(D[AQR+['BIL']],STRAT_W)
current=monthly_rebalanced(D[AQR+['BIL']],CURR_W)
print('QDSIX_LIVE',json.dumps(met(D.QDSIX)))
print('STATIC_STRATEGIC',json.dumps({**met(static),'Corr':float(static.corr(D.QDSIX)),'TEann':float((static-D.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((static-D.QDSIX).mean()*12)}))
print('STATIC_CURRENT',json.dumps({**met(current),'Corr':float(current.corr(D.QDSIX)),'TEann':float((current-D.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((current-D.QDSIX).mean()*12)}))

# Nonnegative sum-to-one regression: descriptive, NOT a tradable OOS model.
X=D[AQR+['BIL']].values;y=D.QDSIX.values
obj=lambda w:np.mean((y-X@w)**2)
cons={'type':'eq','fun':lambda w:w.sum()-1}
fit=minimize(obj,STRAT_W,bounds=[(0,1)]*7,constraints=cons,method='SLSQP',options={'maxiter':10000,'ftol':1e-14})
wfit=fit.x;pred=pd.Series(X@wfit,index=D.index)
print('FULL_FIT_WEIGHTS',json.dumps(dict(zip(AQR+['BIL'],map(float,wfit)))))
print('FULL_FIT',json.dumps({**met(pred),'Corr':float(pred.corr(D.QDSIX)),'TEann':float((pred-D.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((pred-D.QDSIX).mean()*12)}))

# Honest split: fit 2020-07..2022-12, evaluate 2023..2026.
tr=D.loc[:'2022-12-31'];te=D.loc['2023-01-01':]
Xtr=tr[AQR+['BIL']].values;ytr=tr.QDSIX.values
fit2=minimize(lambda w:np.mean((ytr-Xtr@w)**2),STRAT_W,bounds=[(0,1)]*7,constraints=cons,method='SLSQP',options={'maxiter':10000,'ftol':1e-14})
w2=fit2.x
pte=pd.Series(te[AQR+['BIL']].values@w2,index=te.index)
print('SPLIT_FIT_WEIGHTS',json.dumps(dict(zip(AQR+['BIL'],map(float,w2)))))
print('SPLIT_OOS',json.dumps({**met(pte),'Actual':met(te.QDSIX),'Corr':float(pte.corr(te.QDSIX)),'TEann':float((pte-te.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((pte-te.QDSIX).mean()*12)}))

# Calendar-year behavior and rough strategic contribution.
for yr,g in D.groupby(D.index.year):
    q=float((1+g.QDSIX).prod()-1);vals={t:float((1+g[t]).prod()-1) for t in AQR+['BIL']};approx={t:float(STRAT_W[i]*vals[t]) for i,t in enumerate(AQR+['BIL'])};print('YEAR',int(yr),json.dumps({'QDSIX':q,'Underlying':vals,'ApproxWeightedContribution':approx}))

# Underlying correlations and volatility over QDSIX live window.
print('UNDERLYING_CORR',D[AQR].corr().round(4).to_json())
for t in AQR:
    print('UNDERLYING_MET',t,json.dumps(met(D[t])))

# Pseudo-history of published strategic mix, from first common date of all six AQR sleeves (pre-QDSIX is hypothetical).
H=pd.concat({t:MR[t] for t in AQR+['BIL']},axis=1).dropna();hist=monthly_rebalanced(H,STRAT_W)
print('STRATEGIC_PSEUDOHISTORY',json.dumps({'Start':str(hist.index[0].date()),**met(hist)}))
for a,b,n in [('2014-11-01','2019-12-31','PRE_QDSIX'),('2020-01-01','2026-06-30','POST_2020'),('2022-01-01','2022-12-31','Y2022')]:
    z=hist.loc[a:b];print('PSEUDO_SUB',n,json.dumps(met(z)))

# Candidate ETF correlations with each AQR sleeve over overlapping monthly history, require >=24 months.
for target in AQR:
    rows=[]
    for e in ETF_UNIVERSE:
        if e not in MR:continue
        z=pd.concat({'a':MR[target],'e':MR[e]},axis=1).dropna()
        if len(z)<24:continue
        rows.append((e,len(z),float(z.a.corr(z.e)),float((z.a-z.e).std()*np.sqrt(12))))
    rows=sorted(rows,key=lambda x:(-x[2],x[3]))
    print('ETF_CORR',target,json.dumps([{'ETF':a,'N':n,'Corr':c,'TE':te} for a,n,c,te in rows[:8]]))

# Retail clone 1: long-history mapping, available since ~2020.
# ADAIX->MNA; QMNIX->BTAL; QGMIX->DBMF; QMHIX->KMLM; AQRIX->RPAR; QSPIX->QAI; cash->BIL.
MAP_LONG=['MNA','BTAL','DBMF','KMLM','RPAR','QAI','BIL']
# Modern closer mapping uses new ETF structures: NTRL is too new for useful test, QIS for style premia; CTA for HV trend.
MAP_MODERN=['MNA','BTAL','DBMF','CTA','RPAR','QIS','BIL']

def clone_eval(mapping,label,start=None):
    if any(t not in MR for t in mapping):return
    Z=pd.concat({t:MR[t] for t in mapping+['QDSIX']},axis=1).dropna()
    if start:Z=Z.loc[Z.index>=pd.Timestamp(start)]
    if len(Z)<12:return
    r=(Z[mapping].values@STRAT_W);r=pd.Series(r,index=Z.index)
    print('CLONE',label,json.dumps({'Start':str(Z.index[0].date()),'End':str(Z.index[-1].date()),**met(r),'ActualQDSIX':met(Z.QDSIX),'CorrQDSIX':float(r.corr(Z.QDSIX)),'TEann':float((r-Z.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((r-Z.QDSIX).mean()*12)}))
clone_eval(MAP_LONG,'LOGICAL_LONG')
clone_eval(MAP_MODERN,'LOGICAL_MODERN')

# Direct long-only ETF tracking fit to QDSIX using ETFs with >=2020 history. Fit 2020-07..2022-12, OOS 2023+.
CANDS=['MNA','QAI','BTAL','PHDG','DBMF','GMOM','RPAR','BIL']
Z=pd.concat({t:MR[t] for t in CANDS+['QDSIX']},axis=1).dropna();Z=Z.loc[Z.index>=pd.Timestamp('2020-07-31')]
tr=Z.loc[:'2022-12-31'];te=Z.loc['2023-01-01':]
if len(tr)>=20 and len(te)>=12:
    x=tr[CANDS].values;y=tr.QDSIX.values;w0=np.repeat(1/len(CANDS),len(CANDS));c={'type':'eq','fun':lambda w:w.sum()-1};f=minimize(lambda w:np.mean((y-x@w)**2),w0,bounds=[(0,1)]*len(CANDS),constraints=c,method='SLSQP',options={'maxiter':10000,'ftol':1e-14});w=f.x
    po=pd.Series(te[CANDS].values@w,index=te.index)
    print('ETF_FIT_WEIGHTS',json.dumps(dict(zip(CANDS,map(float,w)))))
    print('ETF_FIT_OOS',json.dumps({'Start':str(te.index[0].date()),**met(po),'ActualQDSIX':met(te.QDSIX),'CorrQDSIX':float(po.corr(te.QDSIX)),'TEann':float((po-te.QDSIX).std()*np.sqrt(12)),'MeanDiffAnn':float((po-te.QDSIX).mean()*12)}))

# Compare QDSIX to the user's production strategy over common dates (read-only import).
try:
    buf=io.StringIO()
    with contextlib.redirect_stdout(buf):import analysis_production_summary_dca as prod
    for key in ['DB_ACT','AQ_ACT','DB','AQ']:
        if key not in prod.Z:continue
        u=(1+prod.Z[key].ret).resample('ME').prod()-1
        z=pd.concat({'user':u,'qds':MR['QDSIX']},axis=1).dropna();z=z.loc[z.index<=pd.Timestamp('2026-06-30')]
        print('USER_VS_QDSIX',key,json.dumps({'Start':str(z.index[0].date()),'User':met(z.user),'QDSIX':met(z.qds),'Corr':float(z.user.corr(z.qds))}))
except Exception as e:print('USER_COMPARE_FAIL',repr(e))
