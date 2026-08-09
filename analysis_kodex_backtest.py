import numpy as np
import pandas as pd
import yfinance as yf

TARGET, WIN, FLOOR, CAP, INC = 0.20, 16, 0.20, 1.0, 0.15
DEADBAND = 0.05
GOLD_FRAC, DBMF_FRAC, H_CAP = 0.50, 0.50, 0.60
GATE_MA, GATE_MULT = 200, 0.5

START = "2022-06-01"
TICKERS = ["QLD", "409820.KS", "GLD", "DBMF", "SGOV", "SPY", "KRW=X"]


def dl(t):
    x = yf.download(t, start="2021-01-01", auto_adjust=True, progress=False)
    if isinstance(x.columns, pd.MultiIndex): x.columns = x.columns.get_level_values(0)
    return x["Close"].dropna()

px = pd.concat({t: dl(t) for t in TICKERS}, axis=1).sort_index().ffill()

# Signal stays exactly as in the live strategy: QLD USD return + SPY 200SMA.
rq = px["QLD"].pct_change()
vol = rq.rolling(WIN).std() * np.sqrt(252)
te = np.clip((TARGET / vol).values, FLOOR, CAP)
cur = 0.0; asym=[]
for x in te:
    if np.isnan(x): asym.append(cur); continue
    if x < cur: cur = x
    elif x-cur > INC: cur=x
    asym.append(cur)
asym=np.array(asym)
ma=px["SPY"].rolling(GATE_MA).mean()
gate=(px["SPY"]<ma).values
gated=asym*np.where(gate,GATE_MULT,1.0)
last=gated[0]; w=[]
for g in gated:
    if abs(g-last)>DEADBAND: last=g
    w.append(last)
w=np.array(w)

weights=[]
for q in w:
    hb=min(H_CAP,max(0,1-q))
    weights.append((q, GOLD_FRAC*hb, DBMF_FRAC*hb, max(0,1-q-hb)))
W=pd.DataFrame(weights,index=px.index,columns=["q","g","d","c"]).shift(1)

# Normalize EVERYTHING to KRW investor returns.
ret = px.pct_change()
fx = ret["KRW=X"]  # KRW per USD; + means USD strengthens vs KRW

def usd_to_krw(r_usd):
    return (1+r_usd)*(1+fx)-1

R = pd.DataFrame(index=px.index)
R["QLD_KRW"] = usd_to_krw(ret["QLD"])
R["KODEX_H_KRW"] = ret["409820.KS"]
R["GLD_KRW"] = usd_to_krw(ret["GLD"])
R["DBMF_KRW"] = usd_to_krw(ret["DBMF"])
R["SGOV_KRW"] = usd_to_krw(ret["SGOV"])


def stats(r):
    r=r.dropna(); eq=(1+r).cumprod(); years=len(r)/252
    cagr=eq.iloc[-1]**(1/years)-1
    annvol=r.std()*np.sqrt(252)
    sharpe=(r.mean()*252)/annvol if annvol else np.nan
    mdd=(eq/eq.cummax()-1).min()
    return {"CAGR":cagr,"Vol":annvol,"Sharpe0":sharpe,"MDD":mdd,"End":eq.iloc[-1],"N":len(r)}


def run(qcol):
    r = W.q*R[qcol] + W.g*R.GLD_KRW + W.d*R.DBMF_KRW + W.c*R.SGOV_KRW
    r=r.loc[START:].dropna()
    return stats(r)

print("PORT_Qld_KRW", run("QLD_KRW"))
print("PORT_KodexH_KRW", run("KODEX_H_KRW"))

# Sleeve-level KRW comparison.
common = R[["QLD_KRW","KODEX_H_KRW"]].loc[START:].dropna()
for t in common:
    print("SLEEVE",t,stats(common[t]))

# Direct relative terminal wealth on common observations.
qeq=(1+common.QLD_KRW).cumprod(); keq=(1+common.KODEX_H_KRW).cumprod()
print("REL_KODEX_OVER_QLD", float(keq.iloc[-1]/qeq.iloc[-1]-1))
