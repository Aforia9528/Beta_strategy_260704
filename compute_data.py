# -*- coding: utf-8 -*-
"""IA_BETA_01_2X 신호계산 → docs/data.json 생성 (매일 GitHub Action 실행).
최종 config: QLD vol-target + wq데드밴드5%p + SPY200게이트 + 금60/DBMF40 + 4자산자유리밸(5%밴드).
신호=미국 종가(yfinance). 상태는 400일 재계산(저장 불필요, 경로의존 replay).
데이터 검증/로직스펙 값도 함께 산출해 프론트가 그대로 렌더."""
import json, datetime, sys
import numpy as np, pandas as pd, yfinance as yf

# ===== LOCKED config =====
TARGET, WIN, FLOOR, CAP, INC = 0.20, 16, 0.20, 1.0, 0.15
DEADBAND = 0.05
GOLD_FRAC, DBMF_FRAC, H_CAP = 0.60, 0.40, 0.60
BAND, MIN_TRADE = 0.05, 0.005
GATE_TICKER, GATE_MA, GATE_MULT = "SPY", 200, 0.5
TICKERS = ["QLD", "GLD", "DBMF", "SGOV", "SPY"]

def fetch():
    cols = {}
    for t in TICKERS:
        for _ in range(3):
            try:
                h = yf.download(t, period="400d", interval="1d", auto_adjust=True, progress=False)
                if h is not None and len(h) and "Close" in h.columns:
                    cl = h["Close"]
                    if isinstance(cl, pd.DataFrame): cl = cl.iloc[:, 0]
                    cl = cl.dropna()
                    if len(cl): cols[t] = cl; break
            except Exception:
                pass
    df = pd.DataFrame(cols)
    df.index = pd.to_datetime(df.index)
    try: df.index = df.index.tz_localize(None)
    except Exception: pass
    return df.sort_index().ffill()

def compute(px):
    rq = px["QLD"].pct_change()
    vol_s = (rq.rolling(WIN).std() * np.sqrt(252))
    te_s = np.clip((TARGET / vol_s).values, FLOOR, CAP)
    # 비대칭 스무딩 (전체 경로)
    cur = 0.0; asym = []
    for x in te_s:
        if np.isnan(x): asym.append(cur); continue
        if x < cur: cur = x
        elif x - cur > INC: cur = x
        asym.append(cur)
    asym = np.array(asym)
    # 게이트
    ma_s = px[GATE_TICKER].rolling(GATE_MA).mean()
    gate_on_s = (px[GATE_TICKER] < ma_s).values
    wq_gated = asym * np.where(gate_on_s, GATE_MULT, 1.0)
    # 데드밴드 (마지막 실행 wq서 5%p 넘게 변할때만 갱신)
    last = wq_gated[0]; wq_final = []
    for g in wq_gated:
        if abs(g - last) > DEADBAND: last = g
        wq_final.append(last)
    wq_final = np.array(wq_final)
    return dict(rq=rq, vol_s=vol_s, te_s=te_s, asym=asym, ma_s=ma_s,
                gate_on_s=gate_on_s, wq_gated=wq_gated, wq_final=wq_final)

def build_target(wq):
    hb = min(H_CAP, max(0.0, 1 - wq))
    return {"QLD": wq, "gold": GOLD_FRAC * hb, "dbmf": DBMF_FRAC * hb,
            "cash": max(0.0, 1 - wq - hb)}, hb

def main():
    px = fetch()
    if px.empty or "QLD" not in px.columns:
        print("ERROR: QLD 수집 실패"); sys.exit(1)
    C = compute(px)
    i = -1
    vol = float(C["vol_s"].iloc[i]); te = float(C["te_s"][i]); wqa = float(C["asym"][i])
    spy = float(px[GATE_TICKER].iloc[i]); ma = float(C["ma_s"].iloc[i])
    g_on = bool(C["gate_on_s"][i]); wqg = float(C["wq_gated"][i])
    L = float(C["wq_final"][i - 1]); wq = float(C["wq_final"][i])
    tgt, hb = build_target(wq)
    asof = str(px.index[i].date())
    # 데이터 검증용 최근 종가 + 전일대비
    last_row = px.iloc[i]; chg = px.pct_change().iloc[i]
    prices = {t: {"close": round(float(last_row[t]), 2),
                  "chg": round(float(chg[t]) * 100, 2)} for t in px.columns}
    # 최근 20일 스펙 테이블
    recent = []
    for j in range(-20, 0):
        recent.append({
            "date": str(px.index[j].date()),
            "qld": round(float(px["QLD"].iloc[j]), 2),
            "ret": round(float(C["rq"].iloc[j]) * 100, 2),
            "te": round(float(C["te_s"][j]), 3),
            "spy": round(float(px[GATE_TICKER].iloc[j]), 2),
            "ma": round(float(C["ma_s"].iloc[j]), 2),
            "gate": bool(C["gate_on_s"][j]),
            "wq_asym": round(float(C["asym"][j]), 3),
            "wq_final": round(float(C["wq_final"][j]), 3),
        })
    r16 = list(np.round(C["rq"].iloc[-16:].values * 100, 2))
    dist_pct = (spy / ma - 1) if ma and not np.isnan(ma) else None
    data = {
        "asof": asof,
        "generated_utc": datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "target": {k: round(v, 4) for k, v in tgt.items()},
        "vol": round(vol, 4), "te": round(te, 4),
        "wq_asym": round(wqa, 4), "wq_gated": round(wqg, 4),
        "last_wq": round(L, 4), "wq_final": round(wq, 4), "hb": round(hb, 4),
        "gate": {"on": g_on, "spy": round(spy, 2), "ma": round(ma, 2),
                 "dist_pct": round(dist_pct, 4) if dist_pct is not None else None},
        "prices": prices,
        "r16": r16,
        "recent": recent,
        "config": {"TARGET": TARGET, "WIN": WIN, "FLOOR": FLOOR, "CAP": CAP, "INC": INC,
                   "DEADBAND": DEADBAND, "GOLD_FRAC": GOLD_FRAC, "DBMF_FRAC": DBMF_FRAC,
                   "H_CAP": H_CAP, "BAND": BAND, "MIN_TRADE": MIN_TRADE,
                   "GATE_MA": GATE_MA, "GATE_MULT": GATE_MULT},
        # 실현/전망 성적 (데드밴드 config)
        "stats": {"calmar_bt": 1.20, "robust_bt": 0.97, "cagr_bt": 0.241,
                  "mdd_bt": -0.202, "calmar_mc": 0.84, "mdd_mc": -0.28},
    }
    with open("docs/data.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"OK data.json 생성: {asof} · QLD {wq*100:.1f}% / 금 {tgt['gold']*100:.1f}% / "
          f"DBMF {tgt['dbmf']*100:.1f}% / 현금 {tgt['cash']*100:.1f}% · vol {vol*100:.0f}% · "
          f"게이트 {'ON' if g_on else 'off'}")

if __name__ == "__main__":
    main()
