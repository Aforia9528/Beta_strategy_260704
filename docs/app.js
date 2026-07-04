/* IA_BETA_01_2X 리밸런스 어드바이저 — 정적앱.
   data.json(매일 Action 갱신) 로드 → 목표비율 표시 + 매매판정(브라우저) + 저널 + 로직스펙. */
"use strict";
const KEYS = ["QLD", "gold", "dbmf", "cash"];
const KNAME = { QLD: "QLD", gold: "금", dbmf: "DBMF", cash: "현금" };
const won = (x) => Math.round(x).toLocaleString("ko-KR");
const pct = (x, d = 1) => (x * 100).toFixed(d) + "%";
let DATA = null;

/* ===== 매매판정 — 파이썬 decide()와 동일 (4자산 자유리밸 5%밴드) ===== */
function decide(hold, deposit, tgt, BAND, MIN_TRADE) {
  const NAV = KEYS.reduce((s, a) => s + hold[a], 0) + deposit;
  if (NAV <= 0) return null;
  const tamt = {}, trades = {};
  KEYS.forEach((a) => { tamt[a] = tgt[a] * NAV; trades[a] = 0; });
  const drift = Math.max(...KEYS.map((a) => Math.abs(hold[a] / NAV - tgt[a])));
  if (drift > BAND) {                          // 밴드 초과 → 전체 리밸런스
    KEYS.forEach((a) => { trades[a] = tgt[a] * NAV - hold[a]; });
  } else if (deposit > 0) {                     // 밴드 이내 → 적립금만 저비중 매수(매도X)
    const under = {}; let tot = 0;
    KEYS.forEach((a) => { under[a] = Math.max(0, tgt[a] * NAV - hold[a]); tot += under[a]; });
    if (tot > 0) KEYS.forEach((a) => { trades[a] = (under[a] / tot) * deposit; });
    else trades.cash += deposit;
  }
  KEYS.forEach((a) => { if (Math.abs(trades[a]) < MIN_TRADE * NAV) trades[a] = 0; });
  const cw = {}; KEYS.forEach((a) => { cw[a] = hold[a] / NAV; });
  const no_trade = KEYS.every((a) => Math.abs(trades[a]) < 1e-6);
  return { NAV, tamt, trades, drift, cw, no_trade };
}

/* ===== 렌더 ===== */
function renderBanner(d) {
  const staleDays = Math.floor((Date.now() - new Date(d.asof + "T00:00:00Z")) / 86400000);
  document.getElementById("banner").innerHTML =
    `🕒 기준일 <b>${d.asof}</b> (미국장 종가) · 변동성 ${pct(d.vol, 0)} → QLD ${pct(d.wq_final)}` +
    (staleDays > 5 ? ` <span class="warn">⚠️ ${staleDays}일 전 데이터(휴장/지연 확인)</span>` : "");
  const g = d.gate, box = document.getElementById("gatebox");
  if (g.spy == null) { box.innerHTML = ""; return; }
  if (g.on) {
    box.className = "alert red";
    box.innerHTML = `🔴 <b>추세게이트 발동</b> — SPY ${g.spy} &lt; 200일선 ${g.ma} (하락추세) → QLD 절반 감산. <b>방어 모드</b>.`;
  } else {
    box.className = "alert green";
    box.innerHTML = `🟢 게이트 OFF (상승추세) — SPY ${g.spy} &gt; 200일선 ${g.ma} · ` +
      `여유 <b>+${pct(g.dist_pct)}</b> (플립까지 멀수록 안전) <span class="muted small">※종가기준</span>`;
  }
}
function renderTargets(d) {
  const t = d.target, lab = { QLD: "QLD (ISA)", gold: "금 (ISA)", dbmf: "DBMF (해외)", cash: "현금 (ISA)" };
  document.getElementById("targets").innerHTML = KEYS.map((a) =>
    `<div class="card"><div class="k">${lab[a]}</div><div class="v">${pct(t[a], 0)}</div></div>`).join("");
  document.getElementById("basis").innerHTML =
    `근거: QLD 16일변동성 <b>${pct(d.vol, 0)}</b> → 원시노출 ${d.te.toFixed(2)} → 비대칭·데드밴드 후 ` +
    `QLD <b>${pct(d.wq_final)}</b> (나스닥 ${(d.wq_final * 2).toFixed(2)}배) · 바스켓 금60/DBMF40`;
}
function renderDecision() {
  const hold = {
    QLD: +document.getElementById("v_qld").value || 0,
    gold: +document.getElementById("v_gold").value || 0,
    dbmf: +document.getElementById("v_dbmf").value || 0,
    cash: +document.getElementById("v_cash").value || 0,
  };
  const deposit = +document.getElementById("deposit").value || 0;
  const c = DATA.config;
  const res = decide(hold, deposit, DATA.target, c.BAND, c.MIN_TRADE);
  const box = document.getElementById("decision");
  if (!res) { box.innerHTML = `<p class="muted">보유 평가액을 입력하면 매매 지시가 나옵니다.</p>`; return null; }
  if (res.no_trade) {
    box.innerHTML = `<div class="ok">🟢 오늘 거래 없음 (최대 드리프트 ${pct(res.drift)} &lt; ${pct(c.BAND, 0)})</div>` +
      `<div class="muted small">총자산(입금후) ₩${won(res.NAV)}</div>`;
  } else {
    const rows = KEYS.map((a) => {
      const tr = res.trades[a];
      const act = tr > 0 ? `<span class="buy">+₩${won(tr)} 매수</span>`
        : tr < 0 ? `<span class="sell">₩${won(-tr)} 매도</span>` : `<span class="muted">변동 없음</span>`;
      return `<tr><td>${KNAME[a]}</td><td>${act}</td><td class="muted small">${pct(res.cw[a], 0)}→${pct(DATA.target[a], 0)}</td></tr>`;
    }).join("");
    box.innerHTML = `<table class="trades"><tr><th>자산</th><th>지시</th><th>현재→목표</th></tr>${rows}</table>` +
      (res.drift > c.BAND ? `<div class="muted small">드리프트 ${pct(res.drift)} &gt; ${pct(c.BAND, 0)} → 전체 리밸런스</div>` : "") +
      `<div class="muted small">총자산(입금후) ₩${won(res.NAV)}</div>`;
  }
  return { hold, deposit, res };
}
function renderPrices(d) {
  const rows = Object.entries(d.prices).map(([t, p]) =>
    `<tr><td>${t}</td><td>$${p.close}</td><td class="${p.chg >= 0 ? "buy" : "sell"}">${p.chg >= 0 ? "+" : ""}${p.chg}%</td></tr>`).join("");
  document.getElementById("prices").innerHTML =
    `<table class="trades"><tr><th>티커</th><th>종가</th><th>전일대비</th></tr>${rows}</table>` +
    `<div class="muted small">기준일 ${d.asof} · QLD 16일 실현변동성 ${pct(d.vol, 1)}</div>`;
}
function renderRules(d) {
  const c = d.config, s = d.stats;
  document.getElementById("rules").innerHTML = `
<b>LOCKED 전략</b>: QLD vol-target(target${c.TARGET}/win${c.WIN}/floor${c.FLOOR}/cap${c.CAP}/비대칭inc${c.INC})<br>
· <b>wq 데드밴드 ${pct(c.DEADBAND, 0)}</b> (마지막 실행서 5%p 넘게 변할때만 갱신 — whipsaw 제거)<br>
· <b>추세게이트: SPY&lt;${c.GATE_MA}일SMA → QLD×${c.GATE_MULT}</b><br>
· 바스켓 금${pct(c.GOLD_FRAC, 0)}/DBMF${pct(c.DBMF_FRAC, 0)}(h≤${pct(c.H_CAP, 0)}) · 밴드 ${pct(c.BAND, 0)} · 미세거래&lt;${pct(c.MIN_TRADE, 1)} 무시<br>
· 4자산 자유 리밸런스 (계좌·add-only 제약 없음)<br><br>
<b>검증성적(백테 현실, 비용0.1%)</b>: CAGR ${pct(s.cagr_bt)}, MDD ${pct(s.mdd_bt)}, Calmar ${s.calmar_bt}, robust ${s.robust_bt}<br>
<b>정직한 MC 전망(운 제외)</b>: Calmar 중앙 <b>${s.calmar_mc}</b> · MDD 중앙 <b>${pct(s.mdd_mc)}</b><br>
<b>⚠️ 위험</b>: 2x — 미래 낙폭 중앙 ${pct(s.mdd_mc)}, 위기 -40%+, 회복 ~18개월. 장기자금만. 폭락에 적립 지속이 핵심.`;
}

/* ===== 로직 상세설명서 ===== */
function buildSpec(d) {
  const c = d.config, t = d.target;
  const gateTxt = d.gate.on
    ? `SPY ${d.gate.spy} &lt; 200일SMA ${d.gate.ma} → 게이트 ON (하락추세)`
    : `SPY ${d.gate.spy} ≥ 200일SMA ${d.gate.ma} → 게이트 OFF (상승추세)`;
  const step4 = Math.abs(d.wq_gated - d.last_wq) > c.DEADBAND
    ? `&gt;${c.DEADBAND} → 갱신 → wq=${d.wq_final}`
    : `≤${c.DEADBAND} → 유지 → wq=${d.last_wq}`;
  const rows = d.recent.map((r) =>
    `<tr><td>${r.date}</td><td>${r.qld}</td><td>${r.ret >= 0 ? "+" : ""}${r.ret}</td><td>${r.te}</td><td>${r.spy}</td><td>${r.ma}</td><td>${r.gate ? "ON" : "off"}</td><td>${r.wq_asym}</td><td>${r.wq_final}</td></tr>`).join("");
  return `
<h3>기준일 ${d.asof} — 이 문서만으로 오늘 비율 재현·검증 가능</h3>
<p><b>결과:</b> QLD ${pct(t.QLD, 1)} / 금 ${pct(t.gold, 1)} / DBMF ${pct(t.dbmf, 1)} / 현금 ${pct(t.cash, 1)}</p>
<p><b>매일 변하는 값 = 딱 2개</b></p>
<p>① QLD 16일 실현변동성 = <b>${pct(d.vol, 2)}</b> &nbsp; <code>vol = std(16일 QLD수익률)×√252</code><br>
최근16일 수익률(%): ${d.r16.join(", ")}</p>
<p>② 게이트: ${gateTxt}</p>
<p><b>단계별 (오늘 값)</b><br>
STEP1 원시노출 <code>te=clip(${c.TARGET}/vol,${c.FLOOR},${c.CAP})=${d.te}</code><br>
STEP2 비대칭 스무딩 → 누적상태 <b>wq_asym=${d.wq_asym}</b> (경로의존; 아래 표로 replay)<br>
STEP3 게이트 <code>wq_gated=wq_asym×${d.gate.on ? c.GATE_MULT : 1}=${d.wq_gated}</code><br>
STEP4 데드밴드: 마지막실행 ${d.last_wq}, |${d.wq_gated}−${d.last_wq}|=${Math.abs(d.wq_gated - d.last_wq).toFixed(4)} → ${step4}<br>
STEP5 <code>hb=min(${c.H_CAP},1−wq)=${d.hb}</code>; 금=${c.GOLD_FRAC}×hb=${t.gold.toFixed(4)}; DBMF=${c.DBMF_FRAC}×hb=${t.dbmf.toFixed(4)}; 현금=1−wq−hb=${t.cash.toFixed(4)}<br>
검산 합=${(t.QLD + t.gold + t.dbmf + t.cash).toFixed(4)}</p>
<p><b>최근 20일 (STEP2·4 replay용)</b></p>
<div class="scroll"><table class="spec-tbl"><tr><th>날짜</th><th>QLD</th><th>수익%</th><th>te</th><th>SPY</th><th>200SMA</th><th>게이트</th><th>wq_asym</th><th>wq_final</th></tr>${rows}</table></div>`;
}
function specPrompt(d) {
  const c = d.config, t = d.target;
  return `아래 알고리즘과 오늘 입력값으로 4자산 목표비율을 재계산하고 결과가 일치하는지 검증해줘.
[상수] TARGET=${c.TARGET}, WIN=${c.WIN}, FLOOR=${c.FLOOR}, CAP=${c.CAP}, INC=${c.INC}, DEADBAND=${c.DEADBAND}, GOLD_FRAC=${c.GOLD_FRAC}, DBMF_FRAC=${c.DBMF_FRAC}, H_CAP=${c.H_CAP}, GATE_MULT=${c.GATE_MULT}
[오늘입력] vol=${d.vol}, wq_asym(누적상태)=${d.wq_asym}, 게이트ON=${d.gate.on}, 마지막실행wq=${d.last_wq}
[계산] te=clip(TARGET/vol,FLOOR,CAP); wq_gated=wq_asym*(GATE_MULT if 게이트ON else 1);
       wq = wq_gated if |wq_gated-마지막실행wq|>DEADBAND else 마지막실행wq;
       hb=min(H_CAP,1-wq); 금=GOLD_FRAC*hb; DBMF=DBMF_FRAC*hb; 현금=1-wq-hb
[정답] QLD=${pct(t.QLD, 1)}, 금=${pct(t.gold, 1)}, DBMF=${pct(t.dbmf, 1)}, 현금=${pct(t.cash, 1)}`;
}

/* ===== 저널 (localStorage) ===== */
const JKEY = "beta2x_journal";
function loadJournal() { try { return JSON.parse(localStorage.getItem(JKEY)) || []; } catch { return []; } }
function saveJournal(j) { localStorage.setItem(JKEY, JSON.stringify(j)); }
function renderJournal() {
  const j = loadJournal();
  document.getElementById("journalCount").textContent = j.length;
  document.getElementById("journal").innerHTML = j.length === 0
    ? `<p class="muted small">아직 기록 없음. 거래 후 "기록하기"를 누르세요.</p>`
    : `<table class="trades"><tr><th>날짜</th><th>총자산</th><th>QLD/금/DBMF/현금(현재%)</th><th>지시</th></tr>` +
    j.slice().reverse().map((e) =>
      `<tr><td>${e.ts.slice(0, 10)}</td><td>₩${won(e.nav)}</td><td class="small">${e.cw}</td><td class="small">${e.trades || "없음"}</td></tr>`).join("") + `</table>`;
}
function recordSnapshot() {
  const r = renderDecision();
  if (!r) { alert("보유 평가액을 먼저 입력하세요."); return; }
  const { res } = r;
  const cw = KEYS.map((a) => (res.cw[a] * 100).toFixed(0)).join("/");
  const trades = res.no_trade ? "거래없음" : KEYS.filter((a) => Math.abs(res.trades[a]) > 1e-6)
    .map((a) => `${KNAME[a]}${res.trades[a] > 0 ? "+" : ""}${won(res.trades[a])}`).join(", ");
  const j = loadJournal();
  j.push({
    ts: new Date().toISOString(), asof: DATA.asof, nav: Math.round(res.NAV), cw, trades,
    target: KEYS.map((a) => (DATA.target[a] * 100).toFixed(0)).join("/"),
    hold: KEYS.map((a) => Math.round(r.hold[a])).join("/"), deposit: Math.round(r.deposit),
  });
  saveJournal(j); renderJournal();
}
function exportCSV() {
  const j = loadJournal();
  if (!j.length) { alert("기록이 없습니다."); return; }
  const head = "일시,기준일,총자산,보유(Q/G/D/현),적립,현재비중%,목표비중%,지시\n";
  const body = j.map((e) => `${e.ts},${e.asof},${e.nav},${e.hold},${e.deposit},${e.cw},${e.target},"${e.trades}"`).join("\n");
  const blob = new Blob(["﻿" + head + body], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob); a.download = `beta2x_journal_${DATA.asof}.csv`; a.click();
}

/* ===== init ===== */
async function init() {
  try {
    const res = await fetch("data.json?v=" + Date.now());
    DATA = await res.json();
  } catch (e) {
    document.getElementById("banner").innerHTML = `<span class="warn">⚠️ data.json 로드 실패: ${e}</span>`;
    return;
  }
  renderBanner(DATA); renderTargets(DATA); renderPrices(DATA); renderRules(DATA);
  document.getElementById("spec").innerHTML = buildSpec(DATA);
  document.getElementById("footer").textContent = `생성 ${DATA.generated_utc} UTC · 정적앱(서버 없음)`;
  // 마지막 보유 입력 복원
  try {
    const last = JSON.parse(localStorage.getItem("beta2x_lasthold") || "{}");
    ["v_qld", "v_gold", "v_dbmf", "v_cash"].forEach((id) => { if (last[id]) document.getElementById(id).value = last[id]; });
  } catch {}
  const recalc = () => {
    renderDecision();
    const last = {}; ["v_qld", "v_gold", "v_dbmf", "v_cash"].forEach((id) => { last[id] = document.getElementById(id).value; });
    localStorage.setItem("beta2x_lasthold", JSON.stringify(last));
  };
  ["v_qld", "v_gold", "v_dbmf", "v_cash", "deposit"].forEach((id) =>
    document.getElementById(id).addEventListener("input", recalc));
  recalc(); renderJournal();
  document.getElementById("recordBtn").addEventListener("click", recordSnapshot);
  document.getElementById("exportBtn").addEventListener("click", exportCSV);
  document.getElementById("clearBtn").addEventListener("click", () => {
    if (confirm("저널 전체를 삭제할까요?")) { saveJournal([]); renderJournal(); }
  });
  document.getElementById("copySpecBtn").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText(specPrompt(DATA)); document.getElementById("copySpecBtn").textContent = "✅ 복사됨"; }
    catch { alert(specPrompt(DATA)); }
  });
}
init();
