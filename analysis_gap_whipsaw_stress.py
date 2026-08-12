from pathlib import Path
import contextlib, io, json, math
import numpy as np
import pandas as pd

# Reuse validated long-history Nasdaq synthetic, SPX gate history and production parameters.
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    exec(compile(Path('analysis_30y_runner.py').read_text(),'analysis_30y_runner_runtime.py','exec'),globals(),globals())

RNG=np.random.default_rng(19871019)
N=20000
HORIZON=40

# Historical state pool. Keep dates with fully formed signal/MA/16d history.
idx=syn.index.intersection(gspc.index)
sp=gspc.reindex(idx).ffill();ma=sp.rolling(200).mean();dist=sp/ma-1
qhist=Q.reindex(idx).ffill();er=syn.reindex(idx)
valid=(idx>=pd.Timestamp('1990-01-01')) & dist.notna().values & qhist.notna().values
pool=pd.DataFrame({'q':qhist.values,'dist':dist.values,'gate':(sp<ma).values},index=idx).loc[valid]

# Need 16 prior 2x daily returns per sampled date.
def prior_returns(dt):
    loc=er.index.get_indexer([dt])[0]
    if loc<20:return None
    return er.iloc[loc-20:loc].dropna().values[-16:]

# Recover approximate pre-gate asymmetric signal state from q target.
def recover_asym(q,gate):
    return float(np.clip(q/(GATE_MULT if gate else 1.0),FLOOR,CAP))

def next_q_state(hist_returns,cur_asym,last_q,gate_on):
    vol=float(np.std(hist_returns[-WIN:],ddof=1)*np.sqrt(252))
    te=float(np.clip(TARGET/vol,FLOOR,CAP)) if vol>1e-12 else CAP
    if te<cur_asym:cur_asym=te
    elif te-cur_asym>INC:cur_asym=te
    gated=cur_asym*(GATE_MULT if gate_on else 1.0)
    if abs(gated-last_q)>SIGNAL_DEADBAND:last_q=gated
    return cur_asym,last_q,vol

def event_path(shock,rebound_frac,second_frac,second_day,noise_vol=.018):
    # NDX underlying return path: one-day gap, 5-day rebound, optional second leg, then choppy normalization.
    r=np.zeros(HORIZON)
    r[0]=shock
    # log-return distribute rebound to recover a fraction of the initial percentage loss.
    reb=max(0.0,rebound_frac*abs(shock))
    daily_reb=(1+reb)**(1/5)-1
    r[1:6]=daily_reb
    if second_frac>0:
        sec=-second_frac*abs(shock)
        d=(1+sec)**(1/4)-1
        st=int(np.clip(second_day,6,HORIZON-5));r[st:st+4]=d
    # residual choppy market, higher vol immediately after shock then decaying.
    for i in range(1,HORIZON):
        if r[i]==0:
            sig=noise_vol*(1.8 if i<10 else (1.3 if i<20 else 1.0))
            r[i]=RNG.normal(0.0004,sig)
    return np.clip(r,-.45,.35)

def simulate_one(state_kind,hedge_mode):
    if state_kind=='UNCONDITIONAL': cand=pool
    elif state_kind=='CALM_Q60': cand=pool[(pool.q>=.60)&(~pool.gate)]
    elif state_kind=='MAX_Q80': cand=pool[(pool.q>=.80)&(~pool.gate)]
    else: raise ValueError(state_kind)
    row=cand.iloc[int(RNG.integers(0,len(cand)))];dt=row.name;q0=float(row.q);gate0=bool(row.gate);d0=float(row.dist)
    prev=prior_returns(dt)
    if prev is None:return simulate_one(state_kind,hedge_mode)
    # Event distribution deliberately heavier-tailed than modern normal regimes.
    shock=-float(np.clip(RNG.normal(.155,.055),.06,.30))
    rebound=float(np.clip(RNG.normal(.75,.35),0,1.5))
    second=float(np.clip(RNG.normal(.45,.30),0,1.2)) if RNG.random()<.72 else 0.0
    second_day=int(RNG.integers(6,16))
    ndx=event_path(shock,rebound,second,second_day,float(RNG.uniform(.012,.028)))
    # SPX shock is typically smaller than NDX shock; noise prevents deterministic gate behavior.
    beta=float(RNG.uniform(.55,.82));spr=beta*ndx+RNG.normal(0,.004,size=HORIZON)
    # Approximate 200DMA as locally slow-moving. Start at sampled historical distance to MA.
    sp_rel=(1+d0)*np.cumprod(1+spr)
    gate_path=sp_rel<1.0
    # 2x ETF daily path. Ignore tiny financing over 40d except calibrated annual drag.
    qld=np.clip(2*ndx-(drag_ann/252),-.99,.75)
    # Hedge daily returns. First day can co-crash; later returns are noisy and can whipsaw.
    gold=RNG.normal(.00015,.010,size=HORIZON);mf=RNG.normal(.00010,.008,size=HORIZON);cash=np.zeros(HORIZON)
    if hedge_mode=='NEUTRAL': gold[0]=0;mf[0]=0
    elif hedge_mode=='ADVERSE': gold[0]=-.03;mf[0]=-.05
    elif hedge_mode=='SEVERE': gold[0]=-.07;mf[0]=-.10
    else: raise ValueError(hedge_mode)
    # managed futures can be wrong-footed by rapid reversal: penalize large sign reversals in NDX for first 12d.
    for i in range(1,min(12,HORIZON)):
        if np.sign(ndx[i])!=np.sign(ndx[i-1]) and abs(ndx[i])+abs(ndx[i-1])>.04:
            mf[i]-=float(RNG.uniform(.005,.02))
    hb0=min(H_CAP,max(0,1-q0));w=np.array([q0,.5*hb0,.5*hb0,max(0,1-q0-hb0)],float)
    nav=1.;peak=1.;mdd=0.;rets=[];hist=list(prev);cur_asym=recover_asym(q0,gate0);last_q=q0;q_targets=[q0];trade_count=0
    day_under30=None;gate_day=None
    for i in range(HORIZON):
        # q target for day i is based on information through previous day; day 0 therefore uses pre-event q0.
        if i>0:
            cur_asym,last_q,_=next_q_state(np.array(hist),cur_asym,last_q,bool(gate_path[i-1]));qtar=float(last_q)
            hb=min(H_CAP,max(0,1-qtar));target=np.array([qtar,.5*hb,.5*hb,max(0,1-qtar-hb)],float)
            if np.max(np.abs(w-target))>TRADE_BAND:
                w=target.copy();trade_count+=1
            q_targets.append(qtar)
            if day_under30 is None and qtar<=.30:day_under30=i
            if gate_day is None and bool(gate_path[i-1]):gate_day=i
        a=np.array([qld[i],gold[i],mf[i],cash[i]])
        rp=float(w@a);nav*=1+rp;rets.append(rp);peak=max(peak,nav);mdd=min(mdd,nav/peak-1)
        end=w*(1+a);den=end.sum();w=end/den
        hist.append(qld[i]);hist=hist[-WIN:]
    return {'q0':q0,'shock':shock,'rebound':rebound,'second':second,'mdd':mdd,'day0':rets[0],'final':nav-1,'q1':q_targets[1] if len(q_targets)>1 else q0,'q5':q_targets[5] if len(q_targets)>5 else q_targets[-1],'under30':day_under30 if day_under30 is not None else 99,'gate_day':gate_day if gate_day is not None else 99,'trades':trade_count}

def summarize(arr):
    df=pd.DataFrame(arr)
    return {
      'N':len(df),'avg_q0':float(df.q0.mean()),'med_shock':float(df.shock.median()),
      'MDD_median':float(df.mdd.median()),'MDD_p10':float(df.mdd.quantile(.10)),'MDD_p05':float(df.mdd.quantile(.05)),'MDD_p01':float(df.mdd.quantile(.01)),'MDD_worst':float(df.mdd.min()),
      'P_MDD_lt20':float((df.mdd<-.20).mean()),'P_MDD_lt30':float((df.mdd<-.30).mean()),'P_MDD_lt40':float((df.mdd<-.40).mean()),'P_MDD_lt50':float((df.mdd<-.50).mean()),
      'Day0_median':float(df.day0.median()),'Day0_p05':float(df.day0.quantile(.05)),'Final40d_median':float(df.final.median()),'Final40d_p05':float(df.final.quantile(.05)),
      'q1_median':float(df.q1.median()),'q5_median':float(df.q5.median()),'P_q_under30_by_day2':float((df.under30<=2).mean()),'P_gate_by_day2':float((df.gate_day<=2).mean()),'Trades40d_med':float(df.trades.median())
    }

print('POOL',json.dumps({'all':len(pool),'calm60':len(pool[(pool.q>=.60)&(~pool.gate)]),'max80':len(pool[(pool.q>=.80)&(~pool.gate)]),'q_quantiles':{str(x):float(pool.q.quantile(x)) for x in [.1,.25,.5,.75,.9,.95]}}))
for state in ['UNCONDITIONAL','CALM_Q60','MAX_Q80']:
    for hedge in ['NEUTRAL','ADVERSE','SEVERE']:
        arr=[simulate_one(state,hedge) for _ in range(N)]
        print('MC',state,hedge,json.dumps(summarize(arr)))

# Deterministic shock grid: clean one-day gap followed by 80% rebound and 50% second leg, neutral/adverse hedges.
def deterministic(q0,shock,hedge_mode):
    # Build a synthetic calm 16d history consistent with q0: choose vol roughly TARGET/q0.
    target_vol=float(np.clip(TARGET/max(q0,.2),.08,.50));daily=target_vol/np.sqrt(252);hist=list(RNG.normal(0,daily,size=16));cur=q0;last=q0;d0=.08
    ndx=event_path(shock,.80,.50,9,.012);spr=.70*ndx;sp_rel=(1+d0)*np.cumprod(1+spr);gate=sp_rel<1;qld=np.clip(2*ndx-drag_ann/252,-.99,.75)
    gold=np.zeros(HORIZON);mf=np.zeros(HORIZON);cash=np.zeros(HORIZON)
    if hedge_mode=='ADVERSE':gold[0]=-.03;mf[0]=-.05
    hb=min(H_CAP,1-q0);w=np.array([q0,.5*hb,.5*hb,max(0,1-q0-hb)]);nav=1;peak=1;mdd=0;qs=[q0]
    for i in range(HORIZON):
        if i>0:
            cur,last,_=next_q_state(np.array(hist[-WIN:]),cur,last,bool(gate[i-1]));qt=last;hb=min(H_CAP,1-qt);tar=np.array([qt,.5*hb,.5*hb,max(0,1-qt-hb)])
            if np.max(np.abs(w-tar))>TRADE_BAND:w=tar.copy()
            qs.append(qt)
        a=np.array([qld[i],gold[i],mf[i],cash[i]]);nav*=1+float(w@a);peak=max(peak,nav);mdd=min(mdd,nav/peak-1);e=w*(1+a);w=e/e.sum();hist.append(qld[i])
    return {'q0':q0,'NDXshock':shock,'hedge':hedge_mode,'MDD':mdd,'Final40d':nav-1,'q_day1':qs[1],'q_day5':qs[5],'gate_day1':bool(gate[0])}
for q0 in [.4,.6,.8,1.0]:
    for sh in [-.10,-.15,-.20,-.25]:
        for h in ['NEUTRAL','ADVERSE']:
            print('GRID',json.dumps(deterministic(q0,sh,h)))
