from pathlib import Path
import contextlib, io, json

# Load the validated long-history data construction, then add a hybrid simulator
# that keeps equity/risk-control decisions daily while defensive proxies remain monthly.
exec(compile(Path('analysis_30y_runner.py').read_text(),'analysis_30y_runner_runtime.py','exec'),globals(),globals())

print('HYBRID_VALIDATION')

def expand_monthly_lump(mret, daily_index):
    """Place each monthly return on the final available trading day of that month.
    This avoids inventing a smooth daily path for a monthly academic factor."""
    s=pd.Series(0.0,index=daily_index)
    by_period={p:g.index[-1] for p,g in s.groupby(s.index.to_period('M'))}
    for dt,v in mret.dropna().items():
        p=pd.Timestamp(dt).to_period('M')
        if p in by_period:s.loc[by_period[p]]=float(v)
    return s

def make_monthly_components(mf_scale=None,mf_fee=.01,gold_drag=0.0,fx_mf=True,fx_cash=True):
    sc=med_scale if mf_scale is None else mf_scale
    gret=gm-gold_drag/12
    mf_usd=rfm+sc*tsmom-mf_fee/12
    mfret=(1+mf_usd)*(1+fxm)-1 if fx_mf else mf_usd
    cashret=(1+rfm)*(1+fxm)-1 if fx_cash else rfm
    return gret,mfret,cashret

def hybrid_strategy(extra_financing=0.0,hcap=.70,mf_scale=None,mf_fee=.01,gold_drag=0.0,fx_mf=True,fx_cash=True,with_gate=True):
    er=synth_2x(extra_financing)
    qsig=qsignal(er) if with_gate else qsignal_nogate(er)
    idx=er.index.intersection(qsig.index)
    er=er.reindex(idx);qsig=qsig.reindex(idx)
    gret,mfret,cashret=make_monthly_components(mf_scale,mf_fee,gold_drag,fx_mf,fx_cash)
    gd=expand_monthly_lump(gret,idx);md=expand_monthly_lump(mfret,idx);cd=expand_monthly_lump(cashret,idx)
    A=pd.DataFrame({'q':er,'gold':gd,'mf':md,'cash':cd},index=idx)
    valid_months=set(gret.dropna().index.to_period('M')).intersection(mfret.dropna().index.to_period('M')).intersection(cashret.dropna().index.to_period('M'))
    keep=A.index.to_period('M').isin(valid_months);A=A.loc[keep];qsig=qsig.reindex(A.index)
    wc=None;rows=[];turn=0.;nt=0
    for dt,row in A.iterrows():
        q=float(np.clip(qsig.loc[dt],0,1));hb=min(hcap,max(0,1-q));cash=max(0,1-q-hb);wt=np.array([q,.5*hb,.5*hb,cash],float)
        if wc is None:wc=wt.copy()
        cost=0.
        if np.max(np.abs(wc-wt))>TRADE_BAND:
            to=float(np.sum(np.abs(wt-wc)));cost=COST*to;turn+=to;nt+=1;wc=wt.copy()
        ar=row[['q','gold','mf','cash']].to_numpy(dtype=float);gross=float(wc@ar);net=(1-cost)*(1+gross)-1
        end=wc*(1+ar);wc=end/end.sum();rows.append((dt,net))
    d=pd.Series(dict(rows)).sort_index();d.attrs={'turn':turn,'ntrade':nt}
    m=(1+d).resample('ME').prod()-1;m=m[m.index.to_period('M').isin(valid_months)]
    return d,m

def report_hybrid(label,**kw):
    d,m=hybrid_strategy(**kw);met=metrics_ret(m);met.update(rolling_stats(m));met['DailyTradeCount']=d.attrs['ntrade'];met['DailyTurnover']=d.attrs['turn'];met['NMonths']=len(m);print('HYBRID_FULL',label,json.dumps(met));
    for nm,a,b in [('BLACK_MONDAY','1987-08-01','1988-03-31'),('PRE_DOTCOM','1990-01-01','1999-12-31'),('DOTCOM','2000-01-01','2002-12-31'),('LOST_DECADE','2000-01-01','2009-12-31'),('GFC','2007-01-01','2009-06-30'),('POST_GFC','2010-01-01','2019-12-31'),('RECENT','2020-01-01','2026-12-31')]:
        print('HYBRID_REGIME',label,nm,json.dumps(submetric(m,a,b)))
    print('HYBRID_DD',label,json.dumps(dd_diag(m)))
    return d,m

HOUT={}
for label,kw in {
    'BASE':{},
    'NO_GATE':{'with_gate':False},
    'H60':{'hcap':.60},
    'H80':{'hcap':.80},
    'FIN_PLUS_100BP':{'extra_financing':.01},
    'MF_SCALE_75':{'mf_scale':.75*med_scale},
    'MF_SCALE_125':{'mf_scale':1.25*med_scale},
    'GOLD_DRAG_70BP':{'gold_drag':.007},
    'MF_FEE_150BP':{'mf_fee':.015},
    'FX_HH':{'fx_mf':False,'fx_cash':False},
    'FX_MF_H_CASH_U':{'fx_mf':False,'fx_cash':True},
    'FX_MF_U_CASH_H':{'fx_mf':True,'fx_cash':False},
}.items():
    HOUT[label]=report_hybrid(label,**kw)

def component_diag(a,b):
    gret,mfret,cashret=make_monthly_components();er=month_ret_from_daily(syn);qm_sig=Q.resample('ME').last()
    R=pd.concat({'q2x':er,'gold':gret,'mf':mfret,'cash':cashret,'target_q_end':qm_sig},axis=1).loc[a:b]
    out={}
    for c in ['q2x','gold','mf','cash']:
        z=R[c].dropna();out[c+'_CAGR']=float((1+z).prod()**(12/len(z))-1) if len(z) else np.nan
    out['avg_target_q']=float(R.target_q_end.mean());out['nmonths']=int(len(R))
    return out
for nm,a,b in [('DRAWDOWN_1986_90','1986-04-01','1990-10-31'),('DOTCOM','2000-01-01','2002-12-31'),('GFC','2007-01-01','2009-06-30')]:print('COMPONENT_DIAG',nm,json.dumps(component_diag(a,b)))

# Best-effort in-process bridge; a separate fresh-process bridge is used for final validation.
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import analysis_production_summary_dca as prod
base_hm=HOUT['BASE'][1]
for s in ['RY','AQ','WT','DB']:
    actual=(1+prod.Z[s].ret).resample('ME').prod()-1
    ix=actual.index.intersection(base_hm.index);aa=actual.reindex(ix).dropna();ss=base_hm.reindex(ix).dropna();ix=aa.index.intersection(ss.index);aa=aa.reindex(ix);ss=ss.reindex(ix)
    print('OVERLAP_BRIDGE',s,json.dumps({'N':len(ix),'actual':metrics_ret(aa),'synthetic_hybrid':metrics_ret(ss),'monthly_corr':float(aa.corr(ss)),'mean_return_diff_ann':float((ss-aa).mean()*12)}))

rows=[]
for label,(d,m) in HOUT.items():
    x=metrics_ret(m);x.update(rolling_stats(m));x['Variant']=label;x['NMonths']=len(m);rows.append(x)
pd.DataFrame(rows).to_csv('analysis_30y_hybrid_summary.csv',index=False)
HOUT['BASE'][1].rename('ret').to_csv('analysis_30y_hybrid_base_monthly.csv',index_label='Date')
