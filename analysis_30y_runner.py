from pathlib import Path

p=Path('analysis_30y_synthetic_dotcom.py')
s=p.read_text()

old_gold="rg=requests.get(STOOQ_GOLD,headers={'User-Agent':'Mozilla/5.0'},timeout=45);rg.raise_for_status()\ngold_df=pd.read_csv(io.StringIO(rg.text));gold_df.columns=[c.capitalize() for c in gold_df.columns];gold_df['Date']=pd.to_datetime(gold_df.Date);gold=gold_df.set_index('Date')['Close'].astype(float).sort_index();gold=gold.loc[gold.index>=pd.Timestamp(START)]"
new_gold="gold_df=pd.read_csv('https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv');gold_df['Date']=pd.to_datetime(gold_df['Date'])+pd.offsets.MonthEnd(0);gold=gold_df.set_index('Date')['Price'].astype(float).sort_index();gold=gold.loc[gold.index>=pd.Timestamp(START)]"
if old_gold not in s:
    raise SystemExit('gold source block not found')
s=s.replace(old_gold,new_gold)

old_parser="""            nums={}
            for c in dfi.columns:
                if c==datecol:continue
                z=pd.to_numeric(dfi.loc[ok,c],errors='coerce')
                if z.notna().sum()>100: nums[c]=z
            if not nums:continue
            tmp=pd.DataFrame(nums,index=dd[ok]);tmp=tmp[~tmp.index.duplicated()].sort_index()"""
new_parser="""            nums={}
            for j,c in enumerate(dfi.columns):
                if c==datecol:continue
                z=pd.to_numeric(dfi.loc[ok].iloc[:,j],errors='coerce')
                if z.notna().sum()>100: nums[f'{c}__{j}']=z.to_numpy()
            if not nums:continue
            tmp=pd.DataFrame(nums,index=pd.DatetimeIndex(dd[ok].to_numpy()));tmp=tmp[~tmp.index.duplicated()].sort_index()"""
if old_parser not in s:
    raise SystemExit('AQR parser block not found')
s=s.replace(old_parser,new_parser)

# AQR dates are business month-ends while FRED/gold resamples use calendar month-ends.
# Normalize by month before arithmetic; otherwise weekend month-ends silently disappear.
anchor="ash,ah,acol,tsmom=parse_aqr();print("
replacement="ash,ah,acol,tsmom=parse_aqr();tsmom.index=pd.DatetimeIndex(tsmom.index)+pd.offsets.MonthEnd(0);tsmom=tsmom.groupby(tsmom.index).last();print("
if anchor not in s:
    raise SystemExit('AQR normalization anchor not found')
s=s.replace(anchor,replacement,1)

extra = r'''

print('EXTRA_DIAGNOSTICS')
def dd_diag(r):
    r=pd.Series(r).dropna();eq=(1+r).cumprod();dd=eq/eq.cummax()-1;tr=dd.idxmin();pk=eq.loc[:tr].idxmax();rec=None
    peakval=float(eq.loc[pk]);aft=eq.loc[tr:];rr=aft[aft>=peakval]
    if len(rr):rec=rr.index[0]
    return {'peak':str(pk.date()),'trough':str(tr.date()),'recovery':str(rec.date()) if rec is not None else None,'mdd':float(dd.loc[tr]),'months_peak_to_trough':float((tr-pk).days/30.44),'months_to_recovery':float((rec-pk).days/30.44) if rec is not None else None}

def qsignal_nogate(eqret):
    vol=eqret.rolling(WIN).std()*np.sqrt(252);te=(TARGET/vol).clip(FLOOR,CAP);cur=0.;aa=[]
    for z in te:
        if pd.isna(z):aa.append(cur);continue
        if z<cur:cur=float(z)
        elif z-cur>INC:cur=float(z)
        aa.append(cur)
    last=float(aa[0]);oo=[]
    for z in aa:
        if abs(z-last)>SIGNAL_DEADBAND:last=float(z)
        oo.append(last)
    return pd.Series(oo,index=eqret.index).shift(1).fillna(0)

QNG=qsignal_nogate(syn);control_ng=daily_control(syn,QNG)
print('DAILY_CONTROL_NOGATE_FULL',json.dumps({**metrics_ret(control_ng),**rolling_stats(control_ng),**dd_diag(control_ng)}))
print('DAILY_CONTROL_GATE_DD',json.dumps(dd_diag(control)))
for nm,a,b in [('BLACK_MONDAY','1987-08-01','1988-03-31'),('ASIA_LTCM','1997-07-01','1998-12-31'),('DOTCOM','2000-01-01','2002-12-31'),('GFC','2007-01-01','2009-06-30'),('EURO_2011','2011-01-01','2011-12-31'),('INFLATION_2022','2022-01-01','2022-12-31')]:
    print('DAILY_PAIR_REGIME',nm,json.dumps({'gate':submetric(control,a,b),'nogate':submetric(control_ng,a,b),'avg_q_gate':float(Q.loc[(Q.index>=a)&(Q.index<=b)].mean()),'avg_q_nogate':float(QNG.loc[(QNG.index>=a)&(QNG.index<=b)].mean())}))
for name,z in OUT.items():
    yr=((1+z).groupby(z.index.year).prod()-1).sort_values()
    print('VARIANT_DD',name,json.dumps(dd_diag(z)))
    print('WORST_YEARS',name,json.dumps({str(int(k)):float(v) for k,v in yr.head(7).items()}))
for name in ['BASE','NO_GATE','H60','H80','FIN_PLUS_100BP','GOLD_DRAG_70BP','MF_SCALE_75','MF_FEE_150BP']:
    z=OUT[name]
    for nm,a,b in [('BLACK_MONDAY','1987-08-01','1988-03-31'),('ASIA_LTCM','1997-07-01','1998-12-31')]:
        print('EXTRA_REGIME',name,nm,json.dumps(submetric(z,a,b)))
'''

s += extra
exec(compile(s,'analysis_30y_synthetic_dotcom_runtime.py','exec'),globals(),globals())
