import json, os
import numpy as np
import pandas as pd

CANDIDATE_FILE="walk_forward_all_candidates.csv"; FINAL_FILE="adversarial_final_candidates.csv"
FOLD_FILE="adversarial_multi_oos_folds.csv"; OOS_FILE="adversarial_oos_results.csv"; DEV_ALL_FILE="adversarial_dev_all_combos.csv"
START=pd.Timestamp(os.getenv("WF_START_DATE","2021-01-01")); END=pd.Timestamp(os.getenv("WF_END_DATE","2026-08-22"))
FOLDS=int(os.getenv("WF_OOS_FOLDS","4")); DAYS=max(40,int(os.getenv("WF_OOS_DAYS","252"))//FOLDS); TOP_N=int(os.getenv("WF_TOP_N","10"))
PURGE=int(os.getenv("WF_PURGE_DAYS","7")); EMBARGO=int(os.getenv("WF_EMBARGO_DAYS","7")); CAPITAL=float(os.getenv("WF_INITIAL_CAPITAL","1000000"))
MIN_TRADES=int(os.getenv("WF_MIN_TOTAL_OOS_TRADES","20")); MIN_MONTHLY=float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO","0.55"))*100
MAX_DD=float(os.getenv("WF_MAX_OOS_DD","35")); MIN_POSITIVE=int(os.getenv("WF_MIN_POSITIVE_FOLDS","3")); SMOOTH=12.0
STRATEGY="REGIME_EXPECTED_RETURN_TOP10"

def prepare(x):
    need=["date","ticker","score","up_prob","down_prob","price","take_profit","stop_loss","return"]
    missing=[c for c in need if c not in x]
    if missing: raise RuntimeError("候補CSVの不足列: "+", ".join(missing))
    x=x.copy(); x.date=pd.to_datetime(x.date,errors="coerce").dt.normalize()
    for c in ["score","up_prob","down_prob","price","take_profit","stop_loss","return"]: x[c]=pd.to_numeric(x[c],errors="coerce")
    x["relative_strength"]=pd.to_numeric(x.get("relative_strength",0),errors="coerce").fillna(0)
    if "market_regime" not in x:
        trend=x.get("nikkei_uptrend",False)
        trend=pd.Series(trend,index=x.index).astype(str).str.lower().isin(["true","1","yes"])
        x["market_regime"]=np.where(trend,"RISK_ON","RISK_OFF")
    x["market_regime"]=x.market_regime.astype(str).where(x.market_regime.astype(str).isin(["RISK_ON","NEUTRAL","RISK_OFF"]),"NEUTRAL")
    x=x.dropna(subset=need); x=x[(x.date>=START)&(x.date<=END)].copy()
    x["individual_strength"]=.45*(x.up_prob-x.down_prob)+.40*x.score+.15*(x.relative_strength*100).clip(-20,20)
    x["strength_bucket"]=pd.cut(x.individual_strength,[-np.inf,25,45,65,np.inf],labels=["LOW","MID","HIGH","ELITE"]).astype(str)
    return x

def expectancy(x):
    if x.empty:return {"global":0.0,"regimes":{},"groups":{}}
    g=float(x["return"].mean()); r={}; q={}
    for key,row in x.groupby("market_regime")["return"].agg(["mean","count"]).iterrows():
        n=float(row["count"]); r[str(key)]=(n*row["mean"]+SMOOTH*g)/(n+SMOOTH)
    for (reg,bucket),row in x.groupby(["market_regime","strength_bucket"])["return"].agg(["mean","count"]).iterrows():
        n=float(row["count"]); parent=r.get(str(reg),g); q[f"{reg}|{bucket}"]=(n*row["mean"]+SMOOTH*parent)/(n+SMOOTH)
    return {"global":g,"regimes":r,"groups":q}

def select(x,m):
    x=x.copy(); key=x.market_regime.astype(str)+"|"+x.strength_bucket.astype(str)
    x["expected_return"]=key.map(m["groups"]).fillna(x.market_regime.map(m["regimes"])).fillna(m["global"])
    return x.sort_values(["date","expected_return","individual_strength","score"],ascending=[True,False,False,False]).groupby("date",group_keys=False).head(TOP_N)

def stat(x):
    if x.empty:return dict(signals=0,avg_return=0.,expected_value=0.,pf=0.,dd=0.,monthly_positive_ratio=0.,compound_return=0.,final_capital=CAPITAL)
    z=x[["date","return"]].sort_values("date"); daily=z.groupby("date")["return"].mean(); eq=(1+daily/100).cumprod()
    gains=float(z.loc[z["return"]>0,"return"].sum()); losses=float(-z.loc[z["return"]<0,"return"].sum()); pf=gains/losses if losses else (99. if gains else 0.)
    monthly=daily.groupby(daily.index.to_period("M")).apply(lambda s:((1+s/100).prod()-1)*100)
    return dict(signals=len(z),avg_return=float(z["return"].mean()),expected_value=float(z["return"].mean()),pf=float(pf),dd=float((eq/eq.cummax()-1).min()*100),monthly_positive_ratio=float((monthly>0).mean()*100) if len(monthly) else 0.,compound_return=float((eq.iloc[-1]-1)*100),final_capital=float(CAPITAL*eq.iloc[-1]))

def main():
    if not os.path.exists(CANDIDATE_FILE):raise RuntimeError(f"{CANDIDATE_FILE} がありません")
    c=prepare(pd.read_csv(CANDIDATE_FILE)); dates=sorted(c.date.unique())
    if len(dates)<FOLDS*DAYS+120:raise RuntimeError(f"複数OOSに必要な営業日が不足: {len(dates)}")
    print("REGIME-CONDITIONED EXPECTED-RETURN MULTI-OOS GATE")
    rows=[]; diag=[]
    for fold in range(1,FOLDS+1):
        end=len(dates)-(FOLDS-fold)*DAYS; start=end-DAYS; oos_dates=dates[start:end]; pre=dates[:max(0,start-PURGE-EMBARGO)]
        if len(pre)<120:raise RuntimeError(f"Fold {fold}: 学習期間不足")
        cut=int(len(pre)*.60); dev=pre[:max(1,cut-PURGE)]; val=pre[min(len(pre),cut+EMBARGO):]; ccut=max(40,len(dev)//2)
        cal=c[c.date.isin(dev[:ccut])]; de=c[c.date.isin(dev[ccut:])]; va=c[c.date.isin(val)]; oo=c[c.date.isin(oos_dates)]
        dm=expectancy(cal); ds=select(de,dm); vm=expectancy(pd.concat([cal,de])); vs=select(va,vm); om=expectancy(pd.concat([cal,de,va])); osel=select(oo,om)
        a,b,o=stat(ds),stat(vs),stat(osel); passed=o["signals"]>=MIN_TRADES and o["pf"]>=1 and o["avg_return"]>0 and o["dd"]>=-MAX_DD and o["compound_return"]>0 and o["monthly_positive_ratio"]>=MIN_MONTHLY
        rows.append(dict(fold=fold,strategy=STRATEGY,up=0,score=0,nikkei=False,tp=3.,sl=1.5,hold=5,**{f"dev_{k}":v for k,v in a.items()},**{f"validation_{k}":v for k,v in b.items()},**{f"oos_{k}":v for k,v in o.items()},oos_pass=passed,regime_expectancy_json=json.dumps(om,ensure_ascii=False,sort_keys=True)))
        diag.append(dict(fold=fold,calibration_rows=len(cal),dev_rows=len(de),validation_rows=len(va),oos_rows=len(oo),dev_selected=len(ds),validation_selected=len(vs),oos_selected=len(osel),regime_groups=len(om["groups"])))
        print(f"Fold {fold}: OOS {o['signals']} | PF={o['pf']:.2f} | EV={o['expected_value']:+.3f}%")
    f=pd.DataFrame(rows); f.to_csv(FOLD_FILE,index=False,encoding="utf-8-sig"); f.to_csv(OOS_FILE,index=False,encoding="utf-8-sig"); pd.DataFrame(diag).to_csv(DEV_ALL_FILE,index=False,encoding="utf-8-sig")
    positive=int((f.oos_compound_return>0).sum()); total=int(f.oos_signals.sum()); pf=float(f.oos_pf.replace(np.inf,99).min()); avg=float(f.oos_avg_return.mean()); monthly=float(f.oos_monthly_positive_ratio.mean()); dd=float(f.oos_dd.min()); compound=float(np.prod(1+f.oos_compound_return/100)-1)*100
    passed=positive>=MIN_POSITIVE and total>=MIN_TRADES and pf>=1 and avg>0 and monthly>=MIN_MONTHLY and dd>=-MAX_DD and compound>0
    final=dict(strategy=STRATEGY,folds=FOLDS,positive_oos_folds=positive,oos_signals=total,oos_pf_min=pf,oos_pf_mean=float(f.oos_pf.replace(np.inf,99).mean()),oos_avg_return_mean=avg,oos_monthly_positive_ratio_mean=monthly,oos_worst_dd=dd,oos_compound_return=compound,up=0,score=0,nikkei=False,tp=3.,sl=1.5,hold=5,final_status="PASS" if passed else "FAIL",up_threshold=0,score_threshold=0,nikkei_filter=False,tp_multiplier=3.,sl_multiplier=1.5,hold_days=5,validation_signals=int(f.validation_signals.sum()),validation_win_rate=0.,validation_avg_return=float(f.validation_avg_return.mean()),validation_pf=float(f.validation_pf.replace(np.inf,99).min()),validation_dd=float(f.validation_dd.min()),oos_win_rate=0.,oos_avg_return=avg,oos_pf=pf,oos_dd=dd,oos_validation_pf_ratio=1.,oos_monthly_positive_ratio=monthly,oos_compound_final_capital=CAPITAL*(1+compound/100),oos_expected_value=avg,regime_expectancy_json=json.dumps(expectancy(c),ensure_ascii=False,sort_keys=True),profit_objective=monthly*.45+np.clip(compound,-100,1000)*.35+np.clip(pf,0,8)*2)
    pd.DataFrame([final]).to_csv(FINAL_FILE,index=False,encoding="utf-8-sig")
    print(f"4-FOLD結果: {'PASS' if passed else 'FAIL'} | positive={positive}/{FOLDS} | total={total} | PF={pf:.2f} | EV={avg:+.3f}%")
if __name__=="__main__":main()
