"""6戦略(LONG/SHORT×STANDARD/RELAXED/LOOSE)の最良戦略を採用する。"""
import json, os
from datetime import datetime
import pandas as pd

POLICY_FILE="strategy_policy.json"
INPUT_FILE="intraday_strategy_comparison.csv"
MIN_TRADES=int(os.getenv("ID_MIN_TRADES","10"))
MIN_PF=float(os.getenv("ID_MIN_PF","1.0"))
MIN_AVG=float(os.getenv("ID_MIN_AVG_RETURN","0.0"))
CONFIGS={
 "STANDARD":{"min_score":65.0,"min_prob":55.0,"min_vol_ratio":1.0},
 "RELAXED":{"min_score":60.0,"min_prob":52.0,"min_vol_ratio":0.9},
 "LOOSE":{"min_score":55.0,"min_prob":50.0,"min_vol_ratio":0.8},
}

def load_policy():
    if not os.path.exists(POLICY_FILE): return {}
    try:
        p=json.load(open(POLICY_FILE,encoding="utf-8")); return p if isinstance(p,dict) else {}
    except Exception as e:
        print(f"⚠ policy読込失敗: {e}"); return {}

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"⚠ {INPUT_FILE}なし: 既存policyを維持"); return
    df=pd.read_csv(INPUT_FILE)
    req={"strategy","direction","stage","trades","pf","win_rate","avg_return_pct","total_return_pct","max_dd_pct","final_capital"}
    miss=req-set(df.columns)
    if miss: raise SystemExit(f"❌ 比較結果列不足: {sorted(miss)}")
    for c in req-{"strategy","direction","stage"}: df[c]=pd.to_numeric(df[c],errors="coerce")
    q=df[(df.trades>=MIN_TRADES)&(df.pf>=MIN_PF)&(df.avg_return_pct>MIN_AVG)].copy()
    if q.empty:
        print(f"🟡 採用条件未達。既存デイトレpolicyを維持 (trades>={MIN_TRADES}, PF>={MIN_PF})"); return
    b=q.sort_values(["pf","avg_return_pct","final_capital","trades"],ascending=False).iloc[0]
    direction=str(b.direction).upper(); stage=str(b.stage).upper()
    if direction not in {"LONG","SHORT"} or stage not in CONFIGS: raise SystemExit(f"❌ 不正な採用戦略: {b.strategy}")
    c=CONFIGS[stage]; p=load_policy()
    p.update({
      "intraday_strategy":str(b.strategy), "intraday_direction":direction, "intraday_stage":stage,
      "intraday_min_score":c["min_score"], "intraday_min_prob":c["min_prob"],
      "intraday_min_up_prob":c["min_prob"] if direction=="LONG" else None,
      "intraday_min_down_prob":c["min_prob"] if direction=="SHORT" else None,
      "intraday_min_vol_ratio":c["min_vol_ratio"],
      "intraday_atr_tp":float(os.getenv("AUTO_ENTRY_ATR_TP","2.0")), "intraday_atr_sl":float(os.getenv("AUTO_ENTRY_ATR_SL","1.0")),
      "intraday_trades":int(b.trades), "intraday_pf":float(b.pf), "intraday_win_rate":float(b.win_rate),
      "intraday_avg_return_pct":float(b.avg_return_pct), "intraday_total_return_pct":float(b.total_return_pct),
      "intraday_max_dd_pct":float(b.max_dd_pct), "intraday_final_capital":float(b.final_capital),
      "intraday_updated_at":datetime.now().isoformat(timespec="seconds")
    })
    with open(POLICY_FILE,"w",encoding="utf-8") as f: json.dump(p,f,ensure_ascii=False,indent=2)
    print("✅ デイトレ採用:",b.strategy,direction,"PF=",b.pf,"件数=",int(b.trades))

if __name__=="__main__": main()
