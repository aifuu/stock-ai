"""日足AI TOP3を、日経/先物の地合いに応じてLONGまたはSHORTで監視する。"""
import json, os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, yfinance as yf
from common import COMPANY_NAMES, is_tse_trading_day, send

JST=ZoneInfo("Asia/Tokyo"); TOP3_FILE=os.getenv("TOP3_HISTORY_FILE","intraday_today_top3.csv")
POLICY_FILE="strategy_policy.json"; STATE_FILE="intraday_auto_entry_state.json"; HISTORY_FILE="paper_intraday_history.csv"
START=dtime(9,15); END=dtime(10,0); STAGES={"STANDARD":{"min_score":65.0,"min_prob":55.0,"min_vol_ratio":1.0},"RELAXED":{"min_score":60.0,"min_prob":52.0,"min_vol_ratio":0.9},"LOOSE":{"min_score":55.0,"min_prob":50.0,"min_vol_ratio":0.8}}

def read_json(p):
    try:return json.load(open(p,encoding="utf-8"))
    except Exception:return {}

def norm(d):
    if d is None or d.empty:return None
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    idx=pd.to_datetime(d.index)
    if getattr(idx,"tz",None) is not None:idx=idx.tz_convert(JST).tz_localize(None)
    else:idx=idx.tz_localize("UTC").tz_convert(JST).tz_localize(None)
    d=d.copy();d.index=idx
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in d.columns:return None
        d[c]=pd.to_numeric(d[c],errors="coerce")
    return d.sort_index()

def market_bias():
    def close(t):
        try:return pd.to_numeric(norm(yf.download(t,period="6mo",interval="1d",auto_adjust=True,progress=False,threads=False))["Close"],errors="coerce")
        except Exception:return pd.Series(dtype=float)
    n,f=close("^N225"),close("NIY=F"); w=s=0
    for x in [n,f]:
        if len(x)>=6:
            r=x.iloc[-1]/x.iloc[-6]-1; w+=r<0; s+=r>0
    if len(n)>=75:
        a=n.rolling(25).mean().iloc[-1]; b=n.rolling(75).mean().iloc[-1]; w+=a<b; s+=a>b
    return "WEAK" if w>=2 and w>s else ("STRONG" if s>=2 and s>w else "NEUTRAL")

def load_top3():
    if not os.path.exists(TOP3_FILE):return pd.DataFrame()
    d=pd.read_csv(TOP3_FILE,encoding="utf-8-sig")
    if not {"date","ticker","score"}.issubset(d.columns):return pd.DataFrame()
    d["date"]=pd.to_datetime(d["date"],errors="coerce"); d["score"]=pd.to_numeric(d["score"],errors="coerce")
    if "probability" not in d.columns:d["probability"]=np.nan
    d["probability"]=pd.to_numeric(d["probability"],errors="coerce")
    today=datetime.now(JST).date(); d=d[d.date.dt.date==today].sort_values("score",ascending=False).drop_duplicates("ticker").head(3).copy(); d["rank"]=range(1,len(d)+1); return d

def indicators(d):
    x=norm(d); typ=(x.High+x.Low+x.Close)/3; cv=x.Volume.replace(0,np.nan).cumsum(); x["vwap"]=(typ*x.Volume).cumsum()/cv
    x["vol_ma20"]=x.Volume.rolling(20,min_periods=5).mean(); x["vol_ratio"]=x.Volume/x.vol_ma20.replace(0,np.nan)
    prev=x.Close.shift(1); tr=pd.concat([x.High-x.Low,(x.High-prev).abs(),(x.Low-prev).abs()],axis=1).max(axis=1); x["atr"]=tr.rolling(14,min_periods=5).mean(); x["ema5"]=x.Close.ewm(span=5,adjust=False).mean(); x["ema20"]=x.Close.ewm(span=20,adjust=False).mean(); return x

def find_entry(df,row,direction,cfg,atr_tp,atr_sl):
    x=indicators(df); today=x[x.index.date==datetime.now(JST).date()]; w=today[(today.index.time>=START)&(today.index.time<=END)]
    score=float(row.score); prob=float(row.probability)
    if len(w)<2 or not np.isfinite(score) or score<cfg["min_score"] or not np.isfinite(prob) or prob<cfg["min_prob"]:return None
    for i in range(len(w)-1):
        b,n=w.iloc[i],w.iloc[i+1]
        if pd.isna(b.vol_ratio) or b.vol_ratio<cfg["min_vol_ratio"]:continue
        if direction=="LONG" and b.Close<=b.vwap:continue
        if direction=="SHORT" and b.Close>=b.vwap:continue
        atr=float(b.atr) if pd.notna(b.atr) and b.atr>0 else float(b.Close)*.005; ep=float(n.Open)
        return {"entry_time":n.name.strftime("%H:%M"),"entry_price":ep,"tp":ep+(atr_tp*atr if direction=="LONG" else -atr_tp*atr),"sl":ep+(-atr_sl*atr if direction=="LONG" else atr_sl*atr),"vol_ratio":float(b.vol_ratio),"ema_bull":bool(b.ema5>b.ema20)}
    return None

def save_entry(date,row,entry,direction,strategy,bias):
    rec={"date":str(date),"ticker":str(row.ticker),"rank":int(row.rank),"strategy":strategy,"direction":direction,"market_bias":bias,"score":float(row.score),"probability":float(row.probability),**entry,"exit_time":"","exit_price":np.nan,"exit_reason":"","return_pct":np.nan,"status":"OPEN"}
    hist=pd.read_csv(HISTORY_FILE,encoding="utf-8-sig") if os.path.exists(HISTORY_FILE) else pd.DataFrame()
    for c in rec:
        if c not in hist.columns:hist[c]=np.nan
    if not hist.empty:hist=hist[hist.date.astype(str)!=str(date)]
    pd.concat([hist,pd.DataFrame([rec])],ignore_index=True).to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig")

def main():
    now=datetime.now(JST); date=now.date()
    if not is_tse_trading_day(date) or now.time()<START or now.time()>dtime(15,25):return
    top=load_top3();
    if len(top)!=3:return
    state=read_json(STATE_FILE)
    if state.get("date")==str(date) and state.get("signaled"):return
    p=read_json(POLICY_FILE); strategy=str(p.get("intraday_strategy","")).upper(); direction=str(p.get("intraday_direction","")).upper(); stage=str(p.get("intraday_stage","")).upper()
    if direction not in {"LONG","SHORT"} or stage not in STAGES:
        print("⚠️ LONG/SHORT採用policyなし。見送り");return
    bias=market_bias()
    if (direction=="LONG" and bias=="WEAK") or (direction=="SHORT" and bias=="STRONG"):
        print(f"市場環境={bias} / 採用方向={direction} → 見送り");return
    # LONGは上昇確率、SHORTは下落確率を使う。historyにdown_probabilityがある場合はそちらを優先。
    if direction=="SHORT":
        if "down_probability" not in top.columns:
            print("⚠️ 下落確率データなし。SHORT見送り");return
        top["probability"]=pd.to_numeric(top["down_probability"],errors="coerce")
    cfg=STAGES[stage]; tp=float(p.get("intraday_atr_tp",2.0)); sl=float(p.get("intraday_atr_sl",1.0))
    for _,row in top.sort_values("rank").iterrows():
        d=yf.download(str(row.ticker),period="5d",interval="5m",auto_adjust=False,progress=False,threads=False); d=norm(d)
        if d is None:continue
        entry=find_entry(d,row,direction,cfg,tp,sl)
        if entry is None:continue
        save_entry(date,row,entry,direction,strategy,bias)
        state={"date":str(date),"signaled":True,"closed":False,"strategy":strategy,"direction":direction,"market_bias":bias,"ticker":str(row.ticker),"rank":int(row.rank),**entry}
        with open(STATE_FILE,"w",encoding="utf-8") as f:json.dump(state,f,ensure_ascii=False,indent=2)
        label="買い" if direction=="LONG" else "空売り"; prob_label="上昇" if direction=="LONG" else "下落"
        send(f"🚨 自動デイトレ{label}シグナル\n日付: {date}\n戦略: {strategy}\nTOP{int(row.rank)}: {row.ticker} {COMPANY_NAMES.get(str(row.ticker),'')}\n市場環境: {bias}\nAIスコア: {float(row.score):.1f}\n{prob_label}確率: {float(row.probability):.1f}%\nエントリー: {entry['entry_time']}\n価格: {entry['entry_price']:.1f}\nTP: {entry['tp']:.1f} / SL: {entry['sl']:.1f}\n※ペーパートレード")
        return
    print(f"採用戦略 {strategy}: TOP3すべて条件未成立")

if __name__=="__main__":main()
