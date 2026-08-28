"""LONG/SHORT各3段階=6戦略を同一基準で5分足バックテストする。"""
import os
import time
from datetime import time as dtime
import numpy as np
import pandas as pd
import yfinance as yf

HISTORY_FILE = os.getenv("TOP3_HISTORY_FILE", "prediction_history.csv")
OUT_FILE = "intraday_strategy_comparison.csv"
START_DATE = pd.Timestamp(os.getenv("IS_START_DATE", (pd.Timestamp.today() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")))
END_DATE = pd.Timestamp(os.getenv("IS_END_DATE", pd.Timestamp.today().strftime("%Y-%m-%d")))
TOP_N, ATR_PERIOD = 3, 14
ENTRY_START, ENTRY_END, FORCED_EXIT = dtime(9,15), dtime(10,0), dtime(15,25)
MAX_GAP_PCT = float(os.getenv("AUTO_ENTRY_MAX_GAP", "5.0"))
ATR_TP = float(os.getenv("AUTO_ENTRY_ATR_TP", "2.0"))
ATR_SL = float(os.getenv("AUTO_ENTRY_ATR_SL", "1.0"))
FEE_RATE = float(os.getenv("IT_FEE_RATE", "0.00055"))
INITIAL_CAPITAL = float(os.getenv("IT_INITIAL_CAPITAL", "1000000"))
MIN_TRADES = int(os.getenv("IS_MIN_TRADES", "10"))

STAGES = {
    "STANDARD": {"min_score":65.0,"min_prob":55.0,"min_vol_ratio":1.0},
    "RELAXED": {"min_score":60.0,"min_prob":52.0,"min_vol_ratio":0.9},
    "LOOSE": {"min_score":55.0,"min_prob":50.0,"min_vol_ratio":0.8},
}
STRATEGIES = {**{f"LONG_{k}":{**v,"direction":"LONG","stage":k} for k,v in STAGES.items()},
              **{f"SHORT_{k}":{**v,"direction":"SHORT","stage":k} for k,v in STAGES.items()}}


def norm_ohlcv(df):
    if df is None or df.empty: return None
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    need=["Open","High","Low","Close","Volume"]
    if any(c not in df.columns for c in need): return None
    idx=pd.to_datetime(df.index)
    if getattr(idx,"tz",None) is not None: idx=idx.tz_convert("Asia/Tokyo").tz_localize(None)
    else: idx=idx.tz_localize("UTC").tz_convert("Asia/Tokyo").tz_localize(None)
    df=df.copy(); df.index=idx
    for c in need: df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_index()


def load_top3_history():
    if not os.path.exists(HISTORY_FILE): raise ValueError(f"{HISTORY_FILE} がありません")
    df=pd.read_csv(HISTORY_FILE,encoding="utf-8-sig")
    req={"date","ticker","score"}; miss=req-set(df.columns)
    if miss: raise ValueError(f"必要列不足: {sorted(miss)}")
    df["date"]=pd.to_datetime(df["date"],errors="coerce")
    df["score"]=pd.to_numeric(df["score"],errors="coerce")
    for c in ["probability","up_probability","down_probability"]:
        if c in df.columns: df[c]=pd.to_numeric(df[c],errors="coerce")
        else: df[c]=np.nan
    df["up_prob_used"]=df["up_probability"].where(df["up_probability"].notna(),df["probability"])
    df["short_prob_used"]=df["down_probability"]
    df=df.dropna(subset=["date","ticker","score"])
    df=df[(df["date"]>=START_DATE)&(df["date"]<=END_DATE)].copy()
    if df.empty:return df
    df=df.sort_values(["date","score"],ascending=[True,False]).drop_duplicates(["date","ticker"],keep="first")
    df["calc_rank"]=df.groupby("date")["score"].rank(method="first",ascending=False)
    return df[df["calc_rank"]<=TOP_N].copy()


def daily_market_features(ticker,start,end):
    try: d=yf.download(ticker,start=start,end=end,interval="1d",auto_adjust=True,progress=False,threads=False)
    except Exception as e: print(f"{ticker}取得失敗: {e}"); return pd.DataFrame()
    d=norm_ohlcv(d)
    if d is None:return pd.DataFrame()
    c=d["Close"]; out=pd.DataFrame(index=d.index); out["close"]=c
    out["ma25"]=c.rolling(25).mean(); out["ma75"]=c.rolling(75).mean(); out["ret5"]=c.pct_change(5)*100
    delta=c.diff(); gain=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean(); loss=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    out["rsi"]=100-(100/(1+(gain/loss.replace(0,np.nan))))
    return out


def build_market_bias(start,end):
    s=start-pd.Timedelta(days=120); e=end+pd.Timedelta(days=2)
    n=daily_market_features("^N225",s,e); f=daily_market_features("NIY=F",s,e)
    dates=sorted(set(n.index.date)|set(f.index.date)); rows=[]
    for d in dates:
        nr=n[n.index.date==d]; fr=f[f.index.date==d]; ns=nr.iloc[-1] if not nr.empty else None; fs=fr.iloc[-1] if not fr.empty else None
        weak=strong=0
        if ns is not None:
            if pd.notna(ns["ret5"]): weak+=int(ns["ret5"]<0); strong+=int(ns["ret5"]>0)
            if pd.notna(ns["rsi"]): weak+=int(ns["rsi"]<50); strong+=int(ns["rsi"]>50)
            if pd.notna(ns["ma25"]) and pd.notna(ns["ma75"]): weak+=int(ns["ma25"]<ns["ma75"]); strong+=int(ns["ma25"]>ns["ma75"])
        if fs is not None and pd.notna(fs["ret5"]): weak+=int(fs["ret5"]<0); strong+=int(fs["ret5"]>0)
        bias="WEAK" if weak>=2 and weak>strong else ("STRONG" if strong>=2 and strong>weak else "NEUTRAL")
        rows.append({"date":pd.Timestamp(d),"market_bias":bias,"weak_score":weak,"strong_score":strong})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def download_5m(ticker,trade_date,cache):
    key=(ticker,trade_date.date())
    if key in cache:return cache[key]
    try: d=yf.download(ticker,start=pd.Timestamp(trade_date).tz_localize("Asia/Tokyo"),end=(pd.Timestamp(trade_date)+pd.Timedelta(days=1)).tz_localize("Asia/Tokyo"),interval="5m",auto_adjust=False,progress=False,threads=False)
    except Exception as e: print(f"{ticker} {trade_date:%Y-%m-%d} 5m失敗: {e}"); cache[key]=None; return None
    cache[key]=norm_ohlcv(d); time.sleep(0.15); return cache[key]


def previous_close(ticker,trade_date,cache):
    key=("prev",ticker,trade_date.date())
    if key in cache:return cache[key]
    try: d=yf.download(ticker,start=trade_date-pd.Timedelta(days=10),end=trade_date+pd.Timedelta(days=1),interval="1d",auto_adjust=False,progress=False,threads=False)
    except Exception: cache[key]=np.nan; return np.nan
    d=norm_ohlcv(d)
    if d is None: cache[key]=np.nan; return np.nan
    d=d[d.index<trade_date]; v=float(d["Close"].iloc[-1]) if not d.empty else np.nan; cache[key]=v; return v


def indicators(df):
    x=df.copy(); typ=(x["High"]+x["Low"]+x["Close"])/3
    cv=x["Volume"].replace(0,np.nan).cumsum(); x["vwap"]=(typ*x["Volume"]).cumsum()/cv
    x["vol_ma20"]=x["Volume"].rolling(20,min_periods=5).mean(); x["vol_ratio"]=x["Volume"]/x["vol_ma20"].replace(0,np.nan)
    x["ema5"]=x["Close"].ewm(span=5,adjust=False).mean(); x["ema20"]=x["Close"].ewm(span=20,adjust=False).mean()
    prev=x["Close"].shift(1); tr=pd.concat([x["High"]-x["Low"],(x["High"]-prev).abs(),(x["Low"]-prev).abs()],axis=1).max(axis=1); x["atr"]=tr.rolling(ATR_PERIOD,min_periods=5).mean()
    return x


def find_entry(intra,prev_close,score,prob,config,direction,trade_date):
    x=indicators(intra); today=x[x.index.date==trade_date.date()]; window=today[(today.index.time>=ENTRY_START)&(today.index.time<=ENTRY_END)]
    if len(window)<2 or not np.isfinite(score) or score<config["min_score"] or not np.isfinite(prob) or prob<config["min_prob"]: return None
    first_open=float(today["Open"].iloc[0]); gap=np.nan
    if np.isfinite(prev_close) and prev_close>0:
        gap=(first_open/prev_close-1)*100
        if abs(gap)>MAX_GAP_PCT:return None
    for i in range(len(window)-1):
        bar=window.iloc[i]; nxt=window.iloc[i+1]
        if pd.isna(bar["vol_ratio"]) or float(bar["vol_ratio"])<config["min_vol_ratio"]:continue
        atr=float(bar["atr"]) if pd.notna(bar["atr"]) and float(bar["atr"])>0 else float(bar["Close"])*0.005
        ep=float(nxt["Open"])
        if direction=="LONG":
            if float(bar["Close"])<=float(bar["vwap"]):continue
            tp,sl=ep+ATR_TP*atr,ep-ATR_SL*atr
        else:
            if float(bar["Close"])>=float(bar["vwap"]):continue
            tp,sl=ep-ATR_TP*atr,ep+ATR_SL*atr
        return {"entry_time":nxt.name,"entry_price":ep,"tp":tp,"sl":sl,"gap_pct":gap,"vol_ratio":float(bar["vol_ratio"]),"ema_bull":bool(float(bar["ema5"])>float(bar["ema20"]))}
    return None


def exit_trade(intra,entry,direction):
    x=indicators(intra); future=x[x.index>=entry["entry_time"]]
    if future.empty:return None
    for ts,bar in future.iterrows():
        h,l=float(bar["High"]),float(bar["Low"]); tp,sl=entry["tp"],entry["sl"]
        if direction=="LONG":
            if l<=sl and h>=tp:return ts,sl,"SL_BOTH"
            if h>=tp:return ts,tp,"TP"
            if l<=sl:return ts,sl,"SL"
        else:
            if l<=tp and h>=sl:return ts,sl,"SL_BOTH"
            if l<=tp:return ts,tp,"TP"
            if h>=sl:return ts,sl,"SL"
        if ts.time()>=FORCED_EXIT:return ts,float(bar["Close"]),"EOD"
    last=future.iloc[-1]; return future.index[-1],float(last["Close"]),"EOD"


def backtest_strategy(name,config,top3,bias_map,cache):
    direction=config["direction"]; rows=[]
    for trade_date,day in top3.groupby("date"):
        key=pd.Timestamp(trade_date).normalize(); bias=str(bias_map.loc[key,"market_bias"]) if not bias_map.empty and key in bias_map.index else "NEUTRAL"
        if direction=="LONG" and bias=="WEAK":continue
        if direction=="SHORT" and bias=="STRONG":continue
        for _,r in day.sort_values("calc_rank").head(TOP_N).iterrows():
            intra=download_5m(str(r["ticker"]),trade_date,cache)
            if intra is None:continue
            pc=previous_close(str(r["ticker"]),trade_date,cache); prob=float(r["up_prob_used"] if direction=="LONG" else r["short_prob_used"])
            entry=find_entry(intra,pc,float(r["score"]),prob,config,direction,trade_date)
            if entry is None:continue
            ex=exit_trade(intra,entry,direction)
            if ex is None:continue
            xt,xp,reason=ex; gross=((xp/entry["entry_price"]-1)*100) if direction=="LONG" else ((entry["entry_price"]/xp-1)*100); net=gross-FEE_RATE*200
            rows.append({"date":trade_date,"direction":direction,"strategy":name,"stage":config["stage"],"market_bias":bias,"rank":int(r["calc_rank"]),"ticker":r["ticker"],"entry_time":entry["entry_time"],"entry_price":entry["entry_price"],"exit_time":xt,"exit_price":xp,"reason":reason,"return_pct":net})
            break
    t=pd.DataFrame(rows)
    if t.empty:return {"strategy":name,"direction":direction,"stage":config["stage"],"trades":0,"win_rate":0.0,"avg_return_pct":0.0,"pf":0.0,"total_return_pct":0.0,"max_dd_pct":0.0,"final_capital":INITIAL_CAPITAL},t
    ret=pd.to_numeric(t["return_pct"],errors="coerce").dropna(); win=float((ret>0).mean()*100); avg=float(ret.mean()); gp=float(ret[ret>0].sum()); gl=float(-ret[ret<0].sum()); pf=gp/gl if gl>0 else float("inf"); cap=INITIAL_CAPITAL; peak=cap; mdd=0
    for r in t.sort_values("date")["return_pct"]: cap*=1+r/100; peak=max(peak,cap); mdd=min(mdd,(cap/peak-1)*100)
    return {"strategy":name,"direction":direction,"stage":config["stage"],"trades":len(t),"win_rate":win,"avg_return_pct":avg,"pf":pf,"total_return_pct":(cap/INITIAL_CAPITAL-1)*100,"max_dd_pct":mdd,"final_capital":cap},t


def main():
    top3=load_top3_history()
    if top3.empty:raise ValueError("対象期間にTOP3履歴がありません")
    bias=build_market_bias(START_DATE,END_DATE); cache={}; summaries=[]
    for name,cfg in STRATEGIES.items():
        s,t=backtest_strategy(name,cfg,top3,bias,cache); summaries.append(s)
        if not t.empty:t.to_csv(f"intraday_trades_{name}.csv",index=False,encoding="utf-8-sig")
    out=pd.DataFrame(summaries); out.to_csv(OUT_FILE,index=False,encoding="utf-8-sig"); bias.reset_index().to_csv("intraday_market_bias.csv",index=False,encoding="utf-8-sig")
    print("="*70); print("📊 LONG / SHORT 6戦略 デイトレ比較"); print("="*70); print(out.to_string(index=False))
    q=out[out["trades"]>=MIN_TRADES]
    if not q.empty:
        b=q.sort_values(["pf","avg_return_pct","final_capital","trades"],ascending=[False,False,False,False]).iloc[0]
        print(f"✅ 最良候補: {b['strategy']} / PF={b['pf']:.3f} / 件数={int(b['trades'])} / 勝率={b['win_rate']:.1f}%")
    else: print(f"⚠ {MIN_TRADES}件以上の戦略なし")

if __name__=="__main__":main()
