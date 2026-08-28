"""LONG/SHORT×STANDARD/RELAXED/LOOSEの6戦略を同一基準でバックテストする。"""
import os, time
from datetime import time as dtime
import numpy as np, pandas as pd, yfinance as yf

HISTORY_FILE=os.getenv("TOP3_HISTORY_FILE","prediction_history.csv")
OUT_FILE="intraday_strategy_comparison.csv"
START_DATE=pd.Timestamp(os.getenv("IS_START_DATE",(pd.Timestamp.today()-pd.Timedelta(days=60)).strftime("%Y-%m-%d")))
END_DATE=pd.Timestamp(os.getenv("IS_END_DATE",pd.Timestamp.today().strftime("%Y-%m-%d")))
TOP_N=3; ATR_PERIOD=14; ENTRY_START=dtime(9,15); ENTRY_END=dtime(10,0); FORCED_EXIT=dtime(15,25)
MAX_GAP_PCT=float(os.getenv("AUTO_ENTRY_MAX_GAP","5.0")); ATR_TP=float(os.getenv("AUTO_ENTRY_ATR_TP","2.0")); ATR_SL=float(os.getenv("AUTO_ENTRY_ATR_SL","1.0")); FEE_RATE=float(os.getenv("IT_FEE_RATE","0.00055")); INITIAL_CAPITAL=float(os.getenv("IT_INITIAL_CAPITAL","1000000")); MIN_TRADES=int(os.getenv("IS_MIN_TRADES","10"))
STOCK_TREND_MIN=int(os.getenv("IS_STOCK_TREND_MIN","2"))
STAGES={"STANDARD":{"min_score":65.0,"min_prob":55.0,"min_vol_ratio":1.0},"RELAXED":{"min_score":60.0,"min_prob":52.0,"min_vol_ratio":0.9},"LOOSE":{"min_score":55.0,"min_prob":50.0,"min_vol_ratio":0.8}}
STRATEGIES={**{f"LONG_{k}":{**v,"direction":"LONG","stage":k} for k,v in STAGES.items()},**{f"SHORT_{k}":{**v,"direction":"SHORT","stage":k} for k,v in STAGES.items()}}

def norm(df):
    if df is None or df.empty:return None
    if isinstance(df.columns,pd.MultiIndex):df.columns=df.columns.get_level_values(0)
    need=["Open","High","Low","Close","Volume"]
    if any(c not in df.columns for c in need):return None
    idx=pd.to_datetime(df.index)
    if getattr(idx,"tz",None) is not None:idx=idx.tz_convert("Asia/Tokyo").tz_localize(None)
    else:idx=idx.tz_localize("UTC").tz_convert("Asia/Tokyo").tz_localize(None)
    df=df.copy();df.index=idx
    for c in need:df[c]=pd.to_numeric(df[c],errors="coerce")
    return df.sort_index()

def load_top3():
    if not os.path.exists(HISTORY_FILE):raise ValueError(f"{HISTORY_FILE} がありません")
    d=pd.read_csv(HISTORY_FILE,encoding="utf-8-sig"); req={"date","ticker","score"}; miss=req-set(d.columns)
    if miss:raise ValueError(f"必要列不足: {sorted(miss)}")
    d["date"]=pd.to_datetime(d["date"],errors="coerce"); d["score"]=pd.to_numeric(d["score"],errors="coerce")
    for c in ["probability","up_probability","down_probability"]:
        d[c]=pd.to_numeric(d[c],errors="coerce") if c in d.columns else np.nan
    d["up_prob_used"]=d["up_probability"].where(d["up_probability"].notna(),d["probability"]); d["short_prob_used"]=d["down_probability"]
    d=d.dropna(subset=["date","ticker","score"]); d=d[(d.date>=START_DATE)&(d.date<=END_DATE)].copy()
    if d.empty:return d
    d=d.sort_values(["date","score"],ascending=[True,False]).drop_duplicates(["date","ticker"],keep="first")
    d["calc_rank"]=d.groupby("date")["score"].rank(method="first",ascending=False)
    return d[d.calc_rank<=TOP_N].copy()

def daily_features(ticker,start,end):
    try:d=norm(yf.download(ticker,start=start,end=end,interval="1d",auto_adjust=True,progress=False,threads=False))
    except Exception:return pd.DataFrame()
    if d is None:return pd.DataFrame()
    c=d.Close; o=pd.DataFrame(index=d.index); o["close"]=c; o["ma25"]=c.rolling(25).mean(); o["ma75"]=c.rolling(75).mean(); o["ma25_slope5"]=o.ma25/o.ma25.shift(5)-1; o["ret5"]=c.pct_change(5); o["ret20"]=c.pct_change(20); return o

def stock_trend_score(feat,date,direction):
    if feat.empty:return 0
    x=feat[feat.index<date]
    if x.empty:return 0
    r=x.iloc[-1]; vals=[]
    if direction=="LONG":
        vals=[r.close>r.ma25, r.ma25>r.ma75, r.ma25_slope5>0, r.ret5>0, r.ret20>0]
    else:
        vals=[r.close<r.ma25, r.ma25<r.ma75, r.ma25_slope5<0, r.ret5<0, r.ret20<0]
    return int(sum(bool(v) for v in vals if pd.notna(v)))

def market_bias_map(start,end):
    s=start-pd.Timedelta(days=120); e=end+pd.Timedelta(days=2); n=daily_features("^N225",s,e); f=daily_features("NIY=F",s,e)
    dates=sorted(set(n.index.date)|set(f.index.date)); rows=[]
    for d in dates:
        day=pd.Timestamp(d); ns=n[n.index.date==d]; fs=f[f.index.date==d]; nrow=ns.iloc[-1] if not ns.empty else None; frow=fs.iloc[-1] if not fs.empty else None; w=st=0
        if nrow is not None:
            if pd.notna(nrow.ret5):w+=int(nrow.ret5<0);st+=int(nrow.ret5>0)
            if pd.notna(nrow.ma25) and pd.notna(nrow.ma75):w+=int(nrow.ma25<nrow.ma75);st+=int(nrow.ma25>nrow.ma75)
            if pd.notna(nrow.ret20):w+=int(nrow.ret20<0);st+=int(nrow.ret20>0)
        if frow is not None and pd.notna(frow.ret5):w+=int(frow.ret5<0);st+=int(frow.ret5>0)
        bias="WEAK" if w>=2 and w>st else ("STRONG" if st>=2 and st>w else "NEUTRAL")
        rows.append({"date":day,"market_bias":bias,"weak_score":w,"strong_score":st})
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()

def download_5m(ticker,date,cache):
    key=(ticker,date.date())
    if key in cache:return cache[key]
    try:d=yf.download(ticker,start=pd.Timestamp(date).tz_localize("Asia/Tokyo"),end=(pd.Timestamp(date)+pd.Timedelta(days=1)).tz_localize("Asia/Tokyo"),interval="5m",auto_adjust=False,progress=False,threads=False)
    except Exception:cache[key]=None;return None
    cache[key]=norm(d);time.sleep(.1);return cache[key]

def previous_close(ticker,date,cache):
    key=("pc",ticker,date.date())
    if key in cache:return cache[key]
    try:d=norm(yf.download(ticker,start=date-pd.Timedelta(days=10),end=date+pd.Timedelta(days=1),interval="1d",auto_adjust=False,progress=False,threads=False))
    except Exception:cache[key]=np.nan;return np.nan
    if d is None:cache[key]=np.nan;return np.nan
    d=d[d.index<date];v=float(d.Close.iloc[-1]) if not d.empty else np.nan;cache[key]=v;return v

def indicators(df):
    x=df.copy();typ=(x.High+x.Low+x.Close)/3;cv=x.Volume.replace(0,np.nan).cumsum();x["vwap"]=(typ*x.Volume).cumsum()/cv;x["vol_ma20"]=x.Volume.rolling(20,min_periods=5).mean();x["vol_ratio"]=x.Volume/x.vol_ma20.replace(0,np.nan);x["ema5"]=x.Close.ewm(span=5,adjust=False).mean();x["ema20"]=x.Close.ewm(span=20,adjust=False).mean();p=x.Close.shift(1);tr=pd.concat([x.High-x.Low,(x.High-p).abs(),(x.Low-p).abs()],axis=1).max(axis=1);x["atr"]=tr.rolling(ATR_PERIOD,min_periods=5).mean();return x

def find_entry(intra,prev_close,score,prob,cfg,direction,date):
    x=indicators(intra);today=x[x.index.date==date.date()];w=today[(today.index.time>=ENTRY_START)&(today.index.time<=ENTRY_END)]
    if len(w)<2 or not np.isfinite(score) or score<cfg["min_score"] or not np.isfinite(prob) or prob<cfg["min_prob"]:return None
    if np.isfinite(prev_close) and prev_close>0 and not np.isfinite(float(today.Open.iloc[0])):return None
    if np.isfinite(prev_close) and prev_close>0:
        gap=(float(today.Open.iloc[0])/prev_close-1)*100
        if abs(gap)>MAX_GAP_PCT:return None
    else:gap=np.nan
    for i in range(len(w)-1):
        b,n=w.iloc[i],w.iloc[i+1]
        if pd.isna(b.vol_ratio) or b.vol_ratio<cfg["min_vol_ratio"]:continue
        if direction=="LONG" and b.Close<=b.vwap:continue
        if direction=="SHORT" and b.Close>=b.vwap:continue
        atr=float(b.atr) if pd.notna(b.atr) and b.atr>0 else float(b.Close)*.005;ep=float(n.Open)
        return {"entry_time":n.name,"entry_price":ep,"tp":ep+(ATR_TP*atr if direction=="LONG" else -ATR_TP*atr),"sl":ep+(-ATR_SL*atr if direction=="LONG" else ATR_SL*atr),"gap_pct":gap,"vol_ratio":float(b.vol_ratio),"ema_bull":bool(b.ema5>b.ema20)}
    return None

def exit_trade(intra,e,direction):
    x=indicators(intra);f=x[x.index>=e["entry_time"]]
    for ts,b in f.iterrows():
        h,l=float(b.High),float(b.Low);tp,sl=e["tp"],e["sl"]
        if direction=="LONG":
            if l<=sl and h>=tp:return ts,sl,"SL_BOTH"
            if h>=tp:return ts,tp,"TP"
            if l<=sl:return ts,sl,"SL"
        else:
            if l<=tp and h>=sl:return ts,sl,"SL_BOTH"
            if l<=tp:return ts,tp,"TP"
            if h>=sl:return ts,sl,"SL"
        if ts.time()>=FORCED_EXIT:return ts,float(b.Close),"EOD"
    if not f.empty:return f.index[-1],float(f.Close.iloc[-1]),"EOD"
    return None

def backtest(name,cfg,top,bias_map,stock_maps,cache):
    direction=cfg["direction"];rows=[]
    for date,day in top.groupby("date"):
        key=pd.Timestamp(date).normalize();bias=str(bias_map.loc[key,"market_bias"]) if not bias_map.empty and key in bias_map.index else "NEUTRAL"
        if direction=="LONG" and bias=="WEAK":continue
        if direction=="SHORT" and bias=="STRONG":continue
        for _,r in day.sort_values("calc_rank").head(TOP_N).iterrows():
            ticker=str(r.ticker);trend=stock_trend_score(stock_maps.get(ticker,pd.DataFrame()),date,direction)
            if trend<STOCK_TREND_MIN:continue
            intra=download_5m(ticker,date,cache)
            if intra is None:continue
            prob=float(r.up_prob_used if direction=="LONG" else r.short_prob_used)
            entry=find_entry(intra,previous_close(ticker,date,cache),float(r.score),prob,cfg,direction,date)
            if entry is None:continue
            ex=exit_trade(intra,entry,direction)
            if ex is None:continue
            xt,xp,reason=ex;gross=(xp/entry["entry_price"]-1)*100 if direction=="LONG" else (entry["entry_price"]/xp-1)*100;net=gross-FEE_RATE*200
            rows.append({"date":date,"direction":direction,"strategy":name,"stage":cfg["stage"],"market_bias":bias,"stock_trend_score":trend,"ticker":ticker,"rank":int(r.calc_rank),"entry_time":entry["entry_time"],"entry_price":entry["entry_price"],"exit_time":xt,"exit_price":xp,"reason":reason,"return_pct":net});break
    t=pd.DataFrame(rows)
    if t.empty:return {"strategy":name,"direction":direction,"stage":cfg["stage"],"trades":0,"win_rate":0.0,"avg_return_pct":0.0,"pf":0.0,"total_return_pct":0.0,"max_dd_pct":0.0,"final_capital":INITIAL_CAPITAL},t
    ret=pd.to_numeric(t.return_pct,errors="coerce").dropna();gp=float(ret[ret>0].sum());gl=float(-ret[ret<0].sum());cap=INITIAL_CAPITAL;peak=cap;mdd=0
    for z in t.sort_values("date").return_pct:cap*=1+z/100;peak=max(peak,cap);mdd=min(mdd,(cap/peak-1)*100)
    return {"strategy":name,"direction":direction,"stage":cfg["stage"],"trades":len(t),"win_rate":float((ret>0).mean()*100),"avg_return_pct":float(ret.mean()),"pf":gp/gl if gl>0 else float("inf"),"total_return_pct":(cap/INITIAL_CAPITAL-1)*100,"max_dd_pct":mdd,"final_capital":cap},t

def main():
    top=load_top3();bias=market_bias_map(START_DATE,END_DATE);cache={};stock_maps={}
    for ticker in top.ticker.astype(str).unique():stock_maps[ticker]=daily_features(ticker,START_DATE-pd.Timedelta(days=120),END_DATE+pd.Timedelta(days=1))
    summaries=[]
    for name,cfg in STRATEGIES.items():
        s,t=backtest(name,cfg,top,bias,stock_maps,cache);summaries.append(s)
        if not t.empty:t.to_csv(f"intraday_trades_{name}.csv",index=False,encoding="utf-8-sig")
    out=pd.DataFrame(summaries);out.to_csv(OUT_FILE,index=False,encoding="utf-8-sig")
    if not bias.empty:bias.reset_index().to_csv("intraday_market_bias.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame([{"stock_trend_min":STOCK_TREND_MIN,"max_gap_pct":MAX_GAP_PCT,"atr_tp":ATR_TP,"atr_sl":ATR_SL}]).to_csv("intraday_backtest_config.csv",index=False,encoding="utf-8-sig")
    print("="*72);print("📊 LONG / SHORT 6戦略 + 銘柄チャート強弱バックテスト");print("="*72);print(out.to_string(index=False))
    q=out[out.trades>=MIN_TRADES]
    if not q.empty:
        b=q.sort_values(["pf","avg_return_pct","final_capital","trades"],ascending=False).iloc[0];print(f"✅ 最良候補: {b.strategy} / PF={b.pf:.3f} / 件数={int(b.trades)} / 勝率={b.win_rate:.1f}% / 株価トレンド条件={STOCK_TREND_MIN}/5")
    else:print(f"⚠ {MIN_TRADES}件以上の戦略なし")

if __name__=="__main__":main()
