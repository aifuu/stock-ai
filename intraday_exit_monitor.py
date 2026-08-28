"""LONG/SHORTペーパー取引のTP/SL/15:25決済を監視する。"""
import json, os
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, yfinance as yf
from common import is_tse_trading_day, send, COMPANY_NAMES

JST=ZoneInfo("Asia/Tokyo"); STATE_FILE="intraday_auto_entry_state.json"; HISTORY_FILE="paper_intraday_history.csv"; FORCED=dtime(15,25)
FEE=float(os.getenv("IT_FEE_RATE","0.00055"))

def read_state():
    try:return json.load(open(STATE_FILE,encoding="utf-8"))
    except Exception:return {}

def norm(d):
    if d is None or d.empty:return None
    if isinstance(d.columns,pd.MultiIndex):d.columns=d.columns.get_level_values(0)
    idx=pd.to_datetime(d.index)
    if getattr(idx,"tz",None) is not None:idx=idx.tz_convert(JST).tz_localize(None)
    else:idx=idx.tz_localize("UTC").tz_convert(JST).tz_localize(None)
    d=d.copy();d.index=idx
    return d.sort_index()

def download(ticker):
    try:d=norm(yf.download(ticker,period="5d",interval="5m",auto_adjust=False,progress=False,threads=False)); return d
    except Exception as e:print(f"{ticker}: 5m取得失敗 {e}");return None

def resolve(df,date,entry_time,tp,sl,direction,now):
    today=df[df.index.date==date]; entry_dt=pd.Timestamp(f"{date} {entry_time}"); bars=today[(today.index>=entry_dt)&(today.index<=now.replace(tzinfo=None))]
    if bars.empty:return None
    for ts,b in bars.iterrows():
        h,l=float(b.High),float(b.Low)
        if direction=="LONG":
            if l<=sl and h>=tp:return ts,sl,"SL_BOTH"
            if h>=tp:return ts,tp,"TP"
            if l<=sl:return ts,sl,"SL"
        else:
            if l<=tp and h>=sl:return ts,sl,"SL_BOTH"
            if l<=tp:return ts,tp,"TP"
            if h>=sl:return ts,sl,"SL"
        if ts.time()>=FORCED:return ts,float(b.Close),"EOD"
    if now.time()>=FORCED:
        b=bars.iloc[-1];return bars.index[-1],float(b.Close),"EOD"
    return None

def ensure_row(state):
    cols=["date","ticker","rank","strategy","direction","market_bias","score","probability","entry_time","entry_price","tp","sl","vol_ratio","ema_bull","exit_time","exit_price","exit_reason","return_pct","status"]
    if os.path.exists(HISTORY_FILE):
        try:h=pd.read_csv(HISTORY_FILE,encoding="utf-8-sig")
        except Exception:h=pd.DataFrame()
    else:h=pd.DataFrame()
    for c in cols:
        if c not in h.columns:h[c]=np.nan
    m=(h.date.astype(str)==str(state["date"]))&(h.ticker.astype(str)==str(state["ticker"]))
    rec={c:state.get(c,np.nan) for c in cols};rec.update({"date":state["date"],"ticker":state["ticker"],"exit_time":"","exit_price":np.nan,"exit_reason":"","return_pct":np.nan,"status":"OPEN"})
    if m.any():
        i=h.index[m][0]
        for c,v in rec.items():h.at[i,c]=v
    else:h=pd.concat([h,pd.DataFrame([rec])],ignore_index=True)
    h.to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig");return h

def main():
    now=datetime.now(JST);date=now.date()
    if not is_tse_trading_day(date) or now.time()<dtime(9,15):return
    s=read_state()
    if s.get("date")!=str(date) or not s.get("signaled") or s.get("closed"):return
    needed=["ticker","entry_time","entry_price","tp","sl","direction"]
    if any(k not in s for k in needed):print("❌ state不足");return
    h=ensure_row(s);m=(h.date.astype(str)==str(date))&(h.ticker.astype(str)==str(s["ticker"]))
    row=h[m].iloc[0]
    if pd.notna(row.get("exit_price")) and str(row.get("exit_price")).strip() not in ("","nan"):
        return
    d=download(str(s["ticker"]));
    if d is None:return
    ex=resolve(d,date,str(s["entry_time"]),float(s["tp"]),float(s["sl"]),str(s["direction"]).upper(),now)
    if ex is None:return
    et,ep,reason=ex; direction=str(s["direction"]).upper(); entry=float(s["entry_price"])
    gross=((ep/entry-1)*100) if direction=="LONG" else ((entry/ep-1)*100); net=gross-FEE*200
    h.loc[m,"exit_time"]=et.strftime("%H:%M");h.loc[m,"exit_price"]=ep;h.loc[m,"exit_reason"]=reason;h.loc[m,"return_pct"]=round(net,3);h.loc[m,"status"]="CLOSED";h.to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig")
    s.update({"closed":True,"exit_time":et.strftime("%H:%M"),"exit_price":ep,"exit_reason":reason,"return_pct":round(net,3)})
    with open(STATE_FILE,"w",encoding="utf-8") as f:json.dump(s,f,ensure_ascii=False,indent=2)
    label="買い" if direction=="LONG" else "空売り"; why={"TP":"🎯 利確","SL":"🛑 損切り","SL_BOTH":"🛑 損切り(同一足両到達)","EOD":"⏰ 15:25強制決済"}.get(reason,reason)
    send(f"✅ 自動デイトレ決済結果\n日付: {date}\n銘柄: {s['ticker']} {COMPANY_NAMES.get(str(s['ticker']),'')}\n方向: {label}\n戦略: {s.get('strategy','')}\n市場環境: {s.get('market_bias','')}\nエントリー: {entry:.1f} → 決済: {ep:.1f}\n理由: {why}\n時刻: {et:%H:%M}\n損益率(手数料込み): {net:+.2f}%\n※ペーパートレード")

if __name__=="__main__":main()
