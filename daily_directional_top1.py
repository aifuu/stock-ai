import os
import json
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# DAILY DIRECTIONAL TOP1 PAPER TRADER
# ============================================================
# 毎営業日、100銘柄をスキャンしてTOP1を必ず選出。
# 上昇優勢なら BUY、下落優勢なら SHORT を選ぶ。
# 実注文は一切しない。ペーパートレード専用。
# ============================================================

TZ = ZoneInfo("Asia/Tokyo")
MODEL_FILE = "model.pkl"
TRAIN_FILE = "train_data.csv"
POLICY_FILE = "strategy_policy.json"
HISTORY_FILE = "directional_paper_history.csv"
STATE_FILE = "directional_paper_state.json"
MONTHLY_FILE = "directional_monthly_performance.csv"
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
HOLD_DAYS = 5
TP_MULT = 3.0
SL_MULT = 1.5

FEATURES = [
    "ret1","ma25","ma75","vol_ratio","rsi","adx","macd","signal",
    "from_high","from_low","relative_strength","ret5","ret20","ma25_slope5",
    "volume_surge","breakout20","trend_alignment","momentum_score","bb_position",
    "bb_width","obv_change","atr_ratio","volatility20","avg_volume_ratio",
    "nikkei_kairi25","nikkei_rsi","nikkei_macd","nikkei_return_5d",
    "future_return","future_ma5","future_rsi","future_gap",
]

# 現行stock_scan.pyと同じ100銘柄
TICKERS = [
    "7203.T","7269.T","285A.T","9984.T","4980.T","8031.T","8058.T","9509.T","9501.T","8362.T",
    "8306.T","5803.T","6526.T","6613.T","6758.T","6861.T","6857.T","8035.T","6920.T","6146.T",
    "6501.T","6503.T","6701.T","6702.T","6902.T","6901.T","7270.T","7267.T","7201.T","7202.T",
    "7205.T","7211.T","7261.T","7272.T","8316.T","8411.T","8331.T","8308.T","8309.T","8354.T",
    "8355.T","7182.T","7186.T","8697.T","8001.T","8002.T","8015.T","2768.T","8053.T","8056.T",
    "8032.T","8012.T","8014.T","8037.T","9432.T","9433.T","9434.T","9613.T","9983.T","4755.T",
    "4689.T","6098.T","2413.T","3659.T","4063.T","4188.T","4005.T","4004.T","4204.T","4502.T",
    "4503.T","4519.T","4523.T","4568.T","5401.T","5411.T","5711.T","5801.T","5802.T","5713.T",
    "6301.T","6302.T","6367.T","7011.T","7012.T","7013.T","9101.T","9104.T","9107.T","9020.T",
    "9021.T","9022.T","8801.T","8802.T","2914.T","3382.T","6762.T","7735.T","6981.T","4543.T",
]

NAMES = {
    "7203.T":"トヨタ自動車","7269.T":"スズキ","285A.T":"キオクシアHD","9984.T":"ソフトバンクG","4980.T":"デクセリアルズ",
    "8031.T":"三井物産","8058.T":"三菱商事","9509.T":"北海道電力","9501.T":"東京電力HD","8362.T":"福井銀行",
    "8306.T":"三菱UFJ","5803.T":"フジクラ","6526.T":"ソシオネクスト","6613.T":"QDレーザ","6758.T":"ソニーグループ",
    "6861.T":"キーエンス","6857.T":"アドバンテスト","8035.T":"東京エレクトロン","6920.T":"レーザーテック","6146.T":"ディスコ",
    "6501.T":"日立製作所","6503.T":"三菱電機","6701.T":"日本電気","6702.T":"富士通","6902.T":"デンソー",
    "7270.T":"SUBARU","7267.T":"本田技研工業","7201.T":"日産自動車","7202.T":"いすゞ自動車","7205.T":"日野自動車",
    "7211.T":"三菱自動車工業","7261.T":"マツダ","7272.T":"ヤマハ発動機","8316.T":"三井住友FG","8411.T":"みずほFG",
    "8331.T":"千葉銀行","8308.T":"りそなHD","8309.T":"三井住友トラスト","7182.T":"ゆうちょ銀行","7186.T":"コンコルディアFG",
    "8697.T":"日本取引所グループ","8001.T":"伊藤忠商事","8002.T":"丸紅","8015.T":"豊田通商","2768.T":"双日",
    "8053.T":"住友商事","8056.T":"BIPROGY","9432.T":"NTT","9433.T":"KDDI","9434.T":"ソフトバンク",
    "9613.T":"NTTデータグループ","9983.T":"ファーストリテイリング","4755.T":"楽天グループ","4689.T":"LINEヤフー","6098.T":"リクルートHD",
    "2413.T":"エムスリー","3659.T":"ネクソン","4063.T":"信越化学工業","4188.T":"三菱ケミカル","4005.T":"住友化学",
    "4004.T":"レゾナックHD","4204.T":"積水化学工業","4502.T":"武田薬品","4503.T":"アステラス製薬","4519.T":"中外製薬",
    "4523.T":"エーザイ","4568.T":"第一三共","5401.T":"日本製鉄","5411.T":"JFE","5711.T":"三菱マテリアル",
    "5801.T":"古河電工","5802.T":"住友電工","5713.T":"住友金属鉱山","6301.T":"コマツ","6302.T":"住友重機械",
    "6367.T":"ダイキン","7011.T":"三菱重工","7012.T":"川崎重工","7013.T":"IHI","9101.T":"日本郵船",
    "9104.T":"商船三井","9107.T":"川崎汽船","9020.T":"JR東日本","9021.T":"JR西日本","9022.T":"JR東海",
    "8801.T":"三井不動産","8802.T":"三菱地所","2914.T":"JT","3382.T":"セブン&アイ","6762.T":"TDK",
    "7735.T":"SCREEN","6981.T":"村田製作所","4543.T":"テルモ",
}


def download(ticker, period="3y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as e:
        print(f"{ticker}: {e}")
        return None


def rsi(close, period=14):
    d = close.diff()
    gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-d).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    return (100 - 100/(1 + gain/loss)).where(loss != 0, 100)


def atr(df, period=14):
    h, l, c = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()


def adx(df, period=14):
    h, l, c = df["High"].squeeze(), df["Low"].squeeze(), df["Close"].squeeze()
    pc = c.shift(1)
    tr = pd.concat([h-l, (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    up, down = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=h.index)
    a = tr.ewm(alpha=1/period, adjust=False).mean()
    p = plus.ewm(alpha=1/period, adjust=False).mean()
    m = minus.ewm(alpha=1/period, adjust=False).mean()
    pdi, mdi = 100*p/a.replace(0, np.nan), 100*m/a.replace(0, np.nan)
    dx = (pdi-mdi).abs()/(pdi+mdi).replace(0, np.nan)*100
    return dx.ewm(alpha=1/period, adjust=False).mean()


def features(df, nikkei):
    x = df.copy()
    c, v = x["Close"].squeeze(), x["Volume"].squeeze()
    x["ret1"] = c.pct_change()
    x["ma25"] = c.rolling(25).mean()
    x["ma75"] = c.rolling(75).mean()
    x["vol_ratio"] = v / v.rolling(20).mean()
    x["rsi"] = rsi(c)
    x["adx"] = adx(x)
    e12 = c.ewm(span=12, adjust=False).mean(); e26 = c.ewm(span=26, adjust=False).mean()
    x["macd"] = e12-e26; x["signal"] = x["macd"].ewm(span=9, adjust=False).mean()
    hi, lo = c.rolling(252).max(), c.rolling(252).min()
    x["from_high"] = (c/hi-1)*100; x["from_low"] = (c/lo-1)*100
    x["_stock_ret5"] = c.pct_change(5)
    x["ret5"] = c.pct_change(5)*100; x["ret20"] = c.pct_change(20)*100
    x["ma25_slope5"] = (x["ma25"]/x["ma25"].shift(5)-1)*100
    x["volume_surge"] = v/v.rolling(5).mean()
    rh = c.shift(1).rolling(20).max(); x["breakout20"] = (c/rh-1)*100
    x["trend_alignment"] = (c>x["ma25"]).astype(int)+(x["ma25"]>x["ma75"]).astype(int)+(x["ma25_slope5"]>0).astype(int)
    ms = pd.Series(0.0, index=x.index)
    ms += np.where(c>x["ma25"],20,0)+np.where(x["ma25"]>x["ma75"],20,0)+np.where(x["ma25_slope5"]>0,15,0)
    ms += np.where(x["ret5"]>0,10,0)+np.where(x["ret20"]>0,10,0)+np.where(x["volume_surge"]>=1.2,10,0)
    ms += np.where(x["from_high"]>=-10,10,0)+np.where(x["breakout20"]>=0,5,0)
    x["momentum_score"] = ms.clip(0,100)
    bbm, bbs = c.rolling(20).mean(), c.rolling(20).std()
    upper, lower = bbm+2*bbs, bbm-2*bbs
    x["bb_position"] = (c-lower)/(upper-lower); x["bb_width"] = (upper-lower)/bbm*100
    direction = np.sign(c.diff()); obv = (v*direction).fillna(0).cumsum()
    x["obv_change"] = obv.diff(5)/v.rolling(5).sum()*100
    a = atr(x); x["atr_ratio"] = a/c*100
    av20, av60 = v.rolling(20).mean(), v.rolling(60).mean()
    x["volatility20"] = x["ret1"].rolling(20).std()*100; x["avg_volume_ratio"] = av20/av60.replace(0,np.nan)
    n = nikkei.reindex(x.index).ffill()
    x["nikkei_kairi25"] = n["kairi25"]; x["nikkei_rsi"] = n["rsi"]; x["nikkei_macd"] = n["macd"]; x["nikkei_return_5d"] = n["ret5"]
    x["relative_strength"] = x["_stock_ret5"] - n["ret5_raw"]
    # 先物データは取得できない環境でもスキャンを止めない
    x["future_return"] = 0.0; x["future_ma5"] = 0.0; x["future_rsi"] = 50.0; x["future_gap"] = 0.0
    return x


def make_nikkei():
    n = download("^N225")
    if n is None: return None
    c = n["Close"].squeeze()
    ma25, ma75 = c.rolling(25).mean(), c.rolling(75).mean()
    return pd.DataFrame({"kairi25":(c-ma25)/ma25*100,"rsi":rsi(c),"macd":c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(),"ret5":c.pct_change(5)*100,"ret5_raw":c.pct_change(5)}, index=n.index)


def load_model():
    if os.path.exists(MODEL_FILE):
        try:
            m = joblib.load(MODEL_FILE)
            if np.array_equal(m.classes_, np.array([0,1,2])): return m
        except Exception:
            pass
    if not os.path.exists(TRAIN_FILE): return None
    df = pd.read_csv(TRAIN_FILE)
    df = df.dropna(subset=FEATURES+["target"])
    if len(df) < 100 or df["target"].nunique() != 3: return None
    m = RandomForestClassifier(n_estimators=300,max_depth=7,random_state=42,class_weight="balanced",n_jobs=-1)
    m.fit(df[FEATURES], df["target"].astype(int)); joblib.dump(m,MODEL_FILE); return m


def directional_score(row, up, down):
    # 現行AIのロングスコアを土台に、ショート側は反転したテクニカルを評価。
    r = float(row["rsi"]); macd=float(row["macd"]); sig=float(row["signal"]); ma25=float(row["ma25"]); ma75=float(row["ma75"])
    vol=float(row["vol_ratio"]); low=float(row["from_low"]); hi=float(row["from_high"])
    short_tech = 0
    if r > 65: short_tech += 25
    if macd < sig: short_tech += 25
    if ma25 < ma75: short_tech += 20
    if vol > 1.5: short_tech += 20
    if low < 10: short_tech += 15
    elif low < 20: short_tech += 8
    tech_long = 0
    if r < 35: tech_long += 25
    if macd > sig: tech_long += 25
    if ma25 > ma75: tech_long += 20
    if vol > 1.5: tech_long += 20
    if hi > -10: tech_long += 15
    elif hi > -20: tech_long += 8
    long_score = tech_long/105*100*0.525 + up*100*0.225 + float(row["momentum_score"])*0.25
    short_score = short_tech/105*100*0.525 + down*100*0.225 + (100-float(row["momentum_score"]))*0.25
    return long_score, short_score


def load_state():
    if os.path.exists(STATE_FILE):
        try: return json.load(open(STATE_FILE,encoding="utf-8"))
        except Exception: pass
    return {"capital":INITIAL_CAPITAL,"position":None,"peak":INITIAL_CAPITAL,"max_dd":0.0}


def save_state(s):
    with open(STATE_FILE,"w",encoding="utf-8") as f: json.dump(s,f,ensure_ascii=False,indent=2)


def append_history(row):
    df = pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        old=pd.read_csv(HISTORY_FILE); df=pd.concat([old,df],ignore_index=True)
    df.to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig")


def update_open_position(state):
    p=state.get("position")
    if not p: return None
    df=download(p["ticker"],period="3mo")
    if df is None or df.empty: return None
    entry_date=pd.Timestamp(p["entry_date"]); days=pd.bdate_range(entry_date+pd.Timedelta(days=1),pd.Timestamp.now(tz=TZ).tz_localize(None).normalize())
    if len(days)==0: return None
    bars=df[df.index.normalize().isin(days)]
    if bars.empty: return None
    exit_reason=None; exit_price=None; exit_date=None
    for idx,bar in bars.iterrows():
        h=float(bar["High"]); l=float(bar["Low"])
        if p["direction"]=="BUY":
            if l<=p["sl"] and h>=p["tp"]: exit_reason="SL"; exit_price=p["sl"]
            elif h>=p["tp"]: exit_reason="TP"; exit_price=p["tp"]
            elif l<=p["sl"]: exit_reason="SL"; exit_price=p["sl"]
        else:
            if h>=p["sl"] and l<=p["tp"]: exit_reason="SL"; exit_price=p["sl"]
            elif l<=p["tp"]: exit_reason="TP"; exit_price=p["tp"]
            elif h>=p["sl"]: exit_reason="SL"; exit_price=p["sl"]
        if exit_reason:
            exit_date=idx; break
    if exit_reason is None and len(bars)>=HOLD_DAYS:
        exit_date=bars.index[HOLD_DAYS-1]; exit_price=float(bars.iloc[HOLD_DAYS-1]["Close"]); exit_reason="TIME"
    if exit_reason is None: return None
    entry=float(p["entry_price"]); ret=((exit_price-entry)/entry*100) if p["direction"]=="BUY" else ((entry-exit_price)/entry*100)
    pnl=state["capital"]*ret/100
    state["capital"] += pnl
    state["position"]=None
    state["peak"]=max(state.get("peak",state["capital"]),state["capital"])
    dd=(state["peak"]-state["capital"])/state["peak"]*100 if state["peak"] else 0
    state["max_dd"]=max(state.get("max_dd",0),dd)
    append_history({"entry_date":p["entry_date"],"exit_date":str(pd.Timestamp(exit_date).date()),"ticker":p["ticker"],"company":p["company"],"direction":p["direction"],"entry_price":entry,"exit_price":exit_price,"tp":p["tp"],"sl":p["sl"],"score":p["score"],"up_probability":p["up_probability"],"down_probability":p["down_probability"],"return_pct":round(ret,3),"pnl":round(pnl,2),"result":exit_reason,"hold_days":len(pd.bdate_range(entry_date,pd.Timestamp(exit_date)))})
    return f"決済 {p['direction']} {p['ticker']} {exit_reason} {ret:+.2f}%"


def monthly_report():
    if not os.path.exists(HISTORY_FILE): return None
    df=pd.read_csv(HISTORY_FILE)
    if df.empty: return None
    df["exit_date"]=pd.to_datetime(df["exit_date"],errors="coerce"); df["pnl"]=pd.to_numeric(df["pnl"],errors="coerce")
    df=df.dropna(subset=["exit_date","pnl"])
    if df.empty:return None
    m=df.assign(month=df["exit_date"].dt.to_period("M")).groupby("month").agg(trades=("pnl","size"),pnl=("pnl","sum"),avg_return=("return_pct","mean"),wins=("pnl",lambda s:int((s>0).sum()))).reset_index()
    m["win_rate_pct"]=m["wins"]/m["trades"]*100
    m.to_csv(MONTHLY_FILE,index=False,encoding="utf-8-sig")
    return m.iloc[-1].to_dict()


def send(msg):
    print(msg)
    if WEBHOOK_URL:
        try: requests.post(WEBHOOK_URL,json={"content":msg[:1900]},timeout=30)
        except Exception as e: print("Discord error",e)


def main():
    today=datetime.now(TZ).strftime("%Y-%m-%d")
    state=load_state()
    closed=update_open_position(state)
    if closed: print(closed)
    if state.get("position"):
        save_state(state)
        send(f"📝 DAILY TOP1｜保有継続\n現在ポジション: {state['position']['direction']} {state['position']['ticker']} {state['position']['company']}\n最大保有: {HOLD_DAYS}営業日\n資産: {state['capital']:,.0f}円\n⚠️ 実注文なし")
        return

    nikkei=make_nikkei()
    model=load_model()
    if nikkei is None or model is None:
        send("❌ DAILY TOP1スキャン失敗｜日経またはAIモデルを取得できませんでした")
        return

    candidates=[]
    for ticker in TICKERS:
        df=download(ticker)
        if df is None or len(df)<150: continue
        x=features(df,nikkei).dropna(subset=FEATURES)
        if x.empty: continue
        last=x.iloc[-1]
        if float(last["Volume"] if "Volume" in last else df["Volume"].iloc[-1]) <= 0: continue
        latest=x[FEATURES].iloc[-1:]
        try:
            probs=model.predict_proba(latest)[0]; classes=list(model.classes_)
            down=float(probs[classes.index(0)]); up=float(probs[classes.index(2)])
            long_s,short_s=directional_score(last,up,down)
            direction="BUY" if long_s>=short_s else "SHORT"
            score=max(long_s,short_s)
            price=float(df["Close"].iloc[-1]); a=float(atr(df).iloc[-1])
            if not np.isfinite(a) or a<=0: continue
            if direction=="BUY": tp=price+a*TP_MULT; sl=price-a*SL_MULT
            else: tp=price-a*TP_MULT; sl=price+a*SL_MULT
            candidates.append({"ticker":ticker,"company":NAMES.get(ticker,ticker),"direction":direction,"score":score,"long_score":long_s,"short_score":short_s,"up_probability":up*100,"down_probability":down*100,"price":price,"tp":tp,"sl":sl,"data_date":str(x.index[-1].date())})
        except Exception as e: print(ticker,"predict",e)

    if not candidates:
        send("❌ DAILY TOP1｜有効な候補なし"); return
    candidates.sort(key=lambda z:z["score"],reverse=True); top=candidates[0]
    state["position"]={"entry_date":today,"ticker":top["ticker"],"company":top["company"],"direction":top["direction"],"entry_price":top["price"],"tp":top["tp"],"sl":top["sl"],"score":top["score"],"up_probability":top["up_probability"],"down_probability":top["down_probability"]}
    save_state(state)
    last_month=monthly_report()
    month_text="確定取引なし"
    if last_month: month_text=f"今月損益 {last_month['pnl']:+,.0f}円｜取引 {int(last_month['trades'])}｜勝率 {last_month['win_rate_pct']:.1f}%"
    msg=(f"🤖 DAILY TOP1｜方向選択ペーパートレード\n━━━━━━━━━━━━━━\n"
         f"📅 {today}\n⚠️ 実注文なし\n\n"
         f"🏆 本日のTOP1\n{top['direction']}｜{top['ticker']} {top['company']}\n"
         f"総合方向スコア: {top['score']:.1f}\n買い側: {top['long_score']:.1f}｜空売り側: {top['short_score']:.1f}\n"
         f"上昇確率: {top['up_probability']:.1f}%｜下落確率: {top['down_probability']:.1f}%\n"
         f"エントリー: {top['price']:,.0f}\nTP: {top['tp']:,.0f}\nSL: {top['sl']:,.0f}\n保有: 最大{HOLD_DAYS}営業日\n\n"
         f"💰 仮想資産: {state['capital']:,.0f}円\n{month_text}\n\n"
         f"📊 候補数: {len(candidates)}\n"
         f"1位以下: {candidates[1]['direction']} {candidates[1]['ticker']} {candidates[1]['score']:.1f} / {candidates[2]['direction']} {candidates[2]['ticker']} {candidates[2]['score']:.1f}\n\n"
         f"📌 毎営業日100銘柄をスキャンし、買い/空売りのうちTOP1だけを仮想売買します。")
    send(msg)


if __name__=="__main__": main()
