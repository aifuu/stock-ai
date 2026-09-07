import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

TZ = ZoneInfo("Asia/Tokyo")
MODEL_FILE = "directional_model.pkl"
TRAIN_FILE = "train_data.csv"
HISTORY_FILE = "directional_paper_history.csv"
STATE_FILE = "directional_paper_state.json"
MONTHLY_FILE = "directional_monthly_performance.csv"
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")
INITIAL_CAPITAL = float(os.getenv("AI_INITIAL_CAPITAL", "1000000"))
HOLD_DAYS = 5
TP_MULT = 3.0
SL_MULT = 1.5
NORMAL_UP_MIN = float(os.getenv("TOP1_NORMAL_UP_MIN", "0.40"))
NORMAL_SCORE_MIN = float(os.getenv("TOP1_NORMAL_SCORE_MIN", "55.0"))
FORCED_TOP1_ENABLED = os.getenv("DAILY_TOP1_FORCED_ENABLED", "true").lower() == "true"

FEATURES = [
    "ret1","ma25","ma75","vol_ratio","rsi","adx","macd","signal","from_high","from_low","relative_strength",
    "ret5","ret20","ma25_slope5","volume_surge","breakout20","trend_alignment","momentum_score","bb_position",
    "bb_width","obv_change","atr_ratio","volatility20","avg_volume_ratio","nikkei_kairi25","nikkei_rsi",
    "nikkei_macd","nikkei_return_5d","future_return","future_ma5","future_rsi","future_gap",
]

TICKERS = [
    "2002.T",
    "2269.T",
    "2282.T",
    "2501.T",
    "2502.T",
    "2503.T",
    "2801.T",
    "2802.T",
    "2871.T",
    "2914.T",
    "3401.T",
    "3402.T",
    "3861.T",
    "3405.T",
    "3407.T",
    "4004.T",
    "4005.T",
    "4021.T",
    "4042.T",
    "4043.T",
    "4061.T",
    "4063.T",
    "4183.T",
    "4188.T",
    "4208.T",
    "4452.T",
    "4901.T",
    "4911.T",
    "6988.T",
    "4151.T",
    "4502.T",
    "4503.T",
    "4506.T",
    "4507.T",
    "4519.T",
    "4523.T",
    "4568.T",
    "4578.T",
    "5019.T",
    "5020.T",
    "5101.T",
    "5108.T",
    "5201.T",
    "5214.T",
    "5233.T",
    "5301.T",
    "5332.T",
    "5333.T",
    "5401.T",
    "5406.T",
    "5411.T",
    "3436.T",
    "5706.T",
    "5711.T",
    "5713.T",
    "5714.T",
    "5801.T",
    "5802.T",
    "5803.T",
    "5631.T",
    "6103.T",
    "6113.T",
    "6273.T",
    "6301.T",
    "6302.T",
    "6305.T",
    "6326.T",
    "6361.T",
    "6367.T",
    "6471.T",
    "6472.T",
    "6473.T",
    "7004.T",
    "7011.T",
    "7013.T",
    "285A.T",
    "4062.T",
    "6479.T",
    "6501.T",
    "6503.T",
    "6504.T",
    "6506.T",
    "6526.T",
    "6645.T",
    "6701.T",
    "6702.T",
    "6723.T",
    "6724.T",
    "6752.T",
    "6753.T",
    "6758.T",
    "6762.T",
    "6770.T",
    "6841.T",
    "6857.T",
    "6861.T",
    "6902.T",
    "6920.T",
    "6954.T",
    "6963.T",
    "6971.T",
    "6976.T",
    "6981.T",
    "7735.T",
    "7751.T",
    "7752.T",
    "8035.T",
    "7012.T",
    "543A.T",
    "7201.T",
    "7202.T",
    "7203.T",
    "7211.T",
    "7261.T",
    "7267.T",
    "7269.T",
    "7270.T",
    "7272.T",
    "4543.T",
    "4902.T",
    "6146.T",
    "7731.T",
    "7733.T",
    "7741.T",
    "7832.T",
    "7911.T",
    "7912.T",
    "7951.T",
    "1332.T",
    "1605.T",
    "1721.T",
    "1801.T",
    "1802.T",
    "1803.T",
    "1808.T",
    "1812.T",
    "1925.T",
    "1928.T",
    "1963.T",
    "2768.T",
    "8001.T",
    "8002.T",
    "8015.T",
    "8031.T",
    "8053.T",
    "8058.T",
    "3086.T",
    "3092.T",
    "3099.T",
    "3382.T",
    "7453.T",
    "7532.T",
    "8233.T",
    "8252.T",
    "8267.T",
    "9843.T",
    "9983.T",
    "5831.T",
    "7186.T",
    "8304.T",
    "8306.T",
    "8308.T",
    "8309.T",
    "8316.T",
    "8331.T",
    "8354.T",
    "8411.T",
    "8601.T",
    "8604.T",
    "8630.T",
    "8725.T",
    "8750.T",
    "8766.T",
    "8795.T",
    "8253.T",
    "8591.T",
    "8697.T",
    "3289.T",
    "8801.T",
    "8802.T",
    "8804.T",
    "8830.T",
    "9001.T",
    "9005.T",
    "9007.T",
    "9008.T",
    "9009.T",
    "9020.T",
    "9021.T",
    "9022.T",
    "9064.T",
    "9147.T",
    "9101.T",
    "9104.T",
    "9107.T",
    "9201.T",
    "9202.T",
    "9432.T",
    "9433.T",
    "9434.T",
    "9984.T",
    "9501.T",
    "9502.T",
    "9503.T",
    "9531.T",
    "9532.T",
    "2413.T",
    "2432.T",
    "3659.T",
    "3697.T",
    "4307.T",
    "4324.T",
    "4385.T",
    "4661.T",
    "4689.T",
    "4704.T",
    "4751.T",
    "4755.T",
    "6098.T",
    "6178.T",
    "6532.T",
    "7974.T",
    "9602.T",
    "9735.T",
    "9766.T",
]
NAMES = {
    "2002.T": "日清製粉グループ本社",
    "2269.T": "明治ホールディングス",
    "2282.T": "日本ハム",
    "2501.T": "サッポロホールディングス",
    "2502.T": "アサヒグループホールディングス",
    "2503.T": "キリンホールディングス",
    "2801.T": "キッコーマン",
    "2802.T": "味の素",
    "2871.T": "ニチレイ",
    "2914.T": "日本たばこ産業",
    "3401.T": "帝人",
    "3402.T": "東レ",
    "3861.T": "王子ホールディングス",
    "3405.T": "クラレ",
    "3407.T": "旭化成",
    "4004.T": "レゾナック・ホールディングス",
    "4005.T": "住友化学",
    "4021.T": "日産化学",
    "4042.T": "東ソー",
    "4043.T": "トクヤマ",
    "4061.T": "デンカ",
    "4063.T": "信越化学工業",
    "4183.T": "三井化学",
    "4188.T": "三菱ケミカルグループ",
    "4208.T": "UBE",
    "4452.T": "花王",
    "4901.T": "富士フイルムホールディングス",
    "4911.T": "資生堂",
    "6988.T": "日東電工",
    "4151.T": "協和キリン",
    "4502.T": "武田薬品工業",
    "4503.T": "アステラス製薬",
    "4506.T": "住友ファーマ",
    "4507.T": "塩野義製薬",
    "4519.T": "中外製薬",
    "4523.T": "エーザイ",
    "4568.T": "第一三共",
    "4578.T": "大塚ホールディングス",
    "5019.T": "出光興産",
    "5020.T": "ENEOSホールディングス",
    "5101.T": "横浜ゴム",
    "5108.T": "ブリデストン",
    "5201.T": "AGC",
    "5214.T": "日本電気硯子",
    "5233.T": "太平洋セメント",
    "5301.T": "東海カーボン",
    "5332.T": "TOTO",
    "5333.T": "NGK",
    "5401.T": "日本製鉄",
    "5406.T": "神戸製鉜所",
    "5411.T": "JFEホールディングス",
    "3436.T": "SUMCO",
    "5706.T": "三井金属鉱業",
    "5711.T": "三菱マテリアル",
    "5713.T": "住友金属鉱山",
    "5714.T": "DOWAホールディングス",
    "5801.T": "古河電気工業",
    "5802.T": "住友電気工業",
    "5803.T": "フジクラ",
    "5631.T": "日本製鉜所",
    "6103.T": "オークマ",
    "6113.T": "アマダ",
    "6273.T": "SMC",
    "6301.T": "小松製作所",
    "6302.T": "住友重機械工業",
    "6305.T": "日立建機",
    "6326.T": "クボタ",
    "6361.T": "荒原製作所",
    "6367.T": "ダイキン工業",
    "6471.T": "日本精工",
    "6472.T": "NTN",
    "6473.T": "ジェイテクト",
    "7004.T": "カナデビア",
    "7011.T": "三菱重工業",
    "7013.T": "IHI",
    "285A.T": "キオクシアホールディングス",
    "4062.T": "イビデン",
    "6479.T": "ミネベアミツミ",
    "6501.T": "日立製作所",
    "6503.T": "三菱電機",
    "6504.T": "富士電機",
    "6506.T": "安川電機",
    "6526.T": "ソシオネクスト",
    "6645.T": "オムロン",
    "6701.T": "日本電気",
    "6702.T": "富士通",
    "6723.T": "ルネサスエレクトロニクス",
    "6724.T": "セイコーエプソン",
    "6752.T": "パナソニックホールディングス",
    "6753.T": "シャープ",
    "6758.T": "ソニーグループ",
    "6762.T": "TDK",
    "6770.T": "アルプスアルパイン",
    "6841.T": "横河電機",
    "6857.T": "アドバンテスト",
    "6861.T": "キーエンス",
    "6902.T": "デンソー",
    "6920.T": "レーザーテック",
    "6954.T": "ファナック",
    "6963.T": "ローム",
    "6971.T": "京セラ",
    "6976.T": "太陽誘電",
    "6981.T": "村田製作所",
    "7735.T": "SCREENホールディングス",
    "7751.T": "キヤノン",
    "7752.T": "リコー",
    "8035.T": "東京エレクトロン",
    "7012.T": "川崎重工業",
    "543A.T": "ARCHION",
    "7201.T": "日産自動車",
    "7202.T": "いすご自動車",
    "7203.T": "トヨタ自動車",
    "7211.T": "三菱自動車工業",
    "7261.T": "マツダ",
    "7267.T": "本田技研工業",
    "7269.T": "スズキ",
    "7270.T": "SUBARU",
    "7272.T": "ヤマハ発動機",
    "4543.T": "テルモ",
    "4902.T": "コニカミノルタ",
    "6146.T": "ディスコ",
    "7731.T": "ニコン",
    "7733.T": "オリンパス",
    "7741.T": "HOYA",
    "7832.T": "バンダイナムコホールディングス",
    "7911.T": "TOPPANホールディングス",
    "7912.T": "大日本印刷",
    "7951.T": "ヤマハ",
    "1332.T": "ニッスイ",
    "1605.T": "INPEX",
    "1721.T": "コムシスホールディングス",
    "1801.T": "大成建設",
    "1802.T": "大林組",
    "1803.T": "清水建設",
    "1808.T": "長谷工コーポレーション",
    "1812.T": "鹿島建設",
    "1925.T": "大和ハウス工業",
    "1928.T": "積水ハウス",
    "1963.T": "日揮ホールディングス",
    "2768.T": "双日",
    "8001.T": "伊藤忠商事",
    "8002.T": "丸紅",
    "8015.T": "豊田通商",
    "8031.T": "三井物産",
    "8053.T": "住友商事",
    "8058.T": "三菱商事",
    "3086.T": "J.フロント リテイリング",
    "3092.T": "ZOZO",
    "3099.T": "三越伊勢丹ホールディングス",
    "3382.T": "セブン&アイ・ホールディングス",
    "7453.T": "良品計画",
    "7532.T": "パン・パシフィック・インターナショナルホールディングス",
    "8233.T": "高島屋",
    "8252.T": "丸井グループ",
    "8267.T": "イオン",
    "9843.T": "ニトリホールディングス",
    "9983.T": "ファーストリテイリング",
    "5831.T": "しずおかフィナンシャルグループ",
    "7186.T": "横浜フィナンシャルグループ",
    "8304.T": "あおぞら銀行",
    "8306.T": "三菱UFJフィナンシャル・グループ",
    "8308.T": "りそなホールディングス",
    "8309.T": "三井住友トラスト・ホールディングス",
    "8316.T": "三井住友フィナンシャルグループ",
    "8331.T": "千葉銀行",
    "8354.T": "ふくおかフィナンシャルグループ",
    "8411.T": "みずほフィナンシャルグループ",
    "8601.T": "大和証券グループ本社",
    "8604.T": "野村ホールディングス",
    "8630.T": "SOMPOホールディングス",
    "8725.T": "MS&ADインシュアランスグループホールディングス",
    "8750.T": "第一ライフグループ",
    "8766.T": "東京海上ホールディングス",
    "8795.T": "T&Dホールディングス",
    "8253.T": "クレディセゾン",
    "8591.T": "オリックス",
    "8697.T": "日本取引所グループ",
    "3289.T": "東急不動産ホールディングス",
    "8801.T": "三井不動産",
    "8802.T": "三菱地所",
    "8804.T": "東京建物",
    "8830.T": "住友不動産",
    "9001.T": "東武鉄道",
    "9005.T": "東急",
    "9007.T": "小田急電鉄",
    "9008.T": "京王電鉄",
    "9009.T": "京成電鉄",
    "9020.T": "東日本旅客鉄道",
    "9021.T": "西日本旅客鉄道",
    "9022.T": "東海旅客鉄道",
    "9064.T": "ヤマトホールディングス",
    "9147.T": "NIPPON EXPRESSホールディングス",
    "9101.T": "日本郵船",
    "9104.T": "商船三井",
    "9107.T": "川崎汽船",
    "9201.T": "日本航空",
    "9202.T": "ANAホールディングス",
    "9432.T": "NTT",
    "9433.T": "KDDI",
    "9434.T": "ソフトバンク",
    "9984.T": "ソフトバンクグループ",
    "9501.T": "東京電力ホールディングス",
    "9502.T": "中部電力",
    "9503.T": "関西電力",
    "9531.T": "東京ガス",
    "9532.T": "大阪ガス",
    "2413.T": "エムスリー",
    "2432.T": "ディー・エヌ・エー",
    "3659.T": "ネクソン",
    "3697.T": "SHIFT",
    "4307.T": "野村総合研究所",
    "4324.T": "電通グループ",
    "4385.T": "メルカリ",
    "4661.T": "オリエンタルランド",
    "4689.T": "LINEヤフー",
    "4704.T": "トレンドマイクロ",
    "4751.T": "サイバーエージェント",
    "4755.T": "楽天",
    "6098.T": "リクルートホールディングス",
    "6178.T": "日本郵政",
    "6532.T": "ベイカレント",
    "7974.T": "任天堂",
    "9602.T": "東宝",
    "9735.T": "セコム",
    "9766.T": "コナミグループ",
}
_FUTURES_FEATURE_CACHE = None


def download(ticker, period="3y"):
    try:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False, threads=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        print(f"{ticker}: {exc}")
        return None


def rsi(close, period=14):
    d = close.diff(); gain = d.clip(lower=0).ewm(alpha=1/period, adjust=False).mean(); loss = (-d).clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    return (100 - 100/(1 + gain/loss)).where(loss != 0, 100)


def atr(df, period=14):
    h,l,c=df["High"].squeeze(),df["Low"].squeeze(),df["Close"].squeeze(); pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1)
    return tr.ewm(alpha=1/period,adjust=False).mean()


def adx(df, period=14):
    h,l,c=df["High"].squeeze(),df["Low"].squeeze(),df["Close"].squeeze(); pc=c.shift(1)
    tr=pd.concat([h-l,(h-pc).abs(),(l-pc).abs()],axis=1).max(axis=1); up,down=h.diff(),-l.diff()
    plus=pd.Series(np.where((up>down)&(up>0),up,0.0),index=h.index); minus=pd.Series(np.where((down>up)&(down>0),down,0.0),index=h.index)
    a=tr.ewm(alpha=1/period,adjust=False).mean(); p=plus.ewm(alpha=1/period,adjust=False).mean(); m=minus.ewm(alpha=1/period,adjust=False).mean()
    pdi,mdi=100*p/a.replace(0,np.nan),100*m/a.replace(0,np.nan); dx=(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)*100
    return dx.ewm(alpha=1/period,adjust=False).mean()


def make_futures_features():
    f = download("NIY=F")
    if f is None or f.empty:
        print("⚠ 日経225先物(NIY=F)取得失敗: future_*特徴量はNaNにして予測を止めます")
        return None
    c = f["Close"].squeeze()
    out = pd.DataFrame(index=pd.to_datetime(f.index).normalize())
    out["future_return"] = c.to_numpy() / c.shift(1).to_numpy() - 1.0
    out["future_ma5"] = c.rolling(5).mean().to_numpy()
    out["future_rsi"] = rsi(c).to_numpy()
    out["future_gap"] = (c - c.shift(1)).to_numpy() / c.shift(1).to_numpy()
    out = out.shift(1)
    out = out[~out.index.duplicated(keep="last")]
    return out


def _get_futures_features():
    global _FUTURES_FEATURE_CACHE
    if _FUTURES_FEATURE_CACHE is None:
        _FUTURES_FEATURE_CACHE = make_futures_features()
    return _FUTURES_FEATURE_CACHE


def features(df, nikkei, futures_df=None):
    x=df.copy(); c,v=x["Close"].squeeze(),x["Volume"].squeeze()
    x["ret1"]=c.pct_change(); x["ma25"]=c.rolling(25).mean(); x["ma75"]=c.rolling(75).mean(); x["vol_ratio"]=v/v.rolling(20).mean(); x["rsi"]=rsi(c); x["adx"]=adx(x)
    e12,e26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean(); x["macd"]=e12-e26; x["signal"]=x["macd"].ewm(span=9,adjust=False).mean()
    hi,lo=c.rolling(252).max(),c.rolling(252).min(); x["from_high"]=(c/hi-1)*100; x["from_low"]=(c/lo-1)*100; x["_stock_ret5"]=c.pct_change(5)
    x["ret5"]=c.pct_change(5)*100; x["ret20"]=c.pct_change(20)*100; x["ma25_slope5"]=(x["ma25"]/x["ma25"].shift(5)-1)*100; x["volume_surge"]=v/v.rolling(5).mean(); rh=c.shift(1).rolling(20).max(); x["breakout20"]=(c/rh-1)*100
    x["trend_alignment"]=(c>x["ma25"]).astype(int)+(x["ma25"]>x["ma75"]).astype(int)+(x["ma25_slope5"]>0).astype(int)
    ms=pd.Series(0.0,index=x.index); ms+=np.where(c>x["ma25"],20,0)+np.where(x["ma25"]>x["ma75"],20,0)+np.where(x["ma25_slope5"]>0,15,0)+np.where(x["ret5"]>0,10,0)+np.where(x["ret20"]>0,10,0)+np.where(x["volume_surge"]>=1.2,10,0)+np.where(x["from_high"]>=-10,10,0)+np.where(x["breakout20"]>=0,5,0)
    x["momentum_score"]=ms.clip(0,100); bbm,bbs=c.rolling(20).mean(),c.rolling(20).std(); upper,lower=bbm+2*bbs,bbm-2*bbs; x["bb_position"]=(c-lower)/(upper-lower); x["bb_width"]=(upper-lower)/bbm*100
    direction=np.sign(c.diff()); obv=(v*direction).fillna(0).cumsum(); x["obv_change"]=obv.diff(5)/v.rolling(5).sum()*100; a=atr(x); x["atr_ratio"]=a/c*100; av20,av60=v.rolling(20).mean(),v.rolling(60).mean(); x["volatility20"]=x["ret1"].rolling(20).std()*100; x["avg_volume_ratio"]=av20/av60.replace(0,np.nan)
    n=nikkei.reindex(x.index).ffill(); x["nikkei_kairi25"]=n["kairi25"]; x["nikkei_rsi"]=n["rsi"]; x["nikkei_macd"]=n["macd"]; x["nikkei_return_5d"]=n["ret5"]; x["relative_strength"]=x["_stock_ret5"]-n["ret5_raw"]
    if futures_df is None:
        futures_df = _get_futures_features()
    if futures_df is None:
        x["future_return"]=np.nan; x["future_ma5"]=np.nan; x["future_rsi"]=np.nan; x["future_gap"]=np.nan
    else:
        aligned=futures_df.reindex(pd.to_datetime(x.index).normalize()).ffill()
        aligned.index=x.index
        x["future_return"]=aligned["future_return"].to_numpy(); x["future_ma5"]=aligned["future_ma5"].to_numpy(); x["future_rsi"]=aligned["future_rsi"].to_numpy(); x["future_gap"]=aligned["future_gap"].to_numpy()
    return x


def make_nikkei():
    n=download("^N225")
    if n is None:return None
    c=n["Close"].squeeze(); ma25,ma75=c.rolling(25).mean(),c.rolling(75).mean()
    return pd.DataFrame({"kairi25":(c-ma25)/ma25*100,"rsi":rsi(c),"macd":c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean(),"ret5":c.pct_change(5)*100,"ret5_raw":c.pct_change(5)},index=n.index)


def load_model():
    expected = list(FEATURES)
    if os.path.exists(MODEL_FILE):
        try:
            m=joblib.load(MODEL_FILE); actual=list(getattr(m,"feature_names_in_",[]))
            if np.array_equal(m.classes_,np.array([0,1,2])) and actual==expected:return m
            print("⚠️ directional_model.pkl の特徴量セット不一致 → 現行FEATURESで再学習します")
        except Exception as exc:print(f"⚠️ directional_model.pkl 読込失敗 → 再学習します: {exc}")
    if not os.path.exists(TRAIN_FILE):return None
    try:df=pd.read_csv(TRAIN_FILE)
    except Exception:return None
    required=expected+["target"]
    if any(col not in df.columns for col in required):
        print("❌ train_data.csv に現行TOP1特徴量が不足しているため、旧モデルへフォールバックしません"); return None
    df=df.dropna(subset=required)
    if len(df)<100 or df["target"].nunique()!=3:return None
    m=RandomForestClassifier(n_estimators=300,max_depth=7,random_state=42,class_weight="balanced",n_jobs=-1); m.fit(df[expected],df["target"].astype(int)); joblib.dump(m,MODEL_FILE); return m


def directional_score(row,up,down):
    r,macd,sig,ma25,ma75,vol=float(row["rsi"]),float(row["macd"]),float(row["signal"]),float(row["ma25"]),float(row["ma75"]),float(row["vol_ratio"]); low,hi=float(row["from_low"]),float(row["from_high"])
    short_tech=(25 if r>65 else 0)+(25 if macd<sig else 0)+(20 if ma25<ma75 else 0)+(20 if vol>1.5 else 0)+(15 if low<10 else (8 if low<20 else 0)); tech_long=(25 if r<35 else 0)+(25 if macd>sig else 0)+(20 if ma25>ma75 else 0)+(20 if vol>1.5 else 0)+(15 if hi>-10 else (8 if hi>-20 else 0))
    return tech_long/105*100*0.50+up*100*0.05+float(row["momentum_score"])*0.45, short_tech/105*100*0.50+down*100*0.05+(100-float(row["momentum_score"]))*0.45


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE,encoding="utf-8") as f:return json.load(f)
        except Exception:pass
    return {"capital":INITIAL_CAPITAL,"position":None,"peak":INITIAL_CAPITAL,"max_dd":0.0,"trades_today":0,"trade_count_date":None,"daily_start_capital":INITIAL_CAPITAL,"forced_top1_used_date":None}


def save_state(state):
    tmp=STATE_FILE+".tmp"
    with open(tmp,"w",encoding="utf-8") as f:json.dump(state,f,ensure_ascii=False,indent=2); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,STATE_FILE)


def append_history(row):
    df=pd.DataFrame([row])
    if os.path.exists(HISTORY_FILE):
        try:df=pd.concat([pd.read_csv(HISTORY_FILE),df],ignore_index=True)
        except Exception:pass
    df.to_csv(HISTORY_FILE,index=False,encoding="utf-8-sig")


def update_open_position(state):
    p=state.get("position")
    if not p:return None
    df=download(p["ticker"],period="3mo")
    if df is None or df.empty:return None
    entry_date=pd.Timestamp(p["entry_date"]); days=pd.bdate_range(entry_date+pd.Timedelta(days=1),pd.Timestamp.now(tz=TZ).tz_localize(None).normalize())
    if len(days)==0:return None
    bars=df[df.index.normalize().isin(days)]
    if bars.empty:return None
    exit_reason=exit_price=exit_date=None
    for idx,bar in bars.iterrows():
        h,l=float(bar["High"]),float(bar["Low"])
        if p["direction"]=="BUY":
            if l<=p["sl"] and h>=p["tp"]:exit_price,exit_reason=p["sl"],"SL"
            elif h>=p["tp"]:exit_price,exit_reason=p["tp"],"TP"
            elif l<=p["sl"]:exit_price,exit_reason=p["sl"],"SL"
        else:
            if h>=p["sl"] and l<=p["tp"]:exit_price,exit_reason=p["sl"],"SL"
            elif l<=p["tp"]:exit_price,exit_reason=p["tp"],"TP"
            elif h>=p["sl"]:exit_price,exit_reason=p["sl"],"SL"
        if exit_reason:exit_date=idx;break
    if exit_reason is None and len(bars)>=HOLD_DAYS:exit_date,exit_price,exit_reason=bars.index[HOLD_DAYS-1],float(bars.iloc[HOLD_DAYS-1]["Close"]),"TIME"
    if exit_reason is None:return None
    entry=float(p["entry_price"]); ret=(exit_price-entry)/entry*100 if p["direction"]=="BUY" else (entry-exit_price)/entry*100; pnl=state["capital"]*ret/100; state["capital"]+=pnl; state["position"]=None; state["peak"]=max(float(state.get("peak",state["capital"])),state["capital"]); state["max_dd"]=max(float(state.get("max_dd",0)),((state["peak"]-state["capital"])/state["peak"]*100 if state["peak"] else 0))
    hold_days=len(pd.bdate_range(entry_date,pd.Timestamp(exit_date)))
    append_history({"entry_date":p["entry_date"],"exit_date":str(pd.Timestamp(exit_date).date()),"ticker":p["ticker"],"company":p["company"],"direction":p["direction"],"selection_mode":p.get("selection_mode","unknown"),"entry_price":entry,"exit_price":exit_price,"tp":p["tp"],"sl":p["sl"],"score":p["score"],"up_probability":p["up_probability"],"down_probability":p["down_probability"],"return_pct":round(ret,3),"pnl":round(pnl,2),"result":exit_reason,"hold_days":hold_days,"capital_after":round(state["capital"],2)})
    result_label={"TP":"利確(TP)","SL":"損切(SL)","TIME":"期限到達"}.get(exit_reason,exit_reason)
    emoji="✅" if pnl>=0 else "❌"
    text=(f"{emoji} DAILY TOP1｜決済\n━━━━━━━━━━━━━━\n"
          f"📅 {str(pd.Timestamp(exit_date).date())}\n"
          f"選定区分: {p.get('selection_mode','unknown')}\n"
          f"{p['direction']}｜{p['ticker']} {p['company']}\n"
          f"エントリー: {entry:,.0f} → 決済: {exit_price:,.0f}\n"
          f"結果: {result_label}（{ret:+.2f}%）\n"
          f"損益: {pnl:+,.0f}円\n"
          f"保有: {hold_days}営業日\n\n"
          f"💰 仮想資産: {state['capital']:,.0f}円")
    send(text)
    return text


def monthly_report():
    if not os.path.exists(HISTORY_FILE):return None
    try:df=pd.read_csv(HISTORY_FILE)
    except Exception:return None
    if df.empty:return None
    if "selection_mode" not in df.columns:df["selection_mode"]="unknown"
    df["exit_date"]=pd.to_datetime(df["exit_date"],errors="coerce"); df["pnl"]=pd.to_numeric(df["pnl"],errors="coerce"); df["return_pct"]=pd.to_numeric(df.get("return_pct"),errors="coerce")
    df=df.dropna(subset=["exit_date","pnl"])
    if df.empty:return None
    base=df.assign(month=df["exit_date"].dt.to_period("M").astype(str))
    m=base.groupby("month").agg(trades=("pnl","size"),pnl=("pnl","sum"),normal_trades=("selection_mode",lambda s:int((s=="normal").sum())),forced_trades=("selection_mode",lambda s:int((s=="forced_top1").sum())),normal_pnl=("pnl",lambda s:float(base.loc[s.index].loc[base.loc[s.index,"selection_mode"].eq("normal"),"pnl"].sum()) if len(s) else 0.0),forced_pnl=("pnl",lambda s:float(base.loc[s.index].loc[base.loc[s.index,"selection_mode"].eq("forced_top1"),"pnl"].sum()) if len(s) else 0.0),avg_return=("return_pct","mean"),wins=("pnl",lambda s:int((s>0).sum()))).reset_index()
    m["win_rate_pct"]=np.where(m["trades"]>0,m["wins"]/m["trades"]*100,0.0); m.to_csv(MONTHLY_FILE,index=False,encoding="utf-8-sig"); return m.iloc[-1].to_dict()


def send(msg):
    text=str(msg); print(text)
    if not WEBHOOK_URL:return False
    text=text if len(text)<=1900 else text[:1897]+"..."
    r=requests.post(WEBHOOK_URL,json={"content":text},timeout=30)
    if not 200<=r.status_code<300:raise RuntimeError(f"Discord HTTP {r.status_code}: {r.text[:300]}")
    return True


def main():
    today=datetime.now(TZ).strftime("%Y-%m-%d"); state=load_state(); state.setdefault("forced_top1_used_date",None); update_open_position(state)
    state=load_state(); state.setdefault("forced_top1_used_date",None)
    if state.get("position"):
        save_state(state); return
    nikkei=make_nikkei(); model=load_model()
    if nikkei is None or model is None:send("❌ DAILY TOP1｜日経またはAIモデル取得失敗");return
    candidates=[]
    for ticker in TICKERS:
        df=download(ticker)
        if df is None or len(df)<150:continue
        x=features(df,nikkei).dropna(subset=FEATURES)
        if x.empty:continue
        last=x.iloc[-1]
        try:
            probs=model.predict_proba(x[FEATURES].iloc[-1:])[0]; classes=list(model.classes_); down=float(probs[classes.index(0)]); up=float(probs[classes.index(2)]); long_s,short_s=directional_score(last,up,down); direction="BUY" if long_s>=short_s else "SHORT"; score=max(long_s,short_s); price=float(df["Close"].iloc[-1]); a=float(atr(df).iloc[-1])
            if not np.isfinite(a) or a<=0:continue
            tp,sl=(price+a*TP_MULT,price-a*SL_MULT) if direction=="BUY" else (price-a*TP_MULT,price+a*SL_MULT)
            candidates.append({"ticker":ticker,"company":NAMES.get(ticker,ticker),"direction":direction,"score":score,"long_score":long_s,"short_score":short_s,"up_probability":up*100,"down_probability":down*100,"price":price,"tp":tp,"sl":sl,"data_date":str(x.index[-1].date())})
        except Exception as exc:print(ticker,"predict",exc)
    if not candidates:send("❌ DAILY TOP1｜有効候補なし（先物特徴量取得/整合性を確認）");return
    candidates.sort(key=lambda z:z["score"],reverse=True)
    normal=[]
    for c in candidates:
        if c["direction"]=="BUY":ok=(c["up_probability"]>=NORMAL_UP_MIN*100 and c["up_probability"]>c["down_probability"] and c["score"]>=NORMAL_SCORE_MIN)
        else:ok=(c["down_probability"]>=NORMAL_UP_MIN*100 and c["down_probability"]>c["up_probability"] and c["score"]>=NORMAL_SCORE_MIN)
        if ok:normal.append(c)
    if normal:top=normal[0]; selection_mode="normal"
    elif FORCED_TOP1_ENABLED and state.get("forced_top1_used_date")!=today:top=candidates[0]; selection_mode="forced_top1"
    else:
        save_state(state); send(f"🟡 DAILY TOP1｜通常候補なし・forced_top1本日使用済み\n候補数: {len(candidates)}\n※ペーパートレード");return
    state["position"]={"entry_date":today,"entry_time":datetime.now(TZ).strftime("%H:%M"),"ticker":top["ticker"],"company":top["company"],"direction":top["direction"],"entry_price":top["price"],"tp":top["tp"],"sl":top["sl"],"score":top["score"],"up_probability":top["up_probability"],"down_probability":top["down_probability"],"selection_mode":selection_mode}
    state["trade_count_date"]=today; state["trades_today"]=int(state.get("trades_today",0))+1
    if selection_mode=="forced_top1":state["forced_top1_used_date"]=today
    save_state(state)
    month=monthly_report(); month_text="確定取引なし" if not month else f"今月累計 {month['pnl']:+,.0f}円｜normal {int(month['normal_trades'])}件/{month['normal_pnl']:+,.0f}円｜forced {int(month['forced_trades'])}件/{month['forced_pnl']:+,.0f}円"
    send(f"🤖 DAILY TOP1｜方向選択ペーパートレード\n━━━━━━━━━━━━━━\n📅 {today}\n⚠️ 実注文なし\n選定区分: {selection_mode}\nTOP1: {top['direction']}｜{top['ticker']} {top['company']}\n総合方向スコア: {top['score']:.1f}\n買い側: {top['long_score']:.1f}｜空売り側: {top['short_score']:.1f}\n上昇確率: {top['up_probability']:.1f}%｜下落確率: {top['down_probability']:.1f}%\nエントリー: {top['price']:,.0f}\nTP: {top['tp']:,.0f}\nSL: {top['sl']:,.0f}\n保有: 最大{HOLD_DAYS}営業日\n\n💰 仮想資産: {state['capital']:,.0f}円\n{month_text}\n\n📊 全候補: {len(candidates)}｜通常候補: {len(normal)}\n📌 forced_top1は稼働率検証専用で、normalと分離集計します。")


if __name__=="__main__":
    main()
