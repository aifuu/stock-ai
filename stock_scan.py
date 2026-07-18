import csv
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import joblib

from sklearn.ensemble import RandomForestClassifier

# =====================
# Discord
# =====================
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

def send(msg):
    if not WEBHOOK_URL:
        print("❌ Webhookなし")
        return

    if len(msg) > 1900:
        msg = msg[:1900]

    r = requests.post(
        WEBHOOK_URL,
        json={"content": msg},
        timeout=30
    )

    print("Discord status =", r.status_code)

    if r.status_code == 204:
        print("✅ Discord送信成功")
    else:
        print("❌ Discord送信失敗")
        print(r.text)


# =====================
# 銘柄
# =====================
TICKERS = [
    "7203.T",
    "7269.T",
    "285A.T",
    "9984.T",
    "4980.T",
    "8031.T",
    "8058.T",
    "9509.T",
    "9501.T",
    "8362.T",
    "8306.T",
    "5803.T",
    "6526.T",
    "6613.T"
]

COMPANY_NAMES = {

    "7203.T": "トヨタ自動車",
    "7269.T": "スズキ",
    "285A.T": "キオクシアHD",
    "9984.T": "ソフトバンクG",
    "4980.T": "デクセリアルズ",
    "8031.T": "三井物産",
    "8058.T": "三菱商事",
    "9509.T": "北海道電力",
    "9501.T": "東京電力HD",
    "8362.T": "福井銀行",
    "8306.T": "三菱UFJ",
    "5803.T": "フジクラ",
    "6526.T": "ソシオネクスト",
    "6613.T": "QDレーザ",
}

TRAIN_FILE = "train_data.csv"
MODEL_FILE = "model.pkl"


# =====================
# RSI
# =====================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, 0.0001)
    return 100 - (100 / (1 + rs))


# =====================
# 日経平均
# =====================
nikkei = yf.download(
    "^N225",
    period="3y",
    interval="1d",
    auto_adjust=True
)

nikkei_close = nikkei["Close"].squeeze()

nikkei["nikkei_ma25"] = nikkei_close.rolling(25).mean()
nikkei["nikkei_kairi25"] = (
    (nikkei_close - nikkei["nikkei_ma25"])
    / nikkei["nikkei_ma25"] * 100
)

nikkei["nikkei_rsi"] = calc_rsi(nikkei_close)

ema12_n = nikkei_close.ewm(span=12).mean()
ema26_n = nikkei_close.ewm(span=26).mean()

nikkei["nikkei_macd"] = ema12_n - ema26_n

nikkei["nikkei_return_5d"] = (
    nikkei_close.pct_change(5) * 100
)





# =====================
# 学習データ読み込み
# =====================
def load_training_data():
    if not os.path.exists(TRAIN_FILE):
        return None, None

    df = pd.read_csv(TRAIN_FILE).dropna()

    X = df[[
    "rsi",
    "macd",
    "signal",
    "ma25",
    "ma75",
    "vol_ratio",
    "from_high",
    "from_low",
    "nikkei_kairi25",
    "nikkei_rsi",
    "nikkei_macd",
    "nikkei_return_5d"
]]

    y = df["target"]

    return X, y


# =====================
# モデル
# =====================
if os.path.exists(MODEL_FILE):
    model = joblib.load(MODEL_FILE)
    print("✅ 既存モデル")
else:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42
    )
    print("🆕 新規モデル")


results = []


# =====================
# メイン処理
# =====================
for ticker in TICKERS:

    try:
        print("解析中:", ticker)

        df = yf.download(
            ticker,
            period="3y",
            interval="1d",
            auto_adjust=True
        )

        if df is None or len(df) < 150:
            continue

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

        # ===== 特徴量 =====
        df["ret1"] = close.pct_change()
        df["ma25"] = close.rolling(25).mean()
        df["ma75"] = close.rolling(75).mean()
        df["vol_ratio"] = volume / volume.rolling(20).mean()
        df["rsi"] = calc_rsi(close)

        df["high252"] = close.rolling(252).max()
        df["low252"] = close.rolling(252).min()
        df["from_high"] = (
            (close / df["high252"] - 1) * 100
        )
        df["from_low"] = (
            (close / df["low252"] - 1) * 100
        )

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9).mean()

        df["high252"] = close.rolling(252).max()
        df["low252"] = close.rolling(252).min()
        df["from_high"] = (
            (close / df["high252"] - 1) * 100
        )
        df["from_low"] = (
            (close / df["low252"] - 1) * 100
        )


        # 日経平均特徴量を結合
        df = df.join(
            nikkei[
            [
                "nikkei_kairi25",
                "nikkei_rsi",
                "nikkei_macd",
                "nikkei_return_5d"
            ]
            ],
            how="left"
        )


        
        df = df.dropna()

        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)


        features = [
            "ret1",
            "ma25",
            "ma75",
            "vol_ratio",
            "rsi",
            "macd",
            "signal",
            "from_high",
            "from_low",
            "nikkei_kairi25",
            "nikkei_rsi",
            "nikkei_macd","nikkei_return_5d"
        ]





        X = df[features]
        y = df["target"]

        if len(X) < 100:
            continue

        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)

        latest = X.iloc[-1:]
        prob = model.predict_proba(latest)[0][1]
        price = float(np.asarray(close)[-1])
        rsi = float(np.asarray(df["rsi"])[-1])
        macd = float(np.asarray(df["macd"])[-1])
        signal = float(np.asarray(df["signal"])[-1])
        ma25 = float(np.asarray(df["ma25"])[-1])
        ma75 = float(np.asarray(df["ma75"])[-1])
        vol_ratio = float(np.asarray(df["vol_ratio"])[-1])

        high52 = float(close.rolling(252).max().iloc[-1])
        distance = (price / high52 - 1) * 100
        
        take_profit = price * 1.08
        stop_loss = price * 0.95

        score = 0

        if rsi < 35:
            score += 25

        if macd > signal:
            score += 25

        if ma25 > ma75:
            score += 20

        if vol_ratio > 1.5:
            score += 20

        if distance > -10:
            score += 15
        elif distance > -20:
            score += 8

        nikkei_rsi = float(np.asarray(df["nikkei_rsi"])[-1])
        nikkei_return = float(np.asarray(df["nikkei_return_5d"])[-1])

        if nikkei_rsi > 50:
            score += 5

        if nikkei_return > 0:
            score += 5

        score += prob * 50


        



 

       

        results.append({
            "ticker": ticker,
            "score": round(score, 1),
            "prob": round(prob * 100, 1),
            "price": round(price, 0),
            "rsi": round(rsi, 1),
            "vol": round(vol_ratio, 2),
            "take_profit": round(take_profit, 0),
            "stop_loss": round(stop_loss, 0)
        })
        print(
            ticker,
            "score=", round(score,1),
            "prob=", round(prob*100,1)
        )


    except Exception as e:
        print(ticker, "エラー:", e)


# =====================
# CSV全履歴学習
# =====================
try:
    X_all, y_all = load_training_data()

    if X_all is not None and len(X_all) > 200:
        model.fit(X_all, y_all)
        joblib.dump(model, MODEL_FILE)
        print("✅ CSV全履歴で再学習完了")

except Exception as e:
    print("CSV学習スキップ:", e)


# =====================
# 結果
# =====================
if not results:
    send("⚪ データなし")
    exit()

results = sorted(results, key=lambda x: x["score"], reverse=True)
top = results[:3]

msg = f"⏰ JST: {datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y-%m-%d %H:%M:%S')}\n\n"
msg += "📊 AI株スキャン結果\n\n"

for i, r in enumerate(top):

    if r["score"] >= 60:
        rank = "🔥 強い買い"
    elif r["score"] >= 45:
        rank = "🟢 買い候補"
    elif r["score"] >= 35:
        rank = "🟡 監視"
    else:
        continue

    msg += f"""
━━━━━━━━━━━━━━
#{i+1} {r['ticker']} {COMPANY_NAMES.get(r['ticker'],'')}

{rank}

AIスコア: {r['score']}
上昇確率: {r['prob']}%

買値: {r['price']}
利確: {r['take_profit']}
損切: {r['stop_loss']}

RSI: {r['rsi']}
出来高倍率: {r['vol']}
━━━━━━━━━━━━━━
"""

print(msg)
send(msg)
