```python
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os

from sklearn.ensemble import RandomForestClassifier

# =====================
# Discord
# =====================

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

def send(msg):

    if not WEBHOOK_URL:
        print("Webhookなし")
        return

    if len(msg) > 1900:
        msg = msg[:1900]

    requests.post(
        WEBHOOK_URL,
        json={"content": msg}
    )

# =====================
# 監視銘柄
# =====================

TICKERS = [
    "6857.T",
    "8035.T",
    "6920.T",
    "6526.T",
    "6501.T",
    "6503.T",
    "5803.T",
    "7011.T",
    "4980.T",
    "9984.T"
]

# =====================
# RSI
# =====================

def calc_rsi(close, period=14):

    delta = close.diff()

    gain = delta.clip(lower=0).rolling(period).mean()

    loss = (
        -delta.clip(upper=0)
    ).rolling(period).mean()

    rs = gain / loss.replace(0, 0.0001)

    return 100 - (100 / (1 + rs))
```
```python
# =====================
# メイン処理
# =====================

results = []

for ticker in TICKERS:

    try:

        print(f"解析中: {ticker}")

        df = yf.download(
            ticker,
            period="3y",
            interval="1d",
            auto_adjust=True,
            progress=False
        )

        if df is None or len(df) < 150:
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        volume = df["Volume"]

        df["ret1"] = close.pct_change()

        df["ma25"] = close.rolling(25).mean()
        df["ma75"] = close.rolling(75).mean()

        df["vol_ratio"] = (
            volume /
            volume.rolling(20).mean()
        )

        df["rsi"] = calc_rsi(close)

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()

        df["macd"] = ema12 - ema26
        df["signal"] = (
            df["macd"]
            .ewm(span=9)
            .mean()
        )

        df["target"] = (
            close.shift(-1) > close
        ).astype(int)

        features = [
            "ret1",
            "ma25",
            "ma75",
            "vol_ratio",
            "rsi",
            "macd",
            "signal"
        ]

        model_df = df.dropna()

        if len(model_df) < 100:
            continue

        X = model_df[features]
        y = model_df["target"]

        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            random_state=42
        )

        model.fit(X_train, y_train)

        latest = X.iloc[-1:]

        prob = (
            model
            .predict_proba(latest)[0][1]
        )

        score = 0

        rsi = float(
            model_df["rsi"].iloc[-1]
        )

        macd = float(
            model_df["macd"].iloc[-1]
        )

        signal = float(
            model_df["signal"].iloc[-1]
        )

        ma25 = float(
            model_df["ma25"].iloc[-1]
        )

        ma75 = float(
            model_df["ma75"].iloc[-1]
        )

        vol_ratio = float(
            model_df["vol_ratio"].iloc[-1]
        )

        if rsi < 35:
            score += 25

        if macd > signal:
            score += 25

        if ma25 > ma75:
            score += 20

        if vol_ratio > 1.5:
            score += 20

        score += prob * 30

        price = float(close.iloc[-1])

        results.append({
            "ticker": ticker,
            "score": round(score, 1),
            "prob": round(prob * 100, 1),
            "price": round(price, 0),
            "rsi": round(rsi, 1),
            "vol": round(vol_ratio, 2)
        })

    except Exception as e:

        print(
            f"{ticker} エラー: {e}"
        )
```
```python id="r4e2bz"
# =====================
# 結果判定
# =====================

if len(results) == 0:

    send("⚪ データ取得なし")
    exit()

results = sorted(
    results,
    key=lambda x: x["score"],
    reverse=True
)

top = results[:3]

msg = "📊 AI株スキャン結果\n\n"

for i, r in enumerate(top):

    if r["score"] >= 85:
        rank = "🔥 強い買い"

    elif r["score"] >= 70:
        rank = "🟢 買い候補"

    elif r["score"] >= 60:
        rank = "🟡 監視"

    else:
        continue

    buy = r["price"]

    take_profit = round(
        buy * 1.08,
        0
    )

    stop_loss = round(
        buy * 0.95,
        0
    )

    msg += f"""
━━━━━━━━━━━━━━

#{i+1} {r['ticker']}

{rank}

AIスコア: {r['score']}
上昇確率: {r['prob']}%

RSI: {r['rsi']}
出来高倍率: {r['vol']}

買値: {buy}
利確: {take_profit}
損切: {stop_loss}

"""

if msg == "📊 AI株スキャン結果\n\n":

    msg = "⚪ 本日の買い候補なし"

print(msg)

send(msg)
```
