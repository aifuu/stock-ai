```python
import yfinance as yf
import pandas as pd
import requests
import os

TICKERS = [
    "6526.T",  # ソシオネクスト
    "6501.T",  # 日立製作所
    "6503.T",  # 三菱電機
    "5803.T",  # フジクラ
    "4980.T",  # デクセリアルズ
    "9984.T",  # ソフトバンクG
    "6613.T",  # QDレーザ
    "6506.T",  # 安川電機
    "6269.T"   # 三井海洋開発
]

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

def send_discord(message):
    if not WEBHOOK_URL:
        print("Discord Webhook未設定")
        return

    requests.post(
        WEBHOOK_URL,
        json={"content": message}
    )

for ticker in TICKERS:

    try:

        df = yf.download(
            ticker,
            period="6mo",
            progress=False,
            auto_adjust=True
        )

        if len(df) < 100:
            continue

        close = df["Close"]

        ma5 = close.rolling(5).mean().iloc[-1]
        ma25 = close.rolling(25).mean().iloc[-1]
        ma75 = close.rolling(75).mean().iloc[-1]

        # RSI
        delta = close.diff()

        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        rsi = float(rsi.iloc[-1])

        score = 0

        if rsi <= 30:
            score += 25
        elif rsi <= 40:
            score += 15

        if ma5 > ma25:
            score += 20

        if ma25 > ma75:
            score += 20

        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()

        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        if macd.iloc[-1] > signal.iloc[-1]:
            score += 15

        # 出来高
        vol20 = df["Volume"].rolling(20).mean().iloc[-1]
        today_vol = df["Volume"].iloc[-1]

        if today_vol > vol20 * 1.5:
            score += 10

        # 高値更新
        high20 = df["High"].rolling(20).max().iloc[-2]
        today_high = df["High"].iloc[-1]

        if today_high > high20:
            score += 10

        price = float(close.iloc[-1])

        buy_price = round(price * 1.005, 1)
        take_profit1 = round(price * 1.05, 1)
        take_profit2 = round(price * 1.10, 1)
        stop_loss = round(price * 0.97, 1)

        if score >= 80:
            rank = "🔥 強い買い"
        elif score >= 60:
            rank = "🟢 買い候補"
        elif score >= 40:
            rank = "🟡 監視"
        else:
            rank = "⚪ 見送り"

        if score >= 60:

            msg = f"""
{rank}

銘柄: {ticker}
AIスコア: {score}

現在値: {price:,.1f}

買い: {buy_price:,.1f}

利確①: {take_profit1:,.1f}
利確②: {take_profit2:,.1f}

損切り: {stop_loss:,.1f}

RSI: {rsi:.1f}
"""

            print(msg)
            send_discord(msg)

    except Exception as e:
        print(ticker, e)
```
