import yfinance as yf
import pandas as pd
import requests
import os

# =====================
# 銘柄リスト
# =====================
TICKERS = [
    "6526.T",
    "6501.T",
    "6503.T",
    "5803.T",
    "4980.T",
    "9984.T",
    "6613.T",
    "6506.T",
    "6269.T"
]

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

# =====================
# Discord送信
# =====================
def send_discord(message):
    if not WEBHOOK_URL:
        print("Discord Webhook未設定")
        return

    if len(message) > 1900:
        message = message[:1900]

    requests.post(WEBHOOK_URL, json={"content": message})


# =====================
# 株データ分析
# =====================
def analyze_stock(ticker):

    df = yf.download(
        ticker,
        period="6mo",
        interval="1d",
        progress=False,
        auto_adjust=True
    )

    if df is None or df.empty or len(df) < 100:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]

    price = float(close.iloc[-1])

    # =====================
    # 移動平均
    # =====================
    ma5 = close.rolling(5).mean().iloc[-1]
    ma25 = close.rolling(25).mean().iloc[-1]
    ma75 = close.rolling(75).mean().iloc[-1]

    # =====================
    # RSI
    # =====================
    delta = close.diff()

    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()

    rs = gain / loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = float(rsi.iloc[-1])

    # =====================
    # MACD
    # =====================
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    # =====================
    # 出来高
    # =====================
    vol20 = volume.rolling(20).mean().iloc[-1]
    today_vol = volume.iloc[-1]

    # =====================
    # 高値更新
    # =====================
    high20 = high.rolling(20).max().iloc[-2]
    today_high = high.iloc[-1]

    # =====================
    # スコア計算
    # =====================
    score = 0

    # RSI
    if rsi <= 30:
        score += 25
    elif rsi <= 40:
        score += 15

    # トレンド
    if ma5 > ma25:
        score += 15

    if ma25 > ma75:
        score += 15

    if price > ma75:
        score += 10

    # MACD
    if macd.iloc[-1] > signal.iloc[-1]:
        score += 15

    # 出来高
    if today_vol > vol20 * 2:
        score += 15
    elif today_vol > vol20 * 1.5:
        score += 8

    # 高値更新
    if today_high > high20:
        score += 10

    score = min(score, 100)

    # =====================
    # 判定
    # =====================
    if score >= 80:
        rank = "🔥 強い買い"
    elif score >= 60:
        rank = "🟢 買い候補"
    elif score >= 40:
        rank = "🟡 監視"
    else:
        return None

    # =====================
    # 売買目安
    # =====================
    buy_price = round(price * 1.005, 1)
    take_profit1 = round(price * 1.05, 1)
    take_profit2 = round(price * 1.10, 1)
    stop_loss = round(price * 0.97, 1)

    msg = f"""
{rank}

銘柄: {ticker}
AIスコア: {score}

現在値: {price:,.1f}

買い目安: {buy_price:,.1f}

利確①: {take_profit1:,.1f}
利確②: {take_profit2:,.1f}

損切り: {stop_loss:,.1f}

RSI: {rsi:.1f}
"""

    return msg


# =====================
# メイン処理
# =====================
def main():

    results = []

    for ticker in TICKERS:
        try:
            msg = analyze_stock(ticker)
            if msg:
                print(msg)
                results.append(msg)
        except Exception as e:
            print(f"{ticker} エラー: {e}")

    if results:
        send_discord("\n\n".join(results))
    else:
        send_discord("📊 条件一致銘柄なし")


# =====================
# 実行
# =====================
if __name__ == "__main__":
    main()

    send_discord("📊 AI株スキャン完了（正常動作）")
