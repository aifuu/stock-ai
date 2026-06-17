import yfinance as yf
import pandas as pd
import numpy as np
import os
import requests

from sklearn.ensemble import RandomForestClassifier

# =====================
# Discord
# =====================
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")

def send_discord(msg):
    if not WEBHOOK_URL:
        print("Webhookなし")
        return

    if len(msg) > 1900:
        msg = msg[:1900]

    requests.post(WEBHOOK_URL, json={"content": msg})

# =====================
# 銘柄
# =====================
TICKERS = [
    "6857.T","8035.T","6920.T","6526.T","6501.T",
    "6503.T","5803.T","7011.T","4980.T","9984.T"
]

all_data = []

# =====================
# データ作成
# =====================
for t in TICKERS:

    df = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty or len(df) < 200:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]

    df = df.copy()

    df["ret1"] = close.pct_change()
    df["ma5"] = close.rolling(5).mean()
    df["ma25"] = close.rolling(25).mean()
    df["ma75"] = close.rolling(75).mean()
    df["vol_ratio"] = volume / volume.rolling(20).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9).mean()

    df["target"] = (close.shift(-1) > close).astype(int)

    all_data.append(df)

data = pd.concat(all_data).dropna()

features = ["ret1","ma5","ma25","ma75","vol_ratio","rsi","macd","signal"]

X = data[features]
y = data["target"]

split = int(len(data) * 0.7)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =====================
# AIモデル
# =====================
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

# =====================
# 利益シミュレーション
# =====================
test = data.iloc[split:].copy()
test["prob"] = model.predict_proba(X_test)[:, 1]

test["signal"] = (test["prob"] > 0.6).astype(int)

test["market_return"] = data["Close"].pct_change().iloc[split:]
test["strategy_return"] = test["market_return"] * test["signal"]

market = (1 + test["market_return"]).cumprod().iloc[-1]
strategy = (1 + test["strategy_return"]).cumprod().iloc[-1]

# =====================
# 最新予測
# =====================
latest = X.iloc[-1:]
prob = model.predict_proba(latest)[0][1]

# =====================
# Discordメッセージ
# =====================
msg = f"""
📊 AI株スキャン結果

■ AI精度（勝率）
{acc:.3f}

■ 市場リターン
{market:.2f}倍

■ AI戦略リターン
{strategy:.2f}倍

■ 翌日上昇確率
{prob*100:.2f}%

判定:
{"🔥 強い買い" if prob > 0.65 else "🟢 監視" if prob > 0.55 else "⚪ 見送り"}
"""

print(msg)
send_discord(msg)

send_discord("📊 AI株スキャン完了テスト")
