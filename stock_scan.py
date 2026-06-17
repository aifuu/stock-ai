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

def send(msg):
    if not WEBHOOK_URL:
        print("no webhook")
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

# =====================
# ニューススコア関数
# =====================
def news_score(ticker):

    try:
        t = yf.Ticker(ticker)
        news = t.news
    except:
        return 0

    score = 0

    for n in news[:5]:
        title = n.get("title", "")

        if any(k in title for k in ["上方修正", "増益", "好調", "買収", "自社株"]):
            score += 10

        if any(k in title for k in ["下方修正", "減益", "不祥事", "赤字"]):
            score -= 15

        if any(k in title for k in ["AI", "半導体", "需要", "成長"]):
            score += 5

    return score

# =====================
# AIモデル用データ
# =====================
all_data = []

for t in TICKERS:

    df = yf.download(t, period="5y", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty or len(df) < 200:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]

    df = df.copy()

    # =====特徴量=====
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

    # 目的変数
    df["target"] = (close.shift(-1) > close).astype(int)

    all_data.append(df)

data = pd.concat(all_data).dropna()

features = ["ret1","ma5","ma25","ma75","vol_ratio","rsi","macd","signal"]

X = data[features]
y = data["target"]

split = int(len(data)*0.7)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# =====================
# モデル
# =====================
model = RandomForestClassifier(n_estimators=300, max_depth=7, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)

# =====================
# 最新予測
# =====================
latest = X.iloc[-1:]
prob = model.predict_proba(latest)[0][1]

# =====================
# ニューススコア
# =====================
news_total = 0
for t in TICKERS:
    news_total += news_score(t)

# =====================
# 最終スコア
# =====================
final_score = (prob * 100) + news_total

if final_score > 75:
    judge = "🔥 強い買い"
elif final_score > 60:
    judge = "🟢 買い候補"
else:
    judge = "⚪ 見送り"

# =====================
# Discord送信
# =====================
msg = f"""
📊 AI株スキャン（ニュース統合版）

■ AI勝率
{acc:.3f}

■ 翌日上昇確率
{prob*100:.2f}%

■ ニューススコア
{news_total}

■ 最終スコア
{final_score:.1f}

判定:
{judge}
"""

print(msg)
send(msg)

send("📊 ニュース統合AIスキャン完了")
