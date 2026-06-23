import csv
import os
from datetime import datetime

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

COMPANY_NAMES = {
    "6857.T": "アドバンテスト",
    "8035.T": "東京エレクトロン",
    "6920.T": "レーザーテック",
    "6526.T": "ソシオネクスト",
    "6501.T": "日立製作所",
    "6503.T": "三菱電機",
    "5803.T": "フジクラ",
    "7011.T": "三菱重工業",
    "4980.T": "デクセリアルズ",
    "9984.T": "ソフトバンクG"
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
        "vol_ratio"
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

        df = yf.download(ticker, period="3y", interval="1d", auto_adjust=True)

               if df is None or len(df) < 150:
            continue

        close = df["Close"].squeeze()
        volume = df["Volume"].squeeze()

　　　　price = float(close.iloc[-1])

        # ===== 特徴量 =====
        df["ret1"] = close.pct_change()
        df["ma25"] = close.rolling(25).mean()
        df["ma75"] = close.rolling(75).mean()
        df["vol_ratio"] = volume / volume.rolling(20).mean()
        df["rsi"] = calc_rsi(close)

        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd"] = ema12 - ema26
        df["signal"] = df["macd"].ewm(span=9).mean()

        df = df.dropna()

        # ===== ラベル =====
        df["target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

        features = ["ret1","ma25","ma75","vol_ratio","rsi","macd","signal"]

        X = df[features]
        y = df["target"]

        if len(X) < 100:
            continue

        split = int(len(X) * 0.8)

        X_train = X.iloc[:split]
        y_train = y.iloc[:split]

        # =====================
        # 学習
        # =====================
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_FILE)

        latest = X.iloc[-1:]
        prob = model.predict_proba(latest)[0][1]

        # =====================
        # 数値取得（Series事故防止）
        # =====================
        price = float(close.iloc[-1])
        rsi = float(df["rsi"].iloc[-1])
        macd = float(df["macd"].iloc[-1])
        signal = float(df["signal"].iloc[-1])
        ma25 = float(df["ma25"].iloc[-1])
        ma75 = float(df["ma75"].iloc[-1])
        vol_ratio = float(df["vol_ratio"].iloc[-1])

        # =====================
        # 利確・損切
        # =====================
        take_profit = price * 1.08
        stop_loss = price * 0.95

        # =====================
        # CSV保存
        # =====================
        with open(TRAIN_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d"),
                ticker,
                rsi,
                macd,
                signal,
                ma25,
                ma75,
                vol_ratio,
                int(df["target"].iloc[-1])
            ])

        # =====================
        # スコア
        # =====================
        score = 0

        if rsi < 35:
            score += 25
        if macd > signal:
            score += 25
        if ma25 > ma75:
            score += 20
        if vol_ratio > 1.5:
            score += 20

        score += prob * 30

        results.append({
            "ticker": ticker,
            "score": round(score,1),
            "prob": round(prob*100,1),
            "price": round(price,0),
            "rsi": round(rsi,1),
            "vol": round(vol_ratio,2),
            "take_profit": round(take_profit,0),
            "stop_loss": round(stop_loss,0)
        })

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
