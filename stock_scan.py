import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================
# あなたの銘柄
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
# データ統合
# =====================
all_data = []

for ticker in TICKERS:

    df = yf.download(ticker, period="5y", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty or len(df) < 200:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna()

    close = df["Close"]
    volume = df["Volume"]

    df["ticker"] = ticker

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

    # 目的変数（翌日上昇）
    df["target"] = (close.shift(-1) > close).astype(int)

    all_data.append(df)

# =====================
# 統合
# =====================
data = pd.concat(all_data)
data = data.dropna()

features = [
    "ret1",
    "ma5",
    "ma25",
    "ma75",
    "vol_ratio",
    "rsi",
    "macd",
    "signal"
]

X = data[features]
y = data["target"]

# =====================
# 時系列分割（重要）
# =====================
split = int(len(data) * 0.7)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# =====================
# モデル
# =====================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# バックテスト（勝率）
# =====================
accuracy = model.score(X_test, y_test)

print("\n=== バックテスト結果 ===")
print(f"勝率: {accuracy:.3f}")

# =====================
# 最新シグナル（全銘柄）
# =====================
print("\n=== 最新AIシグナル ===")

for ticker in TICKERS:

    df = yf.download(ticker, period="6mo", interval="1d", auto_adjust=True, progress=False)

    if df is None or df.empty or len(df) < 100:
        continue

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]

    latest = pd.DataFrame({
        "ret1": [close.pct_change().iloc[-1]],
        "ma5": [close.rolling(5).mean().iloc[-1]],
        "ma25": [close.rolling(25).mean().iloc[-1]],
        "ma75": [close.rolling(75).mean().iloc[-1]],
        "vol_ratio": [volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]],
        "rsi": [0],
        "macd": [0],
        "signal": [0]
    })

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    latest["rsi"] = 100 - (100 / (1 + rs.iloc[-1]))

    # MACD
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    latest["macd"] = macd.iloc[-1]
    latest["signal"] = signal.iloc[-1]

    prob = model.predict_proba(latest[features])[0][1]

    if prob >= 0.6:
        print(f"🔥 {ticker} 上昇確率: {prob*100:.1f}%")
    elif prob >= 0.55:
        print(f"🟢 {ticker} 監視: {prob*100:.1f}%")
