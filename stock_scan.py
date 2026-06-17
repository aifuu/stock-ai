import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# =====================
# 銘柄（あなたのやつ）
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
# データ作成
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

    # =====目的変数=====
    # 翌日上がるか
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
# 時系列分割
# =====================
split = int(len(data) * 0.7)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# =====================
# AIモデル
# =====================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=7,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# AI勝率（バックテスト）
# =====================
accuracy = model.score(X_test, y_test)

print("\n=== AIバックテスト結果 ===")
print(f"勝率（正解率）: {accuracy:.3f}")

# =====================
# 利益シミュレーション
# =====================
test_prices = data.iloc[split:].copy()

test_prices["prob"] = model.predict_proba(X_test)[:, 1]

test_prices["strategy"] = 0

# シグナル条件
test_prices.loc[test_prices["prob"] > 0.6, "strategy"] = 1

# リターン
test_prices["market_return"] = data["Close"].pct_change().iloc[split:]
test_prices["strategy_return"] = test_prices["market_return"] * test_prices["strategy"]

cum_market = (1 + test_prices["market_return"]).cumprod().iloc[-1]
cum_strategy = (1 + test_prices["strategy_return"]).cumprod().iloc[-1]

print("\n=== 利益バックテスト ===")
print(f"市場リターン: {cum_market:.2f}倍")
print(f"AI戦略リターン: {cum_strategy:.2f}倍")

# =====================
# 最新予測
# =====================
latest = X.iloc[-1:]
prob = model.predict_proba(latest)[0][1]

print("\n=== 最新シグナル ===")
print(f"翌日上昇確率: {prob*100:.2f}%")

if prob > 0.65:
    print("🔥 強い買い")
elif prob > 0.55:
    print("🟢 監視")
else:
    print("⚪ 見送り")
