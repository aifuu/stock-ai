import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# =====================
# 銘柄
# =====================
TICKER = "9984.T"  # まず1銘柄で学習（重要）

# =====================
# データ取得
# =====================
df = yf.download(TICKER, period="5y", interval="1d", auto_adjust=True)

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.dropna()

close = df["Close"]
volume = df["Volume"]

# =====================
# 特徴量作成（AI用）
# =====================
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

# =====================
# 目的変数（翌日上がるか）
# =====================
df["target"] = (close.shift(-1) > close).astype(int)

df = df.dropna()

# =====================
# 学習データ
# =====================
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

X = df[features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, shuffle=False
)

# =====================
# モデル
# =====================
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# 予測精度（勝率）
# =====================
accuracy = model.score(X_test, y_test)

print("\n=== バックテスト結果 ===")
print(f"勝率（精度）: {accuracy:.3f}")

# =====================
# 最新予測（翌日上昇確率）
# =====================
latest = X.iloc[-1:].copy()

prob_up = model.predict_proba(latest)[0][1]

print("\n=== 最新シグナル ===")
print(f"翌日上昇確率: {prob_up*100:.2f}%")

# =====================
# シグナル判定
# =====================
if prob_up >= 0.65:
    print("🔥 強い買いシグナル")
elif prob_up >= 0.55:
    print("🟢 買い候補")
else:
    print("⚪ 見送り")
