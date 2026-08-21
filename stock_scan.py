import os
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


warnings.filterwarnings("ignore")


# =========================================================
# 設定
# =========================================================

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
    "6613.T",
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


# =========================================================
# バックテスト設定
# =========================================================

DATA_PERIOD = "5y"

# 過去何営業日を学習に使うか
TRAIN_DAYS = 500

# 何営業日ごとに再学習するか
RETRAIN_DAYS = 20

# 予測期間(何営業日後の騰落率をターゲットにするか)
FORWARD_DAYS = 3

# 実際の売買判定期間
MAX_HOLD_DAYS = 5

# 3クラス分類の閾値
DOWN_THRESHOLD = -1.5
UP_THRESHOLD = 1.5

# 利確 / 損切
TAKE_PROFIT = 0.08
STOP_LOSS = 0.04

# 流動性フィルター
MIN_AVG_VOLUME = 100000

# 最初の予測開始に必要な最低データ数
MIN_DATA_REQUIRED = 600


# =========================================================
# 特徴量
# =========================================================

FEATURES = [

    # 基本
    "ret1",
    "ma25",
    "ma75",
    "vol_ratio",
    "rsi",
    "adx",
    "macd",
    "signal",
    "from_high",
    "from_low",

    # ATR
    "atr_ratio",

    # 相対強度
    "relative_strength",

    # ボリンジャーバンド
    "bb_position",
    "bb_width",

    # OBV
    "obv_change",

    # 日経平均
    "nikkei_kairi25",
    "nikkei_rsi",
    "nikkei_macd",
    "nikkei_return_5d",

    # 日経225先物
    "future_return",
    "future_ma5",
    "future_rsi",
    "future_gap",
]


# =========================================================
# ダウンロード
# =========================================================

def safe_download(ticker, retries=3, **kwargs):

    for attempt in range(1, retries + 1):

        try:

            df = yf.download(
                ticker,
                **kwargs
            )

            if df is not None and not df.empty:

                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                return df

            print(
                f"{ticker} 空データ "
                f"(試行 {attempt}/{retries})"
            )

        except Exception as e:

            print(
                f"{ticker} 取得失敗 "
                f"(試行 {attempt}/{retries}): {e}"
            )

    return None


# =========================================================
# RSI
# =========================================================

def calc_rsi(close, period=14):

    close = close.squeeze()

    delta = close.diff()

    gain = delta.clip(lower=0)

    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    rsi = 100 - (
        100 / (1 + rs)
    )

    return rsi.fillna(100)


# =========================================================
# ADX
# =========================================================

def calc_adx(df, period=14):

    high = df["High"].squeeze()

    low = df["Low"].squeeze()

    close = df["Close"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0.0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0.0
    )

    plus_dm = pd.Series(plus_dm, index=high.index)
    minus_dm = pd.Series(minus_dm, index=high.index)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    plus_dm_sm = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_sm = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    plus_di = 100 * plus_dm_sm / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_sm / atr.replace(0, np.nan)

    dx = (
        (plus_di - minus_di).abs()
        / (plus_di + minus_di).replace(0, np.nan)
    ) * 100

    adx = dx.ewm(alpha=1 / period, adjust=False).mean()

    return adx


# =========================================================
# ATR
# =========================================================

def calc_atr(df, period=14):

    high = df["High"].squeeze()
    low = df["Low"].squeeze()
    close = df["Close"].squeeze()

    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()

    return atr


# =========================================================
# 特徴量作成
# =========================================================

def create_features(df):

    df = df.copy()

    close = df["Close"].squeeze()
    volume = df["Volume"].squeeze()

    df["ret1"] = close.pct_change()

    df["ma25"] = close.rolling(25).mean()
    df["ma75"] = close.rolling(75).mean()

    df["vol_ratio"] = volume / volume.rolling(20).mean()

    df["rsi"] = calc_rsi(close)
    df["adx"] = calc_adx(df)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    high252 = close.rolling(252).max()
    low252 = close.rolling(252).min()

    df["from_high"] = (close / high252 - 1) * 100
    df["from_low"] = (close / low252 - 1) * 100

    atr = calc_atr(df)
    df["atr_ratio"] = atr / close

    df["_stock_ret5"] = close.pct_change(5)

    bb_ma20 = close.rolling(20).mean()
    bb_std20 = close.rolling(20).std()

    bb_upper = bb_ma20 + bb_std20 * 2
    bb_lower = bb_ma20 - bb_std20 * 2
    band_width = bb_upper - bb_lower

    df["bb_position"] = (
        (close - bb_lower) / band_width.replace(0, np.nan)
    )

    df["bb_width"] = (
        band_width / bb_ma20.replace(0, np.nan) * 100
    )

    direction = np.sign(close.diff())
    obv = (volume * direction).fillna(0).cumsum()

    df["obv_change"] = (
        obv.diff(5) / volume.rolling(5).sum() * 100
    )

    return df


# =========================================================
# 日経特徴量
# =========================================================

def create_nikkei_features(nikkei):

    nikkei = nikkei.copy()

    close = nikkei["Close"].squeeze()

    ma25 = close.rolling(25).mean()

    nikkei["nikkei_kairi25"] = (
        (close - ma25) / ma25.replace(0, np.nan) * 100
    )

    nikkei["nikkei_rsi"] = calc_rsi(close)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()

    nikkei["nikkei_macd"] = ema12 - ema26

    nikkei["nikkei_return_5d"] = close.pct_change(5) * 100
    nikkei["nikkei_ret5_raw"] = close.pct_change(5)

    return nikkei


# =========================================================
# 先物特徴量
# =========================================================

def create_future_features(futures):

    futures = futures.copy()

    close = futures["Close"].squeeze()

    futures["future_return"] = close.pct_change()
    futures["future_ma5"] = close.rolling(5).mean()
    futures["future_rsi"] = calc_rsi(close)

    futures["future_gap"] = (
        (close - close.shift(1)) / close.shift(1)
    )

    # 先物だけ1日ラグ(データ配信タイミングのズレ対策)
    futures["future_return"] = futures["future_return"].shift(1)
    futures["future_ma5"] = futures["future_ma5"].shift(1)
    futures["future_rsi"] = futures["future_rsi"].shift(1)
    futures["future_gap"] = futures["future_gap"].shift(1)

    return futures


# =========================================================
# ターゲット作成
# =========================================================

def create_target(df):

    close = df["Close"].squeeze()

    future_price = close.shift(-FORWARD_DAYS)

    future_return = (future_price / close - 1) * 100

    target = np.select(
        [
            future_return <= DOWN_THRESHOLD,
            future_return >= UP_THRESHOLD,
        ],
        [0, 2],
        default=1
    )

    return pd.Series(target, index=df.index), future_return


# =========================================================
# 流動性チェック
# =========================================================

def liquidity_ok(df):

    volume = df["Volume"].squeeze()
    avg_volume = volume.tail(20).mean()

    return avg_volume >= MIN_AVG_VOLUME


# =========================================================
# 売買結果判定
# =========================================================

def evaluate_trade(df, entry_index, entry_price):

    future = df.iloc[
        entry_index + 1: entry_index + 1 + MAX_HOLD_DAYS
    ]

    if len(future) == 0:
        return None

    take_profit = entry_price * (1 + TAKE_PROFIT)
    stop_loss = entry_price * (1 - STOP_LOSS)

    for day_number, (_, row) in enumerate(future.iterrows(), start=1):

        high = float(row["High"])
        low = float(row["Low"])

        # 同日に両方到達した場合は保守的にLOSS扱い
        if low <= stop_loss and high >= take_profit:
            return {
                "result": "LOSS",
                "return": -STOP_LOSS * 100,
                "hold_days": day_number,
            }

        if high >= take_profit:
            return {
                "result": "WIN",
                "return": TAKE_PROFIT * 100,
                "hold_days": day_number,
            }

        if low <= stop_loss:
            return {
                "result": "LOSS",
                "return": -STOP_LOSS * 100,
                "hold_days": day_number,
            }

    last_close = float(future.iloc[-1]["Close"])
    return_rate = (last_close / entry_price - 1) * 100

    result = "TIMEOUT_LOSS" if return_rate < 0 else "HOLD"

    return {
        "result": result,
        "return": return_rate,
        "hold_days": len(future),
    }


# =========================================================
# ウォークフォワード
# =========================================================

def run_walk_forward(ticker, df):

    print("")
    print("=" * 60)
    print(f"🚀 ウォークフォワード開始 {ticker} {COMPANY_NAMES.get(ticker, '')}")
    print("=" * 60)

    df = df.copy()

    target, future_return = create_target(df)

    df["target"] = target
    df["_future_return"] = future_return

    if len(df) < MIN_DATA_REQUIRED:
        print(f"{ticker}: データ不足 {len(df)}日")
        return []

    results = []

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=7,
        random_state=42,
        class_weight="balanced"
    )

    start = TRAIN_DAYS

    last_valid_index = len(df) - FORWARD_DAYS - MAX_HOLD_DAYS

    prediction_count = 0

    current = start

    while current < last_valid_index:

        train_start = current - TRAIN_DAYS
        train_end = current

        test_start = current
        test_end = min(current + RETRAIN_DAYS, last_valid_index)

        # =====================================================
        # 【修正①】学習データの境界リーク対策
        #
        # train_end 直前の行は、そのtarget(FORWARD_DAYS後の
        # 騰落率)を計算する際に test_start 以降(=これから
        # 予測する未来期間)の価格を参照してしまっている。
        #
        # つまり「学習ラベルが検証期間の値を覗き見ている」状態
        # になるため、target が train_end より前の価格だけで
        # 確定している行(= train_end - FORWARD_DAYS まで)に
        # 限定する。
        # =====================================================
        safe_train_end = train_end - FORWARD_DAYS

        if safe_train_end <= train_start:
            current = test_end
            continue

        train_df = df.iloc[train_start:safe_train_end].copy()

        train_df = train_df.dropna(subset=FEATURES + ["target"])

        if len(train_df) < 100:
            current = test_end
            continue

        if train_df["target"].nunique() < 3:
            print(f"{ticker}: {df.index[current].date()} 3クラス不足")
            current = test_end
            continue

        X_train = train_df[FEATURES]
        y_train = train_df["target"].astype(int)

        model.fit(X_train, y_train)

        print(
            f"{ticker} | "
            f"学習 {df.index[train_start].date()} "
            f"～ {df.index[safe_train_end - 1].date()} "
            f"(target確定分のみ) | "
            f"予測 {df.index[test_start].date()} "
            f"～ {df.index[test_end - 1].date()}"
        )

        for i in range(test_start, test_end):

            row = df.iloc[i]

            if row[FEATURES].isna().any():
                continue

            X_test = row[FEATURES].to_frame().T

            try:
                probabilities = model.predict_proba(X_test)[0]
                classes = list(model.classes_)

                up_prob = probabilities[classes.index(2)] if 2 in classes else 0
                flat_prob = probabilities[classes.index(1)] if 1 in classes else 0
                down_prob = probabilities[classes.index(0)] if 0 in classes else 0

                prediction = int(model.predict(X_test)[0])

            except Exception as e:
                print(f"{ticker} 予測失敗: {e}")
                continue

            actual = int(df.iloc[i]["target"])

            entry_price = float(df.iloc[i]["Close"])

            trade = evaluate_trade(df, i, entry_price)

            if trade is None:
                continue

            result_row = {
                "date": df.index[i].strftime("%Y-%m-%d"),
                "ticker": ticker,
                "prediction": prediction,
                "actual": actual,
                "correct": int(prediction == actual),
                "down_probability": round(down_prob * 100, 2),
                "flat_probability": round(flat_prob * 100, 2),
                "up_probability": round(up_prob * 100, 2),
                "price": round(entry_price, 2),
                "result": trade["result"],
                "return": round(trade["return"], 2),
                "hold_days": trade["hold_days"],
                "train_start": df.index[train_start].strftime("%Y-%m-%d"),
                "train_end": df.index[safe_train_end - 1].strftime("%Y-%m-%d"),
            }

            results.append(result_row)

            prediction_count += 1

        current = test_end

    print(f"✅ {ticker}: {prediction_count}件の予測完了")

    return results


# =========================================================
# 【修正②】ポートフォリオ資産曲線(日次集計版)
#
# 全取引を単純に時系列順へ1本の口座として複利計算すると、
# 実際には複数銘柄を同時に保有している実態と食い違う。
# ここではエントリー日ごとに、その日建てた全ポジションの
# 平均リターンを求め、それを日次リターンとして複利計算する。
# =========================================================

def build_portfolio_equity_curve(result_df):

    daily = (
        result_df
        .groupby("date")["return"]
        .mean()
        .sort_index()
    )

    equity = (1 + daily / 100).cumprod()

    peak = equity.cummax()
    drawdown = equity / peak - 1

    max_drawdown = drawdown.min() * 100

    return equity, max_drawdown


# =========================================================
# メイン
# =========================================================

def main():

    print("")
    print("=" * 60)
    print("📊 ウォークフォワード・バックテスト(修正版)")
    print("=" * 60)

    print(f"学習期間: {TRAIN_DAYS}営業日")
    print(f"再学習間隔: {RETRAIN_DAYS}営業日")
    print(f"予測期間: {FORWARD_DAYS}営業日")
    print(f"売買判定: 最大{MAX_HOLD_DAYS}営業日")
    print(f"利確: +{TAKE_PROFIT * 100:.1f}%")
    print(f"損切: -{STOP_LOSS * 100:.1f}%")
    print(f"流動性フィルター: 20日平均出来高 {MIN_AVG_VOLUME:,}")
    print("")

    # =====================================================
    # 日経平均
    # =====================================================

    nikkei = safe_download(
        "^N225", period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False
    )

    if nikkei is None:
        print("❌ 日経平均取得失敗")
        return

    nikkei = create_nikkei_features(nikkei)

    # =====================================================
    # 日経225先物
    # =====================================================

    futures = safe_download(
        "NIY=F", period=DATA_PERIOD, interval="1d",
        auto_adjust=True, progress=False
    )

    if futures is None:
        print("⚠ 先物取得失敗")
        futures = pd.DataFrame()
    else:
        futures = create_future_features(futures)

    # =====================================================
    # 全銘柄
    # =====================================================

    all_results = []
    skipped_liquidity = []

    for ticker in TICKERS:

        print("")
        print(f"📥 データ取得: {ticker}")

        stock = safe_download(
            ticker, period=DATA_PERIOD, interval="1d",
            auto_adjust=False, progress=False
        )

        if stock is None:
            print(f"{ticker}: データ取得失敗")
            continue

        if len(stock) < MIN_DATA_REQUIRED:
            print(f"{ticker}: データ不足 {len(stock)}日")
            continue

        if not liquidity_ok(stock):
            avg_volume = stock["Volume"].squeeze().tail(20).mean()
            print(f"⚠ {ticker}: 流動性不足 平均出来高={avg_volume:,.0f}")
            skipped_liquidity.append(ticker)
            continue

        stock = create_features(stock)

        stock = stock.join(
            nikkei[[
                "nikkei_kairi25",
                "nikkei_rsi",
                "nikkei_macd",
                "nikkei_return_5d",
                "nikkei_ret5_raw",
            ]],
            how="left"
        )

        if not futures.empty:
            stock = stock.join(
                futures[[
                    "future_return",
                    "future_ma5",
                    "future_rsi",
                    "future_gap",
                ]],
                how="left"
            )
        else:
            for col in ["future_return", "future_ma5", "future_rsi", "future_gap"]:
                stock[col] = np.nan

        stock["relative_strength"] = (
            stock["_stock_ret5"] - stock["nikkei_ret5_raw"]
        )

        stock = stock.dropna(subset=FEATURES + ["Close"])

        if len(stock) < MIN_DATA_REQUIRED:
            print(f"{ticker}: 特徴量作成後データ不足")
            continue

        ticker_results = run_walk_forward(ticker, stock)

        all_results.extend(ticker_results)

    if not all_results:
        print("❌ バックテスト結果なし")
        return

    result_df = pd.DataFrame(all_results)

    result_df.to_csv(
        "walk_forward_results.csv", index=False, encoding="utf-8-sig"
    )

    print("")
    print("✅ walk_forward_results.csv 保存")

    # =====================================================
    # Accuracy / Precision
    # =====================================================

    accuracy = result_df["correct"].mean() * 100

    try:
        precision = precision_score(
            result_df["actual"],
            result_df["prediction"],
            labels=[0, 1, 2],
            average="macro",
            zero_division=0
        ) * 100
    except Exception:
        precision = 0

    # =====================================================
    # 売買結果
    # =====================================================

    wins = (result_df["result"] == "WIN").sum()
    losses = result_df["result"].isin(["LOSS", "TIMEOUT_LOSS"]).sum()
    holds = (result_df["result"] == "HOLD").sum()

    trades = wins + losses
    win_rate = wins / trades * 100 if trades > 0 else 0

    avg_return = result_df["return"].mean()
    total_return = result_df["return"].sum()

    # =====================================================
    # 【修正②】ポートフォリオ資産曲線(日次集計)
    # =====================================================

    equity, max_drawdown = build_portfolio_equity_curve(result_df)

    portfolio_total_return = (equity.iloc[-1] - 1) * 100

    # =====================================================
    # 結果表示
    # =====================================================

    print("")
    print("=" * 60)
    print("📊 ウォークフォワード結果")
    print("=" * 60)

    print(f"予測件数       : {len(result_df)}")
    print(f"Accuracy       : {accuracy:.2f}%")
    print(f"Precision      : {precision:.2f}%")

    print("")
    print(f"WIN            : {wins}")
    print(f"LOSS           : {losses}")
    print(f"HOLD           : {holds}")
    print(f"勝率           : {win_rate:.2f}%")

    print("")
    print(f"平均リターン(1トレードあたり) : {avg_return:.2f}%")
    print(f"単純合計リターン(参考値)      : {total_return:.2f}%")
    print("")
    print("--- ポートフォリオ換算(日次・並行ポジション考慮) ---")
    print(f"ポートフォリオ累計リターン    : {portfolio_total_return:+.2f}%")
    print(f"最大ドローダウン              : {max_drawdown:.2f}%")

    print("")

    # =====================================================
    # 銘柄別成績
    # =====================================================

    print("=" * 60)
    print("📊 銘柄別成績")
    print("=" * 60)

    for ticker, group in result_df.groupby("ticker"):

        ticker_accuracy = group["correct"].mean() * 100

        ticker_wins = (group["result"] == "WIN").sum()
        ticker_losses = group["result"].isin(["LOSS", "TIMEOUT_LOSS"]).sum()
        ticker_trades = ticker_wins + ticker_losses

        ticker_win_rate = (
            ticker_wins / ticker_trades * 100 if ticker_trades > 0 else 0
        )

        ticker_avg_return = group["return"].mean()

        print(
            f"{ticker:8s} "
            f"{COMPANY_NAMES.get(ticker, ''):10s} "
            f"Accuracy={ticker_accuracy:5.1f}% "
            f"勝率={ticker_win_rate:5.1f}% "
            f"平均={ticker_avg_return:+.2f}% "
            f"件数={len(group)}"
        )

    # =====================================================
    # 資産曲線CSV保存(グラフ化用)
    # =====================================================

    equity.to_csv(
        "portfolio_equity_curve.csv",
        header=["equity"],
        encoding="utf-8-sig"
    )

    print("")
    print("✅ portfolio_equity_curve.csv 保存(日次資産曲線)")

    # =====================================================
    # 流動性で除外された銘柄
    # =====================================================

    if skipped_liquidity:
        print("")
        print("⚠ 流動性フィルターで除外:")
        for ticker in skipped_liquidity:
            print(f"  {ticker} {COMPANY_NAMES.get(ticker, '')}")

    print("")
    print("=" * 60)
    print("✅ ウォークフォワード・バックテスト完了")
    print("=" * 60)

    print("結果ファイル:")
    print("  walk_forward_results.csv     (取引ごとの明細)")
    print("  portfolio_equity_curve.csv   (日次資産曲線)")


# =========================================================
# 実行
# =========================================================

if __name__ == "__main__":
    main()
