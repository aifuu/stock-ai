"""
単体テクニカル指標バックテスト（研究・診断専用スクリプト）
=================================================

★このスクリプトは本番トレードロジック(daily_directional_top1.py の main()、
profit_top10_paper.py の scan()/open_positions()/mark_and_close() など)を
一切呼び出さず、一切変更もしない。どのGitHub Actionsワークフロー
(.github/workflows/*.yml)からも実行されない、完全に独立した研究・診断専用
スクリプト。adversarial_oos_diagnostic.py などと同じ位置付け(本番非依存・
本番に一切影響しない)。

背景:
  train_data.csvのtarget列(0/1/2の3値分類ラベル)を使ったクイックな
  探索的分析で「個別テクニカル指標単体の予測力」を調べたことがあったが、
  targetは実際の売買損益(TP/SL早期決済・手数料・保有日数)を反映していない。
  本スクリプトは、日経225全銘柄を対象に、本番と同じATR倍率のTP/SL・
  最大保有営業日数を使った現実的な売買シミュレーションで、以下14指標を
  「他の指標と組み合わせず単体で使った場合」の実際の取引成績を検証する
  (今後も使い回せる再現可能な検証ツールとして残す)。

  対象14指標: ゴールデンクロス/デッドクロス、ma25_slope5, rsi, adx, macd,
  signal, momentum_score, bb_position, bb_width, atr_ratio, volatility20,
  upper_wick_pct, lower_wick_pct, breakout20

設計上の注記:
  - 特徴量計算はdaily_directional_top1.pyのdownload()/make_nikkei()/
    features()/atr()をそのまま流用し、二重実装しない。
  - エントリーはBUY方向のみ(SHORTは今回のスコープ外)。
  - 連続値指標13個については、「直近1年(252営業日)の分布で上位20%」と
    「下位20%」の両方を候補として実際にバックテストし、どちら側に
    エッジがあるかは過去のアドホックな分析結果を決め打ちで埋め込むのではなく、
    本スクリプト自身のTP/SLシミュレーション結果から判定する(そのほうが
    再現性・検証可能性が高いため)。
  - ゴールデンクロス/デッドクロスはイベント発生日(ma25とma75の差の符号が
    前日から変化した日)にBUYエントリー。
  - 累積複利%は、同一銘柄内では決済まで重複エントリーしないが、
    銘柄間の並行保有(資金制約)は考慮せず、全銘柄の全トレードを
    エントリー日時系列順に単純に複利計算する簡略化。指標間の相対比較用の
    診断値であり、ポートフォリオ全体の厳密な資金シミュレーションではない。
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from daily_directional_top1 import TICKERS, atr, download, features, make_nikkei  # noqa: E402

PERIOD = "2y"
LOOKBACK_WINDOW = 252  # 直近1年(営業日)の分布で上位/下位20%を判定する窓
PERCENTILE = 0.20
MIN_ROWS = LOOKBACK_WINDOW + 30  # 判定窓+シグナル発生後の余裕

# 本番承認済みpolicy(strategy_policy.json想定)と同じATR倍率・保有日数
TP_MULT = 3.5
SL_MULT = 2.0
HOLD_DAYS = 3
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))

CONTINUOUS_INDICATORS = [
    "ma25_slope5", "rsi", "adx", "macd", "signal", "momentum_score",
    "bb_position", "bb_width", "atr_ratio", "volatility20",
    "upper_wick_pct", "lower_wick_pct", "breakout20",
]


def continuous_entry_signals(x, col):
    s = x[col]
    roll = s.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW)
    hi_thr = roll.quantile(1 - PERCENTILE)
    lo_thr = roll.quantile(PERCENTILE)
    high_signal = (s >= hi_thr).fillna(False)
    low_signal = (s <= lo_thr).fillna(False)
    return high_signal, low_signal


def cross_entry_signals(x):
    gap = x["ma25"] - x["ma75"]
    sign = np.sign(gap)
    prev_sign = sign.shift(1)
    golden = ((sign > 0) & (prev_sign <= 0) & prev_sign.notna()).fillna(False)
    dead = ((sign < 0) & (prev_sign >= 0) & prev_sign.notna()).fillna(False)
    return golden, dead


def simulate_trades(x, entry_signal, ticker):
    """entry_signalがTrueの日にBUYエントリーし、本番と同じATR倍率のTP/SL・
    最大保有営業日数で決済する現実的なシミュレーション
    (profit_top10_paper.pyのmark_and_close()の決済ロジックを参考にした簡易実装。
    本番コードは呼び出さない)。"""
    trades = []
    close = x["Close"]
    high = x["High"]
    low = x["Low"]
    atr_series = x["_atr"]
    idx = x.index
    n = len(idx)
    i = 0
    while i < n:
        if not bool(entry_signal.iloc[i]) or i + HOLD_DAYS > n - 1:
            i += 1
            continue
        entry_price = float(close.iloc[i])
        a = float(atr_series.iloc[i])
        if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(a) or a <= 0:
            i += 1
            continue
        tp = entry_price + a * TP_MULT
        sl = entry_price - a * SL_MULT
        exit_price = None
        reason = None
        exit_i = None
        for j in range(i + 1, i + HOLD_DAYS + 1):
            hi, lo = float(high.iloc[j]), float(low.iloc[j])
            if lo <= sl and hi >= tp:
                exit_price, reason = sl, "SL"
            elif hi >= tp:
                exit_price, reason = tp, "TP"
            elif lo <= sl:
                exit_price, reason = sl, "SL"
            if reason:
                exit_i = j
                break
        if reason is None:
            exit_i = i + HOLD_DAYS
            exit_price = float(close.iloc[exit_i])
            reason = "HOLD_LIMIT"
        gross_ret_pct = (exit_price - entry_price) / entry_price * 100
        fee_pct = (entry_price + exit_price) / entry_price * FEE_RATE * 100
        net_ret_pct = gross_ret_pct - fee_pct
        trades.append({
            "ticker": ticker,
            "entry_date": idx[i],
            "exit_date": idx[exit_i],
            "return_pct": net_ret_pct,
            "reason": reason,
        })
        i = exit_i + 1  # 同一銘柄内は決済まで重複エントリーしない
    return trades


def build_strategies():
    strategies = [("golden_cross_buy", None, "golden"), ("dead_cross_buy", None, "dead")]
    for col in CONTINUOUS_INDICATORS:
        strategies.append((f"{col}_top20pct", col, "high"))
        strategies.append((f"{col}_bottom20pct", col, "low"))
    return strategies


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append("-" if pd.isna(v) else f"{v:.2f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def aggregate(name, trades):
    if not trades:
        return {
            "indicator_strategy": name, "trades": 0, "win_rate_pct": np.nan,
            "avg_return_pct": np.nan, "cumulative_compound_pct": np.nan, "annualized_pct": np.nan,
        }
    tdf = pd.DataFrame(trades).sort_values("entry_date")
    n_trades = len(tdf)
    win_rate = float((tdf["return_pct"] > 0).mean() * 100)
    avg_ret = float(tdf["return_pct"].mean())
    equity = 1.0
    for r in tdf["return_pct"]:
        equity *= (1 + r / 100)
    cum_pct = (equity - 1) * 100
    span_days = (pd.Timestamp(tdf["exit_date"].max()) - pd.Timestamp(tdf["entry_date"].min())).days
    years = span_days / 365.25 if span_days > 0 else np.nan
    if years and years > 0 and equity > 0:
        annual_pct = (equity ** (1 / years) - 1) * 100
    else:
        annual_pct = np.nan
    return {
        "indicator_strategy": name, "trades": n_trades, "win_rate_pct": win_rate,
        "avg_return_pct": avg_ret, "cumulative_compound_pct": cum_pct, "annualized_pct": annual_pct,
    }


def main(tickers=None):
    tickers = tickers or TICKERS
    nikkei = make_nikkei()
    if nikkei is None:
        print("⚠ 日経225データ取得失敗(ネットワーク不通の可能性)。終了します。")
        return None

    strategies = build_strategies()
    all_trades = {name: [] for name, _, _ in strategies}

    processed = 0
    for n_idx, ticker in enumerate(tickers, 1):
        df = download(ticker, period=PERIOD)
        if df is None or len(df) < MIN_ROWS:
            continue
        x = features(df, nikkei)
        x["_atr"] = atr(x)
        processed += 1
        golden_sig, dead_sig = cross_entry_signals(x)
        for name, col, kind in strategies:
            if kind == "golden":
                sig = golden_sig
            elif kind == "dead":
                sig = dead_sig
            else:
                high_sig, low_sig = continuous_entry_signals(x, col)
                sig = high_sig if kind == "high" else low_sig
            all_trades[name].extend(simulate_trades(x, sig, ticker))
        if n_idx % 20 == 0 or n_idx == len(tickers):
            print(f"  進捗: {n_idx}/{len(tickers)}銘柄処理済み(有効データ{processed}件)")

    if processed == 0:
        print("⚠ 有効なデータを取得できた銘柄が0件でした。終了します。")
        return None

    rows = [aggregate(name, all_trades[name]) for name, _, _ in strategies]
    result_df = pd.DataFrame(rows).sort_values(
        "annualized_pct", ascending=False, na_position="last"
    ).reset_index(drop=True)

    print(f"\n対象銘柄数: {len(tickers)}｜有効データ取得: {processed}件｜"
          f"期間: 直近{PERIOD}｜TP×{TP_MULT}/SL×{SL_MULT}(ATR)｜最大保有{HOLD_DAYS}営業日\n")
    print(to_markdown(result_df))
    return result_df


if __name__ == "__main__":
    main()
