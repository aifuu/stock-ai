"""
nikkei_macd_dip_paper.py TP/SL比較バックテスト(研究・診断専用スクリプト)
====================================================================

★このスクリプトは本番トレードロジック(daily_directional_top1.py の main()、
profit_top10_paper.py の scan()/open_positions()/mark_and_close()、
nikkei_macd_dip_paper.py の _run() 等)を一切呼び出さず、一切変更もしない。
本番の状態ファイル(strategy_policy.json, profit_top10_paper_state.json,
nikkei_macd_dip_paper_state.json 等)も一切読み書きしない、完全に独立した
研究・診断専用スクリプト(single_indicator_backtest.py, adversarial_oos_diagnostic.py
と同じ位置付け)。どのGitHub Actionsワークフロー(既存の *.yml)からも実行されない。

背景:
  nikkei_macd_dip_paper.py の TOP_MULT/SL_MULT は当初 3.5/2.0 (ATR倍率、
  single_indicator_backtest.py の検証値) で作成されたが、その後ユーザーが
  手動で 0.22/0.1257 (利確 約+1%/損切 約-0.56%、RR比1.75:1) に変更した
  (commit ce4383b)。この新しいTP/SLは一度も実データで検証されていない。
  本スクリプトは、nikkei_macd_dip_paper.py と完全に同一のシグナル判定
  (日経225自身のMACDが直近252営業日ローリング分位点で下位10%以下)・
  TOP1選定ロジック(daily_directional_top1.py の directional_score() を
  そのまま流用)を使い、過去全期間(データが取得できる範囲)を日次で
  ウォークフォワード・シミュレーションし、旧TP/SL(3.5/2.0)と
  新TP/SL(0.22/0.1257)の実績を比較する。

設計上の注記:
  - シグナル判定・TOP1選定・決済(SL優先の同日判定、最大保有3営業日)ロジックは
    nikkei_macd_dip_paper.py / single_indicator_backtest.py と同一の計算式を
    このスクリプト内に再実装している(本番コード・状態ファイルは未変更)。
  - AIモデルは directional_model.pkl を1つだけロードし、全期間(過去〜現在)に
    そのまま適用する簡易版(その時点までに学習された過去のモデルを都度
    再現することはしない)。本来は各時点までのデータで学習したモデルを
    使うのが理想だが、モデルのバージョン管理履歴が無いためこの簡易版とした。
    このため過去期間ほど「未来のモデルを使った先読み」のバイアスが
    かかっている可能性がある点に留意。
  - daily_directional_top1.py の features() が内部で参照する日経平均先物
    (NIY=F)・TOPIX代替(1306.T)のキャッシュは、素の make_futures_features()/
    make_topix_features() だとデフォルトで直近3年分しか取得しないため、
    過去期間の future_*/vs_topix_1d_pt 列がNaNになり候補から除外されて
    しまう。本スクリプトでは同じ計算式のまま取得期間だけをPERIODに
    合わせて再取得し、モジュール内のキャッシュ変数を差し替えることで
    過去期間でも欠損しないようにしている。
  - ポジションは常に最大1件(TOP1集中投資)。シグナル成立日でもポジションを
    保有中はエントリーしない(nikkei_macd_dip_paper.py と同じ)。同一日での
    決済→即再エントリーは行わず、決済日の翌営業日以降のシグナル成立日から
    次のエントリーを探す(single_indicator_backtest.py の simulate_trades()
    と同じ「決済まで重複エントリーしない」方針を踏襲した簡略化。本番の
    日次実行では稀に決済と同日の再エントリーが起こり得るが、影響は軽微)。
  - 手数料は nikkei_macd_dip_paper.py と同じ FEE_RATE 定数を使用。
"""

import bisect
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import daily_directional_top1 as ddt  # noqa: E402
from daily_directional_top1 import (  # noqa: E402
    FEATURES,
    TICKERS,
    atr,
    directional_score,
    download,
    features,
    load_model,
    rsi,
)

PERIOD = "10y"  # データが取得できる限り遡る(目安2019年〜現在をカバーする狙い)
LOOKBACK_WINDOW = 252
PERCENTILE = 0.10
HOLD_DAYS = 3
FEE_RATE = float(os.getenv("INTRADAY_FEE_RATE", "0.00055"))

CONFIGS = [
    {"key": "old", "name": "旧TP/SL (ATR×3.5 / ATR×2.0)", "tp_mult": 3.5, "sl_mult": 2.0},
    {"key": "new", "name": "新TP/SL (ATR×0.22 / ATR×0.1257)", "tp_mult": 0.22, "sl_mult": 0.1257},
]


def log(msg):
    print(msg, flush=True)


def make_nikkei_long(period=PERIOD):
    """daily_directional_top1.make_nikkei() と完全に同一の計算式を、
    デフォルトの3年ではなくPERIOD分のデータで計算する。"""
    n = download("^N225", period=period)
    if n is None or n.empty:
        return None
    c = n["Close"].squeeze()
    ma25, ma75 = c.rolling(25).mean(), c.rolling(75).mean()
    return pd.DataFrame(
        {
            "kairi25": (c - ma25) / ma25 * 100,
            "rsi": rsi(c),
            "macd": c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean(),
            "ret5": c.pct_change(5) * 100,
            "ret5_raw": c.pct_change(5),
            "ret1": c.pct_change(),
            "nikkei_uptrend": ma25 > ma75,
        },
        index=n.index,
    )


def nikkei_macd_signal_series(nikkei):
    """nikkei_macd_dip_paper.py の nikkei_macd_signal() と同一の計算式
    (直近252営業日ローリング分位点の下位10%以下)を全期間の系列として計算する。"""
    macd = nikkei["macd"]
    roll = macd.rolling(LOOKBACK_WINDOW, min_periods=LOOKBACK_WINDOW)
    lo_thr = roll.quantile(PERCENTILE)
    signal = (macd <= lo_thr) & lo_thr.notna()
    return signal, lo_thr


def prime_long_period_side_features(period=PERIOD):
    """features()内部でキャッシュされるfutures_df/topix_dfを、デフォルトの3年分
    ではなくPERIOD分に差し替える(過去期間のfuture_*/vs_topix_1d_pt列がNaNに
    なって候補から除外されるのを防ぐため)。計算式はdaily_directional_top1.py
    のmake_futures_features()/make_topix_features()と同一。"""
    f = download("NIY=F", period=period)
    if f is not None and not f.empty:
        c = f["Close"].squeeze()
        out = pd.DataFrame(index=pd.to_datetime(f.index).normalize())
        out["future_return"] = c.to_numpy() / c.shift(1).to_numpy() - 1.0
        out["future_ma5"] = c.rolling(5).mean().to_numpy()
        out["future_rsi"] = rsi(c).to_numpy()
        out["future_gap"] = (c - c.shift(1)).to_numpy() / c.shift(1).to_numpy()
        out = out.shift(1)
        out = out[~out.index.duplicated(keep="last")]
        ddt._FUTURES_FEATURE_CACHE = out
        log(f"日経225先物(NIY=F): {len(out)}件 ({out.index.min().date()}〜{out.index.max().date()})")
    else:
        log("⚠ NIY=F取得失敗。future_*系特徴量はデフォルト(直近3年)またはNaNのまま")

    t = download(ddt.TOPIX_PROXY, period=period)
    if t is not None and not t.empty:
        c = t["Close"].squeeze()
        ddt._TOPIX_FEATURE_CACHE = pd.DataFrame({"ret1": c.pct_change()}, index=t.index)
        log(f"TOPIX代替({ddt.TOPIX_PROXY}): {len(t)}件")
    else:
        log("⚠ TOPIX代替取得失敗。vs_topix_1d_pt特徴量はデフォルト(直近3年)またはNaNのまま")


def load_all_ticker_frames(nikkei, cols):
    frames = {}
    n_tickers = len(TICKERS)
    for idx, ticker in enumerate(TICKERS, 1):
        df = download(ticker, period=PERIOD)
        if df is None or len(df) < LOOKBACK_WINDOW + HOLD_DAYS + 5:
            continue
        x = features(df, nikkei)
        x["_atr"] = atr(x)
        x["_valid"] = (
            x[cols].notna().all(axis=1)
            & x["_atr"].notna()
            & (x["_atr"] > 0)
            & x["Close"].notna()
            & (x["Close"] > 0)
        )
        frames[ticker] = x
        if idx % 20 == 0 or idx == n_tickers:
            log(f"  進捗(データ取得): {idx}/{n_tickers}銘柄処理済み(有効{len(frames)}件)")
    return frames


def scan_candidates(t, frames, pos_maps, model, cols):
    rows = []
    meta = []
    for ticker, x in frames.items():
        row_i = pos_maps[ticker].get(t)
        if row_i is None:
            continue
        if not bool(x["_valid"].iloc[row_i]):
            continue
        if row_i + HOLD_DAYS > len(x) - 1:
            continue
        rows.append(x[cols].iloc[row_i])
        meta.append((ticker, row_i))
    if not rows:
        return None, 0

    batch = pd.DataFrame(rows)
    probs = model.predict_proba(batch)
    classes = list(model.classes_)
    down_idx, up_idx = classes.index(0), classes.index(2)

    best = None
    for k, (ticker, row_i) in enumerate(meta):
        x = frames[ticker]
        row = x.iloc[row_i]
        up = float(probs[k][up_idx])
        down = float(probs[k][down_idx])
        long_s, _short_s = directional_score(row, up, down)
        price = float(row["Close"])
        a = float(row["_atr"])
        if not np.isfinite(a) or a <= 0 or price <= 0:
            continue
        if best is None or long_s > best["score"]:
            best = {
                "ticker": ticker,
                "row_i": row_i,
                "score": float(long_s),
                "up": up,
                "down": down,
                "price": price,
                "atr": a,
            }
    return best, len(meta)


def simulate(config, sig_dates, sig_bool, frames, pos_maps, model, cols):
    tp_mult, sl_mult = config["tp_mult"], config["sl_mult"]
    trades = []
    n_dates = len(sig_dates)
    i = 0
    scanned_events = 0
    while i < n_dates:
        if not sig_bool[i]:
            i += 1
            continue
        t = sig_dates[i]
        best, n_candidates = scan_candidates(t, frames, pos_maps, model, cols)
        scanned_events += 1
        if best is None:
            i += 1
            continue

        entry_price, a = best["price"], best["atr"]
        tp = entry_price + a * tp_mult
        sl = max(0.01, entry_price - a * sl_mult)
        x = frames[best["ticker"]]
        row_i = best["row_i"]

        exit_price = reason = exit_date = None
        for j in range(row_i + 1, row_i + HOLD_DAYS + 1):
            hi, lo = float(x["High"].iloc[j]), float(x["Low"].iloc[j])
            if lo <= sl and hi >= tp:
                exit_price, reason = sl, "SL"
            elif hi >= tp:
                exit_price, reason = tp, "TP"
            elif lo <= sl:
                exit_price, reason = sl, "SL"
            if reason:
                exit_date = x.index[j]
                break
        if reason is None:
            exit_j = row_i + HOLD_DAYS
            exit_date = x.index[exit_j]
            exit_price = float(x["Close"].iloc[exit_j])
            reason = "HOLD_LIMIT"

        gross_ret_pct = (exit_price - entry_price) / entry_price * 100
        fee_pct = (entry_price + exit_price) / entry_price * FEE_RATE * 100
        net_ret_pct = gross_ret_pct - fee_pct

        trades.append(
            {
                "entry_date": t,
                "exit_date": exit_date,
                "ticker": best["ticker"],
                "score": best["score"],
                "up_probability": best["up"] * 100,
                "down_probability": best["down"] * 100,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "tp": tp,
                "sl": sl,
                "atr": a,
                "return_pct": net_ret_pct,
                "reason": reason,
                "candidates_scanned": n_candidates,
            }
        )

        j2 = bisect.bisect_right(sig_dates, exit_date)
        i = max(j2, i + 1)

    log(f"  シグナル成立→候補スキャン回数: {scanned_events}回 / 実エントリー: {len(trades)}件")
    return trades


def aggregate_config(config, trades):
    base = {
        "config": config["name"],
        "trades": len(trades),
        "years": np.nan,
        "win_rate_pct": np.nan,
        "avg_return_pct": np.nan,
        "cumulative_compound_pct": np.nan,
        "annualized_pct": np.nan,
        "max_drawdown_pct": np.nan,
        "profit_factor": np.nan,
        "period_start": None,
        "period_end": None,
        "tp_count": 0,
        "sl_count": 0,
        "hold_limit_count": 0,
    }
    if not trades:
        return base

    tdf = pd.DataFrame(trades).sort_values("exit_date").reset_index(drop=True)
    n_trades = len(tdf)
    win_rate = float((tdf["reason"] == "TP").mean() * 100)
    avg_ret = float(tdf["return_pct"].mean())

    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in tdf["return_pct"]:
        equity *= 1 + r / 100
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100 if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    cum_pct = (equity - 1) * 100

    period_start = tdf["entry_date"].min()
    period_end = tdf["exit_date"].max()
    span_days = (pd.Timestamp(period_end) - pd.Timestamp(period_start)).days
    years = span_days / 365.25 if span_days > 0 else np.nan
    if years and years > 0 and equity > 0:
        annual_pct = (equity ** (1 / years) - 1) * 100
    else:
        annual_pct = np.nan

    gains = tdf.loc[tdf["return_pct"] > 0, "return_pct"].sum()
    losses = tdf.loc[tdf["return_pct"] < 0, "return_pct"].sum()
    profit_factor = (gains / abs(losses)) if losses < 0 else np.nan

    base.update(
        {
            "trades": n_trades,
            "years": years,
            "win_rate_pct": win_rate,
            "avg_return_pct": avg_ret,
            "cumulative_compound_pct": cum_pct,
            "annualized_pct": annual_pct,
            "max_drawdown_pct": max_dd,
            "profit_factor": profit_factor,
            "period_start": str(pd.Timestamp(period_start).date()),
            "period_end": str(pd.Timestamp(period_end).date()),
            "tp_count": int((tdf["reason"] == "TP").sum()),
            "sl_count": int((tdf["reason"] == "SL").sum()),
            "hold_limit_count": int((tdf["reason"] == "HOLD_LIMIT").sum()),
        }
    )
    return base


def to_markdown(rows, cols_order, headers=None):
    headers = headers or cols_order
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        vals = []
        for c in cols_order:
            v = row.get(c)
            if isinstance(v, float):
                vals.append("-" if pd.isna(v) else f"{v:.3f}")
            else:
                vals.append("-" if v is None else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main():
    t0 = time.time()
    log("=== nikkei_macd_dip_paper.py TP/SL比較バックテスト(研究専用) ===")
    log(f"データ取得期間: period={PERIOD}｜LOOKBACK_WINDOW={LOOKBACK_WINDOW}｜PERCENTILE={PERCENTILE}｜HOLD_DAYS={HOLD_DAYS}｜FEE_RATE={FEE_RATE}")
    for c in CONFIGS:
        log(f"  {c['name']}")

    nikkei = make_nikkei_long()
    if nikkei is None:
        log("❌ 日経225データ取得失敗(ネットワーク不通の可能性)。終了します。")
        return
    log(f"日経225データ: {len(nikkei)}件 ({nikkei.index.min().date()}〜{nikkei.index.max().date()})")

    prime_long_period_side_features()

    sig, lo_thr = nikkei_macd_signal_series(nikkei)
    sig_dates = list(sig.index)
    sig_bool = sig.to_numpy()
    n_judgeable = int(lo_thr.notna().sum())
    log(f"シグナル判定可能日数: {n_judgeable}日｜シグナル成立日数: {int(sig_bool.sum())}日")

    model = load_model()
    if model is None:
        log("❌ directional_model.pkl 読込失敗、または現行featuresと非互換。終了します。")
        return
    cols = list(getattr(model, "feature_names_in_", [])) or list(FEATURES)
    log(f"モデル使用特徴量数: {len(cols)}")

    log("=== 銘柄データ取得・特徴量計算 ===")
    frames = load_all_ticker_frames(nikkei, cols)
    log(f"有効銘柄: {len(frames)}/{len(TICKERS)}")
    if not frames:
        log("❌ 有効な銘柄データが0件でした。終了します。")
        return
    pos_maps = {ticker: {ts: i for i, ts in enumerate(x.index)} for ticker, x in frames.items()}
    log(f"経過時間(データ準備完了): {(time.time()-t0)/60:.1f}分")

    results = []
    all_trades = {}
    for config in CONFIGS:
        log(f"\n=== シミュレーション開始: {config['name']} ===")
        trades = simulate(config, sig_dates, sig_bool, frames, pos_maps, model, cols)
        all_trades[config["key"]] = trades
        agg = aggregate_config(config, trades)
        results.append(agg)
        log(
            f"  → 取引数 {agg['trades']}件｜勝率(TP判定比率) {agg['win_rate_pct']:.2f}%｜"
            f"平均リターン {agg['avg_return_pct']:.3f}%｜累積複利 {agg['cumulative_compound_pct']:.2f}%｜"
            f"年率換算 {agg['annualized_pct']:.2f}%｜最大DD {agg['max_drawdown_pct']:.2f}%｜"
            f"PF {agg['profit_factor']:.2f}"
            if agg["trades"] > 0
            else f"  → 取引数 0件"
        )
        log(f"  経過時間: {(time.time()-t0)/60:.1f}分")

    log("\n=== 比較結果サマリー ===")
    cols_order = [
        "config", "trades", "period_start", "period_end", "years",
        "win_rate_pct", "avg_return_pct", "cumulative_compound_pct", "annualized_pct",
        "max_drawdown_pct", "profit_factor", "tp_count", "sl_count", "hold_limit_count",
    ]
    headers = [
        "設定", "取引数", "開始", "終了", "年数",
        "勝率%(TP比率)", "平均リターン%", "累積複利%", "年率%",
        "最大DD%", "PF", "TP件数", "SL件数", "期限到達件数",
    ]
    table = to_markdown(results, cols_order, headers)
    log(table)

    for key, trades in all_trades.items():
        out_csv = f"nikkei_macd_dip_compare_trades_{key}.csv"
        pd.DataFrame(trades).to_csv(out_csv, index=False, encoding="utf-8-sig")
        log(f"取引明細を保存: {out_csv} ({len(trades)}件)")

    summary_csv = "nikkei_macd_dip_compare_summary.csv"
    pd.DataFrame(results).to_csv(summary_csv, index=False, encoding="utf-8-sig")
    log(f"サマリーを保存: {summary_csv}")

    log(f"\n総経過時間: {(time.time()-t0)/60:.1f}分")


if __name__ == "__main__":
    main()
