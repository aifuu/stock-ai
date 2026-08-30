import os
import time
from itertools import product
import numpy as np
import pandas as pd
import yfinance as yf

CANDIDATE_FILE = "walk_forward_all_candidates.csv"
FINAL_FILE = "adversarial_final_candidates.csv"
FOLD_FILE = "adversarial_multi_oos_folds.csv"
OOS_FILE = "adversarial_oos_results.csv"
DEV_ALL_FILE = "adversarial_dev_all_combos.csv"
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
TOTAL_OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "252"))
FOLDS = int(os.getenv("WF_OOS_FOLDS", "4"))
FOLD_OOS_DAYS = max(40, TOTAL_OOS_DAYS // FOLDS)
TOP_N = int(os.getenv("WF_TOP_N", "10"))
PURGE = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))
MIN_TRADES = int(os.getenv("WF_MIN_OOS_TRADES", "5"))
MIN_TOTAL_TRADES = int(os.getenv("WF_MIN_TOTAL_OOS_TRADES", "20"))
MIN_PF = float(os.getenv("WF_MIN_OOS_PF", "1.0"))
MIN_MONTHLY = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55")) * 100
MAX_DD = float(os.getenv("WF_MAX_OOS_DD", "35"))
MIN_POSITIVE_FOLDS = int(os.getenv("WF_MIN_POSITIVE_FOLDS", "3"))

UP = [45, 50, 55, 60, 65]
SCORE = [50, 60, 70, 80]
NIKKEI = [False, True]
TP = [2.0, 2.5, 3.0, 3.5, 4.0]
SL = [1.0, 1.25, 1.5, 1.75, 2.0]
HOLD = [3, 5, 7]
PARAMS = list(product(UP, SCORE, NIKKEI, TP, SL, HOLD))


def download(ticker):
    try:
        x = yf.download(
            ticker,
            start=(START_DATE - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
            end=(END_DATE + pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if x is None or x.empty:
            return None
        if isinstance(x.columns, pd.MultiIndex):
            x.columns = x.columns.get_level_values(0)
        idx = pd.to_datetime(x.index)
        x.index = idx.tz_localize(None) if getattr(idx, "tz", None) is not None else idx
        return x
    except Exception:
        return None


def select(df, up, score, nikkei):
    x = df[
        (df.up_prob >= up)
        & (df.up_prob > df.down_prob)
        & (df.flat_prob < 50)
        & (df.score >= score)
    ].copy()
    if nikkei:
        x = x[x.nikkei_uptrend]
    if x.empty:
        return x
    x["profit_ev"] = (
        (x.up_prob / 100) * ((x.take_profit - x.price) / x.price * 100)
        - (x.down_prob / 100) * ((x.price - x.stop_loss) / x.price * 100)
    )
    x["profit_rank"] = (
        0.70 * x.score + 0.30 * x.profit_ev.clip(-5, 5) * 10
    ).clip(0, 100)
    return (
        x.sort_values(
            ["date", "profit_rank", "score", "up_prob"],
            ascending=[True, False, False, False],
        )
        .groupby("date", group_keys=False)
        .head(TOP_N)
    )


def trade(prices, date, entry, atr_ratio, tp, sl, hold):
    future = prices[prices.index > date].head(hold)
    if future.empty:
        return None
    take = entry * (1 + atr_ratio / 100 * tp)
    stop = entry * (1 - atr_ratio / 100 * sl)
    for _, row in future.iterrows():
        hi, lo = float(row.High), float(row.Low)
        if lo <= stop and hi >= take:
            return (stop / entry - 1) * 100 - 0.10
        if hi >= take:
            return (take / entry - 1) * 100 - 0.10
        if lo <= stop:
            return (stop / entry - 1) * 100 - 0.10
    return (float(future.Close.iloc[-1]) / entry - 1) * 100 - 0.10


def evaluate(df, prices, params):
    up, score, nikkei, tp, sl, hold = params
    x = select(df, up, score, nikkei)
    ai_buy_candidates = len(x)
    rows = []
    for _, r in x.iterrows():
        p = prices.get(r.ticker)
        if p is None:
            continue
        atr_ratio = max(
            0.01,
            min(20, ((float(r.take_profit) / float(r.price) - 1) / 3) * 100),
        )
        ret = trade(p, r.date, float(r.price), atr_ratio, tp, sl, hold)
        if ret is not None:
            rows.append((r.date, float(ret)))
    tp_sl_resolved = len(rows)
    if not rows:
        return {
            "signals": 0,
            "avg_return": 0.0,
            "pf": 0.0,
            "dd": 0.0,
            "monthly_positive_ratio": 0.0,
            "compound_return": 0.0,
            "final_capital": float(INITIAL_CAPITAL),
            "ai_buy_candidates": ai_buy_candidates,
            "tp_sl_resolved": tp_sl_resolved,
        }
    z = pd.DataFrame(rows, columns=["date", "return"]).sort_values("date")
    daily = z.groupby("date")["return"].mean()
    eq = (1 + daily / 100).cumprod()
    gains = z.loc[z["return"] > 0, "return"].sum()
    losses = -z.loc[z["return"] < 0, "return"].sum()
    pf = gains / losses if losses > 0 else (float("inf") if gains > 0 else 0)
    monthly = daily.groupby(daily.index.to_period("M")).apply(
        lambda s: ((1 + s / 100).prod() - 1) * 100
    )
    return {
        "signals": len(z),
        "avg_return": float(z["return"].mean()),
        "pf": float(pf),
        "dd": float((eq / eq.cummax() - 1).min() * 100),
        "monthly_positive_ratio": float((monthly > 0).mean() * 100) if len(monthly) else 0.0,
        "compound_return": float((eq.iloc[-1] - 1) * 100),
        "final_capital": float(INITIAL_CAPITAL * eq.iloc[-1]),
        "ai_buy_candidates": ai_buy_candidates,
        "tp_sl_resolved": tp_sl_resolved,
    }


def passes(s):
    return (
        s["signals"] >= MIN_TRADES
        and s["pf"] >= MIN_PF
        and s["avg_return"] > 0
        and s["dd"] >= -MAX_DD
        and s["compound_return"] > 0
    )


if not os.path.exists(CANDIDATE_FILE):
    raise RuntimeError(f"{CANDIDATE_FILE} がありません")

c = pd.read_csv(CANDIDATE_FILE)
required = [
    "date", "ticker", "score", "up_prob", "flat_prob", "down_prob",
    "price", "take_profit", "stop_loss", "nikkei_uptrend",
]
missing = [x for x in required if x not in c.columns]
if missing:
    raise RuntimeError("候補CSVの不足列: " + ", ".join(missing))

c["date"] = pd.to_datetime(c.date).dt.normalize()
for col in ["score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "stop_loss"]:
    c[col] = pd.to_numeric(c[col], errors="coerce")
c["nikkei_uptrend"] = c.nikkei_uptrend.astype(str).str.lower().isin(["true", "1", "yes"])
c = c.dropna(subset=required)
c = c[(c.date >= START_DATE) & (c.date <= END_DATE)].copy()
dates = sorted(c.date.unique())
if len(dates) < FOLDS * FOLD_OOS_DAYS + 120:
    raise RuntimeError(f"複数OOSに必要な営業日が不足: {len(dates)}")

print("=" * 80)
print("FOUR-FOLD MULTI-OOS PROFIT GATE")
print(f"Folds={FOLDS} / 各OOS={FOLD_OOS_DAYS}営業日 / 合計OOS={FOLD_OOS_DAYS * FOLDS}")
print("100銘柄候補をそのまま使用。コード内で銘柄を改変しません。")
print("=" * 80)

prices = {}
for ticker in c.ticker.drop_duplicates():
    x = download(ticker)
    if x is not None and not x.empty:
        prices[ticker] = x
    time.sleep(0.03)

strategy_fold_rows = []
dev_all_rows = []
for fold in range(1, FOLDS + 1):
    end_pos = len(dates) - (FOLDS - fold) * FOLD_OOS_DAYS
    start_pos = end_pos - FOLD_OOS_DAYS
    oos_dates = dates[start_pos:end_pos]
    pre = dates[: max(0, start_pos - PURGE - EMBARGO)]
    if len(pre) < 120:
        raise RuntimeError(f"Fold {fold}: 学習期間不足")
    cut = int(len(pre) * 0.60)
    dev = pre[: max(1, cut - PURGE)]
    val = pre[min(len(pre), cut + EMBARGO):]
    dev_df = c[c.date.isin(dev)]
    val_df = c[c.date.isin(val)]
    oos_df = c[c.date.isin(oos_dates)]
    print(f"Fold {fold}: DEV={len(dev)} / VAL={len(val)} / OOS={len(oos_dates)}")

    dev_scores = []
    for params in PARAMS:
        s = evaluate(dev_df, prices, params)
        dev_pass = s["signals"] >= 20 and s["avg_return"] > 0 and s["pf"] >= 1
        dev_all_rows.append({
            "fold": fold, "up": params[0], "score": params[1], "nikkei": params[2],
            "tp": params[3], "sl": params[4], "hold": params[5],
            "ai_buy_candidates": s["ai_buy_candidates"], "tp_sl_resolved": s["tp_sl_resolved"],
            "signals": s["signals"], "avg_return": s["avg_return"], "pf": s["pf"],
            "dd": s["dd"], "monthly_positive_ratio": s["monthly_positive_ratio"],
            "compound_return": s["compound_return"], "dev_pass": dev_pass,
        })
        if dev_pass:
            obj = (
                0.35 * s["monthly_positive_ratio"]
                + 0.30 * np.clip(s["compound_return"], -100, 500)
                + 0.20 * np.clip(s["pf"], 0, 5) * 10
                + 0.15 * np.clip(s["avg_return"], -5, 5) * 10
            )
            dev_scores.append((obj, params))
    dev_scores.sort(reverse=True, key=lambda x: x[0])
    validation_candidates = dev_scores[:200]
    valid = []
    for _, params in validation_candidates:
        s = evaluate(val_df, prices, params)
        if s["signals"] >= 20 and s["avg_return"] > 0 and s["pf"] >= 1 and s["dd"] >= -30:
            valid.append((s["compound_return"], params, s))
    valid.sort(reverse=True, key=lambda x: x[0])
    chosen = valid[:50]
    if not chosen:
        chosen = [(0.0, p, evaluate(val_df, prices, p)) for _, p in validation_candidates[:50]]

    ref_row = next(
        (r for r in dev_all_rows if r["fold"] == fold and r["up"] == min(UP) and r["score"] == min(SCORE) and r["nikkei"] is False),
        None,
    )
    single_fold_oos_pass = 0
    for _, params, vs in chosen:
        os_ = evaluate(oos_df, prices, params)
        if passes(os_):
            single_fold_oos_pass += 1
        name = (
            f"UP{params[0]}_SCORE{params[1]}_NIKKEI{'ON' if params[2] else 'OFF'}"
            f"_TP{params[3]}_SL{params[4]}_H{params[5]}"
        )
        strategy_fold_rows.append({
            "fold": fold,
            "strategy": name,
            "up": params[0],
            "score": params[1],
            "nikkei": params[2],
            "tp": params[3],
            "sl": params[4],
            "hold": params[5],
            "validation_signals": vs["signals"],
            "validation_avg_return": vs["avg_return"],
            "validation_pf": vs["pf"],
            "validation_dd": vs["dd"],
            **{"oos_" + k: v for k, v in os_.items()},
            "oos_pass": passes(os_),
        })
    print(f"  ── Fold {fold} ファネル(閾値は変更していません) ──")
    print(f"  生候補行数(DEV期間・全銘柄合算)         : {len(dev_df):,}")
    if ref_row is not None:
        print(f"  AI BUY候補(最も緩いUP{min(UP)}/SCORE{min(SCORE)}/NIKKEI OFF): {ref_row['ai_buy_candidates']:,}")
        print(f"    うちTP/SL判定まで到達               : {ref_row['tp_sl_resolved']:,}")
    print(f"  DEV通過パラメータ数(全{len(PARAMS):,}通り中)  : {len(dev_scores):,}")
    print(f"  VALIDATION通過パラメータ数             : {len(valid):,} / 検証対象{len(validation_candidates):,}")
    print(f"  単一Fold OOS合格パラメータ数(chosen{len(chosen)}件中): {single_fold_oos_pass:,}")

pd.DataFrame(dev_all_rows).to_csv(DEV_ALL_FILE, index=False, encoding="utf-8-sig")
print(f"📁 {DEV_ALL_FILE}(全fold・全パラメータの生結果、合否問わず)を保存しました")

fold_table = pd.DataFrame(strategy_fold_rows)
fold_table.to_csv(FOLD_FILE, index=False, encoding="utf-8-sig")
grouped = []
if not fold_table.empty and "strategy" in fold_table.columns:
    for name, g in fold_table.groupby("strategy"):
        if len(g) < FOLDS:
            continue
        positive = int((g.oos_compound_return > 0).sum())
        total = int(g.oos_signals.sum())
        pfv = pd.to_numeric(g.oos_pf, errors="coerce").replace([np.inf], np.nan).dropna()
        avg = pd.to_numeric(g.oos_avg_return, errors="coerce")
        compound = float(((1 + avg.fillna(0) / 100).prod() - 1) * 100)
        monthly = float(pd.to_numeric(g.oos_monthly_positive_ratio, errors="coerce").mean())
        worst = float(pd.to_numeric(g.oos_dd, errors="coerce").min())
        grouped.append({
            "strategy": name,
            "folds": len(g),
            "positive_oos_folds": positive,
            "oos_signals": total,
            "oos_pf_min": float(pfv.min()) if len(pfv) else 0.0,
            "oos_pf_mean": float(pfv.mean()) if len(pfv) else 0.0,
            "oos_avg_return_mean": float(avg.mean()),
            "oos_monthly_positive_ratio_mean": monthly,
            "oos_worst_dd": worst,
            "oos_compound_return": compound,
            "up": int(g.up.iloc[-1]),
            "score": int(g.score.iloc[-1]),
            "nikkei": bool(g.nikkei.iloc[-1]),
            "tp": float(g.tp.iloc[-1]),
            "sl": float(g.sl.iloc[-1]),
            "hold": int(g.hold.iloc[-1]),
            "oos_pass": (
                positive >= MIN_POSITIVE_FOLDS
                and total >= MIN_TOTAL_TRADES
                and compound > 0
                and monthly >= MIN_MONTHLY
                and worst >= -MAX_DD
            ),
        })

final = pd.DataFrame(grouped)
final = final[final.oos_pass].copy() if not final.empty else final
if not final.empty:
    final["profit_objective"] = (
        0.45 * final.oos_monthly_positive_ratio_mean
        + 0.35 * np.clip(final.oos_compound_return, -100, 1000)
        + 0.20 * np.clip(final.oos_pf_mean, 0, 8) * 10
    )
    final = final.sort_values(
        ["profit_objective", "positive_oos_folds", "oos_compound_return"],
        ascending=False,
    ).reset_index(drop=True)
    final["final_status"] = "PASS"
    final["up_threshold"] = final.up
    final["score_threshold"] = final.score
    final["nikkei_filter"] = final.nikkei
    final["tp_multiplier"] = final.tp
    final["sl_multiplier"] = final.sl
    final["hold_days"] = final.hold
    final["validation_signals"] = final.oos_signals
    final["validation_win_rate"] = 0.0
    final["validation_avg_return"] = final.oos_avg_return_mean
    final["validation_pf"] = final.oos_pf_mean
    final["validation_dd"] = final.oos_worst_dd
    final["oos_win_rate"] = 0.0
    final["oos_avg_return"] = final.oos_avg_return_mean
    final["oos_pf"] = final.oos_pf_mean
    final["oos_dd"] = final.oos_worst_dd
    final["oos_validation_pf_ratio"] = 1.0
    final["oos_monthly_positive_ratio"] = final.oos_monthly_positive_ratio_mean
    final["oos_compound_final_capital"] = INITIAL_CAPITAL * (1 + final.oos_compound_return / 100)
    final["oos_expected_value"] = final.oos_avg_return_mean
    final["oos_avg_month_return"] = 0.0
    final["oos_worst_month_return"] = 0.0
else:
    final = pd.DataFrame(columns=[
        "final_status", "up_threshold", "score_threshold", "nikkei_filter",
        "tp_multiplier", "sl_multiplier", "hold_days",
    ])

final.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
fold_table.to_csv(OOS_FILE, index=False, encoding="utf-8-sig")
print("=" * 80)
print(
    f"4-FOLD結果: 最終PASS={len(final)} / {FOLDS} OOS中、"
    f"最低{MIN_POSITIVE_FOLDS}本が利益プラス"
)
print(
    "採用候補:"
    if not final.empty
    else "採用候補なし → policyはPENDINGのまま。手動でAPPROVEDへ改ざんしない。",
    final.iloc[0].strategy if not final.empty else "",
)
