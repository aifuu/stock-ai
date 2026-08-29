import os
import math
import time
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# ADVERSARIAL STRATEGY VALIDATOR / PROFIT OPTIMIZER
# =========================================================
# 目的:
#   勝率最大化ではなく、OOSで
#   1) 月間損益プラス率
#   2) OOS累積利益
#   3) 複利資産
#   4) Profit Factor
#   5) 最大DD
#   6) 1トレード期待利益
#   を優先して戦略を選ぶ。
#
# 入力: walk_forward_all_candidates.csv
# DEVで候補探索 → Validation → OOS → Final PASS
# =========================================================

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "90"))
TOP_N = int(os.getenv("WF_TOP_N", "10"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))

MIN_VALIDATION_TRADES = 30
MIN_TRADES_HARD = 20
MIN_PF_LOWER = 1.0
MIN_RETURN_LOWER = 0.0
MAX_VALIDATION_DD = 30.0
MIN_ANNUAL_SIGNALS = 20
MIN_OOS_TRADES = 20
MIN_OOS_PF = 1.0
MIN_OOS_AVG_RETURN = 0.0
MIN_OOS_TO_VALIDATION_PF = 0.60
MIN_MONTHLY_POSITIVE_RATIO = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55"))
MAX_OOS_DD = float(os.getenv("WF_MAX_OOS_DD", "35.0"))

BOOTSTRAP_ITERATIONS = int(os.getenv("WF_BOOTSTRAP_ITERATIONS", "3000"))
RANDOM_SEED = 42

UP_THRESHOLDS = [45, 50, 55, 60, 65]
SCORE_THRESHOLDS = [50, 60, 70, 80]
NIKKEI_FILTERS = [False, True]
TP_MULTIPLIERS = [2.0, 2.5, 3.0, 3.5, 4.0]
SL_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]
HOLD_DAYS_LIST = [3, 5, 7]

WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send_discord(msg):
    if not WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOKなし")
        return
    try:
        import requests
        requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=30)
    except Exception as e:
        print("Discord送信エラー:", e)


def safe_download(ticker, start, end):
    for _ in range(3):
        try:
            x = yf.download(ticker, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
            if x is not None and not x.empty:
                if isinstance(x.columns, pd.MultiIndex):
                    x.columns = x.columns.get_level_values(0)
                x.index = pd.to_datetime(x.index)
                if getattr(x.index, "tz", None) is not None:
                    x.index = x.index.tz_localize(None)
                return x
        except Exception as e:
            print(ticker, "download error:", e)
        time.sleep(1)
    return None


if not os.path.exists(CANDIDATE_FILE):
    raise RuntimeError(f"{CANDIDATE_FILE} がありません。先に walk_forward.py を実行してください。")

candidates = pd.read_csv(CANDIDATE_FILE)
if candidates.empty:
    raise RuntimeError("候補CSVが空です。")

required = [
    "date", "ticker", "score", "up_prob", "flat_prob", "down_prob",
    "price", "take_profit", "stop_loss", "nikkei_uptrend"
]
missing = [c for c in required if c not in candidates.columns]
if missing:
    raise RuntimeError("候補CSVの不足列: " + ", ".join(missing))

candidates["date"] = pd.to_datetime(candidates["date"]).dt.normalize()
for c in ["score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "stop_loss"]:
    candidates[c] = pd.to_numeric(candidates[c], errors="coerce")
candidates["nikkei_uptrend"] = candidates["nikkei_uptrend"].astype(str).str.lower().isin(["true", "1", "yes"])
candidates = candidates.dropna(subset=required).copy()
candidates = candidates[(candidates["date"] >= START_DATE) & (candidates["date"] <= END_DATE)].copy()

# walk_forward.pyの既存TP式からATR比率を逆算
candidates["atr_ratio"] = (((candidates["take_profit"] / candidates["price"]) - 1) / 3.0 * 100).clip(0.01, 20.0)

all_dates = sorted(candidates["date"].drop_duplicates().tolist())
if len(all_dates) <= OOS_DAYS:
    raise RuntimeError("OOS_DAYSが予測日数以上です。")

oos_dates = all_dates[-OOS_DAYS:]
pre_oos = all_dates[:-OOS_DAYS]
split = int(len(pre_oos) * 0.60)
dev_dates_raw = pre_oos[:split]
validation_dates_raw = pre_oos[split:]

# ★修正(2026-08): PURGE_DAYS/EMBARGO_DAYSを各フェーズ境界に
# 明示適用する。境界付近のシグナルが次フェーズの価格を参照して
# 評価確定する境界リークを除去する。
def _purge_embargo(dates_before, dates_after, purge_days, embargo_days):
    purged_before = (
        dates_before[:-purge_days]
        if purge_days > 0 and len(dates_before) > purge_days
        else dates_before
    )
    embargoed_after = (
        dates_after[embargo_days:]
        if embargo_days > 0 and len(dates_after) > embargo_days
        else dates_after
    )
    return purged_before, embargoed_after

dev_dates, validation_dates_raw = _purge_embargo(
    dev_dates_raw, validation_dates_raw, PURGE_DAYS, EMBARGO_DAYS
)
validation_dates, oos_dates = _purge_embargo(
    validation_dates_raw, oos_dates, PURGE_DAYS, EMBARGO_DAYS
)

if not dev_dates or not validation_dates or not oos_dates:
    raise RuntimeError("Purge/Embargo後にDEV/Validation/OOS期間が空です")

phase_map = {d: "DEV" for d in dev_dates}
phase_map.update({d: "VALIDATION" for d in validation_dates})
phase_map.update({d: "OOS" for d in oos_dates})
candidates["phase"] = candidates["date"].map(phase_map)
N_EFFECTIVE_STRATEGIES = max(1, int(np.ceil(np.sqrt(len(UP_THRESHOLDS) * len(SCORE_THRESHOLDS) * len(NIKKEI_FILTERS) * len(TP_MULTIPLIERS) * len(SL_MULTIPLIERS) * len(HOLD_DAYS_LIST)))))
MULTIPLE_TEST_ALPHA = 0.05 / N_EFFECTIVE_STRATEGIES

base_business = sorted(candidates["date"].drop_duplicates().tolist())
idx_map = {d: i for i, d in enumerate(base_business)}


def shift_date(d, n):
    p = idx_map.get(d)
    if p is None:
        return d
    return base_business[min(max(p + n, 0), len(base_business) - 1)]


candidates["target_end_date"] = candidates["date"].map(lambda d: shift_date(d, 3))
candidates["trade_end_date"] = candidates["date"].map(lambda d: shift_date(d, max(HOLD_DAYS_LIST)))

print("=" * 80)
print("🛡️ AI PROFIT OPTIMIZER")
print("探索期間:", START_DATE.date(), "～", END_DATE.date())
print("DEV:", len(dev_dates), "VALIDATION:", len(validation_dates), "OOS:", len(oos_dates))
print("TOP_N:", TOP_N)
print("目標: 月間損益プラス率 + OOS複利資産 + 期待利益")
print("=" * 80)

price_data = {}
for ticker in candidates["ticker"].drop_duplicates().tolist():
    print("📥", ticker)
    x = safe_download(
        ticker,
        (START_DATE - pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
        (END_DATE + pd.Timedelta(days=20)).strftime("%Y-%m-%d"),
    )
    if x is not None and not x.empty:
        price_data[ticker] = x


def evaluate_trade(ticker, date, entry, atr_ratio, tp, sl, hold_days, slippage=0.001):
    if ticker not in price_data:
        return None
    data = price_data[ticker]
    future = data[data.index > date].head(hold_days)
    if future.empty:
        return None
    take = entry * (1 + atr_ratio / 100 * tp)
    stop = entry * (1 - atr_ratio / 100 * sl)
    for day_no, (_, row) in enumerate(future.iterrows(), 1):
        high = float(row["High"])
        low = float(row["Low"])
        # 同一日でTP/SL両方に触れた場合は保守的にLOSS
        if low <= stop and high >= take:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
        if high >= take:
            return "WIN", (take / entry - 1) * 100 - slippage * 100, day_no
        if low <= stop:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
    close = float(future.iloc[-1]["Close"])
    ret = (close / entry - 1) * 100 - slippage * 100
    return ("TIMEOUT_LOSS" if ret < 0 else "HOLD", ret, len(future))


def select_for_phase(phase_df, up, score, nikkei):
    x = phase_df.copy()
    x = x[x["up_prob"] >= up]
    x = x[x["up_prob"] > x["down_prob"]]
    x = x[x["flat_prob"] < 50]
    x = x[x["score"] >= score]
    if nikkei:
        x = x[x["nikkei_uptrend"]]
    if x.empty:
        return x
    # 日ごとにTOP_N。スコアだけでなくup_probも同点時の優先順位に使う。
    return (
        x.sort_values(["date", "score", "up_prob"], ascending=[True, False, False])
        .groupby("date", group_keys=False)
        .head(TOP_N)
        .copy()
    )


def run_strategy(phase_df, up, score, nikkei, tp, sl, hold):
    selected = select_for_phase(phase_df, up, score, nikkei)
    rows = []
    for _, r in selected.iterrows():
        result = evaluate_trade(
            r["ticker"], r["date"], float(r["price"]), float(r["atr_ratio"]), tp, sl, hold
        )
        if result is None:
            continue
        name, ret, days = result
        rows.append({
            "date": r["date"],
            "ticker": r["ticker"],
            "score": r["score"],
            "up_prob": r["up_prob"],
            "result": name,
            "return": ret,
            "hold_days": days,
            "phase": r["phase"],
                "risk_unit": max(1e-8, float(atr_ratio) / 100.0 * float(sl)),
        })
    return pd.DataFrame(rows)


def stats(x):
    empty = {
        "signals": 0, "wins": 0, "losses": 0, "holds": 0,
        "win_rate": 0.0, "avg_return": 0.0, "pf": 0.0, "dd": 0.0,
        "annual_signals": 0.0, "positive_months": 0, "months": 0,
        "monthly_positive_ratio": 0.0, "avg_month_return": 0.0,
        "worst_month_return": 0.0, "oos_cumulative_return": 0.0,
        "compound_return": 0.0, "compound_final_capital": INITIAL_CAPITAL,
        "expected_value": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
    }
    if x.empty:
        return empty

    x = x.copy().sort_values("date")
    wins = int((x.result == "WIN").sum())
    losses = int(x.result.isin(["LOSS", "TIMEOUT_LOSS"]).sum())
    holds = int((x.result == "HOLD").sum())
    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.0

    r = pd.to_numeric(x["return"], errors="coerce").dropna()
    avg = float(r.mean()) if len(r) else 0.0
    gains = float(r[r > 0].sum()) if len(r) else 0.0
    loss = float(-r[r < 0].sum()) if len(r) else 0.0
    pf = gains / loss if loss > 0 else (np.inf if gains > 0 else 0.0)

    # TOP_Nを同日に複数持つため、日次ポートフォリオは同日リターンの平均。
    daily = x.groupby("date")["return"].mean().sort_index()
    equity = (1.0 + daily / 100.0).cumprod()
    compound_return = float((equity.iloc[-1] - 1.0) * 100) if len(equity) else 0.0
    final_capital = INITIAL_CAPITAL * (1.0 + compound_return / 100.0)
    dd = float((equity / equity.cummax() - 1.0).min() * 100) if len(equity) else 0.0

    monthly = daily.groupby(daily.index.to_period("M")).apply(
        lambda s: float(((1.0 + s / 100.0).prod() - 1.0) * 100)
    )
    months = int(len(monthly))
    positive_months = int((monthly > 0).sum()) if months else 0
    monthly_positive_ratio = positive_months / months * 100 if months else 0.0
    avg_month_return = float(monthly.mean()) if months else 0.0
    worst_month_return = float(monthly.min()) if months else 0.0

    avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(r[r < 0].mean()) if (r < 0).any() else 0.0
    expected_value = avg

    years = max((x.date.max() - x.date.min()).days / 365.25, 0.5)
    annual = len(x) / years

    return {
        "signals": len(x), "wins": wins, "losses": losses, "holds": holds,
        "win_rate": float(win_rate), "avg_return": avg, "pf": float(pf),
        "dd": dd, "annual_signals": float(annual), "positive_months": positive_months,
        "months": months, "monthly_positive_ratio": float(monthly_positive_ratio),
        "avg_month_return": avg_month_return, "worst_month_return": worst_month_return,
        "oos_cumulative_return": compound_return, "compound_return": compound_return,
        "compound_final_capital": float(final_capital), "expected_value": expected_value,
        "avg_win": avg_win, "avg_loss": avg_loss,
    }


def block_bootstrap_lower(values, block_len=10, n_iter=BOOTSTRAP_ITERATIONS, alpha=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 10:
        return np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(values)
    starts = np.arange(max(1, n - block_len + 1))
    out = []
    for _ in range(n_iter):
        sample = []
        while len(sample) < n:
            start = int(rng.choice(starts))
            sample.extend(values[start:start + block_len])
        out.append(float(np.mean(sample[:n])))
    return float(np.quantile(out, max(1e-6, min(0.5, alpha))))


def monte_carlo_risk_gate(trades, initial_capital=INITIAL_CAPITAL, target_capital=100000000.0, iterations=None):
    iterations = int(iterations or int(os.getenv("WF_MONTE_CARLO_ITERATIONS", "5000")))
    if trades is None or trades.empty or "return" not in trades or "risk_unit" not in trades:
        return None
    ret = pd.to_numeric(trades["return"], errors="coerce").to_numpy(dtype=float)
    unit = pd.to_numeric(trades["risk_unit"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(ret) & np.isfinite(unit) & (unit > 0)
    r_mult = (ret[mask] / 100.0) / unit[mask]
    r_mult = r_mult[np.isfinite(r_mult)]
    if len(r_mult) < 20:
        return None
    years = max(0.5, (trades["date"].max() - trades["date"].min()).days / 365.25)
    annual_signals = max(1.0, len(r_mult) / years)
    rng = np.random.default_rng(RANDOM_SEED)
    best = None
    diagnostic = None
    for sizing in (0.01, 0.0075, 0.005, 0.0025):
        path = np.full(iterations, float(initial_capital))
        peak = path.copy()
        max_dd = np.zeros(iterations)
        checkpoints = {}
        for year in range(1, 21):
            n = max(1, int(round(annual_signals)))
            sample = rng.choice(r_mult, size=(iterations, n), replace=True)
            growth = np.prod(np.clip(1.0 + sizing * sample, 0.01, 5.0), axis=1)
            path *= growth
            peak = np.maximum(peak, path)
            max_dd = np.maximum(max_dd, 1.0 - path / np.maximum(peak, 1e-9))
            if year in (10, 15, 20):
                checkpoints[year] = float(np.mean(path >= target_capital) * 100.0)
        bankruptcy = float(np.mean(path <= initial_capital * 0.50) * 100.0)
        p90_dd = float(np.quantile(max_dd, 0.90) * 100.0)
        diagnostic = {"sizing": sizing, "prob_10y": checkpoints[10], "prob_15y": checkpoints[15], "prob_20y": checkpoints[20], "bankruptcy_prob": bankruptcy, "p90_max_dd": p90_dd}
        if bankruptcy < 5.0 and p90_dd <= 30.0:
            best = diagnostic
            break
    return best or diagnostic


# ---------------------------------------------------------
# DEV探索
# ---------------------------------------------------------
dev_df = candidates[candidates.phase == "DEV"].copy()
validation_df = candidates[candidates.phase == "VALIDATION"].copy()
oos_df = candidates[candidates.phase == "OOS"].copy()

param_space = list(product(UP_THRESHOLDS, SCORE_THRESHOLDS, NIKKEI_FILTERS, TP_MULTIPLIERS, SL_MULTIPLIERS, HOLD_DAYS_LIST))
all_dev_rows = []

for i, (up, score, nikkei, tp, sl, hold) in enumerate(param_space, 1):
    if i % 100 == 0:
        print(f"DEV探索 {i}/{len(param_space)}")
    name = f"UP{up}_SCORE{score}_NIKKEI{'ON' if nikkei else 'OFF'}_TP{tp}_SL{sl}_H{hold}"
    rd = run_strategy(dev_df, up, score, nikkei, tp, sl, hold)
    st = stats(rd)
    all_dev_rows.append({
        "strategy": name, "up": up, "score": score, "nikkei": nikkei,
        "tp": tp, "sl": sl, "hold": hold,
        **{f"dev_{k}": v for k, v in st.items()},
    })

dev_summary = pd.DataFrame(all_dev_rows)
dev_summary.to_csv("adversarial_dev_all_results.csv", index=False, encoding="utf-8-sig")

# DEVでは過学習防止のため、件数・年間頻度・期待利益を最低条件にする。
dev_candidates = dev_summary[
    (dev_summary.dev_signals >= MIN_TRADES_HARD)
    & (dev_summary.dev_annual_signals >= MIN_ANNUAL_SIGNALS)
    & (dev_summary.dev_avg_return > MIN_RETURN_LOWER)
    & (dev_summary.dev_pf >= MIN_PF_LOWER)
].copy()

# 勝率ではなく、月間安定性→複利→PF→期待利益を優先。
dev_candidates["dev_objective"] = (
    dev_candidates["dev_monthly_positive_ratio"] * 0.35
    + np.clip(dev_candidates["dev_compound_return"], -100, 500) * 0.25
    + np.clip(dev_candidates["dev_pf"], 0, 5) * 10.0 * 0.20
    + np.clip(dev_candidates["dev_avg_return"], -5, 5) * 10.0 * 0.20
)
dev_candidates = dev_candidates.sort_values("dev_objective", ascending=False).head(50).copy()
dev_candidates.to_csv("adversarial_dev_selected_candidates.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Validation
# ---------------------------------------------------------
validation_results = []
for _, row in dev_candidates.iterrows():
    rd = run_strategy(validation_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    st = stats(rd)
    lower_avg = block_bootstrap_lower(rd["return"].values, alpha=MULTIPLE_TEST_ALPHA) if not rd.empty else np.nan
    validation_results.append({
        **row.to_dict(),
        **{f"validation_{k}": v for k, v in st.items()},
        "validation_avg_lower": lower_avg,
    })

validation_summary = pd.DataFrame(validation_results)
if not validation_summary.empty:
    validation_summary["validation_pass"] = (
        (validation_summary.validation_signals >= MIN_VALIDATION_TRADES)
        & (validation_summary.validation_pf >= MIN_PF_LOWER)
        & (validation_summary.validation_avg_return > MIN_RETURN_LOWER)
        & (validation_summary.validation_dd >= -MAX_VALIDATION_DD)
        & (validation_summary.validation_annual_signals >= MIN_ANNUAL_SIGNALS)
        & (validation_summary.validation_monthly_positive_ratio >= MIN_MONTHLY_POSITIVE_RATIO * 100)
        & (validation_summary.validation_avg_lower > 0)
    )
else:
    validation_summary["validation_pass"] = False
validation_summary.to_csv("adversarial_validation_results.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# OOS
# ---------------------------------------------------------
oos_results = []
passed_validation = validation_summary[validation_summary.validation_pass].copy() if not validation_summary.empty else pd.DataFrame()
for _, row in passed_validation.iterrows():
    rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    st = stats(rd)
    oos_results.append({
        **row.to_dict(),
        **{f"oos_{k}": v for k, v in st.items()},
    })

oos_summary = pd.DataFrame(oos_results)
if not oos_summary.empty:
    oos_summary["oos_pf_ratio"] = oos_summary["oos_pf"] / oos_summary["validation_pf"].replace(0, np.nan)
    oos_summary["oos_pass"] = (
        (oos_summary.oos_signals >= MIN_OOS_TRADES)
        & (oos_summary.oos_pf >= MIN_OOS_PF)
        & (oos_summary.oos_avg_return > MIN_OOS_AVG_RETURN)
        & (oos_summary.oos_pf_ratio >= MIN_OOS_TO_VALIDATION_PF)
        & (oos_summary.oos_monthly_positive_ratio >= MIN_MONTHLY_POSITIVE_RATIO * 100)
        & (oos_summary.oos_dd >= -MAX_OOS_DD)
        & (oos_summary.oos_compound_return > 0)
    )
else:
    oos_summary["oos_pf_ratio"] = pd.Series(dtype=float)
    oos_summary["oos_pass"] = False

oos_summary.to_csv("adversarial_oos_results.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Final selection: OOS月間安定性 + 複利資産を最優先
# ---------------------------------------------------------
final_pass = oos_summary[oos_summary.oos_pass].copy() if not oos_summary.empty else pd.DataFrame()
if not final_pass.empty:
    final_pass["profit_objective"] = (
        final_pass["oos_monthly_positive_ratio"] * 0.40
        + np.clip(final_pass["oos_compound_return"], -100, 1000) * 0.30
        + np.clip(final_pass["oos_pf"], 0, 8) * 5.0 * 0.15
        + np.clip(final_pass["oos_avg_return"], -5, 5) * 10.0 * 0.10
        + np.clip(final_pass["oos_expected_value"], -5, 5) * 10.0 * 0.05
    )
    final_pass = final_pass.sort_values(
        ["profit_objective", "oos_monthly_positive_ratio", "oos_compound_return", "oos_pf", "oos_avg_return"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

# Mandatory Monte Carlo risk gate.
if not final_pass.empty:
    mc_records = []
    mc_limit = int(os.getenv("WF_MC_CANDIDATES", "20"))
    for _, row in final_pass.head(mc_limit).iterrows():
        rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
        mc = monte_carlo_risk_gate(rd)
        if mc is not None:
            mc_records.append({"strategy": row.strategy, **mc})
    mc_df = pd.DataFrame(mc_records)
    final_pass = final_pass.merge(mc_df, on="strategy", how="inner") if not mc_df.empty else pd.DataFrame()

# build_strategy_policy.pyが読む正式な最終候補CSVを必ず生成
if not final_pass.empty:
    final_pass["final_status"] = "PASS"
    final_pass["up_threshold"] = final_pass["up"]
    final_pass["score_threshold"] = final_pass["score"]
    final_pass["nikkei_filter"] = final_pass["nikkei"]
    final_pass["tp_multiplier"] = final_pass["tp"]
    final_pass["sl_multiplier"] = final_pass["sl"]
    final_pass["hold_days"] = final_pass["hold"]
    final_pass["oos_validation_pf_ratio"] = final_pass["oos_pf_ratio"]
else:
    final_pass = pd.DataFrame(columns=[
        "final_status", "up_threshold", "score_threshold", "nikkei_filter",
        "tp_multiplier", "sl_multiplier", "hold_days",
    ])

final_pass.to_csv("adversarial_final_candidates.csv", index=False, encoding="utf-8-sig")

print("\n" + "=" * 80)
print("🛡️ AI PROFIT OPTIMIZER RESULT")
print("期間:", START_DATE.date(), "～", END_DATE.date())
print("探索数:", len(param_space))
print("N_eff:", N_EFFECTIVE_STRATEGIES)
print("Purge/Embargo:", PURGE_DAYS, EMBARGO_DAYS)
print("TOP_N:", TOP_N)
print("DEV候補:", len(dev_candidates))
print("Validation PASS:", int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0)
print("OOS PASS:", int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0)
print("Final PASS:", len(final_pass))

msg = (
    "🛡️ AI PROFIT OPTIMIZER\n"
    f"期間: {START_DATE.date()} ～ {END_DATE.date()}\n"
    f"TOP_N: {TOP_N}\n"
    f"DEV候補: {len(dev_candidates)}\n"
    f"Validation PASS: {int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0}\n"
    f"OOS PASS: {int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0}\n"
    f"Final PASS: {len(final_pass)}\n"
    "目標: 月間損益プラス率・OOS複利資産・期待利益を優先"
)

if not final_pass.empty:
    msg += "\n\n🏆 BEST STRATEGIES\n"
    for _, r in final_pass.head(10).iterrows():
        msg += (
            f"{r['strategy']}\n"
            f"  月間プラス率={r['oos_monthly_positive_ratio']:.1f}% "
            f"OOS複利={r['oos_compound_return']:+.2f}% "
            f"最終資産=¥{r['oos_compound_final_capital']:,.0f}\n"
            f"  PF={r['oos_pf']:.2f} 期待利益={r['oos_expected_value']:+.3f}% "
            f"最大DD={r['oos_dd']:.2f}% 勝率={r['oos_win_rate']:.1f}%\n"
        )
else:
    msg += "\n\n該当するFinal PASS戦略なし。既存policyは自動変更しません。"

send_discord(msg)
