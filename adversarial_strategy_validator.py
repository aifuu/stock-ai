import os
import time
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "90"))
TOP_N = int(os.getenv("WF_TOP_N", "10"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))

MIN_VALIDATION_TRADES = int(os.getenv("WF_MIN_VALIDATION_TRADES", "30"))
MIN_TRADES_HARD = int(os.getenv("WF_MIN_TRADES_HARD", "20"))
MIN_PF_LOWER = 1.0
MIN_RETURN_LOWER = 0.0
MAX_VALIDATION_DD = 30.0
MIN_ANNUAL_SIGNALS = 20
MIN_OOS_TRADES = 20
MIN_OOS_PF = 1.0
MIN_OOS_AVG_RETURN = 0.0
MIN_OOS_TO_VALIDATION_PF = 0.60
MAX_VALIDATION_PF_FOR_RATIO = float(os.getenv("WF_MAX_VALIDATION_PF_FOR_RATIO", "10.0"))
MIN_MONTHLY_POSITIVE_RATIO = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55"))
MAX_OOS_DD = float(os.getenv("WF_MAX_OOS_DD", "35.0"))
BOOTSTRAP_ITERATIONS = int(os.getenv("WF_BOOTSTRAP_ITERATIONS", "3000"))
RANDOM_SEED = 42

UP_THRESHOLDS = [45, 50, 55, 60, 65]
SCORE_THRESHOLDS = [50, 60, 70, 80]
NIKKEI_FILTERS = [False, True]
TP_MULTIPLIERS = [2.0, 2.5, 3.0, 3.5, 4.0]
SL_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]
HOLD_DAYS_LIST = [1, 3, 5]
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK")


def send_discord(msg):
    if not WEBHOOK_URL:
        print("⚠ DISCORD_WEBHOOKなし")
        return
    try:
        import requests
        r = requests.post(WEBHOOK_URL, json={"content": msg[:1900]}, timeout=30)
        r.raise_for_status()
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
required = ["date", "ticker", "score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "stop_loss", "nikkei_uptrend"]
missing = [c for c in required if c not in candidates.columns]
if missing:
    raise RuntimeError("候補CSVの不足列: " + ", ".join(missing))

candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce").dt.normalize()
for c in ["score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "stop_loss"]:
    candidates[c] = pd.to_numeric(candidates[c], errors="coerce")
candidates["nikkei_uptrend"] = candidates["nikkei_uptrend"].astype(str).str.lower().isin(["true", "1", "yes"])
candidates = candidates.dropna(subset=required)
candidates = candidates[(candidates.date >= START_DATE) & (candidates.date <= END_DATE)].copy()
if candidates.empty:
    raise RuntimeError("指定期間に有効な候補データがありません。")

candidates["atr_ratio"] = (((candidates.take_profit / candidates.price) - 1) / 3.0 * 100).clip(0.01, 20.0)
all_dates = sorted(candidates.date.drop_duplicates().tolist())
if len(all_dates) <= OOS_DAYS:
    raise RuntimeError("OOS_DAYSが予測日数以上です。")

oos_dates = all_dates[-OOS_DAYS:]
pre_oos = all_dates[:-OOS_DAYS]
split = int(len(pre_oos) * 0.60)
dev_dates_raw, validation_dates_raw = pre_oos[:split], pre_oos[split:]


def purge_embargo(dates_before, dates_after, purge_days, embargo_days):
    before = dates_before[:-purge_days] if purge_days > 0 and len(dates_before) > purge_days else dates_before
    after = dates_after[embargo_days:] if embargo_days > 0 and len(dates_after) > embargo_days else dates_after
    return before, after


dev_dates, validation_dates_raw = purge_embargo(dev_dates_raw, validation_dates_raw, PURGE_DAYS, EMBARGO_DAYS)
validation_dates, oos_dates = purge_embargo(validation_dates_raw, oos_dates, PURGE_DAYS, EMBARGO_DAYS)
if not dev_dates or not validation_dates or not oos_dates:
    raise RuntimeError("Purge/Embargo後にDEV/Validation/OOS期間が空です")

phase_map = {d: "DEV" for d in dev_dates}
phase_map.update({d: "VALIDATION" for d in validation_dates})
phase_map.update({d: "OOS" for d in oos_dates})
candidates["phase"] = candidates.date.map(phase_map)

n_strategies = len(UP_THRESHOLDS) * len(SCORE_THRESHOLDS) * len(NIKKEI_FILTERS) * len(TP_MULTIPLIERS) * len(SL_MULTIPLIERS) * len(HOLD_DAYS_LIST)
N_EFFECTIVE_STRATEGIES = max(1, int(np.ceil(np.sqrt(n_strategies))))
MULTIPLE_TEST_ALPHA = 0.05 / N_EFFECTIVE_STRATEGIES

price_data = {}
for ticker in candidates.ticker.drop_duplicates().tolist():
    print("📥", ticker)
    x = safe_download(ticker, (START_DATE - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), (END_DATE + pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
    if x is not None and not x.empty:
        price_data[ticker] = x


def evaluate_trade(ticker, date, entry, atr_ratio, tp, sl, hold_days, slippage=0.001):
    if ticker not in price_data:
        return None
    future = price_data[ticker][price_data[ticker].index > date].head(hold_days)
    if future.empty:
        return None
    take = entry * (1 + atr_ratio / 100 * tp)
    stop = entry * (1 - atr_ratio / 100 * sl)
    for day_no, (_, row) in enumerate(future.iterrows(), 1):
        high, low = float(row["High"]), float(row["Low"])
        if low <= stop and high >= take:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
        if high >= take:
            return "WIN", (take / entry - 1) * 100 - slippage * 100, day_no
        if low <= stop:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
    close = float(future.iloc[-1]["Close"])
    ret = (close / entry - 1) * 100 - slippage * 100
    return ("TIMEOUT_LOSS" if ret < 0 else "HOLD"), ret, len(future)


def select_for_phase(phase_df, up, score, nikkei):
    x = phase_df[(phase_df.up_prob >= up) & (phase_df.up_prob > phase_df.down_prob) & (phase_df.flat_prob < 50) & (phase_df.score >= score)].copy()
    if nikkei:
        x = x[x.nikkei_uptrend]
    if x.empty:
        return x
    return x.sort_values(["date", "score", "up_prob"], ascending=[True, False, False]).groupby("date", group_keys=False).head(TOP_N).copy()


def run_strategy(phase_df, up, score, nikkei, tp, sl, hold):
    rows = []
    for _, r in select_for_phase(phase_df, up, score, nikkei).iterrows():
        result = evaluate_trade(r.ticker, r.date, float(r.price), float(r.atr_ratio), tp, sl, hold)
        if result is None:
            continue
        name, ret, days = result
        rows.append({"date": r.date, "ticker": r.ticker, "score": r.score, "up_prob": r.up_prob, "result": name, "return": ret, "hold_days": days, "phase": r.phase, "risk_unit": max(1e-8, float(r.atr_ratio) / 100.0 * float(sl))})
    return pd.DataFrame(rows)


# 月間+5%目標(元本100万円なら+5万円/月)の達成率を判定する閾値。勝率ではなく月次収益率で戦略を評価する。
MONTHLY_TARGET_PCT = float(os.getenv("WF_MONTHLY_TARGET_PCT", "5.0"))


def stats(x):
    # 注意: 「勝率(win_rate)」「勝ちトレード数(wins)」は選定・ランキング・合否判定に一切使わないため、
    # ここでは計算・保持しない(方針: 利益・収益率基準への統一)。
    empty = {"signals": 0, "losses": 0, "holds": 0, "avg_return": 0.0, "pf": 0.0, "dd": 0.0, "annual_signals": 0.0, "positive_months": 0, "months": 0, "monthly_positive_ratio": 0.0, "monthly_plus5_ratio": 0.0, "avg_month_return": 0.0, "avg_month_profit_jpy": 0.0, "worst_month_return": 0.0, "oos_cumulative_return": 0.0, "compound_return": 0.0, "compound_final_capital": INITIAL_CAPITAL, "expected_value": 0.0, "avg_win": 0.0, "avg_loss": 0.0}
    if x.empty:
        return empty
    x = x.copy().sort_values("date")
    losses = int(x.result.isin(["LOSS", "TIMEOUT_LOSS"]).sum())
    holds = int((x.result == "HOLD").sum())
    r = pd.to_numeric(x["return"], errors="coerce").dropna()
    gains, loss = float(r[r > 0].sum()), float(-r[r < 0].sum())
    pf = gains / loss if loss > 0 else (np.inf if gains > 0 else 0.0)
    daily = x.groupby("date")["return"].mean().sort_index()
    equity = (1 + daily / 100).cumprod()
    compound = float((equity.iloc[-1] - 1) * 100)
    dd = float((equity / equity.cummax() - 1).min() * 100)
    monthly = daily.groupby(daily.index.to_period("M")).apply(lambda s: float(((1 + s / 100).prod() - 1) * 100))
    months = len(monthly)
    years = max((x.date.max() - x.date.min()).days / 365.25, 0.5)
    avg_month_return = float(monthly.mean()) if months else 0.0
    return {"signals": len(x), "losses": losses, "holds": holds, "avg_return": float(r.mean()), "pf": float(pf), "dd": dd, "annual_signals": len(x) / years, "positive_months": int((monthly > 0).sum()), "months": months, "monthly_positive_ratio": float((monthly > 0).mean() * 100) if months else 0.0, "monthly_plus5_ratio": float((monthly >= MONTHLY_TARGET_PCT).mean() * 100) if months else 0.0, "avg_month_return": avg_month_return, "avg_month_profit_jpy": INITIAL_CAPITAL * avg_month_return / 100.0, "worst_month_return": float(monthly.min()) if months else 0.0, "oos_cumulative_return": compound, "compound_return": compound, "compound_final_capital": INITIAL_CAPITAL * (1 + compound / 100), "expected_value": float(r.mean()), "avg_win": float(r[r > 0].mean()) if (r > 0).any() else 0.0, "avg_loss": float(r[r < 0].mean()) if (r < 0).any() else 0.0}


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
    iterations = int(iterations or os.getenv("WF_MONTE_CARLO_ITERATIONS", "5000"))
    if trades is None or trades.empty:
        return None
    ret = pd.to_numeric(trades["return"], errors="coerce").to_numpy(float)
    unit = pd.to_numeric(trades["risk_unit"], errors="coerce").to_numpy(float)
    mask = np.isfinite(ret) & np.isfinite(unit) & (unit > 0)
    r_mult = ((ret[mask] / 100) / unit[mask])
    if len(r_mult) < 20:
        return None
    years = max(0.5, (trades.date.max() - trades.date.min()).days / 365.25)
    annual_signals = max(1.0, len(r_mult) / years)
    rng = np.random.default_rng(RANDOM_SEED)
    diagnostics = []
    for sizing in (0.01, 0.0075, 0.005, 0.0025):
        path = np.full(iterations, float(initial_capital))
        peak = path.copy()
        max_dd = np.zeros(iterations)
        checkpoints = {}
        for year in range(1, 21):
            n = max(1, int(round(annual_signals)))
            sample = rng.choice(r_mult, size=(iterations, n), replace=True)
            growth = np.prod(np.clip(1 + sizing * sample, 0.01, 5.0), axis=1)
            path *= growth
            peak = np.maximum(peak, path)
            max_dd = np.maximum(max_dd, 1 - path / np.maximum(peak, 1e-9))
            if year in (10, 15, 20):
                checkpoints[year] = float(np.mean(path >= target_capital) * 100)
        bankruptcy = float(np.mean(path <= initial_capital * 0.50) * 100)
        p90_dd = float(np.quantile(max_dd, 0.90) * 100)
        d = {"sizing": sizing, "prob_10y": checkpoints[10], "prob_15y": checkpoints[15], "prob_20y": checkpoints[20], "bankruptcy_prob": bankruptcy, "p90_max_dd": p90_dd}
        diagnostics.append(d)
        if bankruptcy < 5.0 and p90_dd <= 30.0:
            return d
    return None


# DEV exploration
dev_df = candidates[candidates.phase == "DEV"].copy()
validation_df = candidates[candidates.phase == "VALIDATION"].copy()
oos_df = candidates[candidates.phase == "OOS"].copy()
param_space = list(product(UP_THRESHOLDS, SCORE_THRESHOLDS, NIKKEI_FILTERS, TP_MULTIPLIERS, SL_MULTIPLIERS, HOLD_DAYS_LIST))
all_dev_rows = []
for i, (up, score, nikkei, tp, sl, hold) in enumerate(param_space, 1):
    if i % 100 == 0:
        print(f"DEV探索 {i}/{len(param_space)}")
    rd = run_strategy(dev_df, up, score, nikkei, tp, sl, hold)
    st = stats(rd)
    all_dev_rows.append({"strategy": f"UP{up}_SCORE{score}_NIKKEI{'ON' if nikkei else 'OFF'}_TP{tp}_SL{sl}_H{hold}", "up": up, "score": score, "nikkei": nikkei, "tp": tp, "sl": sl, "hold": hold, **{f"dev_{k}": v for k, v in st.items()}})
dev_summary = pd.DataFrame(all_dev_rows)
dev_summary.to_csv("adversarial_dev_all_results.csv", index=False, encoding="utf-8-sig")
dev_candidates = dev_summary[(dev_summary.dev_signals >= MIN_TRADES_HARD) & (dev_summary.dev_annual_signals >= MIN_ANNUAL_SIGNALS) & (dev_summary.dev_avg_return > MIN_RETURN_LOWER) & (dev_summary.dev_pf >= MIN_PF_LOWER)].copy()
# 優先順位: ①月間収益率(=月間利益額と線形同値) ②月間+5%達成率 ③OOS系累積収益率(=複利最終資産と線形同値)
# ④平均利益率/期待利益率 ⑤Profit Factor ⑥最大DD(ペナルティ)。勝率(win_rate)は一切使わない。
dev_candidates["dev_objective"] = (
    np.clip(dev_candidates.dev_avg_month_return, -20, 20) * 0.30
    + dev_candidates.dev_monthly_plus5_ratio * 0.20
    + np.clip(dev_candidates.dev_compound_return, -100, 500) * 0.25
    + np.clip(dev_candidates.dev_avg_return, -5, 5) * 10 * 0.15
    + np.clip(dev_candidates.dev_pf, 0, 5) * 10 * 0.07
    - np.clip(-dev_candidates.dev_dd, 0, 100) * 0.03
)
dev_candidates = dev_candidates.sort_values("dev_objective", ascending=False).head(50).copy()
dev_candidates.to_csv("adversarial_dev_selected_candidates.csv", index=False, encoding="utf-8-sig")

# Validation
validation_results = []
for _, row in dev_candidates.iterrows():
    rd = run_strategy(validation_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    st = stats(rd)
    lower_avg = block_bootstrap_lower(rd["return"].values, alpha=MULTIPLE_TEST_ALPHA) if not rd.empty else np.nan
    validation_results.append({**row.to_dict(), **{f"validation_{k}": v for k, v in st.items()}, "validation_avg_lower": lower_avg})
validation_summary = pd.DataFrame(validation_results)
if not validation_summary.empty:
    validation_summary["validation_pass"] = ((validation_summary.validation_signals >= MIN_VALIDATION_TRADES) & (validation_summary.validation_pf >= MIN_PF_LOWER) & (validation_summary.validation_avg_return > MIN_RETURN_LOWER) & (validation_summary.validation_dd >= -MAX_VALIDATION_DD) & (validation_summary.validation_annual_signals >= MIN_ANNUAL_SIGNALS) & (validation_summary.validation_monthly_positive_ratio >= MIN_MONTHLY_POSITIVE_RATIO * 100) & (validation_summary.validation_avg_lower > 0))
else:
    validation_summary["validation_pass"] = False
validation_summary.to_csv("adversarial_validation_results.csv", index=False, encoding="utf-8-sig")

# OOS
passed_validation = validation_summary[validation_summary.validation_pass].copy() if not validation_summary.empty else pd.DataFrame()
oos_results = []
for _, row in passed_validation.iterrows():
    rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    oos_results.append({**row.to_dict(), **{f"oos_{k}": v for k, v in stats(rd).items()}})
oos_summary = pd.DataFrame(oos_results)
if not oos_summary.empty:
    oos_summary["oos_pf_ratio"] = oos_summary.oos_pf / oos_summary.validation_pf.replace(0, np.nan)
    oos_summary["oos_insufficient_data"] = oos_summary.oos_signals < MIN_OOS_TRADES
    ratio_ok = (oos_summary.oos_pf_ratio >= MIN_OOS_TO_VALIDATION_PF) | (oos_summary.validation_pf > MAX_VALIDATION_PF_FOR_RATIO)
    oos_summary["oos_pass"] = ((oos_summary.oos_signals >= MIN_OOS_TRADES) & (oos_summary.oos_pf >= MIN_OOS_PF) & (oos_summary.oos_avg_return > MIN_OOS_AVG_RETURN) & ratio_ok & (oos_summary.oos_monthly_positive_ratio >= MIN_MONTHLY_POSITIVE_RATIO * 100) & (oos_summary.oos_dd >= -MAX_OOS_DD) & (oos_summary.oos_compound_return > 0))
else:
    oos_summary["oos_pf_ratio"] = pd.Series(dtype=float)
    oos_summary["oos_insufficient_data"] = pd.Series(dtype=bool)
    oos_summary["oos_pass"] = False
oos_summary.to_csv("adversarial_oos_results.csv", index=False, encoding="utf-8-sig")

# Final ranking + mandatory MC risk gate
final_pass = oos_summary[oos_summary.oos_pass].copy() if not oos_summary.empty else pd.DataFrame()
if not final_pass.empty:
    # 優先順位: ①月間収益率 ②月間+5%達成率 ③OOS累積収益率(=複利最終資産と線形同値) ④平均利益率/期待利益率 ⑤PF ⑥最大DD(ペナルティ)
    final_pass["profit_objective"] = (
        np.clip(final_pass.oos_avg_month_return, -20, 20) * 0.30
        + final_pass.oos_monthly_plus5_ratio * 0.20
        + np.clip(final_pass.oos_compound_return, -100, 1000) * 0.25
        + np.clip(final_pass.oos_avg_return, -5, 5) * 10 * 0.15
        + np.clip(final_pass.oos_pf, 0, 8) * 5 * 0.07
        - np.clip(-final_pass.oos_dd, 0, 100) * 0.03
    )
    final_pass = final_pass.sort_values(["profit_objective", "oos_avg_month_return", "oos_compound_return", "oos_pf", "oos_avg_return"], ascending=False).reset_index(drop=True)
    mc_limit = int(os.getenv("WF_MC_CANDIDATES", "20"))
    mc_records = []
    for _, row in final_pass.head(mc_limit).iterrows():
        rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
        mc = monte_carlo_risk_gate(rd)
        if mc is not None:
            mc_records.append({"strategy": row.strategy, **mc})
    if mc_records:
        final_pass = final_pass.merge(pd.DataFrame(mc_records), on="strategy", how="inner")
    else:
        final_pass = pd.DataFrame()

if not final_pass.empty:
    final_pass["final_status"] = "PASS"
    final_pass["up_threshold"] = final_pass.up
    final_pass["score_threshold"] = final_pass.score
    final_pass["nikkei_filter"] = final_pass.nikkei
    final_pass["tp_multiplier"] = final_pass.tp
    final_pass["sl_multiplier"] = final_pass.sl
    final_pass["hold_days"] = final_pass.hold
    final_pass["oos_validation_pf_ratio"] = final_pass.oos_pf_ratio
else:
    final_pass = pd.DataFrame(columns=["final_status", "up_threshold", "score_threshold", "nikkei_filter", "tp_multiplier", "sl_multiplier", "hold_days"])
final_pass.to_csv("adversarial_final_candidates.csv", index=False, encoding="utf-8-sig")

validation_n = int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0
oos_n = int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0
oos_insufficient_n = int(oos_summary.oos_insufficient_data.sum()) if not oos_summary.empty else 0
print("=" * 80)
print("🛡️ AI PROFIT OPTIMIZER RESULT")
print("期間:", START_DATE.date(), "～", END_DATE.date())
print("探索数:", len(param_space), "N_eff:", N_EFFECTIVE_STRATEGIES)
print("Purge/Embargo:", PURGE_DAYS, EMBARGO_DAYS, "TOP_N:", TOP_N)
print("DEV候補:", len(dev_candidates), "Validation PASS:", validation_n, "OOS PASS:", oos_n)
print(f"OOS 判定不能(シグナル数<{MIN_OOS_TRADES}):", oos_insufficient_n, "/", len(oos_summary))
print("Final PASS:", len(final_pass))

msg = (f"🛡️ AI PROFIT OPTIMIZER\n期間: {START_DATE.date()} ～ {END_DATE.date()}\nTOP_N: {TOP_N}\nDEV候補: {len(dev_candidates)}\nValidation PASS: {validation_n}\nOOS PASS: {oos_n}(うちOOSシグナル数不足で判定不能: {oos_insufficient_n}件)\nFinal PASS: {len(final_pass)}\n目標: 月間損益プラス率・OOS複利資産・期待利益を優先")
if not final_pass.empty:
    msg += "\n\n🏆 BEST STRATEGIES\n"
    for _, r in final_pass.head(10).iterrows():
        msg += (f"{r.strategy}\n  月間収益率={r.oos_avg_month_return:+.2f}% 月間利益額=¥{r.oos_avg_month_profit_jpy:+,.0f} 月間+5%達成率={r.oos_monthly_plus5_ratio:.1f}%\n  OOS複利={r.oos_compound_return:+.2f}% 最終資産=¥{r.oos_compound_final_capital:,.0f}\n  PF={r.oos_pf:.2f} 期待利益={r.oos_expected_value:+.3f}% 最大DD={r.oos_dd:.2f}%\n")
else:
    msg += "\n\n該当するFinal PASS戦略なし。既存policyは自動変更しません。"
    if not oos_summary.empty:
        near_miss = oos_summary[(oos_summary.oos_signals >= MIN_OOS_TRADES) & (oos_summary.oos_pf >= MIN_OOS_PF) & (oos_summary.oos_avg_return > MIN_OOS_AVG_RETURN) & (oos_summary.oos_compound_return > 0) & (oos_summary.oos_pf_ratio < MIN_OOS_TO_VALIDATION_PF)]
        if not near_miss.empty:
            msg += f"\n⚠ {len(near_miss)}件はOOS実績自体は黒字だがoos_pf_ratio<{MIN_OOS_TO_VALIDATION_PF}。validation_pf外れ値を要確認。"
send_discord(msg)
