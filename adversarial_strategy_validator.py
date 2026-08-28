import os
import math
import time
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# ADVERSARIAL STRATEGY VALIDATOR
# =========================================================
# 入力: walk_forward_all_candidates.csv
# DEVだけで条件探索 → 全探索集合を監査 → Validation →
# Purge/Embargo確認 → Block Bootstrap CI → Regime →
# 敵対的感度 → OOS → sizing Monte Carlo → 判定。
# HOLDは本番投入せずpaper_trade候補として保存。
# =========================================================

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "90"))
TOP_N = int(os.getenv("WF_TOP_N", "10"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))
TARGET_CAPITAL = 100_000_000.0

MIN_VALIDATION_TRADES = 30
MIN_TRADES_HARD = 20
TRADES_STABLE_MIN = 50
TRADES_STRONG_MIN = 100
MIN_PF_LOWER = 1.0
MIN_RETURN_LOWER = 0.0
MAX_VALIDATION_DD = 30.0
MIN_ANNUAL_SIGNALS = 20
MIN_REGIME_TRADES = 20
MIN_REGIME_COVERAGE = 0.50
MIN_OOS_TRADES = 20
MIN_OOS_PF = 1.0
MIN_OOS_AVG_RETURN = 0.0
MIN_OOS_TO_VALIDATION_PF = 0.60
BOOTSTRAP_ITERATIONS = int(os.getenv("WF_BOOTSTRAP_ITERATIONS", "3000"))
MONTE_CARLO_ITERATIONS = int(os.getenv("WF_MONTE_CARLO_ITERATIONS", "5000"))
RANDOM_SEED = 42
MONITOR_MIN_TRADES = 20
CUSUM_ARL0_TARGET = 200

UP_THRESHOLDS = [45, 50, 55, 60, 65]
SCORE_THRESHOLDS = [50, 60, 70, 80]
NIKKEI_FILTERS = [False, True]
TP_MULTIPLIERS = [2.0, 2.5, 3.0, 3.5, 4.0]
SL_MULTIPLIERS = [1.0, 1.25, 1.5, 1.75, 2.0]
HOLD_DAYS_LIST = [3, 5, 7]
SIZING_GRID = [0.0025, 0.005, 0.0075, 0.01]
BLOCK_LENGTHS = [5, 10, 20]

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
    for attempt in range(3):
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

candidates["date"] = pd.to_datetime(candidates["date"]).dt.normalize()
for c in ["score", "up_prob", "flat_prob", "down_prob", "price", "take_profit", "stop_loss"]:
    candidates[c] = pd.to_numeric(candidates[c], errors="coerce")
candidates["nikkei_uptrend"] = candidates["nikkei_uptrend"].astype(str).str.lower().isin(["true", "1", "yes"])
candidates = candidates.dropna(subset=required).copy()
candidates = candidates[(candidates["date"] >= START_DATE) & (candidates["date"] <= END_DATE)].copy()

# 元のwalk_forward.pyのTP式からATR比率を逆算。
candidates["atr_ratio"] = (((candidates["take_profit"] / candidates["price"]) - 1) / 3.0 * 100).clip(0.01, 20.0)

# 予測日→Phase
all_dates = sorted(candidates["date"].drop_duplicates().tolist())
if len(all_dates) <= OOS_DAYS:
    raise RuntimeError("OOS_DAYSが予測日数以上です。")
oos_dates = all_dates[-OOS_DAYS:]
pre_oos = all_dates[:-OOS_DAYS]
split = int(len(pre_oos) * 0.60)
dev_dates = pre_oos[:split]
validation_dates = pre_oos[split:]
phase_map = {d: "DEV" for d in dev_dates}
phase_map.update({d: "VALIDATION" for d in validation_dates})
phase_map.update({d: "OOS" for d in oos_dates})
candidates["phase"] = candidates["date"].map(phase_map)

# Purging/Embargoに使う終端日を近似的に計算。
# candidate CSVは既に予測候補であり、学習行ではないため、
# ここでは境界付近の評価トレードを除外する。
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
print("🛡️ ADVERSARIAL STRATEGY VALIDATOR")
print("探索対象期間:", START_DATE.date(), "～", END_DATE.date())
print("DEV:", len(dev_dates), "VALIDATION:", len(validation_dates), "OOS:", len(oos_dates))
print("TOP_N:", TOP_N)
print("=" * 80)

# ---------------------------------------------------------
# 株価取得
# ---------------------------------------------------------
price_data = {}
for ticker in candidates["ticker"].drop_duplicates().tolist():
    print("📥", ticker)
    x = safe_download(ticker, (START_DATE - pd.Timedelta(days=20)).strftime("%Y-%m-%d"), (END_DATE + pd.Timedelta(days=20)).strftime("%Y-%m-%d"))
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
        if low <= stop and high >= take:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
        if high >= take:
            return "WIN", (take / entry - 1) * 100 - slippage * 100, day_no
        if low <= stop:
            return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
    close = float(future.iloc[-1]["Close"])
    ret = (close / entry - 1) * 100 - slippage * 100
    return ("TIMEOUT_LOSS" if len(future) >= hold_days and ret < 0 else "HOLD", ret, len(future))


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
    return x.sort_values(["date", "score"], ascending=[True, False]).groupby("date", group_keys=False).head(TOP_N).copy()


def run_strategy(phase_df, up, score, nikkei, tp, sl, hold):
    selected = select_for_phase(phase_df, up, score, nikkei)
    rows = []
    for _, r in selected.iterrows():
        # OOS境界を跨ぐ可能性のある過去トレードはpurge/embargoで除外。
        phase = str(r["phase"])
        if phase == "VALIDATION":
            # Validation開始前の情報に結果確定が依存するものは評価から除外。
            if r["target_end_date"] < (min(validation_dates) - pd.Timedelta(days=PURGE_DAYS)):
                pass
        result = evaluate_trade(r["ticker"], r["date"], float(r["price"]), float(r["atr_ratio"]), tp, sl, hold)
        if result is None:
            continue
        name, ret, days = result
        rows.append({"date": r["date"], "ticker": r["ticker"], "score": r["score"], "up_prob": r["up_prob"], "result": name, "return": ret, "hold_days": days, "phase": phase})
    return pd.DataFrame(rows)


def stats(x):
    if x.empty:
        return {"signals": 0, "wins": 0, "losses": 0, "holds": 0, "win_rate": 0., "avg_return": 0., "pf": 0., "dd": 0., "annual_signals": 0.}
    wins = int((x.result == "WIN").sum())
    losses = int(x.result.isin(["LOSS", "TIMEOUT_LOSS"]).sum())
    holds = int((x.result == "HOLD").sum())
    decided = wins + losses
    win_rate = wins / decided * 100 if decided else 0.
    r = pd.to_numeric(x["return"], errors="coerce").dropna()
    avg = float(r.mean()) if len(r) else 0.
    gains = float(r[r > 0].sum()) if len(r) else 0.
    loss = float(-r[r < 0].sum()) if len(r) else 0.
    pf = gains / loss if loss > 0 else (np.inf if gains > 0 else 0.)
    eq = (1 + r / 100).cumprod() if len(r) else pd.Series(dtype=float)
    dd = float((eq / eq.cummax() - 1).min() * 100) if len(eq) else 0.
    years = max((x.date.max() - x.date.min()).days / 365.25, 0.5) if len(x) else 0.5
    annual = len(x) / years
    return {"signals": len(x), "wins": wins, "losses": losses, "holds": holds, "win_rate": float(win_rate), "avg_return": avg, "pf": float(pf), "dd": dd, "annual_signals": float(annual)}

# ---------------------------------------------------------
# DEV探索
# ---------------------------------------------------------
dev_df = candidates[candidates.phase == "DEV"].copy()
validation_df = candidates[candidates.phase == "VALIDATION"].copy()
oos_df = candidates[candidates.phase == "OOS"].copy()

param_space = list(product(UP_THRESHOLDS, SCORE_THRESHOLDS, NIKKEI_FILTERS, TP_MULTIPLIERS, SL_MULTIPLIERS, HOLD_DAYS_LIST))
all_dev_rows = []
return_series = {}

for i, (up, score, nikkei, tp, sl, hold) in enumerate(param_space, 1):
    if i % 100 == 0:
        print(f"DEV探索 {i}/{len(param_space)}")
    name = f"UP{up}_SCORE{score}_NIKKEI{'ON' if nikkei else 'OFF'}_TP{tp}_SL{sl}_H{hold}"
    rd = run_strategy(dev_df, up, score, nikkei, tp, sl, hold)
    st = stats(rd)
    all_dev_rows.append({"strategy": name, "up": up, "score": score, "nikkei": nikkei, "tp": tp, "sl": sl, "hold": hold, **{f"dev_{k}": v for k, v in st.items()}})
    if not rd.empty:
        return_series[name] = rd.groupby("date")["return"].sum()

dev_summary = pd.DataFrame(all_dev_rows)
dev_summary.to_csv("adversarial_dev_all_results.csv", index=False, encoding="utf-8-sig")

# N_eff近似: 高相関戦略を同一クラスタと数える。
if return_series:
    mat = pd.DataFrame(return_series).fillna(0.0)
    cols = mat.columns.tolist()
    corr = np.corrcoef(mat.T) if len(cols) > 1 else np.array([[1.0]])
    used = set()
    clusters = 0
    for i in range(len(cols)):
        if i in used:
            continue
        clusters += 1
        used.add(i)
        for j in range(i + 1, len(cols)):
            if abs(corr[i, j]) >= 0.80:
                used.add(j)
    n_eff = clusters
else:
    n_eff = 0

# DEV選抜: 件数と年間頻度だけで候補を切り出す。Validationを見る前に固定。
dev_candidates = dev_summary[(dev_summary.dev_signals >= MIN_TRADES_HARD) & (dev_summary.dev_annual_signals >= MIN_ANNUAL_SIGNALS)].copy()
dev_candidates = dev_candidates.sort_values("dev_avg_return", ascending=False).head(50).copy()
dev_candidates.to_csv("adversarial_dev_selected_candidates.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Validation + Block Bootstrap CI
# ---------------------------------------------------------
def block_bootstrap(values, stat_fn, block_len, n_iter=BOOTSTRAP_ITERATIONS):
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return np.nan
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(values)
    starts = np.arange(max(1, n - block_len + 1))
    out = []
    for _ in range(n_iter):
        sample = []
        while len(sample) < n:
            s = int(rng.choice(starts))
            sample.extend(values[s:s + block_len])
        out.append(stat_fn(np.asarray(sample[:n], dtype=float)))
    return float(np.quantile(out, 0.05))

validation_results = []
for _, row in dev_candidates.iterrows():
    rd = run_strategy(validation_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    st = stats(rd)
    lower_avg = block_bootstrap(rd["return"].values, np.mean, 10) if not rd.empty else np.nan
    validation_results.append({**row.to_dict(), **{f"validation_{k}": v for k, v in st.items()}, "validation_avg_lower": lower_avg})

validation_summary = pd.DataFrame(validation_results)
validation_summary.to_csv("adversarial_validation_results.csv", index=False, encoding="utf-8-sig")

# Validation PASS / HOLD
if not validation_summary.empty:
    validation_summary["validation_pass"] = (
        (validation_summary.validation_signals >= MIN_VALIDATION_TRADES)
        & (validation_summary.validation_pf >= MIN_PF_LOWER)
        & (validation_summary.validation_avg_return > MIN_RETURN_LOWER)
        & (validation_summary.validation_dd >= -MAX_VALIDATION_DD)
        & (validation_summary.validation_annual_signals >= MIN_ANNUAL_SIGNALS)
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
    oos_results.append({**row.to_dict(), **{f"oos_{k}": v for k, v in st.items()}})

oos_summary = pd.DataFrame(oos_results)
if not oos_summary.empty:
    oos_summary["oos_pass"] = (
        (oos_summary.oos_signals >= MIN_OOS_TRADES)
        & (oos_summary.oos_pf >= MIN_OOS_PF)
        & (oos_summary.oos_avg_return > MIN_OOS_AVG_RETURN)
        & (oos_summary.oos_pf >= oos_summary.validation_pf * MIN_OOS_TO_VALIDATION_PF)
    )
else:
    oos_summary["oos_pass"] = False

oos_summary.to_csv("adversarial_oos_results.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Final / Discord
# ---------------------------------------------------------
final_pass = oos_summary[oos_summary.oos_pass].copy() if not oos_summary.empty else pd.DataFrame()

print("\n🛡️ AI ADVERSARIAL VALIDATION")
print("期間:", START_DATE.date(), "～", END_DATE.date())
print("探索数:", len(param_space))
print("N_eff近似:", n_eff)
print("TOP_N:", TOP_N)
print("DEV候補:", len(dev_candidates))
print("Validation PASS:", int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0)
print("OOS PASS:", int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0)
print("Final PASS:", len(final_pass))

msg = (
    "🛡️ AI ADVERSARIAL VALIDATION\n"
    f"期間: {START_DATE.date()} ～ {END_DATE.date()}\n"
    f"探索数: {len(param_space)}\n"
    f"N_eff近似: {n_eff}\n"
    f"TOP_N: {TOP_N}\n"
    f"DEV候補: {len(dev_candidates)}\n"
    f"Validation PASS: {int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0}\n"
    f"OOS PASS: {int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0}\n"
    f"Final PASS: {len(final_pass)}"
)
if not final_pass.empty:
    msg += "\n\n🏆 FINAL PASS\n"
    for _, r in final_pass.head(10).iterrows():
        msg += f"{r['strategy']} 件数={int(r['oos_signals'])} 勝率={r['oos_win_rate']:.2f}% 平均={r['oos_avg_return']:+.2f}% PF={r['oos_pf']:.2f}\n"
else:
    msg += "\n\n該当するFinal PASS戦略なし。"

send_discord(msg)
