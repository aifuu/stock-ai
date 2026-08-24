import os
import math
import time
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

# =========================================================
# ADVERSARIAL STRATEGY VALIDATOR (修正版)
# =========================================================
# 入力: walk_forward_all_candidates.csv
# DEVだけで条件探索 → 全探索集合を監査 → Validation →
# Purge/Embargo → Block Bootstrap CI → Regime →
# 敵対的感度 → OOS → 銘柄別流動性を考慮したsizing Monte Carlo → 判定。
# HOLDは本番投入せずpaper_trade候補として保存。
#
# 【このスクリプトの責務範囲についての注記】
# up_prob / score などの特徴量そのものの生成・特徴量重要度に基づく
# 選定・per-foldのモデル再学習(=Rolling Walk-Forwardの本体)は、
# 上流の walk_forward.py 側の責務という前提。このスクリプトは、
# 上流が出力した候補に対する「条件(ルール)探索」と「統計的検証」
# を担当する。上流側でも同様のリーク対策(特徴量重要度をtrain内だけ
# で計算する等)が取られているか、別途必ず確認すること。
# =========================================================

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "90"))
TOP_N = int(os.getenv("WF_TOP_N", "3"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))
TARGET_CAPITAL = 100_000_000.0
TARGET_YEARS = [10, 15, 20]  # FIX A: 未定義だったため追加(10/15/20年の3段階目標)

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

# FIX G: キャパシティ制約を「全銘柄共通の固定5000万円」から
# 「銘柄ごとの実測平均売買代金に対する参加率」に変更するための定数。
# あくまでヒューリスティックであり実測のマーケットインパクトモデルではない。
MAX_PARTICIPATION = 0.05        # ポジションが平均売買代金の何%を超えたら追加ペナルティを課すか
CAPACITY_PENALTY_COEF = 0.50    # 超過分をどれだけリターンから差し引くか(係数)

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

VALIDATION_START = min(validation_dates)
OOS_START = min(oos_dates)

# Purging/Embargoに使う終端日を近似的に計算。
base_business = sorted(candidates["date"].drop_duplicates().tolist())
idx_map = {d: i for i, d in enumerate(base_business)}


def shift_date(d, n):
    p = idx_map.get(d)
    if p is None:
        return d
    return base_business[min(max(p + n, 0), len(base_business) - 1)]


candidates["target_end_date"] = candidates["date"].map(lambda d: shift_date(d, 3))
# 保守的に「試すパラメータの中で最も長い保有日数」を使って結果確定日を近似する。
# (どのhold値のバリエーションでも境界をまたがないよう安全側に倒すための選択)
candidates["trade_end_date"] = candidates["date"].map(lambda d: shift_date(d, max(HOLD_DAYS_LIST)))

print("=" * 80)
print("🛡️ ADVERSARIAL STRATEGY VALIDATOR (修正版)")
print("探索対象期間:", START_DATE.date(), "～", END_DATE.date())
print("DEV:", len(dev_dates), "VALIDATION:", len(validation_dates), "OOS:", len(oos_dates))
print(f"Purge={PURGE_DAYS}日 Embargo={EMBARGO_DAYS}日 (VALIDATION開始:{VALIDATION_START.date()} OOS開始:{OOS_START.date()})")
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


def build_ticker_liquidity(pdata, start, end):
    """FIX G: 銘柄ごとの平均売買代金(円)を実データから算出する。
    キャパシティ制約を全銘柄一律の定数ではなく、銘柄固有の値にするため。"""
    liquidity = {}
    vals = []
    for ticker, df in pdata.items():
        sub = df[(df.index >= start) & (df.index <= end)]
        if sub.empty or "Volume" not in sub.columns:
            continue
        dv = (sub["Close"].astype(float) * sub["Volume"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
        if len(dv):
            liquidity[ticker] = float(dv.mean())
            vals.append(liquidity[ticker])
    fallback = float(np.median(vals)) if vals else 50_000_000.0
    return liquidity, fallback


TICKER_LIQUIDITY, LIQUIDITY_FALLBACK = build_ticker_liquidity(price_data, START_DATE, END_DATE)
print(f"流動性データ取得銘柄数: {len(TICKER_LIQUIDITY)}  フォールバック値: {LIQUIDITY_FALLBACK:,.0f}円/日")


def capacity_penalty_for(ticker, position_value):
    """FIX G: ポジション額が銘柄の平均売買代金に対してどれだけの参加率かを見て、
    MAX_PARTICIPATIONを超えた分だけリターンにペナルティ(パーセンテージポイント)を課す。
    実測のマーケットインパクトモデルではなく簡易ヒューリスティックである点に注意。"""
    adv = TICKER_LIQUIDITY.get(ticker, LIQUIDITY_FALLBACK)
    if not adv or adv <= 0:
        adv = LIQUIDITY_FALLBACK
    participation = position_value / adv
    return max(0.0, participation - MAX_PARTICIPATION) * CAPACITY_PENALTY_COEF


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
        phase = str(r["phase"])
        # --- FIX C: Purge / Embargo ---
        # 従来コードはここが `pass` のみで何も除外していなかった。
        if phase == "DEV":
            # DEVトレードの結果確定日がVALIDATION開始以降にかかる場合は除外(Purge)。
            if r["trade_end_date"] >= VALIDATION_START:
                continue
        elif phase == "VALIDATION":
            # VALIDATION開始直後はDEV側からの情報汚染を避けるため除外(Embargo)。
            if r["date"] < VALIDATION_START + pd.Timedelta(days=EMBARGO_DAYS):
                continue
            # VALIDATIONトレードの結果確定日がOOS開始以降にかかる場合は除外(Purge)。
            if r["trade_end_date"] >= OOS_START:
                continue
        elif phase == "OOS":
            # OOS開始直後も同様にEmbargo。
            if r["date"] < OOS_START + pd.Timedelta(days=EMBARGO_DAYS):
                continue
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

# FIX H: 平均リターンの生値だけでソートすると外れ値1件で歪むため、
# PF(3.0でクリップ)との複合スコアでソートする。
dev_summary["dev_pf_capped"] = dev_summary["dev_pf"].replace([np.inf], 3.0).clip(upper=3.0).fillna(0.0)
dev_summary["dev_composite"] = dev_summary["dev_avg_return"] * dev_summary["dev_pf_capped"]
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

search_multiplicity = (len(param_space) / n_eff) if n_eff > 0 else float("nan")
print(f"探索総数: {len(param_space)}  N_eff近似: {n_eff}  多重度(total/N_eff): {search_multiplicity:.2f}")
print("※ N_effは参考値。厳密なSPA/Reality Check検定はここでは未実装のため、")
print("  最終的な採用判断では多重度が高い(=似た戦略ばかり試している)候補ほど懐疑的に見ること。")

# FIX F: DEV選抜数をN_effに連動させる(固定50件ではなく、実質的な独立戦略数を上限の目安にする)。
dev_candidates_n = int(min(50, max(10, n_eff))) if n_eff > 0 else 50
dev_candidates = dev_summary[(dev_summary.dev_signals >= MIN_TRADES_HARD) & (dev_summary.dev_annual_signals >= MIN_ANNUAL_SIGNALS)].copy()
dev_candidates = dev_candidates.sort_values("dev_composite", ascending=False).head(dev_candidates_n).copy()
dev_candidates.to_csv("adversarial_dev_selected_candidates.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Validation + Block Bootstrap CI
# ---------------------------------------------------------
def block_bootstrap(values, stat_fn, block_len, n_iter=BOOTSTRAP_ITERATIONS):
    values = np.asarray(values, dtype=float)
    if len(values) < 10:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(RANDOM_SEED + block_len)
    out = []
    for _ in range(n_iter):
        sample = []
        while len(sample) < len(values):
            s = int(rng.integers(0, len(values)))
            for j in range(block_len):
                if len(sample) >= len(values):
                    break
                sample.append(values[(s + j) % len(values)])
        out.append(stat_fn(np.asarray(sample)))
    out = np.asarray(out, dtype=float)
    out = out[np.isfinite(out)]
    if not len(out):
        return np.nan, np.nan, np.nan
    return float(np.quantile(out, .025)), float(np.quantile(out, .50)), float(np.quantile(out, .975))


def pf_stat(v):
    g = v[v > 0].sum()
    l = -v[v < 0].sum()
    return g / l if l > 0 else (10.0 if g > 0 else 0.0)


validation_rows = []
for _, s in dev_candidates.iterrows():
    rd = run_strategy(validation_df, int(s.up), int(s.score), bool(s.nikkei), float(s.tp), float(s.sl), int(s.hold))
    st = stats(rd)
    r = rd["return"].dropna().to_numpy() if not rd.empty else np.array([])
    pf_l = []
    mean_l = []
    if len(r) >= 10:
        for bl in BLOCK_LENGTHS:
            a, _, _ = block_bootstrap(r, pf_stat, bl)
            b, _, _ = block_bootstrap(r, np.mean, bl)
            if np.isfinite(a):
                pf_l.append(a)
            if np.isfinite(b):
                mean_l.append(b)
    validation_rows.append({"strategy": s.strategy, "up": s.up, "score": s.score, "nikkei": s.nikkei, "tp": s.tp, "sl": s.sl, "hold": s.hold, **{f"validation_{k}": v for k, v in st.items()}, "pf_ci_lower": min(pf_l) if pf_l else np.nan, "mean_ci_lower": min(mean_l) if mean_l else np.nan})

validation = pd.DataFrame(validation_rows)
validation.to_csv("adversarial_validation_gate.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Regime gate: 20件未満はUNVERIFIED。判定可能50%未満ならHOLD。
# ---------------------------------------------------------
def regime_status(rd):
    if rd.empty:
        return "UNVERIFIED", 0.0
    tmp = rd.merge(validation_df[["date", "ticker", "nikkei_uptrend"]], on=["date", "ticker"], how="left")
    groups = [tmp[tmp.nikkei_uptrend == True], tmp[tmp.nikkei_uptrend == False]]
    evaluable = 0
    fail = 0
    for g in groups:
        if len(g) < MIN_REGIME_TRADES:
            continue
        evaluable += 1
        st = stats(g)
        if not (np.isinf(st["pf"]) or st["pf"] > 1.0) or st["avg_return"] <= 0:
            fail += 1
    coverage = evaluable / 2.0
    if coverage < MIN_REGIME_COVERAGE:
        return "UNVERIFIED", coverage
    return ("PASS" if fail == 0 else "FAIL"), coverage


# ---------------------------------------------------------
# Validation gate
# ---------------------------------------------------------
gate_rows = []
for _, r in validation.iterrows():
    rd = run_strategy(validation_df, int(r.up), int(r.score), bool(r.nikkei), float(r.tp), float(r.sl), int(r.hold))
    regime, coverage = regime_status(rd)
    n = int(r.validation_signals)
    sample_class = "REJECT" if n < MIN_TRADES_HARD else ("HOLD" if n < TRADES_STABLE_MIN else ("STRONG" if n >= TRADES_STRONG_MIN else "NORMAL"))
    pass_base = (n >= MIN_TRADES_HARD and np.isfinite(r.pf_ci_lower) and r.pf_ci_lower > MIN_PF_LOWER and np.isfinite(r.mean_ci_lower) and r.mean_ci_lower > MIN_RETURN_LOWER and abs(r.validation_dd) <= MAX_VALIDATION_DD and r.validation_annual_signals >= MIN_ANNUAL_SIGNALS and regime == "PASS")

    # FIX D: 演算子優先順位バグを修正。
    # 旧: "HOLD" if pass_base and sample_class == "HOLD" or regime == "UNVERIFIED" else "FAIL"
    #     → regime=="UNVERIFIED"なら pass_base が False でも無条件にHOLDになっていた。
    gate_label = "PASS" if pass_base and sample_class in ["NORMAL", "STRONG"] else (
        "HOLD" if pass_base and (sample_class == "HOLD" or regime == "UNVERIFIED") else "FAIL"
    )

    # FIX I: 損失トレードが0件(PFが構造的に信頼できない)場合はPASSを許さずHOLDに格下げ。
    degenerate_pf = bool(r.validation_losses == 0 and r.validation_wins > 0)
    if degenerate_pf and gate_label == "PASS":
        gate_label = "HOLD"

    gate_rows.append({**r.to_dict(), "sample_class": sample_class, "regime_status": regime, "regime_coverage": coverage, "degenerate_pf": degenerate_pf, "validation_gate": gate_label})

gate = pd.DataFrame(gate_rows)
gate.to_csv("adversarial_validation_gate_final.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# 敵対的感度分析: 元条件を評価するだけ。別条件へ乗り換えない。
# ---------------------------------------------------------
def stress_test(row):
    records = []
    for du, ds, tpr, slr, dh, slip in product([-5, 0, 5], [-10, 0, 10], [0.8, 1.0, 1.2], [0.8, 1.0, 1.2], [-2, 0, 2], [1.0, 1.5, 2.0]):
        up = max(1, int(row.up + du))
        score = max(0, int(row.score + ds))
        tp = max(.5, float(row.tp * tpr))
        sl = max(.5, float(row.sl * slr))
        hold = min(10, max(1, int(row.hold + dh)))
        rd = run_strategy(validation_df, up, score, bool(row.nikkei), tp, sl, hold)
        st = stats(rd)
        if st["signals"] >= MIN_TRADES_HARD:
            pfv = 10.0 if np.isinf(st["pf"]) else st["pf"]
            adjusted = st["avg_return"] - .001 * (slip - 1) * 100
            records.append((pfv, adjusted))
    if not records:
        return np.nan, np.nan, 0.0, False
    arr = np.asarray(records)
    return float(np.median(arr[:, 0])), float(np.quantile(arr[:, 0], .10)), float(np.mean(arr[:, 0] > 1.0)), bool(np.median(arr[:, 0]) >= 1.0 and np.quantile(arr[:, 0], .10) >= .90 and np.mean(arr[:, 0] > 1.0) >= .50)


passes = gate[gate.validation_gate == "PASS"].copy()
stress_rows = []
for _, row in passes.iterrows():
    med, p10, ratio, sp = stress_test(row)
    stress_rows.append({"strategy": row.strategy, "median_pf": med, "p10_pf": p10, "pf_gt_1_ratio": ratio, "stress_pass": sp})
stress_df = pd.DataFrame(stress_rows)
stress_df.to_csv("adversarial_sensitivity.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# OOS gate
# ---------------------------------------------------------
oos_rows = []
for _, row in passes.iterrows():
    rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
    st = stats(rd)
    ratio = (st["pf"] / row.validation_pf) if row.validation_pf > 0 and np.isfinite(row.validation_pf) else np.nan
    status = "PASS" if st["signals"] >= MIN_OOS_TRADES and (np.isinf(st["pf"]) or st["pf"] >= MIN_OOS_PF) and st["avg_return"] > MIN_OOS_AVG_RETURN and (np.isnan(ratio) or ratio >= MIN_OOS_TO_VALIDATION_PF) else ("HOLD" if st["signals"] < MIN_OOS_TRADES else "FAIL")
    oos_rows.append({**row.to_dict(), "oos_signals": st["signals"], "oos_win_rate": st["win_rate"], "oos_avg_return": st["avg_return"], "oos_pf": st["pf"], "oos_dd": st["dd"], "oos_val_pf_ratio": ratio, "oos_gate": status})
oos = pd.DataFrame(oos_rows)
oos.to_csv("adversarial_oos_gate.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# Monte Carlo: Block Bootstrap × Sizing × 銘柄別流動性キャパシティ
# ---------------------------------------------------------
def monte_carlo(returns_tickers, sizing, iterations=MONTE_CARLO_ITERATIONS, max_year=20):
    """returns_tickers: [(return_pct, ticker), ...] のリスト。
    FIX G: キャパシティペナルティを銘柄ごとの流動性に応じて課すため、
    リターン値だけでなく対応するtickerも一緒にリサンプルする。"""
    if len(returns_tickers) < 10:
        return None
    v = np.asarray([rt[0] for rt in returns_tickers], dtype=float)
    tk = np.array([rt[1] for rt in returns_tickers], dtype=object)
    rng = np.random.default_rng(RANDOM_SEED)
    trades_per_year = max(len(v) / max(OOS_DAYS / 252.0, 0.5), 1.0)
    results = {y: [] for y in TARGET_YEARS}
    bankrupt = 0
    dd_list = []
    for _ in range(iterations):
        ntrades = int(math.ceil(trades_per_year * max_year))
        seq_idx = []
        while len(seq_idx) < ntrades:
            bl = int(rng.choice(BLOCK_LENGTHS))
            s = int(rng.integers(0, len(v)))
            for j in range(bl):
                if len(seq_idx) >= ntrades:
                    break
                seq_idx.append((s + j) % len(v))

        capital = INITIAL_CAPITAL
        curve = [capital]
        bankrupt_here = False
        for idx in seq_idx:
            ret = v[idx]
            ticker = tk[idx]
            position = capital * sizing
            penalty = capacity_penalty_for(ticker, position)
            realized = ret - penalty
            capital += capital * sizing * realized / 100.0
            curve.append(capital)
            if capital <= INITIAL_CAPITAL * 0.10:
                bankrupt_here = True
                capital = max(capital, 0.0)
                break
        if bankrupt_here:
            bankrupt += 1
        arr = np.asarray(curve)
        dd = arr / np.maximum.accumulate(arr) - 1.0
        dd_list.append(dd.min() * 100)

        for y in TARGET_YEARS:
            idx_cut = min(len(seq_idx), max(1, int(trades_per_year * y)))
            c = INITIAL_CAPITAL
            for k in seq_idx[:idx_cut]:
                ret = v[k]
                ticker = tk[k]
                position = c * sizing
                penalty = capacity_penalty_for(ticker, position)
                c += c * sizing * (ret - penalty) / 100.0
                if c <= INITIAL_CAPITAL * 0.10:
                    c = 0.0
                    break
            results[y].append(c)

    out = {"sizing": sizing, "bankruptcy_prob": bankrupt / iterations * 100, "dd_median": float(np.median(dd_list)), "dd_p90": float(np.quantile(dd_list, .90))}
    for y in TARGET_YEARS:
        arr = np.asarray(results[y])
        out[f"prob_{y}y"] = float((arr >= TARGET_CAPITAL).mean() * 100)
        out[f"median_{y}y"] = float(np.median(arr))
        out[f"p10_{y}y"] = float(np.quantile(arr, .10))
        out[f"p90_{y}y"] = float(np.quantile(arr, .90))
    return out


# 最終PASSは OOS PASS + 感度PASS。
final = []
for _, row in oos.iterrows():
    stress = stress_df[stress_df.strategy == row.strategy]
    stress_pass = bool(stress.iloc[0].stress_pass) if not stress.empty else False
    status = "PASS" if row.oos_gate == "PASS" and stress_pass else ("HOLD" if row.oos_gate == "HOLD" else "REJECT")
    final.append({**row.to_dict(), "stress_pass": stress_pass, "final_status": status})
final_df = pd.DataFrame(final)
final_df.to_csv("adversarial_final_candidates.csv", index=False, encoding="utf-8-sig")

mc_rows = []
if not final_df.empty:
    for _, row in final_df[final_df.final_status == "PASS"].iterrows():
        rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold))
        if rd.empty:
            continue
        rd_valid = rd.dropna(subset=["return"])
        returns_tickers = list(zip(rd_valid["return"].to_numpy(), rd_valid["ticker"].to_numpy()))
        for sizing in SIZING_GRID:
            mc = monte_carlo(returns_tickers, sizing)
            if mc:
                mc["strategy"] = row.strategy
                mc_rows.append(mc)

mc_df = pd.DataFrame(mc_rows)
mc_df.to_csv("adversarial_monte_carlo.csv", index=False, encoding="utf-8-sig")

# 最適サイジング: 破産<5%、DD90%<=30%、15年到達確率最大
selected = []
if not mc_df.empty:
    for strategy, g in mc_df.groupby("strategy"):
        z = g[(g.bankruptcy_prob < 5.0) & (g.dd_p90.abs() <= 30.0)].copy()
        if z.empty:
            continue
        z = z.sort_values(["prob_15y", "median_15y"], ascending=False)
        selected.append(z.iloc[0].to_dict())
selected_df = pd.DataFrame(selected)
selected_df.to_csv("adversarial_selected_sizing.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# HOLD -> Paper Trade
# FIX J: Validation段階のHOLDだけでなく、OOS段階のHOLD(件数不足)も統合する。
# ---------------------------------------------------------
paper_frames = []
vh = gate[gate.validation_gate == "HOLD"].copy()
if not vh.empty:
    vh["paper_reason"] = "VALIDATION_HOLD"
    paper_frames.append(vh)
oh = oos[oos.oos_gate == "HOLD"].copy()
if not oh.empty:
    oh["paper_reason"] = "OOS_HOLD"
    paper_frames.append(oh)

if paper_frames:
    paper_df = pd.concat(paper_frames, ignore_index=True, sort=False)
    paper_df["paper_status"] = "PAPER_TRADE"
    paper_df["required_new_trades"] = 30
    paper_df.to_csv("paper_trade_candidates.csv", index=False, encoding="utf-8-sig")
else:
    paper_df = pd.DataFrame()

# ---------------------------------------------------------
# Monitoring config / CUSUM
# ---------------------------------------------------------
monitor_rows = []
for _, row in final_df.iterrows():
    monitor_rows.append({"strategy": row.strategy, "min_trades_before_alert": MONITOR_MIN_TRADES, "target_ARL0_trades": CUSUM_ARL0_TARGET, "baseline_validation_avg_return": row.validation_avg_return, "baseline_validation_pf": row.validation_pf, "alert_before_min_trades": False, "cusum_status": "PAPER_CALIBRATION_REQUIRED"})
pd.DataFrame(monitor_rows).to_csv("adversarial_monitoring_config.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------
# まとめ
# ---------------------------------------------------------
print("\n" + "=" * 100)
print("🏁 FINAL ADVERSARIAL VALIDATION")
print("=" * 100)
print("探索数:", len(param_space))  # FIX B: parameter_space -> param_space
print("N_eff近似:", n_eff, f"(多重度 total/N_eff = {search_multiplicity:.2f})")
print("DEV候補:", len(dev_candidates))
print("Validation PASS:", int((gate.validation_gate == "PASS").sum()))
print("Validation HOLD:", int((gate.validation_gate == "HOLD").sum()))
print("OOS PASS:", int((oos.oos_gate == "PASS").sum()) if not oos.empty else 0)
print("Final PASS:", int((final_df.final_status == "PASS").sum()) if not final_df.empty else 0)
print("Final HOLD:", int((final_df.final_status == "HOLD").sum()) if not final_df.empty else 0)
print("Paper Trade候補:", len(paper_df))

if not selected_df.empty:
    print("\n【Monte Carlo最適サイジング】")
    for _, r in selected_df.head(10).iterrows():
        print(f"{r.strategy} size={r.sizing*100:.2f}% 15y到達={r.prob_15y:.2f}% 破産={r.bankruptcy_prob:.2f}% DD90={r.dd_p90:.2f}%")
else:
    print("\nMonte Carloで採用可能なサイジング条件なし")

# ---------------------------------------------------------
# Discord
# ---------------------------------------------------------
lines = [
    "🛡️ AI ADVERSARIAL VALIDATION",
    "━━━━━━━━━━━━━━━━━━",
    f"期間：{START_DATE.date()} ～ {END_DATE.date()}",
    f"探索数：{len(param_space)}",  # FIX B
    f"N_eff近似：{n_eff}（多重度 {search_multiplicity:.2f}）",
    f"DEV候補：{len(dev_candidates)}",
    f"Validation PASS：{int((gate.validation_gate == 'PASS').sum())}",
    f"Validation HOLD：{int((gate.validation_gate == 'HOLD').sum())}",
    f"OOS PASS：{int((oos.oos_gate == 'PASS').sum()) if not oos.empty else 0}",
    f"Final PASS：{int((final_df.final_status == 'PASS').sum()) if not final_df.empty else 0}",
    f"Paper Trade候補：{len(paper_df)}",
]
if not selected_df.empty:
    lines.append("")
    lines.append("【Monte Carlo】")
    for _, r in selected_df.head(5).iterrows():
        lines.append(f"{r.strategy} Size={r.sizing*100:.2f}% 15年100倍={r.prob_15y:.2f}% 破産={r.bankruptcy_prob:.2f}% DD90={r.dd_p90:.2f}%")
if not paper_df.empty:
    lines += ["", "⚠ HOLD候補は本番投入せずPaper Tradeへ（Validation/OOS両方のHOLDを含む）"]
lines += ["", "📁 adversarial_dev_all_results.csv", "📁 adversarial_validation_gate_final.csv", "📁 adversarial_sensitivity.csv", "📁 adversarial_oos_gate.csv", "📁 adversarial_monte_carlo.csv", "📁 paper_trade_candidates.csv"]
message = "\n".join(lines)
print("\n" + message)
send_discord(message)
print("\n✅ ADVERSARIAL VALIDATION COMPLETE")
print("※ stock_scan.pyは変更していません。")
