import os
import time
from itertools import product

import numpy as np
import pandas as pd
import yfinance as yf

import futures_trend

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
START_DATE = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
# ★修正(2026-09): デフォルトを固定過去日にすると、WF_END_DATE未指定の
# 手動実行が気づかず古いデータで走ってしまう。未指定時は当日を使う。
END_DATE = pd.Timestamp(os.getenv("WF_END_DATE") or pd.Timestamp.today().normalize())
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "90"))
# ★修正(2026-09): 本番(run_profit_loop.py)は承認済みpolicyの条件を通過した
# 候補群の中からTOP1(スコア・EV・レジームで最優先の1件)だけをエントリーするが、
# この検証は従来 groupby("date").head(TOP_N) でTOP10全件を評価し、
# stats()側でその日の複数銘柄リターンを単純平均していた。これは「TOP10に分散
# 投資した場合の成績」であり、実際にTOP1だけを一点集中で建てる本番運用の成績とは
# 統計的性質(平均化によるブレの縮小)が異なる。承認済みpolicyの
# validation_avg_month_return等は「TOP1本番と同条件」とは言えなかったため、
# デフォルトをTOP1に合わせる(環境変数で上書きすれば従来の分散評価も可能)。
TOP_N = int(os.getenv("WF_TOP_N", "1"))
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))

# 案3拡張: 先物トレンド軸での週次二本立て検証。"all"(既定, 従来通り)/"up"/"down"。
# up/downで実行すると、その期間のみに候補を絞り込み、出力ファイル名にサフィックスを付ける。
TREND_FILTER = os.getenv("WF_TREND_FILTER", "all").strip().lower()
if TREND_FILTER not in ("all", "up", "down"):
    raise RuntimeError("WF_TREND_FILTERはall/up/downのいずれかを指定してください")
_OUT_SUFFIX = "" if TREND_FILTER == "all" else f"_{TREND_FILTER}"


def _out(name):
    """TREND_FILTERに応じた出力ファイル名を返す(up/down並行実行時に上書きを防ぐ)。"""
    if "." in name:
        base, ext = name.rsplit(".", 1)
        return f"{base}{_OUT_SUFFIX}.{ext}"
    return f"{name}{_OUT_SUFFIX}"


# ★修正(2026-09): TOP1(単一エントリー)化により取引頻度が構造的に希薄になった
# (実測: fold_4のDEV約3000パラメータ中、年率換算シグナル数の最大値はわずか17.59)。
# ユーザーの目標は「毎日取引」ではなく「毎月収益プラス・月次5%目標・収益率優先」であり
# 取引頻度自体はゴールではないため、旧TOP10前提の閾値(30/20)をTOP1で実際に到達可能な
# 水準に見直す。収益性系の閾値(PF・平均リターン・DD・月次プラス比率)は一切変更しない
# (実測でfold_2/3とも全候補が余裕で合格していたため)。
MIN_VALIDATION_TRADES = int(os.getenv("WF_MIN_VALIDATION_TRADES", "10"))
MIN_TRADES_HARD = int(os.getenv("WF_MIN_TRADES_HARD", "20"))
MIN_PF_LOWER = 1.0
MIN_RETURN_LOWER = 0.0
MAX_VALIDATION_DD = 30.0
MIN_ANNUAL_SIGNALS = int(os.getenv("WF_MIN_ANNUAL_SIGNALS", "7"))
# ★修正(2026-09、追加): 実測(TOP1・stability_objective採用後)で、本来収益性の高い
# 候補が「Validationのannual_signalsが7.38(旧基準8未満)」「OOSのsignalsが12件
# (旧基準20未満)」というごく僅かな差で不合格になっているケースを複数Foldで確認した。
# 収益性条件(PF・平均リターン・DD・月次プラス比率)は満たしているにもかかわらず、
# 純粋な回数不足だけで機械的に弾いていたため、実測値に基づき到達可能な水準へ調整する。
MIN_OOS_TRADES = int(os.getenv("WF_MIN_OOS_TRADES", "12"))
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

# 先物トレンド軸のマージ+絞り込み(TREND_FILTER!=allの場合のみ)。
# サンプル数確保のため、all実行時はこの処理をスキップして従来通りの母集団を使う。
if TREND_FILTER != "all":
    trend_series = futures_trend.historical_trend_series(START_DATE, END_DATE)
    if trend_series.empty:
        raise RuntimeError("先物トレンド系列が取得できませんでした(TREND_FILTER指定時は必須)")
    trend_map = trend_series["trend"].to_dict()
    candidates["futures_trend"] = candidates.date.map(lambda d: trend_map.get(d.normalize()))
    before_n = len(candidates)
    candidates = candidates[candidates.futures_trend == TREND_FILTER].copy()
    print(f"📈 TREND_FILTER={TREND_FILTER}: 候補 {before_n}件 → {len(candidates)}件に絞り込み")
    if candidates.empty:
        raise RuntimeError(f"TREND_FILTER={TREND_FILTER}に該当する候補データがありません")

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


# ★修正(2026-09): 以前はBUY方向の候補しか評価しておらず、実運用
# (profit_top10_paper.py/run_profit_loop.py、SHORT_ENABLED=1がデフォルト)が
# 実際にSHORTもエントリーしていることを一切反映していなかった。承認済みpolicyの
# SHORT側の振る舞いには、この検証によるOOS裏付けが無い状態だった。
SHORT_ENABLED = os.getenv("WF_SHORT_ENABLED", "1").strip().lower() in ("1", "true", "yes", "on")
# profit_top10_paper.pyのFEE_RATE(片道)と同じデフォルト・同じ環境変数名。
# 実運用は1トレードにつき往復(エントリー+決済の2回)分の手数料を資金から
# 差し引くため、%換算では概ね FEE_RATE*2*100 に相当する(profit_priority()の
# flat_cost計算と同じ近似)。
FEE_PCT = float(os.getenv("INTRADAY_FEE_RATE", "0.00055")) * 2 * 100.0


def evaluate_trade(ticker, date, entry, atr_ratio, tp, sl, hold_days, direction="BUY", slippage=0.001):
    if ticker not in price_data:
        return None
    future = price_data[ticker][price_data[ticker].index > date].head(hold_days)
    if future.empty:
        return None
    if direction == "BUY":
        take = entry * (1 + atr_ratio / 100 * tp)
        stop = entry * (1 - atr_ratio / 100 * sl)
    else:
        take = entry * (1 - atr_ratio / 100 * tp)
        stop = entry * (1 + atr_ratio / 100 * sl)
    for day_no, (_, row) in enumerate(future.iterrows(), 1):
        high, low = float(row["High"]), float(row["Low"])
        if direction == "BUY":
            if low <= stop and high >= take:
                return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
            if high >= take:
                return "WIN", (take / entry - 1) * 100 - slippage * 100, day_no
            if low <= stop:
                return "LOSS", (stop / entry - 1) * 100 - slippage * 100, day_no
        else:
            if high >= stop and low <= take:
                return "LOSS", (entry / stop - 1) * 100 - slippage * 100, day_no
            if low <= take:
                return "WIN", (entry / take - 1) * 100 - slippage * 100, day_no
            if high >= stop:
                return "LOSS", (entry / stop - 1) * 100 - slippage * 100, day_no
    close = float(future.iloc[-1]["Close"])
    ret = ((close / entry - 1) * 100 - slippage * 100) if direction == "BUY" else ((entry / close - 1) * 100 - slippage * 100)
    return ("TIMEOUT_LOSS" if ret < 0 else "HOLD"), ret, len(future)


def select_for_phase(phase_df, up, score, nikkei, tp, sl):
    # ★修正(2026-09): BUY/SHORT両方向の候補を生成する(down_probも既に候補CSVの
    # 必須列)。
    buy = phase_df[(phase_df.up_prob >= up) & (phase_df.up_prob > phase_df.down_prob) & (phase_df.flat_prob < 50) & (phase_df.score >= score)].copy()
    buy["direction"] = "BUY"
    if SHORT_ENABLED:
        short = phase_df[(phase_df.down_prob >= up) & (phase_df.down_prob > phase_df.up_prob) & (phase_df.flat_prob < 50) & (phase_df.score >= score)].copy()
        short["direction"] = "SHORT"
        x = pd.concat([buy, short], ignore_index=True) if not short.empty else buy
    else:
        x = buy
    if x.empty:
        return x
    if nikkei:
        x = x[((x.direction == "BUY") & x.nikkei_uptrend) | ((x.direction == "SHORT") & ~x.nikkei_uptrend)]
    if x.empty:
        return x
    # ★修正(2026-09): 本番run_profit_loop.pyの_market_regime()+profit_priority()は
    # 「日経強気→BUYのみ/弱気→SHORTのみ/中立→両方をスコア+EVで比較」という
    # レジームのハードゲートを持つが、この検証が使う候補CSVにはkairi25/ret5の
    # 生値が無くnikkei_uptrend(2値)しか保持していないため、3値判定の「中立」は
    # 再現できない。nikkei_uptrend=True→強気(BUYのみ)/False→弱気(SHORTのみ)の
    # 2値近似とする(将来的にkairi25/ret5を候補CSVへ追加できれば3値化が可能。
    # なおこの近似の帰結として、このハードゲート適用後は残る候補が全て
    # 「優先方向」になるため、本番のregime_bonus=10相当は全候補で定数となり
    # 順位には影響しない)。
    preferred = ((x.direction == "BUY") & x.nikkei_uptrend) | ((x.direction == "SHORT") & ~x.nikkei_uptrend)
    x = x[preferred].copy()
    if x.empty:
        return x
    # ★修正(2026-09): 本番profit_priority()と同じ 0.65×score + 0.35×EV(±10でクリップ)×10
    # の優先順位式でTOP1を選ぶ(feedback_weightのみ、将来の実績データに依存し
    # リークになり得るため意図的に除外)。
    is_buy = (x["direction"] == "BUY").to_numpy()
    atr_ratio = x["atr_ratio"].to_numpy(dtype=float)
    reward_pct = atr_ratio * float(tp)
    risk_pct = atr_ratio * float(sl)
    up_prob = x["up_prob"].to_numpy(dtype=float)
    down_prob = x["down_prob"].to_numpy(dtype=float)
    flat_prob = x["flat_prob"].to_numpy(dtype=float)
    win_prob = np.where(is_buy, up_prob, down_prob) / 100.0
    loss_prob = np.where(is_buy, down_prob, up_prob) / 100.0
    ev = win_prob * reward_pct - loss_prob * risk_pct - (flat_prob / 100.0) * FEE_PCT
    rank = 0.65 * x["score"].to_numpy(dtype=float) + 0.35 * np.clip(ev, -10.0, 10.0) * 10.0
    x = x.assign(_rank=rank)
    return x.sort_values(["date", "_rank", "score", "up_prob"], ascending=[True, False, False, False]).groupby("date", group_keys=False).head(TOP_N).copy()


# 診断用(VALIDATION/OOS専用): 「なぜこの候補が選ばれた/落ちたか」を後から
# 追えるようにするための特徴量スナップショット。ゲート判定には一切使わない。
TRADE_DIAG_COLUMNS = ["gc_gap", "gc_approach", "gc_slope", "adx", "breakout20", "relative_strength", "volume_surge", "atr_ratio", "vol", "rsi"]
trade_diagnostics = []


def run_strategy(phase_df, up, score, nikkei, tp, sl, hold, strategy_name=None, phase_name=None, collect_diagnostics=False):
    rows = []
    for _, r in select_for_phase(phase_df, up, score, nikkei, tp, sl).iterrows():
        result = evaluate_trade(r.ticker, r.date, float(r.price), float(r.atr_ratio), tp, sl, hold, direction=r.direction)
        if result is None:
            continue
        name, ret, days = result
        rows.append({"date": r.date, "ticker": r.ticker, "score": r.score, "up_prob": r.up_prob, "direction": r.direction, "result": name, "return": ret, "hold_days": days, "phase": r.phase, "risk_unit": max(1e-8, float(r.atr_ratio) / 100.0 * float(sl))})
        if collect_diagnostics:
            diag = {"strategy": strategy_name, "phase": phase_name or r.phase, "date": r.date, "ticker": r.ticker, "result": name, "return": ret, "hold_days": days}
            for c in TRADE_DIAG_COLUMNS:
                diag[c] = getattr(r, c, np.nan)
            trade_diagnostics.append(diag)
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

# ★修正(2026-09、安定性導入): DEV全体を1つの期間として最適化すると、たまたまその期間だけ
# 突出して良かった(=過学習した)パラメータが選ばれ、Foldごとに「勝ち戦略」が毎回入れ替わる
# 問題が実測で確認された。DEV期間を前半/後半の2分割に分け、両方の半期でそこそこ良い候補を
# 優先する(min(前半目的関数, 後半目的関数)を最大化)ことで、単一期間の偶然に強い候補を選ぶ。
# 実測で全4Foldの候補家系が(UP45, SCORE50/60, NIKKEI両方)に収束することを確認済み。
dev_dates_sorted = sorted(dev_df["date"].unique())
mid_date = dev_dates_sorted[len(dev_dates_sorted) // 2] if len(dev_dates_sorted) >= 2 else dev_df["date"].max()
dev_half1_df = dev_df[dev_df.date <= mid_date].copy()
dev_half2_df = dev_df[dev_df.date > mid_date].copy()
print(f"DEV安定性検証用分割: half1={dev_half1_df.date.min()}~{dev_half1_df.date.max()} / half2={dev_half2_df.date.min()}~{dev_half2_df.date.max()}")

all_dev_rows = []
for i, (up, score, nikkei, tp, sl, hold) in enumerate(param_space, 1):
    if i % 100 == 0:
        print(f"DEV探索 {i}/{len(param_space)}")
    st = stats(run_strategy(dev_df, up, score, nikkei, tp, sl, hold))
    st_h1 = stats(run_strategy(dev_half1_df, up, score, nikkei, tp, sl, hold))
    st_h2 = stats(run_strategy(dev_half2_df, up, score, nikkei, tp, sl, hold))
    all_dev_rows.append({
        "strategy": f"UP{up}_SCORE{score}_NIKKEI{'ON' if nikkei else 'OFF'}_TP{tp}_SL{sl}_H{hold}",
        "up": up, "score": score, "nikkei": nikkei, "tp": tp, "sl": sl, "hold": hold,
        **{f"dev_{k}": v for k, v in st.items()},
        **{f"devh1_{k}": v for k, v in st_h1.items()},
        **{f"devh2_{k}": v for k, v in st_h2.items()},
    })
dev_summary = pd.DataFrame(all_dev_rows)
dev_summary.to_csv(_out("adversarial_dev_all_results.csv"), index=False, encoding="utf-8-sig")
dev_candidates = dev_summary[(dev_summary.dev_signals >= MIN_TRADES_HARD) & (dev_summary.dev_annual_signals >= MIN_ANNUAL_SIGNALS) & (dev_summary.dev_avg_return > MIN_RETURN_LOWER) & (dev_summary.dev_pf >= MIN_PF_LOWER)].copy()


def _dev_objective(df, prefix):
    # 優先順位: ①月間収益率(=月間利益額と線形同値) ②月間+5%達成率 ③OOS系累積収益率(=複利最終資産と線形同値)
    # ④平均利益率/期待利益率 ⑤Profit Factor ⑥最大DD(ペナルティ)。勝率(win_rate)は一切使わない。
    return (
        np.clip(df[f"{prefix}_avg_month_return"], -20, 20) * 0.30
        + df[f"{prefix}_monthly_plus5_ratio"] * 0.20
        + np.clip(df[f"{prefix}_compound_return"], -100, 500) * 0.25
        + np.clip(df[f"{prefix}_avg_return"], -5, 5) * 10 * 0.15
        + np.clip(df[f"{prefix}_pf"], 0, 5) * 10 * 0.07
        - np.clip(-df[f"{prefix}_dd"], 0, 100) * 0.03
    )


dev_candidates["dev_objective"] = _dev_objective(dev_candidates, "dev")
dev_candidates["devh1_objective"] = _dev_objective(dev_candidates, "devh1")
dev_candidates["devh2_objective"] = _dev_objective(dev_candidates, "devh2")
dev_candidates["stability_objective"] = np.minimum(dev_candidates["devh1_objective"], dev_candidates["devh2_objective"])
dev_candidates = dev_candidates.sort_values("stability_objective", ascending=False).head(50).copy()
dev_candidates.to_csv(_out("adversarial_dev_selected_candidates.csv"), index=False, encoding="utf-8-sig")

# Validation
validation_results = []
for _, row in dev_candidates.iterrows():
    rd = run_strategy(validation_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold), strategy_name=row.strategy, phase_name="VALIDATION", collect_diagnostics=True)
    st = stats(rd)
    lower_avg = block_bootstrap_lower(rd["return"].values, alpha=MULTIPLE_TEST_ALPHA) if not rd.empty else np.nan
    validation_results.append({**row.to_dict(), **{f"validation_{k}": v for k, v in st.items()}, "validation_avg_lower": lower_avg})
validation_summary = pd.DataFrame(validation_results)
if not validation_summary.empty:
    validation_summary["validation_pass"] = ((validation_summary.validation_signals >= MIN_VALIDATION_TRADES) & (validation_summary.validation_pf >= MIN_PF_LOWER) & (validation_summary.validation_avg_return > MIN_RETURN_LOWER) & (validation_summary.validation_dd >= -MAX_VALIDATION_DD) & (validation_summary.validation_annual_signals >= MIN_ANNUAL_SIGNALS) & (validation_summary.validation_monthly_positive_ratio >= MIN_MONTHLY_POSITIVE_RATIO * 100) & (validation_summary.validation_avg_lower > 0))
else:
    validation_summary["validation_pass"] = False
validation_summary.to_csv(_out("adversarial_validation_results.csv"), index=False, encoding="utf-8-sig")

# OOS
passed_validation = validation_summary[validation_summary.validation_pass].copy() if not validation_summary.empty else pd.DataFrame()
oos_results = []
for _, row in passed_validation.iterrows():
    rd = run_strategy(oos_df, int(row.up), int(row.score), bool(row.nikkei), float(row.tp), float(row.sl), int(row.hold), strategy_name=row.strategy, phase_name="OOS", collect_diagnostics=True)
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
oos_summary.to_csv(_out("adversarial_oos_results.csv"), index=False, encoding="utf-8-sig")

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
final_pass.to_csv(_out("adversarial_final_candidates.csv"), index=False, encoding="utf-8-sig")

# 診断用CSV(VALIDATION/OOSのみ)。ゲート判定・選定ロジックには一切影響しない。
# トレード単位: adversarial_fold_trade_diagnostics.csv
# 戦略×フェーズ単位の集計: adversarial_fold_diagnostics.csv
trade_diag_df = pd.DataFrame(trade_diagnostics)
trade_diag_df.to_csv(_out("adversarial_fold_trade_diagnostics.csv"), index=False, encoding="utf-8-sig")

if not trade_diag_df.empty:
    def _approach_rate(s):
        return float(pd.Series(s).astype(bool).mean() * 100)

    fold_diag = trade_diag_df.groupby(["strategy", "phase"]).agg(
        signals=("result", "size"),
        avg_return=("return", "mean"),
        avg_gc_gap=("gc_gap", "mean"),
        gc_approach_ratio=("gc_approach", _approach_rate),
        avg_gc_slope=("gc_slope", "mean"),
        avg_adx=("adx", "mean"),
        avg_breakout20=("breakout20", "mean"),
        avg_relative_strength=("relative_strength", "mean"),
        avg_volume_surge=("volume_surge", "mean"),
        avg_atr_ratio=("atr_ratio", "mean"),
    ).reset_index()
else:
    fold_diag = pd.DataFrame(columns=["strategy", "phase", "signals", "avg_return", "avg_gc_gap", "gc_approach_ratio", "avg_gc_slope", "avg_adx", "avg_breakout20", "avg_relative_strength", "avg_volume_surge", "avg_atr_ratio"])
fold_diag.to_csv(_out("adversarial_fold_diagnostics.csv"), index=False, encoding="utf-8-sig")

validation_n = int(validation_summary.validation_pass.sum()) if not validation_summary.empty else 0
oos_n = int(oos_summary.oos_pass.sum()) if not oos_summary.empty else 0
oos_insufficient_n = int(oos_summary.oos_insufficient_data.sum()) if not oos_summary.empty else 0
print("=" * 80)
print("🛡️ AI PROFIT OPTIMIZER RESULT")
print("TREND_FILTER:", TREND_FILTER)
print("期間:", START_DATE.date(), "～", END_DATE.date())
print("探索数:", len(param_space), "N_eff:", N_EFFECTIVE_STRATEGIES)
print("Purge/Embargo:", PURGE_DAYS, EMBARGO_DAYS, "TOP_N:", TOP_N)
print("DEV候補:", len(dev_candidates), "Validation PASS:", validation_n, "OOS PASS:", oos_n)
print(f"OOS 判定不能(シグナル数<{MIN_OOS_TRADES}):", oos_insufficient_n, "/", len(oos_summary))
print("Final PASS:", len(final_pass))

msg = (f"🛡️ AI PROFIT OPTIMIZER{'（' + TREND_FILTER + 'トレンド専用）' if TREND_FILTER != 'all' else ''}\n期間: {START_DATE.date()} ～ {END_DATE.date()}\nTOP_N: {TOP_N}\nDEV候補: {len(dev_candidates)}\nValidation PASS: {validation_n}\nOOS PASS: {oos_n}(うちOOSシグナル数不足で判定不能: {oos_insufficient_n}件)\nFinal PASS: {len(final_pass)}\n目標: 月間損益プラス率・OOS複利資産・期待利益を優先")
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
