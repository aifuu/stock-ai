from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from regime_profit_core import (
    build_expectancy_table,
    enrich_candidate_frame,
    individual_strength,
    rank_candidates,
    strength_bucket,
)

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
FOLD_FILE = "adversarial_multi_oos_folds.csv"
OOS_FILE = "adversarial_oos_results.csv"
FINAL_FILE = "adversarial_final_candidates.csv"
START = pd.Timestamp(os.getenv("WF_START_DATE", "2021-01-01"))
END = pd.Timestamp(os.getenv("WF_END_DATE", "2026-08-22"))
FOLDS = int(os.getenv("WF_OOS_FOLDS", "4"))
OOS_DAYS = int(os.getenv("WF_OOS_DAYS", "252"))
TOP_N = 1
PURGE = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO = int(os.getenv("WF_EMBARGO_DAYS", "7"))
CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))
MIN_TRADES = int(os.getenv("WF_MIN_TOTAL_OOS_TRADES", "20"))
MIN_MONTHLY = float(os.getenv("WF_MIN_MONTHLY_POSITIVE_RATIO", "0.55")) * 100.0
MAX_DD = float(os.getenv("WF_MAX_OOS_DD", "35"))
MIN_POSITIVE = int(os.getenv("WF_MIN_POSITIVE_FOLDS", "3"))
STRATEGY = "REGIME_EXPECTED_RETURN_TOP1"


def stat(x: pd.DataFrame) -> dict:
    if x.empty:
        return {
            "signals": 0, "avg_return": 0.0, "expected_value": 0.0,
            "pf": 0.0, "dd": 0.0, "monthly_positive_ratio": 0.0,
            "compound_return": 0.0, "final_capital": CAPITAL,
        }
    z = x[["date", "return"]].copy().sort_values("date")
    z["return"] = pd.to_numeric(z["return"], errors="coerce")
    z = z.dropna(subset=["return"])
    daily = z.groupby("date")["return"].mean()
    eq = (1.0 + daily / 100.0).cumprod()
    gains = float(z.loc[z["return"] > 0, "return"].sum())
    losses = float(-z.loc[z["return"] < 0, "return"].sum())
    pf = gains / losses if losses else (99.0 if gains else 0.0)
    monthly = daily.groupby(daily.index.to_period("M")).apply(lambda s: ((1.0 + s / 100.0).prod() - 1.0) * 100.0)
    return {
        "signals": int(len(z)),
        "avg_return": float(z["return"].mean()),
        "expected_value": float(z["return"].mean()),
        "pf": float(pf),
        "dd": float((eq / eq.cummax() - 1.0).min() * 100.0),
        "monthly_positive_ratio": float((monthly > 0).mean() * 100.0) if len(monthly) else 0.0,
        "compound_return": float((eq.iloc[-1] - 1.0) * 100.0),
        "final_capital": float(CAPITAL * eq.iloc[-1]),
    }


def add_strength_fields(x: pd.DataFrame) -> pd.DataFrame:
    x = enrich_candidate_frame(x)
    x["individual_strength"] = [
        individual_strength(u, d, s, rs)
        for u, d, s, rs in zip(x["up_prob"], x["down_prob"], x["score"], x["relative_strength"])
    ]
    x["strength_bucket"] = x["individual_strength"].map(strength_bucket)
    return x


def choose_phase(frame: pd.DataFrame, expectancy_table: dict) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    return pd.DataFrame(rank_candidates(frame.to_dict("records"), expectancy_table, top_n=TOP_N))


def gate_ok(o: dict) -> bool:
    return (
        o["signals"] >= MIN_TRADES
        and o["pf"] >= 1.0
        and o["avg_return"] > 0.0
        and o["dd"] >= -MAX_DD
        and o["compound_return"] > 0.0
        and o["monthly_positive_ratio"] >= MIN_MONTHLY
    )


def main() -> None:
    if not Path(CANDIDATE_FILE).exists():
        raise RuntimeError(f"{CANDIDATE_FILE} がありません")

    c = pd.read_csv(CANDIDATE_FILE)
    c["date"] = pd.to_datetime(c["date"], errors="coerce").dt.normalize()
    c = c[(c["date"] >= START) & (c["date"] <= END)].dropna(subset=["date"]).copy()
    c = add_strength_fields(c)
    dates = sorted(c["date"].drop_duplicates().tolist())
    if len(dates) < FOLDS * OOS_DAYS + 120:
        raise RuntimeError(f"複数OOSに必要な営業日が不足: {len(dates)}")

    fold_rows = []
    last_oos_table = {"global": 0.0, "regimes": {}, "groups": {}}
    last_validation = stat(pd.DataFrame())

    for fold in range(1, FOLDS + 1):
        end_pos = len(dates) - (FOLDS - fold) * OOS_DAYS
        oos_dates = dates[end_pos - OOS_DAYS:end_pos]
        pre_dates = dates[:max(0, end_pos - OOS_DAYS)]
        if len(pre_dates) <= PURGE + EMBARGO + 120:
            raise RuntimeError(f"Fold {fold}: 過去学習期間不足")

        # Purge/embargo: OOS直前の期間を期待値キャリブレーションへ混ぜない。
        pre_dates = pre_dates[:max(0, len(pre_dates) - PURGE - EMBARGO)]
        cut = int(len(pre_dates) * 0.60)
        calibration_dates = pre_dates[:max(1, cut // 2)]
        dev_dates = pre_dates[cut // 2:cut]
        validation_dates = pre_dates[cut:]

        cal = c[c.date.isin(calibration_dates)]
        dev = c[c.date.isin(dev_dates)]
        val = c[c.date.isin(validation_dates)]
        oo = c[c.date.isin(oos_dates)]

        # 各フェーズの選定表は、そのフェーズより前のデータだけで作る。
        cal_table = build_expectancy_table(cal)
        dev_selected = choose_phase(dev, cal_table)

        val_table = build_expectancy_table(pd.concat([cal, dev], ignore_index=True))
        val_selected = choose_phase(val, val_table)

        oos_table = build_expectancy_table(pd.concat([cal, dev, val], ignore_index=True))
        oos_selected = choose_phase(oo, oos_table)
        last_oos_table = oos_table
        last_validation = stat(val_selected)

        d = stat(dev_selected)
        v = last_validation
        o = stat(oos_selected)
        passed = gate_ok(o)
        fold_rows.append({
            "fold": fold,
            "strategy": STRATEGY,
            "up": 0,
            "score": 0,
            "nikkei": False,
            "tp": 3.0,
            "sl": 1.5,
            "hold": 5,
            **{f"dev_{k}": value for k, value in d.items()},
            **{f"validation_{k}": value for k, value in v.items()},
            **{f"oos_{k}": value for k, value in o.items()},
            "oos_pass": passed,
            "regime_expectancy_json": json.dumps(oos_table, ensure_ascii=False, sort_keys=True),
        })
        print(
            f"Fold {fold}: OOS={o['signals']} | PF={o['pf']:.2f} | "
            f"EV={o['expected_value']:+.3f}% | {'PASS' if passed else 'FAIL'}"
        )

    folds_df = pd.DataFrame(fold_rows)
    folds_df.to_csv(FOLD_FILE, index=False, encoding="utf-8-sig")
    folds_df.to_csv(OOS_FILE, index=False, encoding="utf-8-sig")

    # 既存の4-foldゲート条件を変更しない。
    positive = int((folds_df["oos_compound_return"] > 0).sum())
    total = int(pd.to_numeric(folds_df["oos_signals"], errors="coerce").sum())
    pf_min = float(pd.to_numeric(folds_df["oos_pf"], errors="coerce").replace(np.inf, 99).min())
    avg_min = float(pd.to_numeric(folds_df["oos_avg_return"], errors="coerce").min())
    monthly_min = float(pd.to_numeric(folds_df["oos_monthly_positive_ratio"], errors="coerce").min())
    dd_worst = float(pd.to_numeric(folds_df["oos_dd"], errors="coerce").min())
    compound = float(np.prod(1.0 + pd.to_numeric(folds_df["oos_compound_return"], errors="coerce") / 100.0))
    compound_return = (compound - 1.0) * 100.0
    final_pass = (
        positive >= MIN_POSITIVE
        and total >= MIN_TRADES
        and pf_min >= 1.0
        and avg_min > 0.0
        and monthly_min >= MIN_MONTHLY
        and dd_worst >= -MAX_DD
        and compound_return > 0.0
    )

    if final_pass:
        final = pd.DataFrame([{
            "strategy": STRATEGY,
            "final_status": "PASS",
            "up_threshold": 0,
            "score_threshold": 0,
            "nikkei_filter": False,
            "tp_multiplier": 3.0,
            "sl_multiplier": 1.5,
            "hold_days": 5,
            "validation_signals": int(folds_df["validation_signals"].sum()),
            "validation_win_rate": 0.0,
            "validation_avg_return": float(folds_df["validation_avg_return"].min()),
            "validation_pf": float(folds_df["validation_pf"].min()),
            "validation_dd": float(folds_df["validation_dd"].min()),
            "oos_signals": total,
            "oos_win_rate": 0.0,
            "oos_avg_return": avg_min,
            "oos_pf": pf_min,
            "oos_dd": dd_worst,
            "oos_validation_pf_ratio": 1.0,
            "oos_monthly_positive_ratio": monthly_min,
            "oos_compound_return": compound_return,
            "oos_compound_final_capital": CAPITAL * compound,
            "oos_expected_value": float(folds_df["oos_expected_value"].min()),
            "multi_oos_folds": FOLDS,
            "positive_oos_folds": positive,
            "selection_mode": "REGIME_EXPECTED_RETURN_TOP1",
            "selection_version": "regime_profit_core_v1",
            "regime_expectancy_json": json.dumps(last_oos_table, ensure_ascii=False, sort_keys=True),
            "sizing": None,
            "prob_10y": None,
            "prob_15y": None,
            "prob_20y": None,
            "bankruptcy_prob": None,
            "p90_max_dd": None,
            "profit_objective": (
                monthly_min * 0.40
                + np.clip(compound_return, -100, 1000) * 0.30
                + np.clip(pf_min, 0, 8) * 5.0 * 0.15
                + np.clip(avg_min, -5, 5) * 10.0 * 0.10
                + np.clip(float(folds_df["oos_expected_value"].min()), -5, 5) * 10.0 * 0.05
            ),
        }])
    else:
        final = pd.DataFrame(columns=["strategy", "final_status", "oos_signals"])

    final.to_csv(FINAL_FILE, index=False, encoding="utf-8-sig")
    print(
        f"4-FOLD結果: {'PASS' if final_pass else 'FAIL'} | "
        f"positive={positive}/{FOLDS} | total={total} | PF(min)={pf_min:.2f} | "
        f"EV(min)={avg_min:+.3f}% | monthly(min)={monthly_min:.1f}% | "
        f"compound={compound_return:+.2f}%"
    )


if __name__ == "__main__":
    main()
