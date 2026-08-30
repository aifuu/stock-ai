import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATE_FILE = os.getenv("WF_CANDIDATE_FILE", "walk_forward_all_candidates.csv")
OOS_DAYS = int(os.getenv("WF_MULTI_OOS_DAYS", "252"))
FOLDS = int(os.getenv("WF_MULTI_OOS_FOLDS", "4"))
TOP_N = int(os.getenv("WF_TOP_N", "10"))
START_DATE = os.getenv("WF_START_DATE", "2021-01-01")
END_DATE = os.getenv("WF_END_DATE", "2026-08-22")
INITIAL_CAPITAL = float(os.getenv("WF_INITIAL_CAPITAL", "1000000"))
OUT_DIR = Path(os.getenv("WF_MULTI_OUT_DIR", "multi_oos_results"))


def run_fold(fold_no, end_date):
    fold_dir = OUT_DIR / f"fold_{fold_no}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "WF_START_DATE": START_DATE,
        "WF_END_DATE": str(end_date.date()),
        "WF_OOS_DAYS": str(OOS_DAYS),
        "WF_TOP_N": str(TOP_N),
        "WF_INITIAL_CAPITAL": str(INITIAL_CAPITAL),
    })
    print("\n" + "=" * 90)
    print(f"FOLD {fold_no}/{FOLDS}  学習/Validation → OOS")
    print(f"OOS終了: {end_date.date()}  OOS営業日: {OOS_DAYS}")
    print("=" * 90)
    subprocess.run([sys.executable, "adversarial_strategy_validator.py"], env=env, check=True)

    required_outputs = [
        "adversarial_final_candidates.csv",
        "adversarial_validation_results.csv",
        "adversarial_oos_results.csv",
    ]
    for name in required_outputs:
        src = Path(name)
        if not src.exists():
            raise RuntimeError(f"Fold {fold_no}: {name} が生成されませんでした")
        shutil.copy2(src, fold_dir / name)

    df = pd.read_csv(fold_dir / "adversarial_final_candidates.csv")
    if df.empty:
        return df
    df["fold"] = fold_no
    df["oos_end"] = str(end_date.date())
    return df


def weighted_average(rows, value_col, weight_col="oos_signals", default=0.0):
    vals = pd.to_numeric(rows.get(value_col, pd.Series(dtype=float)), errors="coerce")
    weights = pd.to_numeric(rows.get(weight_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    mask = vals.notna() & weights.gt(0)
    if not mask.any():
        return default
    return float(np.average(vals[mask], weights=weights[mask]))


def aggregate(fold_frames):
    usable = [x for x in fold_frames if x is not None and not x.empty and "strategy" in x.columns]
    if len(usable) != FOLDS:
        print(f"⚠️ 全{FOLDS} FoldのFinal PASSが揃っていません: {len(usable)}/{FOLDS}")
        return pd.DataFrame()

    common = set(usable[0]["strategy"].astype(str))
    for df in usable[1:]:
        common &= set(df["strategy"].astype(str))

    print(f"\n共通戦略（全{FOLDS} OOS PASS）: {len(common)}")
    if not common:
        return pd.DataFrame()

    rows = []
    for strategy in sorted(common):
        parts = [df[df["strategy"].astype(str) == strategy].iloc[0] for df in usable]
        first = parts[0]
        out = first.to_dict()
        out["multi_oos_folds"] = FOLDS
        out["multi_oos_pass"] = True
        out["final_status"] = "PASS"

        # Validationは全Foldで崩れないことを優先。
        out["validation_signals"] = int(sum(float(p.get("validation_signals", 0)) for p in parts))
        out["validation_win_rate"] = weighted_average(pd.DataFrame([p.to_dict() for p in parts]), "validation_win_rate", "validation_signals")
        out["validation_avg_return"] = float(min(float(p.get("validation_avg_return", 0)) for p in parts))
        out["validation_pf"] = float(min(float(p.get("validation_pf", 0)) for p in parts))
        out["validation_dd"] = float(min(float(p.get("validation_dd", 0)) for p in parts))

        # 4つの独立OOSを時系列順に複利連結。最弱FoldのPF/DD/期待値も保持。
        oos_signals = [float(p.get("oos_signals", 0)) for p in parts]
        out["oos_signals"] = int(sum(oos_signals))
        out["oos_win_rate"] = weighted_average(pd.DataFrame([p.to_dict() for p in parts]), "oos_win_rate", "oos_signals")
        out["oos_avg_return"] = float(min(float(p.get("oos_avg_return", 0)) for p in parts))
        out["oos_pf"] = float(min(float(p.get("oos_pf", 0)) for p in parts))
        out["oos_dd"] = float(min(float(p.get("oos_dd", 0)) for p in parts))
        out["oos_validation_pf_ratio"] = float(min(float(p.get("oos_validation_pf_ratio", p.get("oos_pf_ratio", 0))) for p in parts))
        out["oos_monthly_positive_ratio"] = float(min(float(p.get("oos_monthly_positive_ratio", 0)) for p in parts))
        out["oos_avg_month_return"] = float(min(float(p.get("oos_avg_month_return", 0)) for p in parts))
        out["oos_worst_month_return"] = float(min(float(p.get("oos_worst_month_return", 0)) for p in parts))
        out["oos_expected_value"] = float(min(float(p.get("oos_expected_value", 0)) for p in parts))

        compound = 1.0
        for p in parts:
            compound *= 1.0 + float(p.get("oos_compound_return", 0)) / 100.0
        out["oos_compound_return"] = (compound - 1.0) * 100.0
        out["oos_compound_final_capital"] = INITIAL_CAPITAL * compound

        # Monte Carloは4 Foldすべてで安全側に倒す。
        out["sizing"] = float(min(float(p.get("sizing", 0)) for p in parts))
        out["prob_10y"] = float(min(float(p.get("prob_10y", 0)) for p in parts))
        out["prob_15y"] = float(min(float(p.get("prob_15y", 0)) for p in parts))
        out["prob_20y"] = float(min(float(p.get("prob_20y", 0)) for p in parts))
        out["bankruptcy_prob"] = float(max(float(p.get("bankruptcy_prob", 100)) for p in parts))
        out["p90_max_dd"] = float(max(float(p.get("p90_max_dd", 100)) for p in parts))

        out["profit_objective"] = (
            out["oos_monthly_positive_ratio"] * 0.40
            + np.clip(out["oos_compound_return"], -100, 1000) * 0.30
            + np.clip(out["oos_pf"], 0, 8) * 5.0 * 0.15
            + np.clip(out["oos_avg_return"], -5, 5) * 10.0 * 0.10
            + np.clip(out["oos_expected_value"], -5, 5) * 10.0 * 0.05
        )
        out["up_threshold"] = out.get("up_threshold", out.get("up"))
        out["score_threshold"] = out.get("score_threshold", out.get("score"))
        out["nikkei_filter"] = out.get("nikkei_filter", out.get("nikkei"))
        out["tp_multiplier"] = out.get("tp_multiplier", out.get("tp"))
        out["sl_multiplier"] = out.get("sl_multiplier", out.get("sl"))
        out["hold_days"] = out.get("hold_days", out.get("hold"))
        rows.append(out)

    result = pd.DataFrame(rows)
    result = result[
        (result["oos_signals"] >= 20)
        & (result["oos_pf"] >= 1.0)
        & (result["oos_avg_return"] > 0)
        & (result["oos_validation_pf_ratio"] >= 0.60)
        & (result["oos_monthly_positive_ratio"] >= 55.0)
        & (result["oos_dd"] >= -35.0)
        & (result["oos_compound_return"] > 0)
        & (result["bankruptcy_prob"] < 5.0)
        & (result["p90_max_dd"] <= 30.0)
    ].copy()
    result = result.sort_values("profit_objective", ascending=False).reset_index(drop=True)
    return result


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATE_FILE)
    if candidates.empty or "date" not in candidates.columns:
        raise RuntimeError("walk_forward_all_candidates.csv が空、またはdate列がありません")
    dates = sorted(pd.to_datetime(candidates["date"], errors="coerce").dropna().dt.normalize().unique())
    required_days = OOS_DAYS * FOLDS
    if len(dates) < required_days + 100:
        raise RuntimeError(f"複数OOSに必要な履歴不足: {len(dates)}営業日 / 必要目安 {required_days + 100}")

    # 非重複の4つのOOS窓。Fold 4が最新。
    end_dates = [dates[-1 - OOS_DAYS * (FOLDS - 1 - i)] for i in range(FOLDS)]
    fold_frames = []
    for i, end_date in enumerate(end_dates, 1):
        fold_frames.append(run_fold(i, pd.Timestamp(end_date)))

    final = aggregate(fold_frames)
    final.to_csv("adversarial_final_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"fold": range(1, FOLDS + 1), "oos_end": [str(x.date()) for x in end_dates], "oos_days": OOS_DAYS}).to_csv("multi_oos_folds.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("🛡️ MULTI-OOS PROFIT GATE")
    print(f"Fold数: {FOLDS} / 各OOS: {OOS_DAYS}営業日")
    print("OOS期間:")
    for i, d in enumerate(end_dates, 1):
        print(f"  Fold {i}: 終了 {pd.Timestamp(d).date()}")
    print(f"全Fold PASS戦略: {len(final)}")
    if final.empty:
        print("⏸ 複数OOSで利益が残る共通戦略なし → APPROVEDにしません")
        return 0
    print("🏆 採用候補:")
    for _, r in final.head(10).iterrows():
        print(f"  {r['strategy']} | 月間+率 {r['oos_monthly_positive_ratio']:.1f}% | 複利 {r['oos_compound_return']:+.2f}% | PF {r['oos_pf']:.2f} | DD {r['oos_dd']:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
