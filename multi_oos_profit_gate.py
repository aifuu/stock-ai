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
PURGE_DAYS = int(os.getenv("WF_PURGE_DAYS", "7"))
EMBARGO_DAYS = int(os.getenv("WF_EMBARGO_DAYS", "7"))

# 診断専用。ゲート条件・探索空間は変更しない。
DIAG_UP = 45
DIAG_SCORE = 50
DIAG_FILE = Path(os.getenv("WF_MULTI_DIAG_FILE", "multi_oos_fold_funnel.csv"))

# adversarial_strategy_validator.py / build_strategy_policy.pyと同じ外れ値免除ロジック。
# 各Foldは既にこの免除込みでoos_pass=Trueと判定済みだが、aggregate()側で
# oos_validation_pf_ratio(=4Fold中のmin)だけを生の0.60閾値で再チェックすると、
# 免除で正しく通ったFoldの分まで巻き込んで機械的に落としてしまうため揃える。
MAX_VALIDATION_PF_FOR_RATIO = float(os.getenv("WF_MAX_VALIDATION_PF_FOR_RATIO", "10.0"))


def _purge_embargo(dates_before, dates_after, purge_days, embargo_days):
    purged = dates_before[:-purge_days] if purge_days > 0 and len(dates_before) > purge_days else dates_before
    embargoed = dates_after[embargo_days:] if embargo_days > 0 and len(dates_after) > embargo_days else dates_after
    return purged, embargoed


def fold_funnel(fold_no, end_date, all_candidates):
    """Foldごとの候補ファネルを診断する。

    adversarial_strategy_validator.py の期間分割と同じ考え方で、
    UP -> UP>DOWN -> FLAT -> SCORE -> NIKKEI の各段階を可視化する。
    ここではゲート条件や閾値を変更せず、原因調査用の集計だけを行う。
    """
    start_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp(end_date)
    c = all_candidates[
        (all_candidates["date"] >= start_date)
        & (all_candidates["date"] <= end_date)
    ].copy()
    all_dates = sorted(c["date"].drop_duplicates().tolist())

    if len(all_dates) <= OOS_DAYS:
        print(
            f"  ⚠ Fold {fold_no}: 候補日数({len(all_dates)}) <= OOS_DAYS({OOS_DAYS}) "
            "のためファネル計算をスキップ"
        )
        return []

    oos_dates = all_dates[-OOS_DAYS:]
    pre_oos = all_dates[:-OOS_DAYS]
    split = int(len(pre_oos) * 0.60)
    dev_dates_raw = pre_oos[:split]
    validation_dates_raw = pre_oos[split:]
    dev_dates, validation_dates_raw = _purge_embargo(
        dev_dates_raw, validation_dates_raw, PURGE_DAYS, EMBARGO_DAYS
    )
    validation_dates, oos_dates = _purge_embargo(
        validation_dates_raw, oos_dates, PURGE_DAYS, EMBARGO_DAYS
    )

    rows = []
    print(f"  ── Fold {fold_no} ファネル診断（閾値・ゲート変更なし） ──")
    for phase_name, phase_dates in [
        ("DEV", dev_dates),
        ("VALIDATION", validation_dates),
        ("OOS", oos_dates),
    ]:
        phase_df = c[c["date"].isin(phase_dates)].copy()
        n0 = len(phase_df)
        x1 = phase_df[phase_df["up_prob"] >= DIAG_UP]
        n1 = len(x1)
        x2 = x1[x1["up_prob"] > x1["down_prob"]]
        n2 = len(x2)
        x3 = x2[x2["flat_prob"] < 50]
        n3 = len(x3)
        x4 = x3[x3["score"] >= DIAG_SCORE]
        n4 = len(x4)
        n5 = int(x4["nikkei_uptrend"].sum())

        rows.append({
            "fold": fold_no,
            "phase": phase_name,
            "phase_start": str(pd.Timestamp(phase_dates[0]).date()) if phase_dates else "",
            "phase_end": str(pd.Timestamp(phase_dates[-1]).date()) if phase_dates else "",
            "phase_days": len(phase_dates),
            "candidate_rows": n0,
            f"UP{DIAG_UP}_plus": n1,
            "UP_excluded": n0 - n1,
            "UP_gt_DOWN": n2,
            "UP_gt_DOWN_excluded": n1 - n2,
            "FLAT_lt_50": n3,
            "FLAT_excluded": n2 - n3,
            f"SCORE{DIAG_SCORE}_plus": n4,
            "SCORE_excluded": n3 - n4,
            "NIKKEI_uptrend_reference": n5,
        })

        print(
            f"    {phase_name:10s} {len(phase_dates):4d}日 | 生候補 {n0:7,d}"
            f" → UP{DIAG_UP}+ {n1:7,d} (-{n0-n1:,})"
            f" → UP>DOWN {n2:7,d} (-{n1-n2:,})"
            f" → FLAT<50 {n3:7,d} (-{n2-n3:,})"
            f" → SCORE{DIAG_SCORE}+ {n4:7,d} (-{n3-n4:,})"
            f" | NIKKEI一致 {n5:,}"
        )

    return rows


def run_fold(fold_no, end_date, all_candidates):
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

    # 最初に診断を出す。失敗しても本体のゲート判定は変えない。
    funnel_rows = fold_funnel(fold_no, end_date, all_candidates)

    subprocess.run(
        [sys.executable, "adversarial_strategy_validator.py"],
        env=env,
        check=True,
    )

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

    # OOS期待利益はprofit_objectiveの5%を担う正式な評価値。
    # 欠落時に .get(..., 0) で黙って0にせず、非空のOOS結果では必須列として検証する。
    oos_path = fold_dir / "adversarial_oos_results.csv"
    oos_df = pd.read_csv(oos_path)
    if not oos_df.empty:
        if "oos_expected_value" not in oos_df.columns:
            raise RuntimeError(
                f"Fold {fold_no}: adversarial_oos_results.csv に oos_expected_value がありません。"
                " profit_objective の期待利益5%が無効化されるため停止します。"
            )
        values = pd.to_numeric(oos_df["oos_expected_value"], errors="coerce")
        if values.isna().any():
            bad = int(values.isna().sum())
            raise RuntimeError(
                f"Fold {fold_no}: oos_expected_value に数値化できない値が {bad} 件あります。"
            )
        print(
            f"  ✅ Fold {fold_no}: oos_expected_value 検証OK "
            f"(min={values.min():+.6f}%, max={values.max():+.6f}%)"
        )
    else:
        print(
            f"  ℹ Fold {fold_no}: OOS結果0件のため oos_expected_value の実値検証は対象外"
        )

    # Fold単位の最終候補が0件でも、診断用に0件として明示的に保存する。
    final_path = fold_dir / "adversarial_final_candidates.csv"
    df = pd.read_csv(final_path)
    if not df.empty:
        if "oos_expected_value" not in df.columns:
            raise RuntimeError(
                f"Fold {fold_no}: Final candidates に oos_expected_value がありません。"
            )
        final_values = pd.to_numeric(df["oos_expected_value"], errors="coerce")
        if final_values.isna().any():
            raise RuntimeError(
                f"Fold {fold_no}: Final candidates の oos_expected_value に欠損/非数値があります。"
            )
        df["fold"] = fold_no
        df["oos_end"] = str(end_date.date())
        df.to_csv(final_path, index=False, encoding="utf-8-sig")
        print(f"  ✅ Fold {fold_no}: Final PASS = {len(df)}")
    else:
        print(f"  ⚠ Fold {fold_no}: Final PASS = 0")

    # 各Foldのファネルを個別保存して、後から比較可能にする。
    if funnel_rows:
        pd.DataFrame(funnel_rows).to_csv(
            fold_dir / "fold_funnel.csv", index=False, encoding="utf-8-sig"
        )

    return df


def weighted_average(rows, value_col, weight_col="oos_signals", default=0.0):
    vals = pd.to_numeric(rows.get(value_col, pd.Series(dtype=float)), errors="coerce")
    weights = pd.to_numeric(rows.get(weight_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
    mask = vals.notna() & weights.gt(0)
    if not mask.any():
        return default
    return float(np.average(vals[mask], weights=weights[mask]))


def aggregate(fold_frames):
    usable = [
        x for x in fold_frames
        if x is not None and not x.empty and "strategy" in x.columns
    ]
    if len(usable) != FOLDS:
        print(f"⚠️ 全{FOLDS} FoldのFinal PASSが揃っていません: {len(usable)}/{FOLDS}")
        return pd.DataFrame()

    # profit_objectiveで使用する評価値は、全Foldで必須。
    # 以前の p.get("oos_expected_value", 0) のような黙った0補完は行わない。
    required_oos_metric_cols = [
        "oos_expected_value",
        "oos_avg_return",
        "oos_pf",
        "oos_monthly_positive_ratio",
        "oos_monthly_plus5_ratio",
        "oos_compound_return",
    ]
    for fold_no, df in enumerate(usable, 1):
        missing = [c for c in required_oos_metric_cols if c not in df.columns]
        if missing:
            raise RuntimeError(
                f"Fold {fold_no}: Final candidates の必須OOS評価列が不足: "
                + ", ".join(missing)
            )
        for col in required_oos_metric_cols:
            values = pd.to_numeric(df[col], errors="coerce")
            if values.isna().any():
                raise RuntimeError(
                    f"Fold {fold_no}: {col} に欠損/非数値があります。"
                )

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

        out["validation_signals"] = int(sum(float(p.get("validation_signals", 0)) for p in parts))
        out["validation_avg_month_return"] = float(min(
            float(p.get("validation_avg_month_return", 0)) for p in parts
        ))
        out["validation_avg_return"] = float(min(float(p.get("validation_avg_return", 0)) for p in parts))
        out["validation_pf"] = float(min(float(p.get("validation_pf", 0)) for p in parts))
        out["validation_dd"] = float(min(float(p.get("validation_dd", 0)) for p in parts))

        oos_signals = [float(p.get("oos_signals", 0)) for p in parts]
        out["oos_signals"] = int(sum(oos_signals))
        out["oos_avg_return"] = float(min(float(p.get("oos_avg_return", 0)) for p in parts))
        out["oos_pf"] = float(min(float(p.get("oos_pf", 0)) for p in parts))
        out["oos_dd"] = float(min(float(p.get("oos_dd", 0)) for p in parts))
        out["oos_validation_pf_ratio"] = float(min(
            float(p.get("oos_validation_pf_ratio", p.get("oos_pf_ratio", 0))) for p in parts
        ))
        out["oos_monthly_positive_ratio"] = float(min(
            float(p.get("oos_monthly_positive_ratio", 0)) for p in parts
        ))
        out["oos_monthly_plus5_ratio"] = float(min(
            float(p.get("oos_monthly_plus5_ratio", 0)) for p in parts
        ))
        out["oos_avg_month_return"] = float(min(float(p.get("oos_avg_month_return", 0)) for p in parts))
        out["oos_avg_month_profit_jpy"] = float(min(
            float(p.get("oos_avg_month_profit_jpy", 0)) for p in parts
        ))
        out["oos_worst_month_return"] = float(min(float(p.get("oos_worst_month_return", 0)) for p in parts))
        out["oos_expected_value"] = float(min(float(p["oos_expected_value"]) for p in parts))

        compound = 1.0
        for p in parts:
            compound *= 1.0 + float(p.get("oos_compound_return", 0)) / 100.0
        out["oos_compound_return"] = (compound - 1.0) * 100.0
        out["oos_compound_final_capital"] = INITIAL_CAPITAL * compound

        out["sizing"] = float(min(float(p.get("sizing", 0)) for p in parts))
        out["prob_10y"] = float(min(float(p.get("prob_10y", 0)) for p in parts))
        out["prob_15y"] = float(min(float(p.get("prob_15y", 0)) for p in parts))
        out["prob_20y"] = float(min(float(p.get("prob_20y", 0)) for p in parts))
        out["bankruptcy_prob"] = float(max(float(p.get("bankruptcy_prob", 100)) for p in parts))
        out["p90_max_dd"] = float(max(float(p.get("p90_max_dd", 100)) for p in parts))

        out["profit_objective"] = (
            np.clip(out["oos_avg_month_return"], -20, 20) * 0.30
            + out["oos_monthly_plus5_ratio"] * 0.20
            + np.clip(out["oos_compound_return"], -100, 1000) * 0.25
            + np.clip(out["oos_avg_return"], -5, 5) * 10.0 * 0.15
            + np.clip(out["oos_pf"], 0, 8) * 5.0 * 0.07
            - np.clip(-out["oos_dd"], 0, 100) * 0.03
        )
        out["up_threshold"] = out.get("up_threshold", out.get("up"))
        out["score_threshold"] = out.get("score_threshold", out.get("score"))
        out["nikkei_filter"] = out.get("nikkei_filter", out.get("nikkei"))
        out["tp_multiplier"] = out.get("tp_multiplier", out.get("tp"))
        out["sl_multiplier"] = out.get("sl_multiplier", out.get("sl"))
        out["hold_days"] = out.get("hold_days", out.get("hold"))
        rows.append(out)

    result = pd.DataFrame(rows)
    ratio_ok = (
        (result["oos_validation_pf_ratio"] >= 0.60)
        | (result["validation_pf"] > MAX_VALIDATION_PF_FOR_RATIO)
    )
    result = result[
        (result["oos_signals"] >= 20)
        & (result["oos_pf"] >= 1.0)
        & (result["oos_avg_return"] > 0)
        & ratio_ok
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

    required_candidate_cols = [
        "date", "up_prob", "down_prob", "flat_prob", "score", "nikkei_uptrend"
    ]
    missing = [x for x in required_candidate_cols if x not in candidates.columns]
    if missing:
        raise RuntimeError("候補CSVの診断用列が不足: " + ", ".join(missing))

    candidates["date"] = pd.to_datetime(candidates["date"], errors="coerce").dt.normalize()
    for col in ["up_prob", "down_prob", "flat_prob", "score"]:
        candidates[col] = pd.to_numeric(candidates[col], errors="coerce")
    candidates["nikkei_uptrend"] = candidates["nikkei_uptrend"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )
    candidates = candidates.dropna(subset=required_candidate_cols).copy()

    dates = sorted(candidates["date"].dropna().unique())
    required_days = OOS_DAYS * FOLDS
    if len(dates) < required_days + 100:
        raise RuntimeError(
            f"複数OOSに必要な履歴不足: {len(dates)}営業日 / 必要目安 {required_days + 100}"
        )

    # 非重複の4つのOOS窓。Fold 4が最新。
    end_dates = [
        dates[-1 - OOS_DAYS * (FOLDS - 1 - i)]
        for i in range(FOLDS)
    ]

    print("=" * 90)
    print("🛡️ MULTI-OOS PROFIT GATE")
    print(f"Fold数: {FOLDS} / 各OOS: {OOS_DAYS}営業日 / TOP_N: {TOP_N}")
    print(f"候補日数: {len(dates)} / 必要目安: {required_days + 100}")
    print("診断: 条件を緩めず、各Foldの候補ファネルを同時保存")
    print("期待利益: oos_expected_value を全Fold必須検証。欠落時は黙って0にしない")
    print("=" * 90)

    fold_frames = []
    all_funnel_rows = []
    for i, end_date in enumerate(end_dates, 1):
        fold_df = run_fold(i, pd.Timestamp(end_date), candidates)
        fold_frames.append(fold_df)
        fold_funnel_rows = fold_funnel(i, pd.Timestamp(end_date), candidates)
        all_funnel_rows.extend(fold_funnel_rows)

    # ファネル診断をルートにも保存。二重計算だが、成果物を確実に残すため明示的に保存する。
    pd.DataFrame(all_funnel_rows).to_csv(
        DIAG_FILE, index=False, encoding="utf-8-sig"
    )
    print(f"📁 {DIAG_FILE} を保存しました")

    final = aggregate(fold_frames)
    final.to_csv("adversarial_final_candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({
        "fold": range(1, FOLDS + 1),
        "oos_end": [str(x.date()) for x in end_dates],
        "oos_days": OOS_DAYS,
        "final_pass_count": [len(x) if x is not None else 0 for x in fold_frames],
    }).to_csv("multi_oos_folds.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("🛡️ MULTI-OOS PROFIT GATE RESULT")
    print(f"Fold数: {FOLDS} / 各OOS: {OOS_DAYS}営業日")
    print("OOS期間:")
    for i, d in enumerate(end_dates, 1):
        print(f"  Fold {i}: 終了 {pd.Timestamp(d).date()} / Final PASS {len(fold_frames[i-1])}")
    print(f"全Fold PASS戦略: {len(final)}")
    if final.empty:
        print("⏸ 複数OOSで利益が残る共通戦略なし → APPROVEDにしません")
        return 0

    print("🏆 採用候補:")
    for _, r in final.head(10).iterrows():
        print(
            f"  {r['strategy']} | 月間+率 {r['oos_monthly_positive_ratio']:.1f}%"
            f" | 複利 {r['oos_compound_return']:+.2f}%"
            f" | PF {r['oos_pf']:.2f} | DD {r['oos_dd']:.2f}%"
            f" | 期待利益 {r['oos_expected_value']:+.3f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
