"""Read-only preview summarizer for the OOS accounting (new vs legacy) comparison.

This script is part of a temporary preview workflow only. It never writes to
strategy_policy.json / strategy_policy_up.json / policy_manual_overrides.json,
never computes or reads AI_POLICY_SIGNING_SECRET, and never commits anything.
It only reads CSV artifacts already produced by the unmodified
multi_oos_profit_gate.py / adversarial_strategy_validator.py pipeline and
writes a JSON + Markdown summary for later comparison between accounting
modes (WF_LEGACY_OVERLAP=0 vs =1).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

ACCOUNTING = os.environ.get("PREVIEW_ACCOUNTING", "unknown")
FOLDS = int(os.environ.get("WF_MULTI_OOS_FOLDS", "4"))
OUT_DIR = Path("multi_oos_results")

# Current production policy parameter combinations, named exactly as
# adversarial_strategy_validator.py builds the "strategy" column:
# f"UP{up}_SCORE{score}_NIKKEI{'ON' if nikkei else 'OFF'}_TP{tp}_SL{sl}_H{hold}"
H1_NAME = "UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1"  # strategy_policy.json
H3_NAME = "UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3"  # strategy_policy_up.json
TARGET_STRATEGIES = {"H1": H1_NAME, "H3": H3_NAME}


def _read_csv(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"WARN: failed to read {path}: {e}", file=sys.stderr)
        return pd.DataFrame()


def _row_for_strategy(df, name):
    if df.empty or "strategy" not in df.columns:
        return None
    hit = df[df["strategy"] == name]
    if hit.empty:
        return None
    return hit.iloc[0].to_dict()


def _clean(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        try:
            if pd.isna(v):
                out[k] = None
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(v, (pd.Timestamp,)):
            out[k] = str(v)
        elif isinstance(v, (bool,)):
            out[k] = bool(v)
        elif isinstance(v, (int,)):
            out[k] = int(v)
        elif isinstance(v, (float,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def combo_lookup(fold_dir):
    validation_df = _read_csv(fold_dir / "adversarial_validation_results.csv")
    oos_df = _read_csv(fold_dir / "adversarial_oos_results.csv")
    fold_final_df = _read_csv(fold_dir / "adversarial_final_candidates.csv")

    result = {}
    for key, name in TARGET_STRATEGIES.items():
        vrow = _row_for_strategy(validation_df, name)
        entry = {
            "in_dev_selected": vrow is not None,
            "validation": None,
            "validation_pass": None,
            "oos": None,
            "oos_pass": None,
            "in_fold_final": False,
            "note": None,
        }
        if vrow is None:
            entry["note"] = "DEV段階のstability上位候補(top<=50)に選ばれず、Validationに進んでいない"
            result[key] = entry
            continue
        entry["validation"] = _clean({k: v for k, v in vrow.items() if k.startswith("validation_")})
        entry["validation_pass"] = bool(vrow.get("validation_pass", False))
        if not entry["validation_pass"]:
            entry["note"] = "Validationゲート不合格のためOOSへ進んでいない"
            result[key] = entry
            continue
        orow = _row_for_strategy(oos_df, name)
        if orow is None:
            entry["note"] = "Validation合格したがOOS結果行が見つからない(想定外)"
            result[key] = entry
            continue
        entry["oos"] = _clean({k: v for k, v in orow.items() if k.startswith("oos_")})
        entry["oos_pass"] = bool(orow.get("oos_pass", False))
        entry["in_fold_final"] = _row_for_strategy(fold_final_df, name) is not None
        if not entry["oos_pass"]:
            entry["note"] = "OOSゲート不合格"
        result[key] = entry
    return result


def main():
    folds_summary = []
    for i in range(1, FOLDS + 1):
        fold_dir = OUT_DIR / f"fold_{i}"
        validation_df = _read_csv(fold_dir / "adversarial_validation_results.csv")
        oos_df = _read_csv(fold_dir / "adversarial_oos_results.csv")
        final_df = _read_csv(fold_dir / "adversarial_final_candidates.csv")
        funnel = {
            "fold": i,
            "exists": fold_dir.exists(),
            "dev_selected": int(len(validation_df)) if not validation_df.empty else 0,
            "validation_pass": int(validation_df["validation_pass"].sum()) if "validation_pass" in validation_df.columns else 0,
            "oos_evaluated": int(len(oos_df)) if not oos_df.empty else 0,
            "oos_pass": int(oos_df["oos_pass"].sum()) if "oos_pass" in oos_df.columns else 0,
            "fold_final": int(len(final_df)) if not final_df.empty else 0,
            "combos": combo_lookup(fold_dir),
        }
        folds_summary.append(funnel)

    agg_final_df = _read_csv(Path("adversarial_final_candidates.csv"))
    aggregated_final_rows = agg_final_df.to_dict("records") if not agg_final_df.empty else []
    aggregated_final_rows = [_clean(r) for r in aggregated_final_rows]

    agg_lookup = {}
    for key, name in TARGET_STRATEGIES.items():
        row = _row_for_strategy(agg_final_df, name)
        agg_lookup[key] = _clean(row) if row is not None else None

    summary = {
        "accounting": ACCOUNTING,
        "folds": folds_summary,
        "aggregated_final_count": len(aggregated_final_rows),
        "aggregated_final_candidates": aggregated_final_rows,
        "aggregated_combo_result": agg_lookup,
    }

    Path(f"preview_summary_{ACCOUNTING}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [f"# OOS accounting preview summary ({ACCOUNTING})", ""]
    lines.append("## Fold funnel (DEV-selected -> Validation PASS -> OOS evaluated -> OOS PASS -> Fold Final)")
    lines.append("")
    lines.append("| fold | dev_selected | validation_pass | oos_evaluated | oos_pass | fold_final |")
    lines.append("|---|---|---|---|---|---|")
    for f in folds_summary:
        lines.append(f"| {f['fold']} | {f['dev_selected']} | {f['validation_pass']} | {f['oos_evaluated']} | {f['oos_pass']} | {f['fold_final']} |")
    lines.append("")
    lines.append(f"## Aggregated (multi-fold) final approved candidates: {len(aggregated_final_rows)}")
    for r in aggregated_final_rows:
        lines.append(f"- {r.get('strategy')}: oos_positive_folds={r.get('oos_positive_folds')} oos_compound_return={r.get('oos_compound_return')} oos_pf={r.get('oos_pf')} oos_dd={r.get('oos_dd')}")
    lines.append("")
    for key, name in TARGET_STRATEGIES.items():
        lines.append(f"## {key} ({name}) per fold")
        for f in folds_summary:
            c = f["combos"][key]
            lines.append(f"- fold {f['fold']}: in_dev_selected={c['in_dev_selected']} validation_pass={c['validation_pass']} oos_pass={c['oos_pass']} in_fold_final={c['in_fold_final']} note={c['note']}")
        agg_row = agg_lookup.get(key)
        lines.append(f"  aggregated_final_present={agg_row is not None}")
        lines.append("")

    Path(f"preview_summary_{ACCOUNTING}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
