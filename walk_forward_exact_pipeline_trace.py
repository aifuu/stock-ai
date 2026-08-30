#!/usr/bin/env python3
"""Run the real walk_forward.py through a temporary source wrapper and trace the exact candidate pipeline. No thresholds/policy are changed."""
from pathlib import Path
import runpy
import csv

BASE=Path(__file__).resolve().with_name("walk_forward.py")
TMP=Path(__file__).resolve().with_name(".walk_forward_exact_trace_tmp.py")
OUT=Path(__file__).resolve().with_name("walk_forward_exact_pipeline_trace.csv")
TARGETS={"2026-08-18","2026-08-19","2026-08-20"}


def patch_source(s):
    marker='candidate_history = []\n'
    injected='''candidate_history = []\n\n# The generated temporary source executes in its own runpy namespace.\n# Define the trace output path inside that namespace so OUT is always available.\n_TRACE_OUT = Path(__file__).resolve().with_name("walk_forward_exact_pipeline_trace.csv")\n_TRACE_ROWS=[]\n_TRACE_TARGETS={"2026-08-18","2026-08-19","2026-08-20"}\n'''
    if marker not in s: raise RuntimeError("candidate_history marker not found")
    s=s.replace(marker,injected,1)

    old='''    candidates = []\n\n    rows_for_batch = []\n    tickers_for_batch = []\n'''
    new='''    candidates = []\n\n    _trace_day=str(pd.Timestamp(prediction_date).date())\n    _trace_expected=len(TICKERS)\n    _trace_present=0\n    _trace_liquid=0\n    _trace_feature_valid=0\n    _trace_prediction_rows=0\n    _trace_signal_rows=0\n    _trace_score_ge_50=0\n    _trace_up_ge_50=0\n    _trace_up_gt_down=0\n    _trace_flat_lt_50=0\n\n    rows_for_batch = []\n    tickers_for_batch = []\n'''
    if old not in s: raise RuntimeError("candidate block not found")
    s=s.replace(old,new,1)

    old='''    for ticker, df in symbol_data.items():\n        if prediction_date not in df.index:\n            continue\n\n        row = df.loc[prediction_date]\n\n        if not bool(row["liquid"]):\n            continue\n\n        x_row = row[FEATURES]\n\n        if x_row.isna().any():\n            continue\n\n        rows_for_batch.append(x_row)\n        tickers_for_batch.append((ticker, row))\n'''
    new='''    for ticker, df in symbol_data.items():\n        if prediction_date not in df.index:\n            continue\n\n        _trace_present += 1\n        row = df.loc[prediction_date]\n\n        if not bool(row["liquid"]):\n            continue\n\n        _trace_liquid += 1\n        x_row = row[FEATURES]\n\n        if x_row.isna().any():\n            continue\n\n        _trace_feature_valid += 1\n        rows_for_batch.append(x_row)\n        tickers_for_batch.append((ticker, row))\n'''
    if old not in s: raise RuntimeError("input loop not found")
    s=s.replace(old,new,1)

    old='''    down_idx, flat_idx, up_idx = classes.index(0), classes.index(1), classes.index(2)\n\n    for (ticker, row), proba in zip(tickers_for_batch, proba_batch):\n'''
    new='''    down_idx, flat_idx, up_idx = classes.index(0), classes.index(1), classes.index(2)\n    _trace_prediction_rows=len(proba_batch)\n\n    for (ticker, row), proba in zip(tickers_for_batch, proba_batch):\n'''
    if old not in s: raise RuntimeError("prediction block not found")
    s=s.replace(old,new,1)

    old='''        signal = calculate_signal(\n            row,\n            up_prob,\n            down_prob,\n            flat_prob\n        )\n\n        signal.update(\n'''
    new='''        signal = calculate_signal(\n            row,\n            up_prob,\n            down_prob,\n            flat_prob\n        )\n\n        _trace_signal_rows += 1\n        if float(signal["score"]) >= 50: _trace_score_ge_50 += 1\n        if float(up_prob)*100 >= 50: _trace_up_ge_50 += 1\n        if float(up_prob) > float(down_prob): _trace_up_gt_down += 1\n        if float(flat_prob)*100 < 50: _trace_flat_lt_50 += 1\n\n        signal.update(\n'''
    if old not in s: raise RuntimeError("signal block not found")
    s=s.replace(old,new,1)

    old='''    if not candidates:\n \n        continue\n \n \n    # =====================================================\n    # AIスコア順\n'''
    new='''    if _trace_day in _TRACE_TARGETS:\n        _TRACE_ROWS.append({"date":_trace_day,"expected_tickers":_trace_expected,"symbol_data_present":_trace_present,"liquid":_trace_liquid,"feature_valid":_trace_feature_valid,"prediction_rows":_trace_prediction_rows,"signal_rows":_trace_signal_rows,"score_ge_50":_trace_score_ge_50,"up_ge_50":_trace_up_ge_50,"up_gt_down":_trace_up_gt_down,"flat_lt_50":_trace_flat_lt_50,"raw_candidates":len(candidates)})\n\n    if not candidates:\n \n        continue\n \n \n    # =====================================================\n    # AIスコア順\n'''
    if old not in s: raise RuntimeError("candidate end block not found")
    s=s.replace(old,new,1)

    marker='# =========================================================\n# 候補が無い場合\n'
    block='''# =========================================================\n# EXACT TRACE OUTPUT (diagnostic only)\n# =========================================================\nif _TRACE_ROWS:\n    with _TRACE_OUT.open("w",encoding="utf-8",newline="") as f:\n        w=csv.DictWriter(f,fieldnames=list(_TRACE_ROWS[0].keys()))\n        w.writeheader(); w.writerows(_TRACE_ROWS)\n    print("\\n🔬 EXACT PIPELINE TRACE")\n    for r in _TRACE_ROWS:\n        print(f"{r['date']} | expected={r['expected_tickers']} present={r['symbol_data_present']} liquid={r['liquid']} feature={r['feature_valid']} predict={r['prediction_rows']} signal={r['signal_rows']} raw_candidate={r['raw_candidates']} score>=50={r['score_ge_50']} up>=50={r['up_ge_50']} up>down={r['up_gt_down']} flat<50={r['flat_lt_50']}")\n    print(f"診断CSV: {_TRACE_OUT.name}")\n\n\n'''
    if marker not in s: raise RuntimeError("output marker not found")
    s=s.replace(marker,block+marker,1)
    return s


def main():
    source=patch_source(BASE.read_text(encoding="utf-8"))
    compile(source,str(TMP),"exec")
    TMP.write_text(source,encoding="utf-8")
    try: runpy.run_path(str(TMP),run_name="__main__")
    finally:
        if TMP.exists(): TMP.unlink()

if __name__=="__main__": main()
