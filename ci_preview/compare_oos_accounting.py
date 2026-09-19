"""Read-only comparison of the new-accounting vs legacy-accounting preview runs.

Consumes preview_summary_new.json and preview_summary_legacy.json produced by
summarize_oos_accounting.py in the two matrix legs of the preview workflow.
Writes only a comparison report (JSON + Markdown) as a workflow artifact.
Never touches strategy_policy*.json, never computes signatures, no secrets.
"""
import json
from pathlib import Path

NAMES = {"H1": "UP20_SCORE40_NIKKEION_TP4.0_SL1.0_H1", "H3": "UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3"}


def load(accounting):
    p = Path(f"preview_summary_{accounting}.json")
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def fmt(v, nd=2):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def main():
    new = load("new")
    legacy = load("legacy")

    lines = ["# OOS accounting preview: new (default) vs legacy (WF_LEGACY_OVERLAP=1)", ""]

    if new is None or legacy is None:
        lines.append(f"⚠ 片方または両方の summary が見つかりません (new={'OK' if new else 'MISSING'}, legacy={'OK' if legacy else 'MISSING'})")
        Path("comparison.md").write_text("\n".join(lines), encoding="utf-8")
        Path("comparison.json").write_text(json.dumps({"new": new, "legacy": legacy}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\n".join(lines))
        return

    lines.append("## Fold funnel: dev_selected -> validation_pass -> oos_evaluated -> oos_pass -> fold_final")
    lines.append("")
    lines.append("| fold | metric | new | legacy |")
    lines.append("|---|---|---|---|")
    nfolds_by_id = {f["fold"]: f for f in new["folds"]}
    lfolds_by_id = {f["fold"]: f for f in legacy["folds"]}
    for fid in sorted(set(nfolds_by_id) | set(lfolds_by_id)):
        nf = nfolds_by_id.get(fid, {})
        lf = lfolds_by_id.get(fid, {})
        for metric in ["dev_selected", "validation_pass", "oos_evaluated", "oos_pass", "fold_final"]:
            lines.append(f"| {fid} | {metric} | {nf.get(metric, '-')} | {lf.get(metric, '-')} |")

    lines.append("")
    lines.append(f"## Aggregated (multi-fold) final approved candidate count: new={new['aggregated_final_count']} / legacy={legacy['aggregated_final_count']}")
    lines.append("")
    lines.append("### new accounting aggregated final candidates")
    for r in new["aggregated_final_candidates"]:
        lines.append(f"- {r.get('strategy')}: folds={r.get('oos_positive_folds')} compound={fmt(r.get('oos_compound_return'))}% pf={fmt(r.get('oos_pf'))} dd={fmt(r.get('oos_dd'))}%")
    lines.append("### legacy accounting aggregated final candidates")
    for r in legacy["aggregated_final_candidates"]:
        lines.append(f"- {r.get('strategy')}: folds={r.get('oos_positive_folds')} compound={fmt(r.get('oos_compound_return'))}% pf={fmt(r.get('oos_pf'))} dd={fmt(r.get('oos_dd'))}%")

    lines.append("")
    for key, name in NAMES.items():
        lines.append(f"## {key} ({name}): new vs legacy")
        lines.append("")
        lines.append("| fold | new: dev_sel/val_pass/oos_pass/fold_final | new oos_signals/avg_month_return/pf/dd/compound | legacy: dev_sel/val_pass/oos_pass/fold_final | legacy oos_signals/avg_month_return/pf/dd/compound |")
        lines.append("|---|---|---|---|---|")
        for fid in sorted(set(nfolds_by_id) | set(lfolds_by_id)):
            nc = nfolds_by_id.get(fid, {}).get("combos", {}).get(key, {})
            lc = lfolds_by_id.get(fid, {}).get("combos", {}).get(key, {})
            n_flags = f"{nc.get('in_dev_selected')}/{nc.get('validation_pass')}/{nc.get('oos_pass')}/{nc.get('in_fold_final')}"
            l_flags = f"{lc.get('in_dev_selected')}/{lc.get('validation_pass')}/{lc.get('oos_pass')}/{lc.get('in_fold_final')}"
            noos = nc.get("oos") or {}
            loos = lc.get("oos") or {}
            n_perf = f"{fmt(noos.get('oos_signals'), 0)}/{fmt(noos.get('oos_avg_month_return'))}/{fmt(noos.get('oos_pf'))}/{fmt(noos.get('oos_dd'))}/{fmt(noos.get('oos_compound_return'))}" if noos else (nc.get("note") or "-")
            l_perf = f"{fmt(loos.get('oos_signals'), 0)}/{fmt(loos.get('oos_avg_month_return'))}/{fmt(loos.get('oos_pf'))}/{fmt(loos.get('oos_dd'))}/{fmt(loos.get('oos_compound_return'))}" if loos else (lc.get("note") or "-")
            lines.append(f"| {fid} | {n_flags} | {n_perf} | {l_flags} | {l_perf} |")
        n_agg = new["aggregated_combo_result"].get(key)
        l_agg = legacy["aggregated_combo_result"].get(key)
        lines.append("")
        lines.append(f"- aggregated (all-fold combined) present: new={n_agg is not None} legacy={l_agg is not None}")
        if n_agg:
            lines.append(f"  - new aggregated: compound={fmt(n_agg.get('oos_compound_return'))}% pf={fmt(n_agg.get('oos_pf'))} dd={fmt(n_agg.get('oos_dd'))}% folds={n_agg.get('oos_positive_folds')}")
        if l_agg:
            lines.append(f"  - legacy aggregated: compound={fmt(l_agg.get('oos_compound_return'))}% pf={fmt(l_agg.get('oos_pf'))} dd={fmt(l_agg.get('oos_dd'))}% folds={l_agg.get('oos_positive_folds')}")
        lines.append("")

    Path("comparison.md").write_text("\n".join(lines), encoding="utf-8")
    Path("comparison.json").write_text(json.dumps({"new": new, "legacy": legacy}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
