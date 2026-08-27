#!/usr/bin/env python3
"""
WALK-FORWARD高速実行ラッパー

既存の walk_forward.py の検証ロジックを変更せず、実行時だけ以下を適用する。
1. 再学習間隔: 20 -> 40営業日
2. RandomForest: 300 -> 150 trees
3. 1銘柄あたり学習履歴を最大750行に制限
4. evaluate_trade の結果をキャッシュして戦略24条件で使い回す
5. 進捗表示に候補累計を追加

元ファイルは変更しない。
GitHub Actionsでは本ファイルを python walk_forward_fast_runner.py で実行する。
"""

from pathlib import Path
import runpy


BASE_FILE = Path(__file__).resolve().with_name("walk_forward.py")
TEMP_FILE = Path(__file__).resolve().with_name(".walk_forward_fast_tmp.py")


def patch_source(source: str) -> str:
    replacements = []

    old = '''REFIT_EVERY_TRADING_DAYS = int(\n    os.getenv(\n        "WF_REFIT_EVERY_TRADING_DAYS",\n        "20"\n    )\n)'''
    new = '''REFIT_EVERY_TRADING_DAYS = int(\n    os.getenv(\n        "WF_REFIT_EVERY_TRADING_DAYS",\n        "40"\n    )\n)'''
    if old in source:
        source = source.replace(old, new, 1)
        replacements.append("REFIT_EVERY_TRADING_DAYS 20->40")
    else:
        raise RuntimeError("REFIT_EVERY_TRADING_DAYS の置換対象が見つかりません")

    old = "N_ESTIMATORS = 300"
    new = '''N_ESTIMATORS = int(\n    os.getenv(\n        "WF_N_ESTIMATORS",\n        "150"\n    )\n)'''
    if old in source:
        source = source.replace(old, new, 1)
        replacements.append("N_ESTIMATORS 300->150")
    else:
        raise RuntimeError("N_ESTIMATORS の置換対象が見つかりません")

    old = '''            if usable_prior.empty:\n\n                continue\n\n\n            train_piece = ('''
    new = '''            if usable_prior.empty:\n\n                continue\n\n\n            # 100銘柄化で学習データが過大にならないよう、\n            # 1銘柄あたり直近750営業日のtarget確定データだけを使用。\n            max_train_rows_per_ticker = int(\n                os.getenv(\n                    "WF_MAX_TRAIN_ROWS_PER_TICKER",\n                    "750"\n                )\n            )\n\n            if len(usable_prior) > max_train_rows_per_ticker:\n\n                usable_prior = (\n                    usable_prior\n                    .tail(max_train_rows_per_ticker)\n                    .copy()\n                )\n\n\n            train_piece = ('''
    if old in source:
        source = source.replace(old, new, 1)
        replacements.append("学習履歴を1銘柄750行へ制限")
    else:
        raise RuntimeError("usable_prior の学習制限挿入位置が見つかりません")

    old = '''if pos % 20 == 0:\n\n        print(\n            f"進捗: "\n            f"{pos + 1}/"\n            f"{len(prediction_dates)} "\n            f"{prediction_date.date()}"\n        )'''
    new = '''if pos % 10 == 0:\n\n        print(\n            f"進捗: "\n            f"{pos + 1}/"\n            f"{len(prediction_dates)} "\n            f"{prediction_date.date()} "\n            f"候補累計={len(candidate_history)}"\n        )'''
    if old in source:
        source = source.replace(old, new, 1)
        replacements.append("進捗表示を10日ごとに強化")
    else:
        raise RuntimeError("進捗表示の置換対象が見つかりません")

    marker = '''# =========================================================\n# START\n# ========================================================='''
    if marker not in source:
        raise RuntimeError("STARTマーカーが見つかりません")

    cache_block = r'''# =========================================================
# 高速化: evaluate_trade キャッシュ
#
# 同じ候補が複数戦略条件で選ばれた場合でも、
# 5営業日のOHLC走査は1回だけ実行して結果を再利用する。
# =========================================================

_original_evaluate_trade = evaluate_trade
_trade_cache = {}


def evaluate_trade(
    day_df,
    entry_date,
    entry_price,
    take_profit,
    stop_loss
):

    key = (
        id(day_df),
        str(pd.Timestamp(entry_date)),
        float(entry_price),
        float(take_profit),
        float(stop_loss),
    )

    if key not in _trade_cache:

        _trade_cache[key] = _original_evaluate_trade(
            day_df,
            entry_date,
            entry_price,
            take_profit,
            stop_loss,
        )

    return _trade_cache[key]


'''
    source = source.replace(marker, cache_block + marker, 1)
    replacements.append("evaluate_trade 結果をキャッシュ")

    print("✅ 高速化パッチ:")
    for item in replacements:
        print(f"  - {item}")

    return source


def main() -> None:
    if not BASE_FILE.exists():
        raise FileNotFoundError(f"{BASE_FILE} が見つかりません")

    source = BASE_FILE.read_text(encoding="utf-8")
    patched = patch_source(source)

    TEMP_FILE.write_text(patched, encoding="utf-8")

    try:
        print("")
        print("===========================================")
        print("🚀 WALK-FORWARD FAST RUNNER")
        print("===========================================")
        print("元ファイル: walk_forward.py")
        print("実行用: .walk_forward_fast_tmp.py")
        print("")

        runpy.run_path(
            str(TEMP_FILE),
            run_name="__main__",
        )

    finally:
        try:
            TEMP_FILE.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
