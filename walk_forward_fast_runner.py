#!/usr/bin/env python3
"""
WALK-FORWARD高速実行ラッパー

walk_forward.py の検証ロジックは変更せず、実行前に安全な正規表現置換で
以下の高速化だけを適用する。

1. 再学習間隔: 20 -> 40営業日
2. RandomForest: 300 -> 150 trees
3. 1銘柄あたり学習履歴を最大750行に制限
4. evaluate_trade の結果をキャッシュして条件比較で再利用
5. 進捗表示に候補累計を追加

文字列の空白・改行に依存した完全一致検索は使わない。
"""

from pathlib import Path
import re
import runpy

BASE_FILE = Path(__file__).resolve().with_name("walk_forward.py")
TEMP_FILE = Path(__file__).resolve().with_name(".walk_forward_fast_tmp.py")


def replace_once(source: str, pattern: str, replacement: str, label: str) -> tuple[str, bool]:
    patched, count = re.subn(pattern, replacement, source, count=1, flags=re.MULTILINE)
    if count != 1:
        raise RuntimeError(f"高速化パッチ失敗: {label} (matched={count})")
    return patched, True


def patch_source(source: str) -> str:
    replacements = []

    # ---------------------------------------------------------
    # 1) 再学習間隔 20 -> 40
    # ---------------------------------------------------------
    pattern = (
        r'REFIT_EVERY_TRADING_DAYS\s*=\s*int\(\s*'
        r'os\.getenv\(\s*'
        r'["\']WF_REFIT_EVERY_TRADING_DAYS["\']\s*,\s*'
        r'["\']20["\']\s*\)\s*\)'
    )
    replacement = '''REFIT_EVERY_TRADING_DAYS = int(
    os.getenv(
        "WF_REFIT_EVERY_TRADING_DAYS",
        "40"
    )
)'''
    source, _ = replace_once(source, pattern, replacement, "REFIT_EVERY_TRADING_DAYS")
    replacements.append("再学習間隔 20→40営業日")

    # ---------------------------------------------------------
    # 2) RandomForest 300 -> 150
    # ---------------------------------------------------------
    pattern = r'^N_ESTIMATORS\s*=\s*300\s*$'
    replacement = '''N_ESTIMATORS = int(
    os.getenv(
        "WF_N_ESTIMATORS",
        "150"
    )
)'''
    source, _ = replace_once(source, pattern, replacement, "N_ESTIMATORS")
    replacements.append("RandomForest 300→150本")

    # ---------------------------------------------------------
    # 3) 各銘柄の学習履歴を最大750行に制限
    #    usable_prior直後の「train_piece」までを正規表現で挿入
    # ---------------------------------------------------------
    pattern = (
        r'(usable_prior\s*=\s*\(\s*prior\[\s*'
        r'prior\[\s*["\']target_valid["\']\s*\]\s*\]\s*'
        r'\.copy\(\)\s*\))'
    )
    insert = r'''\1

            max_train_rows_per_ticker = int(
                os.getenv(
                    "WF_MAX_TRAIN_ROWS_PER_TICKER",
                    "750"
                )
            )

            if len(usable_prior) > max_train_rows_per_ticker:
                usable_prior = (
                    usable_prior
                    .tail(max_train_rows_per_ticker)
                    .copy()
                )'''
    source, _ = replace_once(source, pattern, insert, "usable_prior学習上限")
    replacements.append("1銘柄あたり学習履歴を最大750行")

    # ---------------------------------------------------------
    # 4) 進捗表示 20 -> 10 + 候補累計
    # ---------------------------------------------------------
    pattern = (
        r'if\s+pos\s*%\s*20\s*==\s*0\s*:\s*\n'
        r'(\s*print\(\s*\n'
        r'\s*f["\']進捗:\s*["\']\s*\n'
        r'\s*f["\']\{pos\s*\+\s*1\}/["\']\s*\n'
        r'\s*f["\']\{len\(prediction_dates\)\}\s*["\']\s*\n'
        r'\s*f["\']\{prediction_date\.date\(\)\}["\']\s*\n'
        r'\s*\)\s*)'
    )
    replacement = '''if pos % 10 == 0:

        print(
            f"進捗: "
            f"{pos + 1}/"
            f"{len(prediction_dates)} "
            f"{prediction_date.date()} "
            f"候補累計={len(candidate_history)}"
        )'''
    source, _ = replace_once(source, pattern, replacement, "進捗表示")
    replacements.append("進捗表示を10営業日ごと＋候補累計")

    # ---------------------------------------------------------
    # 5) evaluate_trade のキャッシュ
    #    元関数定義の直後にラッパーを挿入する。
    #    既にパッチ済みなら二重挿入しない。
    # ---------------------------------------------------------
    if "_FAST_TRADE_CACHE" not in source:
        marker_pattern = r'(#\s*=+\s*\n#\s*START\s*\n#\s*=+)'
        cache_block = r'''# =========================================================
# FAST RUNNER: 売買結果キャッシュ
# =========================================================

_FAST_ORIGINAL_EVALUATE_TRADE = evaluate_trade
_FAST_TRADE_CACHE = {}


def evaluate_trade(day_df, entry_date, entry_price, take_profit, stop_loss):
    key = (
        id(day_df),
        str(pd.Timestamp(entry_date)),
        float(entry_price),
        float(take_profit),
        float(stop_loss),
    )

    cached = _FAST_TRADE_CACHE.get(key)
    if cached is None:
        cached = _FAST_ORIGINAL_EVALUATE_TRADE(
            day_df,
            entry_date,
            entry_price,
            take_profit,
            stop_loss,
        )
        _FAST_TRADE_CACHE[key] = cached

    return cached


\1'''
        source, _ = replace_once(source, marker_pattern, cache_block, "evaluate_tradeキャッシュ")
        replacements.append("evaluate_trade結果をキャッシュ")

    print("✅ 高速化パッチ適用:")
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
        compile(patched, str(TEMP_FILE), "exec")
        print("")
        print("===========================================")
        print("🚀 WALK-FORWARD FAST RUNNER")
        print("===========================================")
        print("元ファイル: walk_forward.py")
        print("実行ファイル: .walk_forward_fast_tmp.py")
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
