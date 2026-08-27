#!/usr/bin/env python3
"""
WALK-FORWARD高速実行ラッパー

walk_forward.py 本体を直接書き換えず、実行時だけ高速化する。
内部の特定空白に依存した文字列置換は行わない。

高速化:
- 再学習間隔 20 -> 40営業日
- RandomForest 300 -> 150 trees
- 1回のfitに渡す学習データを最大75,000行へ決定論的サンプリング
- evaluate_trade の結果をキャッシュ
- 進捗表示を10営業日ごとにする
"""

from pathlib import Path
import re
import runpy
import numpy as np

from sklearn import ensemble

BASE_FILE = Path(__file__).resolve().with_name("walk_forward.py")
TEMP_FILE = Path(__file__).resolve().with_name(".walk_forward_fast_tmp.py")


# =========================================================
# RandomForest高速版
# =========================================================

OriginalRandomForestClassifier = ensemble.RandomForestClassifier


class FastRandomForestClassifier(OriginalRandomForestClassifier):
    """大規模学習時だけ決定論的に学習行数を制限する。"""

    MAX_TRAIN_ROWS = 75000
    FAST_N_ESTIMATORS = 150

    def __init__(self, *args, **kwargs):
        kwargs["n_estimators"] = min(
            int(kwargs.get("n_estimators", self.FAST_N_ESTIMATORS)),
            self.FAST_N_ESTIMATORS,
        )
        super().__init__(*args, **kwargs)

    def fit(self, X, y, sample_weight=None):
        n_rows = len(X)

        if n_rows > self.MAX_TRAIN_ROWS:
            rng = np.random.RandomState(42)
            idx = np.sort(
                rng.choice(
                    n_rows,
                    size=self.MAX_TRAIN_ROWS,
                    replace=False,
                )
            )

            X = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

            if sample_weight is not None:
                sample_weight = (
                    sample_weight.iloc[idx]
                    if hasattr(sample_weight, "iloc")
                    else sample_weight[idx]
                )

            print(
                f"  ⚡ RF学習行数を {n_rows} -> "
                f"{self.MAX_TRAIN_ROWS} に制限"
            )

        return super().fit(
            X,
            y,
            sample_weight=sample_weight,
        )


# walk_forward.py の
# from sklearn.ensemble import RandomForestClassifier
# が参照するクラスを実行前に差し替える。
ensemble.RandomForestClassifier = FastRandomForestClassifier


def patch_source(source: str) -> str:
    """元コードの構造を大きく変更せず、軽量化用の小さな変更だけを施す。"""

    # ---------------------------------------------------------
    # 再学習間隔
    # ---------------------------------------------------------
    source2, count = re.subn(
        r'(REFIT_EVERY_TRADING_DAYS\s*=\s*int\(\s*'
        r'os\.getenv\(\s*["\']WF_REFIT_EVERY_TRADING_DAYS["\']\s*,\s*)'
        r'["\']20["\']',
        r'\1"40"',
        source,
        count=1,
        flags=re.DOTALL,
    )

    if count == 1:
        source = source2
        print("  ✅ 再学習間隔 20→40営業日")
    else:
        print("  ℹ 再学習間隔の既定値置換は不要")

    # ---------------------------------------------------------
    # 進捗表示
    # 元コード側が別形式でも失敗しないよう、これは任意。
    # ---------------------------------------------------------
    source2, count = re.subn(
        r'if\s+pos\s*%\s*20\s*==\s*0\s*:',
        'if pos % 10 == 0:',
        source,
        count=1,
    )
    if count == 1:
        source = source2
        print("  ✅ 進捗表示 20→10営業日")

    # ---------------------------------------------------------
    # evaluate_trade キャッシュ
    # STARTマーカーが見つからなくても実行自体は可能にする。
    # ---------------------------------------------------------
    if "_FAST_TRADE_CACHE" not in source:
        cache_block = '''\n# =========================================================\n# FAST RUNNER: evaluate_trade キャッシュ\n# =========================================================\n\n_FAST_ORIGINAL_EVALUATE_TRADE = evaluate_trade\n_FAST_TRADE_CACHE = {}\n\ndef evaluate_trade(day_df, entry_date, entry_price, take_profit, stop_loss):\n    key = (\n        id(day_df),\n        str(pd.Timestamp(entry_date)),\n        float(entry_price),\n        float(take_profit),\n        float(stop_loss),\n    )\n\n    if key not in _FAST_TRADE_CACHE:\n        _FAST_TRADE_CACHE[key] = _FAST_ORIGINAL_EVALUATE_TRADE(\n            day_df,\n            entry_date,\n            entry_price,\n            take_profit,\n            stop_loss,\n        )\n\n    return _FAST_TRADE_CACHE[key]\n\n'''

        # main実行直前の一般的な if __name__ ブロックを狙う。
        main_pattern = r'(?m)^if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*$'
        patched, count = re.subn(
            main_pattern,
            cache_block + '\nif __name__ == "__main__":',
            source,
            count=1,
        )

        if count == 1:
            source = patched
            print("  ✅ evaluate_trade結果をキャッシュ")
        else:
            print("  ⚠ evaluate_tradeキャッシュ挿入をスキップ")

    return source


def main() -> None:
    if not BASE_FILE.exists():
        raise FileNotFoundError(
            f"{BASE_FILE} が見つかりません"
        )

    source = BASE_FILE.read_text(
        encoding="utf-8"
    )

    patched = patch_source(source)

    # 構文チェック
    compile(
        patched,
        str(TEMP_FILE),
        "exec",
    )

    TEMP_FILE.write_text(
        patched,
        encoding="utf-8",
    )

    try:
        print("")
        print("===========================================")
        print("🚀 WALK-FORWARD FAST RUNNER")
        print("===========================================")
        print("元ファイル: walk_forward.py")
        print("高速化: RF150 / 最大学習75,000行 / 再学習40日")
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
