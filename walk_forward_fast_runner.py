#!/usr/bin/env python3
"""
WALK-FORWARD高速実行ラッパー

walk_forward.py 本体を直接変更せず、実行時だけ高速化する。
特定の空白・改行に依存したパッチは行わない。

高速化:
- 再学習間隔 20 -> 40営業日
- RandomForest 300 -> 最大150 trees
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
# 純正RandomForestを使った安全な高速ラッパー
# =========================================================

_ORIGINAL_RF = ensemble.RandomForestClassifier


def FastRandomForestClassifier(*args, **kwargs):
    """
    walk_forward.py からはRandomForestClassifierとして呼ばれるが、
    純正scikit-learn estimatorを生成して返す。
    そのためsklearnのEstimator互換性を壊さない。
    """

    kwargs["n_estimators"] = min(
        int(kwargs.get("n_estimators", 150)),
        150,
    )

    original_fit = _ORIGINAL_RF.fit
    model = _ORIGINAL_RF(*args, **kwargs)

    max_train_rows = 75000

    def fast_fit(X, y, sample_weight=None, **fit_kwargs):
        n_rows = len(X)

        if n_rows > max_train_rows:
            rng = np.random.RandomState(42)

            idx = np.sort(
                rng.choice(
                    n_rows,
                    size=max_train_rows,
                    replace=False,
                )
            )

            X2 = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
            y2 = y.iloc[idx] if hasattr(y, "iloc") else y[idx]

            if sample_weight is not None:
                sample_weight = (
                    sample_weight.iloc[idx]
                    if hasattr(sample_weight, "iloc")
                    else sample_weight[idx]
                )

            print(
                f"  ⚡ RF学習行数 {n_rows:,} -> "
                f"{max_train_rows:,}"
            )

            return original_fit(
                model,
                X2,
                y2,
                sample_weight=sample_weight,
                **fit_kwargs,
            )

        return original_fit(
            model,
            X,
            y,
            sample_weight=sample_weight,
            **fit_kwargs,
        )

    model.fit = fast_fit
    return model


# walk_forward.py の import 後に参照されるクラス名を差し替える。
ensemble.RandomForestClassifier = FastRandomForestClassifier


def patch_source(source: str) -> str:
    """walk_forward.pyの文字列構造に依存しない最小限の高速化パッチ。"""

    # ---------------------------------------------------------
    # 再学習間隔: 20 -> 40
    # ---------------------------------------------------------
    pattern = (
        r'(REFIT_EVERY_TRADING_DAYS\s*=\s*int\(\s*'
        r'os\.getenv\(\s*["\']WF_REFIT_EVERY_TRADING_DAYS["\']\s*,\s*)'
        r'["\']20["\']'
    )

    source, count = re.subn(
        pattern,
        r'\1"40"',
        source,
        count=1,
        flags=re.DOTALL,
    )

    if count == 1:
        print("  ✅ 再学習間隔 20→40営業日")
    else:
        print("  ℹ 再学習間隔は元コードを維持")

    # ---------------------------------------------------------
    # 進捗表示: 20 -> 10
    # ---------------------------------------------------------
    source, count = re.subn(
        r'if\s+pos\s*%\s*20\s*==\s*0\s*:',
        'if pos % 10 == 0:',
        source,
        count=1,
    )

    if count == 1:
        print("  ✅ 進捗表示 20→10営業日")

    # ---------------------------------------------------------
    # evaluate_trade キャッシュ
    # ---------------------------------------------------------
    if "_FAST_TRADE_CACHE" not in source:
        cache_block = '''\n# =========================================================\n# FAST RUNNER: evaluate_trade キャッシュ\n# =========================================================\n\n_FAST_ORIGINAL_EVALUATE_TRADE = evaluate_trade\n_FAST_TRADE_CACHE = {}\n\ndef evaluate_trade(day_df, entry_date, entry_price, take_profit, stop_loss):\n    key = (\n        id(day_df),\n        str(pd.Timestamp(entry_date)),\n        float(entry_price),\n        float(take_profit),\n        float(stop_loss),\n    )\n\n    if key not in _FAST_TRADE_CACHE:\n        _FAST_TRADE_CACHE[key] = _FAST_ORIGINAL_EVALUATE_TRADE(\n            day_df,\n            entry_date,\n            entry_price,\n            take_profit,\n            stop_loss,\n        )\n\n    return _FAST_TRADE_CACHE[key]\n\n'''

        # 通常のPythonエントリポイント直前に差し込む。
        patched, count = re.subn(
            r'(?m)^if\s+__name__\s*==\s*["\']__main__["\']\s*:\s*$',
            cache_block + 'if __name__ == "__main__":',
            source,
            count=1,
        )

        if count == 1:
            source = patched
            print("  ✅ evaluate_trade結果をキャッシュ")
        else:
            print("  ⚠ evaluate_tradeキャッシュは未適用")

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
        print("RF: 最大150本")
        print("学習: 最大75,000行/fit")
        print("再学習: 40営業日")
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
