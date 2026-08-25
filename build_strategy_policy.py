import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd


# =========================================================
# STRATEGY POLICY BUILDER
#
# adversarial_final_candidates.csv
#        ↓
# 採用可能な戦略を判定
#        ↓
# strategy_policy.json
#
# 重要:
#   PASSがない場合は既存policyを変更しない。
#   これにより、本番条件が毎回コロコロ変わるのを防ぐ。
# =========================================================


INPUT_FILE = "adversarial_final_candidates.csv"
POLICY_FILE = "strategy_policy.json"


# =========================================================
# 本番採用ゲート
# =========================================================

MIN_OOS_TRADES = 20
MIN_OOS_PF = 1.00
MIN_OOS_AVG_RETURN = 0.00

MIN_OOS_TO_VALIDATION_PF = 0.60

MAX_VALIDATION_DD = 30.0

MIN_VALIDATION_TRADES = 50

MIN_VALIDATION_PF = 1.00

MIN_VALIDATION_AVG_RETURN = 0.00

MIN_MC_BANKRUPTCY_PROB = 5.0

MAX_MC_DD90 = 30.0


# =========================================================
# デフォルト条件
#
# policyが存在しない初回だけ使用。
# =========================================================

DEFAULT_POLICY = {
    "status": "DEFAULT",
    "updated_at": None,

    "up_threshold": 50,
    "min_score_for_buy": 60,

    "nikkei_filter": False,

    "atr_tp_multiplier": 3.0,
    "atr_sl_multiplier": 1.5,

    "hold_days": 5,

    "validation_signals": 0,
    "validation_win_rate": 0.0,
    "validation_avg_return": 0.0,
    "validation_pf": 0.0,
    "validation_dd": 0.0,

    "oos_signals": 0,
    "oos_win_rate": 0.0,
    "oos_avg_return": 0.0,
    "oos_pf": 0.0,
    "oos_dd": 0.0,
    "oos_validation_pf_ratio": 0.0,

    "mc_sizing": 0.005,
    "mc_10y_probability": 0.0,
    "mc_15y_probability": 0.0,
    "mc_20y_probability": 0.0,
    "mc_bankruptcy_probability": 0.0,
    "mc_p90_max_dd": 0.0,

    "source": "default"
}


# =========================================================
# 数値安全化
# =========================================================

def safe_float(
    value,
    default=0.0
):

    try:

        value = float(value)

        if pd.isna(value):
            return default

        return value

    except Exception:

        return default


def safe_int(
    value,
    default=0
):

    try:

        return int(
            float(value)
        )

    except Exception:

        return default


def safe_bool(
    value,
    default=False
):

    if isinstance(
        value,
        bool
    ):

        return value

    text = str(
        value
    ).strip().lower()

    if text in [
        "true",
        "1",
        "yes",
        "on"
    ]:

        return True

    if text in [
        "false",
        "0",
        "no",
        "off"
    ]:

        return False

    return default


# =========================================================
# policy読み込み
# =========================================================

def load_existing_policy():

    if not os.path.exists(
        POLICY_FILE
    ):

        return DEFAULT_POLICY.copy()

    try:

        with open(
            POLICY_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            policy = json.load(f)

        if not isinstance(
            policy,
            dict
        ):

            print(
                "⚠ strategy_policy.jsonが"
                "辞書形式ではありません"
            )

            return DEFAULT_POLICY.copy()

        # 欠損キーをデフォルトで補完
        merged = DEFAULT_POLICY.copy()

        merged.update(
            policy
        )

        return merged

    except Exception as e:

        print(
            "⚠ 既存policy読み込み失敗:",
            e
        )

        return DEFAULT_POLICY.copy()


# =========================================================
# 入力CSV確認
# =========================================================

if not os.path.exists(
    INPUT_FILE
):

    print(
        f"⚠ {INPUT_FILE} がありません"
    )

    print(
        "今回の本番設定は変更しません"
    )

    raise SystemExit(0)


try:

    df = pd.read_csv(
        INPUT_FILE
    )

except Exception as e:

    print(
        "❌ CSV読み込み失敗:",
        e
    )

    raise SystemExit(1)


if df.empty:

    print(
        "⚠ 候補が0件です"
    )

    print(
        "今回の本番設定は変更しません"
    )

    raise SystemExit(0)


# =========================================================
# 必須列
# =========================================================

required_columns = [

    "final_status",

    "up_threshold",

    "score_threshold",

    "nikkei_filter",

    "tp_multiplier",

    "sl_multiplier",

    "hold_days",

    "validation_signals",

    "validation_win_rate",

    "validation_avg_return",

    "validation_pf",

    "validation_dd",

    "oos_signals",

    "oos_win_rate",

    "oos_avg_return",

    "oos_pf",

    "oos_dd",

    "oos_validation_pf_ratio",
]


missing = [

    col
    for col
    in required_columns
    if col not in df.columns
]


if missing:

    print(
        "❌ strategy候補CSVに"
        "不足列があります:"
    )

    for col in missing:

        print(
            " -",
            col
        )

    raise SystemExit(1)


# =========================================================
# 型
# =========================================================

for col in [

    "up_threshold",

    "score_threshold",

    "tp_multiplier",

    "sl_multiplier",

    "hold_days",

    "validation_signals",

    "validation_win_rate",

    "validation_avg_return",

    "validation_pf",

    "validation_dd",

    "oos_signals",

    "oos_win_rate",

    "oos_avg_return",

    "oos_pf",

    "oos_dd",

    "oos_validation_pf_ratio",

]:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


df["nikkei_filter"] = (
    df["nikkei_filter"]
    .apply(
        safe_bool
    )
)


# =========================================================
# Monte Carlo列
#
# 存在する場合だけ使用。
# =========================================================

mc_columns_exist = all(

    col in df.columns

    for col in [

        "sizing",

        "prob_10y",

        "prob_15y",

        "prob_20y",

        "bankruptcy_prob",

        "p90_max_dd",

    ]

)


if not mc_columns_exist:

    print(
        "⚠ Monte Carlo列がありません"
    )

    print(
        "Monte Carlo条件なしでは"
        "自動採用しません"
    )


# =========================================================
# 最終PASSだけ候補にする
# =========================================================

approved = df[

    df["final_status"]
    .astype(str)
    .str.upper()
    .eq("PASS")

].copy()


if approved.empty:

    print(
        "🟡 PASS候補なし"
    )

    print(
        "strategy_policy.jsonは変更しません"
    )

    raise SystemExit(0)


# =========================================================
# Validation Gate
# =========================================================

approved = approved[
    (
        approved[
            "validation_signals"
        ]
        >=
        MIN_VALIDATION_TRADES
    )
    &
    (
        approved[
            "validation_pf"
        ]
        >=
        MIN_VALIDATION_PF
    )
    &
    (
        approved[
            "validation_avg_return"
        ]
        >
        MIN_VALIDATION_AVG_RETURN
    )
    &
    (
        approved[
            "validation_dd"
        ].abs()
        <=
        MAX_VALIDATION_DD
    )
]


# =========================================================
# OOS Gate
# =========================================================

approved = approved[
    (
        approved[
            "oos_signals"
        ]
        >=
        MIN_OOS_TRADES
    )
    &
    (
        approved[
            "oos_pf"
        ]
        >=
        MIN_OOS_PF
    )
    &
    (
        approved[
            "oos_avg_return"
        ]
        >
        MIN_OOS_AVG_RETURN
    )
    &
    (
        approved[
            "oos_validation_pf_ratio"
        ]
        >=
        MIN_OOS_TO_VALIDATION_PF
    )
]


# =========================================================
# Monte Carlo Gate
# =========================================================

if mc_columns_exist:

    approved = approved[

        (
            approved[
                "bankruptcy_prob"
            ]
            <
            MIN_MC_BANKRUPTCY_PROB
        )
        &
        (
            approved[
                "p90_max_dd"
            ].abs()
            <=
            MAX_MC_DD90
        )

    ]


# =========================================================
# ここまで通過しなければ
# 本番設定は絶対変更しない
# =========================================================

if approved.empty:

    print(
        "🟡 最終採用条件を満たす戦略なし"
    )

    print(
        "strategy_policy.jsonは"
        "変更しません"
    )

    raise SystemExit(0)


# =========================================================
# 候補ランキング
#
# 「一番利益が高い」だけにしない。
# OOS PF → Validation PF → 件数
# の順で安定性を優先。
# =========================================================

sort_columns = [

    "oos_pf",

    "oos_avg_return",

    "validation_pf",

    "validation_signals",

]


approved = (
    approved
    .sort_values(
        sort_columns,
        ascending=[
            False,
            False,
            False,
            False,
        ]
    )
    .reset_index(
        drop=True
    )
)


best = approved.iloc[
    0
]


# =========================================================
# 既存policy
# =========================================================

old_policy = (
    load_existing_policy()
)


# =========================================================
# 新policy
# =========================================================

now = datetime.now().isoformat(
    timespec="seconds"
)


new_policy = {

    "status":
        "APPROVED",

    "updated_at":
        now,


    # -------------------------
    # 本番戦略条件
    # -------------------------

    "up_threshold":
        safe_int(
            best[
                "up_threshold"
            ]
        ),

    "min_score_for_buy":
        safe_int(
            best[
                "score_threshold"
            ]
        ),

    "nikkei_filter":
        safe_bool(
            best[
                "nikkei_filter"
            ]
        ),

    "atr_tp_multiplier":
        safe_float(
            best[
                "tp_multiplier"
            ]
        ),

    "atr_sl_multiplier":
        safe_float(
            best[
                "sl_multiplier"
            ]
        ),

    "hold_days":
        safe_int(
            best[
                "hold_days"
            ]
        ),


    # -------------------------
    # Validation
    # -------------------------

    "validation_signals":
        safe_int(
            best[
                "validation_signals"
            ]
        ),

    "validation_win_rate":
        safe_float(
            best[
                "validation_win_rate"
            ]
        ),

    "validation_avg_return":
        safe_float(
            best[
                "validation_avg_return"
            ]
        ),

    "validation_pf":
        safe_float(
            best[
                "validation_pf"
            ]
        ),

    "validation_dd":
        safe_float(
            best[
                "validation_dd"
            ]
        ),


    # -------------------------
    # OOS
    # -------------------------

    "oos_signals":
        safe_int(
            best[
                "oos_signals"
            ]
        ),

    "oos_win_rate":
        safe_float(
            best[
                "oos_win_rate"
            ]
        ),

    "oos_avg_return":
        safe_float(
            best[
                "oos_avg_return"
            ]
        ),

    "oos_pf":
        safe_float(
            best[
                "oos_pf"
            ]
        ),

    "oos_dd":
        safe_float(
            best[
                "oos_dd"
            ]
        ),

    "oos_validation_pf_ratio":
        safe_float(
            best[
                "oos_validation_pf_ratio"
            ]
        ),


    # -------------------------
    # Monte Carlo
    # -------------------------

    "mc_sizing":
        (
            safe_float(
                best[
                    "sizing"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_sizing",
                0.005
            )
        ),

    "mc_10y_probability":
        (
            safe_float(
                best[
                    "prob_10y"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_10y_probability",
                0.0
            )
        ),

    "mc_15y_probability":
        (
            safe_float(
                best[
                    "prob_15y"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_15y_probability",
                0.0
            )
        ),

    "mc_20y_probability":
        (
            safe_float(
                best[
                    "prob_20y"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_20y_probability",
                0.0
            )
        ),

    "mc_bankruptcy_probability":
        (
            safe_float(
                best[
                    "bankruptcy_prob"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_bankruptcy_probability",
                0.0
            )
        ),

    "mc_p90_max_dd":
        (
            safe_float(
                best[
                    "p90_max_dd"
                ]
            )
            if mc_columns_exist
            else
            old_policy.get(
                "mc_p90_max_dd",
                0.0
            )
        ),


    # -------------------------
    # 管理情報
    # -------------------------

    "strategy_name":
        str(
            best.get(
                "strategy",
                ""
            )
        ),

    "source":
        "adversarial_strategy_validator",

}


# =========================================================
# policy保存
# =========================================================

try:

    with open(
        POLICY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            new_policy,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("")
    print(
        "=" * 60
    )

    print(
        "✅ strategy_policy.json 更新"
    )

    print(
        "=" * 60
    )

    print(
        "採用戦略:",
        new_policy[
            "strategy_name"
        ]
    )

    print(
        "UP:",
        new_policy[
            "up_threshold"
        ]
    )

    print(
        "MIN SCORE:",
        new_policy[
            "min_score_for_buy"
        ]
    )

    print(
        "日経フィルター:",
        new_policy[
            "nikkei_filter"
        ]
    )

    print(
        "ATR TP:",
        new_policy[
            "atr_tp_multiplier"
        ]
    )

    print(
        "ATR SL:",
        new_policy[
            "atr_sl_multiplier"
        ]
    )

    print(
        "Validation PF:",
        new_policy[
            "validation_pf"
        ]
    )

    print(
        "OOS PF:",
        new_policy[
            "oos_pf"
        ]
    )

    print(
        "OOS/Validation PF:",
        new_policy[
            "oos_validation_pf_ratio"
        ]
    )

except Exception as e:

    print(
        "❌ strategy_policy.json保存失敗:",
        e
    )

    raise SystemExit(1)
