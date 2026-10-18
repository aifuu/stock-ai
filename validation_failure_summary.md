# 検証失敗内訳サマリー (validation_failure_breakdown)

本レポートは read-only の診断であり、記載の『この gate を無視した場合に通過したはずの件数』は**あくまで参考情報**であって、いかなるゲート・ポリシー・validator・署名付きpolicyファイルを緩める提案・推奨では一切ない。実際のゲート定数・判定ロジックはadversarial_strategy_validator.py 側で厳格に維持されている。

## 抽出されたゲート定数(adversarial_strategy_validator.pyのソースから正規表現で抽出)

- `MAX_OOS_DD` = 35.0
- `MAX_VALIDATION_DD` = 30.0
- `MAX_VALIDATION_PF_FOR_RATIO` = 10.0
- `MIN_ANNUAL_SIGNALS` = 7.0
- `MIN_MONTHLY_POSITIVE_RATIO` = 0.55
- `MIN_OOS_AVG_RETURN` = 0.0
- `MIN_OOS_PF` = 1.0
- `MIN_OOS_TO_VALIDATION_PF` = 0.6
- `MIN_OOS_TRADES` = 12.0
- `MIN_PF_LOWER` = 1.0
- `MIN_RETURN_LOWER` = 0.0
- `MIN_VALIDATION_TRADES` = 10.0

## UPトレンド

### VALIDATION

- 候補数 n = 50 / 合格 = 50 / 不合格 = 0

| gate | 不合格件数 | この gate のみ無視した場合に追加で通過したはずの件数(情報提供のみ) |
|---|---|---|
| トレード数不足 (gate_signals) | 0 | 0 |
| PF不足 (gate_pf) | 0 | 0 |
| 平均リターン不足 (gate_avg_return) | 0 | 0 |
| 最大DD超過 (gate_dd) | 0 | 0 |
| 年率シグナル数不足 (gate_annual_signals) | 0 | 0 |
| 月次プラス比率不足 (gate_monthly_positive_ratio) | 0 | 0 |
| ブートストラップ下限<=0 (gate_avg_lower) | 0 | 0 |

全50件の戦略のうち、ゲート判定結果(全gateの合否パターン)が完全に一致する重複を除くと、識別可能な『結果パターン』の種類は 1 種類(多くの UP20/25/30 等の行が実質的に同一結果の重複であることを示す)。

### OOS

- 候補数 n = 50 / 合格 = 0 / 不合格 = 50

| gate | 不合格件数 | この gate のみ無視した場合に追加で通過したはずの件数(情報提供のみ) |
|---|---|---|
| トレード数不足 (gate_signals) | 0 | 0 |
| PF不足 (gate_pf) | 0 | 0 |
| 平均リターン不足 (gate_avg_return) | 0 | 0 |
| OOS/Validation PF比不足 (gate_ratio) | 50 | 35 |
| 月次プラス比率不足 (gate_monthly_positive_ratio) | 15 | 0 |
| 最大DD超過 (gate_dd) | 3 | 0 |
| OOS複利リターン<=0 (gate_compound) | 0 | 0 |

上位の失敗ゲート組み合わせ(多い順、上位5件):

- 「OOS/Validation PF比不足」: 35件
- 「OOS/Validation PF比不足;月次プラス比率不足」: 12件
- 「OOS/Validation PF比不足;月次プラス比率不足;最大DD超過」: 3件

全50件の戦略のうち、ゲート判定結果(全gateの合否パターン)が完全に一致する重複を除くと、識別可能な『結果パターン』の種類は 3 種類(多くの UP20/25/30 等の行が実質的に同一結果の重複であることを示す)。

### fold毎の最終合格数の推移 (multi_oos_folds_up.csv)

- fold1=20, fold2=20, fold3=15, fold4=0

### 承認済みポリシーの最新OOS成績

- `strategy_policy.json` (strategy_name=`UP20_SCORE80_NIKKEION_TP4.0_SL1.0_H1`):
  - 最新fold(up)の候補群(DEV top50由来)に、up/score/nikkeiが一致する行が1件も存在しません。この承認済みパラメータ組み合わせは今回のfoldのstability_objective上位50に入らなかったため、最新OOSでの直接の成績はこのファイル群からは導出できません(validatorの再実行が必要であり、本スクリプトはそれを行いません)。
- `strategy_policy_up.json` (strategy_name=`UP20_SCORE70_NIKKEIOFF_TP3.0_SL1.0_H3`):
  - 一致方法: nearest(up/score/nikkei一致,tp/sl/hold違い)
  - oos_signals=173, oos_pf=2.0761051010731695, oos_avg_return=1.2804637362864861, oos_dd=-33.24673072790452, oos_monthly_positive_ratio=55.00000000000001, oos_pf_ratio=0.28510461737703907, oos_pass=False

## DOWNトレンド

### VALIDATION

- 候補数 n = 50 / 合格 = 0 / 不合格 = 50

| gate | 不合格件数 | この gate のみ無視した場合に追加で通過したはずの件数(情報提供のみ) |
|---|---|---|
| トレード数不足 (gate_signals) | 0 | 0 |
| PF不足 (gate_pf) | 0 | 0 |
| 平均リターン不足 (gate_avg_return) | 0 | 0 |
| 最大DD超過 (gate_dd) | 11 | 0 |
| 年率シグナル数不足 (gate_annual_signals) | 0 | 0 |
| 月次プラス比率不足 (gate_monthly_positive_ratio) | 50 | 33 |
| ブートストラップ下限<=0 (gate_avg_lower) | 17 | 0 |

上位の失敗ゲート組み合わせ(多い順、上位5件):

- 「月次プラス比率不足」: 33件
- 「最大DD超過;月次プラス比率不足;ブートストラップ下限<=0」: 11件
- 「月次プラス比率不足;ブートストラップ下限<=0」: 6件

全50件の戦略のうち、ゲート判定結果(全gateの合否パターン)が完全に一致する重複を除くと、識別可能な『結果パターン』の種類は 3 種類(多くの UP20/25/30 等の行が実質的に同一結果の重複であることを示す)。

### OOS

- 対象行なし(ファイルが空、またはヘッダのみ)。

### fold毎の最終合格数の推移 (multi_oos_folds_down.csv)

- fold1=0, fold2=0, fold3=0, fold4=0

### 承認済みポリシーの最新OOS成績

- downトレンド専用の承認済みポリシーファイルは存在しません。

### DOWN固有の参考指標(ゼロ取引月・アクティブ月のみの positive ratio)

- adversarial_validation_results_down.csv / adversarial_oos_results_down.csv には`validation_monthly_positive_ratio`(全暦月に対する比率、ゼロ取引月を0%として含む)と`validation_months`/`validation_positive_months`という**集計済みの件数**のみが保存されており、各月が「ゼロ取引だったのか」「取引はあったが負けただったのか」を区別する月別の明細(per-month trade count)はこのリポジトリのどのCSVにも保存されていません。
- そのため『ゼロ取引月の件数』『アクティブ月(取引が1件以上あった月)のみに限定したpositive ratio』は、現在保存されているファイルからは**導出不可能**です。
- これを導出するには、adversarial_strategy_validator.py側で(validatorを変更せずにこのスクリプトから直接得るのではなく、将来の拡張として)月ごとのtrade件数(signals_per_month)を追加でCSV出力する必要があります(stats()関数内のmonthly Series、または trade_diagnostics の日付をdown trend別に月次集計したものを追加の出力列として export することが考えられます)。本タスクの制約により、validator自体は変更していません。
