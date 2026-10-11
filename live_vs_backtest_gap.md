# ライブ・ペーパートレード vs バックテスト ギャップ分析

⚠️ **サンプルサイズの注意**: 対象トレード数は合計 19 件(TOP1トラック + daytradeトラック、2026-09-01以降の決済済みトレードのみ)。これは統計的に意味のある結論を出すには極めて少ないサンプルであり、以下の『原因の寄与度』はあくまで目視・定性的な傾向であって、厳密な統計的分解ではない。

## TOP1 (profit_top10_paper_history.csv) (n=6)

- ライブ return_pct 合計: -6.099pp (平均 -1.016pp/trade)
- バックテスト規約での再現 return_pct 合計: -8.878pp (平均 -1.480pp/trade, n=6/6件で算出可能)

- 2026-09-03 4812.T BUY: live=FORCED_CLOSE_BUDGET_BUG(-0.145pp) / backtest規約=TIMEOUT_LOSS(-0.135063pp) / gap=-0.009981pp / 原因候補=[FORCED_CLOSE_BUDGET_BUG(資金不足による強制決済、タグ付きデータ);pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード;policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。;exit_reason_diff: ライブ=FORCED_CLOSE_BUDGET_BUG / backtest規約=TIMEOUT_LOSS(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-09-07 6098.T BUY: live=FORCED_CLOSE_BUDGET_BUG(-0.419pp) / backtest規約=SL(-3.021328pp) / gap=2.601997pp / 原因候補=[FORCED_CLOSE_BUDGET_BUG(資金不足による強制決済、タグ付きデータ);pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード;policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。;exit_reason_diff: ライブ=FORCED_CLOSE_BUDGET_BUG / backtest規約=SL(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-09-07 4062.T BUY: live=FORCED_CLOSE_BUDGET_BUG(-0.110pp) / backtest規約=SL(-5.198234pp) / gap=5.088234pp / 原因候補=[FORCED_CLOSE_BUDGET_BUG(資金不足による強制決済、タグ付きデータ);pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード;policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。;exit_reason_diff: ライブ=FORCED_CLOSE_BUDGET_BUG / backtest規約=SL(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-09-24 5019.T BUY: live=SL(-3.178pp) / backtest規約=HOLD(0.535401pp) / gap=-3.713579pp / 原因候補=[pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード;policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。;exit_reason_diff: ライブ=SL / backtest規約=HOLD(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-09-30 7267.T BUY: live=HOLD_LIMIT(0.339pp) / backtest規約=HOLD(1.517735pp) / gap=-1.178611pp / 原因候補=[pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード;policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。]
- 2026-10-08 6472.T BUY: live=SL(-2.586pp) / backtest規約=SL(-2.576878pp) / gap=-0.008638pp / 原因候補=[policy_mismatch_or_unknown: この行はpolicy_file列を持たない旧スキーマ(TOP1 profit_top10_paper_history.csv)のため、現行のstrategy_policy.jsonを代替として仮定した(当時実際に使われていたポリシーファイル・ハッシュは履歴に記録されておらず特定不能)。]

## daytrade (daytrade_tp5000_sl8000_history.csv) (n=13)

- ライブ return_pct 合計: -3.208pp (平均 -0.247pp/trade)
- バックテスト規約での再現 return_pct 合計: 2.193pp (平均 0.219pp/trade, n=10/13件で算出可能)

- 2026-10-02 5301.T BUY: live=TP(0.543pp) / backtest規約=TP(0.552894pp) / gap=-0.010359pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない)]
- 2026-10-02 6302.T BUY: live=SL(-1.279pp) / backtest規約=TP(0.809605pp) / gap=-2.088173pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);exit_reason_diff: ライブ=SL / backtest規約=TP(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-10-02 8766.T BUY: live=SL(-1.012pp) / backtest規約=SL(-0.308915pp) / gap=-0.703258pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない)]
- 2026-10-02 6752.T BUY: live=FORCED_EXIT(0.021pp) / backtest規約=TP(0.557527pp) / gap=-0.53628pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);exit_reason_diff: ライブ=FORCED_EXIT / backtest規約=TP(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-10-05 6752.T BUY: live=SL(-0.849pp) / backtest規約=SL_BOTH_SL_PRIORITY(-0.3136pp) / gap=-0.535116pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない)]
- 2026-10-05 5301.T BUY: live=TP(0.595pp) / backtest規約=TP(0.545pp) / gap=0.050343pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない)]
- 2026-10-05 8766.T BUY: live=TP(0.520pp) / backtest規約=TP(0.530113pp) / gap=-0.010347pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない)]
- 2026-10-05 5301.T BUY: live=TP(0.524pp) / backtest規約=SL_BOTH_SL_PRIORITY(-0.311413pp) / gap=0.835302pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);exit_reason_diff: ライブ=TP / backtest規約=SL_BOTH_SL_PRIORITY(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-10-05 8766.T BUY: live=FORCED_EXIT(-0.247pp) / backtest規約=TP(0.52533pp) / gap=-0.772243pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);exit_reason_diff: ライブ=FORCED_EXIT / backtest規約=TP(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-10-06 6645.T BUY: live=FORCED_EXIT(0.136pp) / backtest規約=SL(-0.393357pp) / gap=0.529489pp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);exit_reason_diff: ライブ=FORCED_EXIT / backtest規約=SL(5分足リアルタイム判定 vs 日足のみの判定の差、またはFORCED_EXIT/HOLD_LIMITの扱いの差)]
- 2026-10-09 8766.T BUY: live=SL(-0.804pp) / backtest規約=None(Nonepp) / gap=Nonepp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);no_price_data: 日足データが取得できず再現不可(ネットワーク環境依存)]
- 2026-10-09 4704.T BUY: live=SL(-1.058pp) / backtest規約=None(Nonepp) / gap=Nonepp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);no_price_data: 日足データが取得できず再現不可(ネットワーク環境依存)]
- 2026-10-09 8766.T BUY: live=FORCED_EXIT(-0.300pp) / backtest規約=None(Nonepp) / gap=Nonepp / 原因候補=[exit_rule_mismatch: daytradeトラックは固定ネット金額(+5000/-8000円)を5分足で判定する決済ルールであり、validatorが検証したATR%ベース・日足判定の決済ルールとは構造的に異なる(本行のbacktest再現値はATR%日足規約での参考値であり、ライブの実際の決済ルールそのものではない);no_price_data: 日足データが取得できず再現不可(ネットワーク環境依存)]

## 原因の順位付け(件数ベース、情報提供のみ)

| 原因候補 | 該当トレード数 | 全体に対する割合 |
|---|---|---|
| exit_rule_mismatch | 13 | 68% |
| exit_reason_diff | 9 | 47% |
| policy_mismatch_or_unknown | 6 | 32% |
| pre-2026-10-06 HOLD_LIMITロールバックバグの影響を受け得る期間のトレード | 5 | 26% |
| FORCED_CLOSE_BUDGET_BUG | 3 | 16% |
| no_price_data | 3 | 16% |

## 選定の正当性(selection validity)について

ライブで選ばれた銘柄が、その取引日においてvalidatorの候補生成ルール(select_for_phase: up_prob/down_prob/flat_prob/score閾値+日経レジームゲート)のもとでも候補になっていたかを検証するには、その日のAI予測全銘柄分のup_prob/down_prob/flat_prob/scoreスナップショット(walk_forward_all_candidates.csv相当のその日の全候補、または少なくとも当日のprediction_history.csv全銘柄分)が必要である。しかしdaytrade_tp5000_sl8000_history.csvのtop10_rank列・top10_source列は『ライブがどのランクで選んだか』のみを記録しており、その日の全候補集合(落選した銘柄を含む全体)のスナップショットはこのリポジトリのどのファイルにも永続化されていない。したがって選定の正当性は本データからは検証不可能であり、検証するには当日全銘柄分の候補スナップショットを追加で保存するようワークフローを変更する必要がある(本タスクでは変更していない)。

## 手数料・スリッページの扱いの違い

- validator (adversarial_strategy_validator.evaluate_trade): リターンから片道0.1%(slippage=0.001)のみ減算。往復手数料(FEE_PCT)は候補優先順位(EV)計算にのみ使われ、実現P&Lには反映されない。
- ライブ profit_top10_paper.py (TOP1): pnl = gross - (entry_price+exit_price)*shares*FEE_RATE で往復手数料を明示的に差し引く(FEE_RATE既定0.00055=0.055%/回、往復で約0.11%)。
- ライブ daytrade_tp5000_sl8000_paper.py: TP/SL自体を『往復手数料差し引き後にネットで+5000円/-8000円』になるように価格から逆算しており、手数料は決済価格そのものに組み込まれている。
- 結論: validatorのバックテストはライブより手数料に関して楽観的(片道0.1%のみ)であり、ライブは毎トレードで往復手数料(約0.11%)を追加で負担している。これは方向として『ライブがバックテストより悪く見える』ことの一部説明になるが、1トレードあたり0.11%程度であり、観測されているギャップの規模(1トレードあたり数%〜)全体を説明するには小さすぎる。

## HOLD_LIMITバグ (pre-2026-10-06) について

profit_top10_paper.py は2026-10-06のコミット(c4782fb)以前、yfinanceのTSE 5分足に15:25のバーが存在しないため、保有上限(hold_days)に達したポジションのHOLD_LIMIT決済が実質的に発火せず、ポジションが無期限に保有され続けるバグがあった(修正: 前営業日の日足を遡るロールバック分岐でHOLD_LIMIT決済するよう追加)。profit_top10_paper_history.csv の2026-09-01以降の行はいずれもこの修正日より前のエントリーであり、このバグの影響を受けた可能性がある(影響の正確な定量化には当時のstateファイルの保有履歴が必要で、本データからは『影響を受け得る期間だった』ことまでしか確認できない)。

## FORCED_CLOSE_BUDGET_BUG タグについて

- 3件がFORCED_CLOSE_BUDGET_BUGタグ付きだった(資金不足等により通常の決済ルールではなく強制決済されたことを示す履歴上のタグ)。これらの行はvalidatorが想定する正常な決済経路を経ていないため、バックテストとの比較対象として本質的に公平ではない。
