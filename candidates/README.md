# candidates/ — 週次オプティマイザの候補policy(未昇格)

## これは何か

`profit-optimizer-validation.yml`(週次オプティマイザ、現在 `disabled_manually`)が
生成するファイルです。このディレクトリの内容は **一度も本番取引に使われません**。
書き込まれるのは次のファイルだけです。

- `candidate_strategy_policy.json` — 案3(先物トレンド無視)の候補policy
- `candidate_strategy_policy_up.json` — 先物トレンド=上昇 専用候補policy
- `candidate_strategy_policy_down.json` — 先物トレンド=下降 専用候補policy
- `latest_comparison.md` / `latest_comparison_up.md` / `latest_comparison_down.md`
  — 直近runでの「LIVE(ルート直下のstrategy_policy*.json) vs CANDIDATE」比較表
  (読み取り専用の比較。署名やライブファイルには一切触れません)

## なぜ分離したか

以前は `profit-optimizer-validation.yml` が `strategy_policy.json` /
`strategy_policy_up.json` / `strategy_policy_down.json`(本番の
`profit_top10_paper.py` が読む、本番取引に直結するファイル)を直接
週次で上書きしていました。ワークフローは現在ユーザーの手動判断で
`disabled_manually` になっていますが、再度有効化したときに「本番へ
自動反映される」挙動を避けるため、出力先を丸ごと `candidates/` 配下の
別ファイル名(`BSP_POLICY_FILE` 環境変数で指定)へ切り替えました。
ワークフロー自身にも、コミット直前にルート直下の
`strategy_policy.json` / `strategy_policy_up.json` /
`strategy_policy_down.json` / `policy_manual_overrides.json` が
変更されていないことを確認して失敗させるガードステップを追加しています
(defense in depth)。

`build_strategy_policy.py` 自体は変更していません。署名対象ペイロード
(`canonical_policy_payload`)は `status` / `updated_at` / 各policy値などの
フィールドのみで構成され、**ファイル名やディレクトリは含まれません**。
そのため、承認済みのcandidateファイルをそのまま本番ファイル名へ
byte-identicalコピーしても署名(`approval_signature`)は有効なままです。

candidateのファイル名は本番ファイルと同じ `strategy_policy*.json` に
しませんでした。`build_strategy_policy.py` の30日保持ロジックは
`BSP_POLICY_FILE` に渡された文字列そのもの(パス込み)で
`policy_manual_overrides.json` のエントリと突き合わせるため、
`candidates/strategy_policy.json` のような同一basenameでも実際には
誤マッチしません(エントリの `policy_file` はパス無しの
`"strategy_policy.json"` で記録されており、`candidates/strategy_policy.json`
という文字列とは一致しないため)。それでも将来どこかのスクリプトが
basename単位で照合するように変わった場合に備え、
`candidate_strategy_policy*.json` という明確に異なるbasenameを採用し、
本番ファイルとの取り違えを構造的に不可能にしています。

## 昇格(promotion)手順 — 必ず人間が確認してから実行する

1. `candidates/latest_comparison*.md` と `$GITHUB_STEP_SUMMARY` の比較表を確認し、
   採用したいcandidateを選ぶ。
2. ルート直下へ **byte-identical** コピーする(署名を保つため、中身を一切
   加工しないこと)。

   ```bash
   cp candidates/candidate_strategy_policy.json strategy_policy.json
   # トレンド版の場合:
   cp candidates/candidate_strategy_policy_up.json strategy_policy_up.json
   cp candidates/candidate_strategy_policy_down.json strategy_policy_down.json
   ```

3. コピーが本当にbyte-identicalであることを検証する。

   ```bash
   sha256sum candidates/candidate_strategy_policy.json strategy_policy.json
   ```

   両者のハッシュが一致することを確認する。

4. 署名検証(任意だが推奨)。`AI_POLICY_SIGNING_SECRET` を持つ環境で
   `python check_policy_metadata.py` (読み取り専用)などを使い、
   コピー後のファイルの `approval_signature` が壊れていないことを
   確認する。

5. 変更をコミットする(このコミットは人間がレビュー・承認した上で行う。
   `profit-optimizer-validation.yml` 自身はこのコミットを作らない)。

## ロールバック手順

昇格後に問題が見つかった場合は、直前のコミットへ戻す。

```bash
# 直前の1コミットだけを打ち消す場合
git revert <昇格コミットのSHA>

# あるいは特定ファイルだけを1つ前の状態に戻す場合
git checkout <昇格前のSHA> -- strategy_policy.json
git commit -m "revert: strategy_policy.json を昇格前の状態へ戻す"
```

いずれの方法でも、本番が読むのはルート直下の `strategy_policy*.json` のみなので、
`candidates/` 配下のファイルは変更・削除しなくても本番挙動には影響しません。
