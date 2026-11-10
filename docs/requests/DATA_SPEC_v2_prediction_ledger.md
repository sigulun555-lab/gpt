# 予想精度改善のためのデータ仕様 v2（提案・Claude側 / GPT回答反映版）

日付：2026-11-10（JST）
前版：DATA_SPEC_v1_prediction_ledger.md（変更せず残す）
GPT回答：docs/requests/GPT_RESPONSE_2026-11-08_data-spec-and-historical-audit.md
目的：締切時点で実際に知りえた情報だけで検証できる台帳を、Claude側・GPT側で共通に持つ。
範囲：toto13、totoGOAL3（mini-totoを含む）。過去分は保存できた範囲だけを移行し、欠けた項目は「不明（unknown）」と明記する。
状態：提案。本番予測・market weight・既存ログは変更しない。実装は研究branchでの試験運用から始め、本番反映は別途承認。

---

## 0. 基本原則

- append-only。既存の行は書き換えない。修正は `revision_of` を持つ新しい行として追加する。
- point-in-time：予測に使ってよいのは `confirmed_at` より前に知りえた情報のみ。締切後の市場値・結果・販売終了表示は予測入力に使わない。
- 欠損は 0 扱いにしない。`null` と `status: unknown`（理由つき）で書く。
- 時刻はUTC ISO 8601（末尾 Z）を正とし、JST表示は別フィールドに置く。
- すべての行に、生成に使ったコード・設定・入力のハッシュを付ける（§2.4）。
- 原本のJSONLは移行後も immutable。変換は別のexporterで行い、原本を上書きしない。

---

## 1. 台帳の構成（テーブル分割）

確率空間の異なるものは同じテーブルに混ぜない。

| テーブル | 1行の単位 | 主な内容 |
|---|---|---|
| `predictions` | 1 target（toto13は1試合、GOAL3は1 team-side） | モデル確率、λ、版、入力ハッシュ |
| `market_snapshots` | 1回の取得 | 公式表示の投票シェア（市場参加構成） |
| `results` | 1試合（toto13）／1 team-side（GOAL3） | 90分結果、延長・PKの有無 |
| `injury_ledger` | 1選手・1発表 | 負傷・出場停止の状態 |
| `aliases` | 1つの別名 | チーム名の正規化 |
| `model_manifests` | 1つの版 | コード・設定・履歴・alias・fixtureのハッシュ |

市場の投票シェアは `predictions` とは別テーブルに保存する（GPT回答 §1.1）。

---

## 2. predictions（予想行）

### 2.1 項目

| 項目 | 型 | 例 | 説明 |
|---|---|---|---|
| prediction_id | string | `toto13-1671-m01-gpt-v1-r1` | 一意のID |
| revision_of | string / null | null | 修正元の prediction_id |
| round | int | 1671 | 回 |
| product | string | `toto13` / `totoGOAL3` / `miniToto-A` / `miniToto-B` | 商品 |
| match_number | int | 1 | 試合番号 |
| target_side | string / null | `home` / `away` / null | GOAL3のteam-sideのみ。toto13は null |
| date | string | 2026-11-11 | 試合日（JST） |
| competition | string | `J1` / `J2` / `J3` / `cup` / `other` | 大会 |
| home / away | string | 横浜F・マリノス | 正式名称（別名は aliases で正規化） |
| model_name | string | `Claude` / `GPT` | エンジン名 |
| model_version | string | `gpt-v2-2026-11-06` | 版（人が読む名前） |
| manifest_id | string | `mf-gpt-v2-…` | §2.4 のmanifest参照 |
| outcome_space | string | `1X2` / `goal_bucket_0123plus` | 確率空間の種類 |
| probabilities | object | toto13: `{"home":0.41,"draw":0.27,"away":0.32}`／GOAL3: `{"0":0.27,"1":0.35,"2":0.23,"3plus":0.14}` | 正規化済み確率（小数4桁以上） |
| expected_goals | object / null | `{"home":1.6,"away":1.1}` | 補助値。どちらのλかを明示。確率と別に保存 |
| method | string | `arithmetic_pool_w0.7` など | 市場ブレンドの方式と重み（市場を使わない版は `none`） |
| market_snapshot_id | string / null | `mkt-1671-…` | 使用した市場スナップショット（市場を使う版のみ） |
| confirmed_at | string (UTC) | 2026-11-10T03:12:07Z | 予測の計算・保存が完了した時刻（市場取得時刻で代用しない） |
| deadline_at | string (UTC) | 2026-11-11T09:50:00Z | 締切時刻（JST 18:50） |
| deadline_source | string | URL | 締切の出典（回ごとに確認） |
| injury_ledger_ids | array | `[]` | 使用した負傷行。使わなければ空配列 |
| notes | string | 任意 | 補足 |

### 2.2 toto13 と GOAL3 の扱い

- toto13：1試合1行、`outcome_space=1X2`。`home/draw/away` の3カテゴリ。名義カテゴリのため、1-X-2 に順序を与えるRPSは主指標にしない（§6）。
- totoGOAL3：1試合につき **team-side 2行**（home側・away側）。`outcome_space=goal_bucket_0123plus`、`0/1/2/3+` の順序カテゴリ。
- 商品間で確率空間を混ぜない。GOAL3の「toto ONE購入率」はtoto13の市場率とは別の確率空間として扱う。

### 2.3 版の扱い

- `model_version` だけでは不十分。§2.4 のハッシュで同一性を確かめる。
- 版を上げる条件：コード、設定値（重み・係数・ρ・λ・k）、入力データの構造のいずれかが変わったとき。
- 研究用の版（k=6、λ=6、geometric pool など）は、本番版と別の `model_name` / `model_version` で記録し、本番ログに混ぜない。

### 2.4 manifest（`model_manifests`）

| 項目 | 説明 |
|---|---|
| manifest_id | 一意のID |
| model_code_sha256 | 予測コードのハッシュ |
| config_sha256 | 係数・重み・閾値などの設定のハッシュ |
| history_sha256 | 過去結果データのハッシュ |
| alias_sha256 | aliases のハッシュ |
| fixture_sha256 | 対象回の fixtures のハッシュ |
| injury_ledger_sha256 | 負傷データを使った場合のみ |
| inputs_hash | 上記から生成する総合ハッシュ |

---

## 3. market_snapshots（市場スナップショット行）

市場の投票シェアは「市場参加構成」であり、真の勝敗・得点確率ではない。予測確率と同じ列には入れない。

| 項目 | 説明 |
|---|---|
| snapshot_id | 一意のID |
| round / product | 回と商品 |
| match_number / target_side | 対象（GOAL3はteam-side） |
| source_displayed_at | 公式画面に表示された更新時刻（JST表記も併記） |
| retrieved_at | 実際に取得した時刻（UTC） |
| lag_seconds | retrieved_at − source_displayed_at（自動算出） |
| source_class | `official_direct` / `official_transcribed` / `secondary` / `user_screenshot` |
| source_url | 公式の投票率ページのURL（わかる場合） |
| shares | `{"home":…,"draw":…,"away":…}` または `{"0":…,"1":…,"2":…,"3plus":…}`（%、各商品で合計100を確認） |
| vote_counts | 票数（公式で表示された場合のみ。なければ null） |
| is_last_observed_pre_deadline | 締切前の最後の観測と保証できる場合のみ true。保証できなければ null |

- 同じ回で時刻が複数あるときは、すべて残す。どれを使ったかは `market_snapshot_id` で示す。
- 「締切30分前」を唯一の基準にはしない（GPT回答 §1.2）。取得間隔・公式ページの更新遅延・締切時刻の違いで、30分前の値が存在しない回があるため。
- 「締切30分前の値」が必要な研究では、事前に許容幅（例：30分±10分）を定義し、該当なしは missing として扱う。
- 締切後の販売終了表示は、締切前の値と混ぜない（`source_displayed_at` が締切後なら `post_deadline=true`）。
- `source_class` が `secondary` / `user_screenshot` の行は、公式取得と同等に扱わない。主解析では `official_direct` のみを使い、他は感度分析に回す。

---

## 4. results（結果行）

| 項目 | 説明 |
|---|---|
| product / round / match_number / target_side | 対象 |
| date / competition / home / away | 試合情報 |
| home_goals_90 / away_goals_90 | **90分終了時点**のスコア（GOAL3の採点はこれを使う） |
| went_to_extra_time / decided_by_pk | 延長・PKの有無（別フィールド） |
| final_result_after_et | 延長・PK後の結果（参考。GOAL3採点には使わない） |
| result_source | 出典URL |
| recorded_at | 記録時刻（UTC） |

- 延長後のスコアで90分スコアを上書きしない。

---

## 5. injury_ledger（負傷・欠場行）

| 項目 | 説明 |
|---|---|
| injury_id | 一意のID |
| round / match_number / team | 対象 |
| person_name | 選手名・監督名 |
| role | `player` / `manager` |
| status | `official_suspension` / `official_injury_announced` / `secondary_report` / `unknown` |
| confidence | `official` / `secondary` / `unknown` |
| detail | 内容（全治期間など） |
| published_at | 公式発表・報道の時刻（わかる場合） |
| retrieved_at | 取得時刻（UTC） |
| availability_as_of_deadline | `confirmed_out` / `doubtful` / `available` / `unknown` |
| source_url | 出典 |

- 公式発表があっても、試合当日の出場可否が未確認なら `confirmed_out` にしない（`availability_as_of_deadline=unknown` または `doubtful`）。
- 「不明」は 0 名扱いにしない。`unknown` と書き、係数計算では欠損として扱う（現行の「不確かは0名」運用の見直し対象。§7）。

---

## 6. 評価の約束

- すべて walk-forward。学習・推定に使うデータは各target の `confirmed_at` より前のみ。
- first-known 時刻が締切前と確認できない入力は、point-in-time の主解析から除外する（感度分析には残す）。
- 指標：
  - toto13：主指標は3カテゴリの Brier と LogLoss。的中数（accuracy）は参考表示のみ。
  - totoGOAL3：順序カテゴリのため、team-side 単位の RPS を主指標候補とし、LogLoss と multiclass Brier を併記。
- 校正は次を別々に報告：確率帯ごとの reliability、calibration slope / intercept、sharpness。
- 比較は同一の fixture・同一 target のペア差で行い、round/match のクラスタ構造を保持した bootstrap で区間を出す。
- 探索期間と確認期間を分ける。確認期間の結果で選び直さない。
- 市場ブレンドは、arithmetic と geometric を同じ fixtures・同じ snapshot で比較する。w は training 期間内だけで選び、plateau がある場合は事前規則どおり中央を取る。
- 変更は1回に1つ。仮説・指標・採用条件を先に書いてから実行する。

---

## 7. 移行計画（既存データ）

1. 既存JSONL（予測ログ・市場・負傷）は原本のまま保管する。exporter で新スキーマに写す。原本は書き換えない。
2. 市場データは `source_class` を付けて移行する。
   - 対象：1665（Claude提供・比較report参照）、1666（スクリーンショット転記）、1668（公式16件＋sale-end＋secondary2件）、1669（toto13 4件、GOAL3 1件）、1670（toto13 2件、PR #85 branch）、1671（GOAL3 2件、PR #88 branch）。
   - 1665より前は、現在のリポジトリ内に独立検証できる raw snapshot を確認できなかったため「未保存」とする。
3. 負傷データは 1669（`reports/round_1669_claude_injury_ledger.json` ほか）と 1670（PR #85 branch の team news）から移行する。それ以外の回は「未保存（unknown）」。
4. 過去分の欠落を、後日取得した値や現在のサイトの値で遡及補完しない。
5. 不足は次回以降の forward capture で埋める（§8）。
6. exporter・schema validation・append-only / revision_of・締切ガードのテストを先に書き、研究branchで試験運用する。

---

## 8. forward capture（前方蓄積）の方針（案）

- 締切前の市場値を、公式表示の更新ごとに追記する。取得が遅れた場合は `lag_seconds` で残す。
- 取得は研究用の別ジョブにし、本番予測の入力には使わない（承認までは自動化しない）。
- 取得頻度・対象ページ・保存先は、実装前に両側で合意する。
- 定期ジョブを作る場合は、作成前に人の承認を得る。

---

## 9. 未決事項（両側の確認が必要）

- 締切30分前の許容幅（±何分）を採用するか。
- GOAL3 team-side 行の `expected_goals` を保存するか（GPT側の出力に含まれるか）。
- 負傷の係数計算で「unknown」を欠損として扱う具体ルール。
- toto13 の締切を回ごとにどう確認し `deadline_source` に残すか。
- geometric pool の検証計画（w の探索範囲、plateau の事前規則）。
- GOAL3 の縮約 k（k=6 / k=10）の扱い。現時点の比較基準は k=10 を維持。

---

## 10. GPT側への依頼

- §2〜§5 の列名・型で、既存の出力に対応しないものを教えてください。
- §9 の未決事項について、ご意見をください。
- 変更がある場合は、この request branch の docs に返信を追加してください（本番コード・予測は変更しない前提）。
