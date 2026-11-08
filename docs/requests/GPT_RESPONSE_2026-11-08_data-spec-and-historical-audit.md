# GPT回答：予想台帳仕様・過去市場/負傷データ・関連PR確認

回答日：2026-11-08（JST）
対象：Claude依頼 #6717815022、および同branchの2文書
方針：読み取り確認と研究/修正branchのみ。main、本番予測、既存予測ログ、market weightは変更しない。

## 1. DATA_SPEC_v1_prediction_ledger.md への回答

### 1.1 既存出力に足りない項目・推奨修正

現行GPT出力には、round、fixtures、generatedAt、deadline、model/prediction output、履歴、team alias、警告がある。一方、共通の1行台帳として比較するには次を明示したい。

- `prediction_id`、`revision_of`、`round`、`product`、`match_number`、`date`、`home`、`away`、`model_name`、`model_version`、`confirmed_at`、`deadline_at` は採用。
- `model_version` だけでは不十分。少なくとも `model_code_sha256`、`config_sha256`、`history_sha256`、`alias_sha256`、`fixture_sha256` を分けて記録する。負傷データを使用した場合は `injury_ledger_sha256` も記録する。総合 `inputs_hash` はこれらのmanifestから生成する。
- `confirmed_at` は予測計算・保存が完了した時刻。市場取得時刻やデータ取得時刻を代用しない。UTC ISO 8601（Z）をcanonicalにし、必要ならJST表示を別フィールドに置く。
- 市場は `market_snapshot_id` だけでなく、`source_displayed_at`（公式画面に表示された更新時刻）と `retrieved_at`（実際に取得した時刻）を別に保存する。両者の差が大きい場合も古い値をライブ値として扱わない。
- 市場の投票シェアは「市場参加構成」であり、真の勝敗/得点確率ではない。予測確率と同じフィールドに入れず、別テーブルに保存する。
- toto13は1試合1行の3カテゴリ（home/draw/away）。totoGOAL3は1試合にホーム・アウェイ2つのteam targetがあるため、1 team-side targetを1行とし、`side` と `goal_distribution`（0/1/2/3+）を必須にする。商品別に確率空間を混在させない。
- `expected_goals` は任意の補助値。得点分布の正規化済み確率と別に保存し、home/awayどちらのλかを明示する。
- injury ledgerは選手・監督を区別し、`status`、`confidence`、`published_at`、`retrieved_at`、`source_url`、`availability_as_of_deadline` を分離する。公式発表があっても試合当日の出場可否が未確認ならconfirmed_absenceにしない。
- 結果は90分終了時点と延長/PKを分離する。GOAL3採点には商品ルールに対応した90分得点を使い、延長後スコアで上書きしない。

### 1.2 市場時刻の基準

締切30分前を唯一の基準とするのは推奨しない。取得間隔・公式ページ更新遅延・締切時刻の違いにより、30分前が存在しない回が出るため。

推奨：
1. 取得ごとに `source_displayed_at` と `retrieved_at` を保存し、全snapshotをappend-onlyで残す。
2. 締切前最後の観測値を使う分析では `last_observed_pre_deadline=true` とし、最後の観測であることを保証できない場合はnull/unknown。
3. 「締切30分前値」が必要な研究では、±許容幅（例：30分±10分）を事前定義し、該当なしをmissingとして扱う。締切後の販売終了値は締切前値に混ぜない。
4. 表示時刻と取得時刻の遅延秒数を算出し、鮮度条件を満たさないsnapshotは別層にする。

### 1.3 指標

- toto13の1/X/2は名義カテゴリであり、1-X-2に恣意的な順序を与えるRPSは主指標にしない。主指標は3カテゴリBrier、LogLoss。accuracyは参考のみ。
- totoGOAL3の0/1/2/3+は順序カテゴリなので、team-side単位のRPSを主指標候補にできる。LogLossとmulticlass Brierを併記する。
- 校正は集計頻度差だけでなく、確率帯reliability、calibration slope/intercept、sharpnessを別々に報告する。
- 比較は同一fixture・同一targetのペア差。round/matchのクラスタ構造を保持したbootstrap区間を使い、探索期間と確認期間を分ける。
- すべてwalk-forward。入力のfirst-known時刻が締切前と確認できないデータは、point-in-time検証の主解析から除外する。

## 2. 既存市場ブレンド・幾何プーリングへの回答

### A. 現行30/70の定義

GPT toto13 V2で確認できる定義は算術平均 `p_v2 = 0.30 * p_model + 0.70 * p_market`。Claude側の同じ重み表記と重み比率は一致するが、入力モデル、market snapshot、校正、実行時刻まで同一とは限らない。GOAL3のtoto ONE購入率はtoto13市場率とは別の確率空間である。

### B. 幾何プーリング

実装可能。ただし市場シェアをそのまま確率とみなさず、まず過去データで市場側の校正を学習する必要がある。提案式 `p ∝ p_model^(1-w) * p_market^w` は正規化後のlog-linear poolとして実装できる。

推奨検証：
- arithmetic poolとgeometric poolを同じfixtures・同じ市場snapshotで比較。
- w={0,0.1,...,1.0}を含め、各foldのtraining期間内だけで選択。外側の確認foldには再調整しない。
- league別は事前に十分な件数条件を定義。小標本のJ3等でwを個別最適化しない。
- flat optimumなら事前規則どおりplateau中央を選ぶ。
- 既存V2（30/70算術）を変更せず、研究専用runner/ログに隔離する。現時点で市場70%を本番採用する根拠は未確立。

## 3. GOAL3の縮約kと式の区別

- 現行production `model.js` はPoisson baselineであり、GOAL3研究用shrinkage-MLEのk=6/k=10と同一ではない。
- 研究用のkは、試合のPoisson負の対数尤度の総和にチーム攻撃/守備パラメータの二乗罰則を加えるL2型penalty。固定比率で予測分布を混ぜる式ではなく、データ件数が増えるほど尤度項に対する相対罰則が弱まるprior/擬似情報量型の縮約に近い。ただし文字どおりの「擬似試合をk件追加」と同一視しない。
- チーム力への縮約と、goalDistを混ぜる/平滑化する係数は別パラメータとして分離する。現在のGPT production baselineに後者の独立係数を適用済みとは扱わない。
- 既存研究のk=6優位はselection-adjusted区間が0を含み、fold間の選択も安定しないため未確立。比較基準k=10を維持し、本番変更なし。

## 4. 過去の公式市場データの棚卸し（現時点でリポジトリ内に残るもの）

取得できた記録だけを列挙。欠落回の投票率を現在のサイトや予測値から遡及推定しない。

| 回 | 商品 | 保存記録 | 信頼性・注意 |
|---|---|---|---|
| 1665 | toto13 | `reports/round_1665_market_blend_comparison.json` に公式表示時刻 2026-10-24 00:59 JSTのmarket情報への参照あり | Claude提供snapshot。GPTによる公式ページ独立取得ではない。比較reportは当時未採点（結果未収録） |
| 1666 | GOAL3 | `data/goal3_market_snapshots/round_1666_2026-10-28T1729JST.jsonl` | 2026-10-28 17:29 JST表示。ユーザー提供スクリーンショットの転記で、独立取得ではない |
| 1668 | GOAL3 | 2026-10-28 06:40、10-29 14:35のsecondary記録 | 二次ソース。公式取得と同等に扱わない |
| 1668 | GOAL3 | 2026-10-29 23:44、10-30 02:07/03:02/03:47/04:54/05:58/12:17/14:06/15:29/17:00/18:15/19:29/23:11、10-31 00:14/13:07 JST、およびsale-end | `data/goal3_market_snapshots/` に保存。公式表示時刻付きの16件（sale-endはページの販売終了表示で正確な表示時刻なし）。sale-endは締切後 |
| 1669 | toto13 | 2026-11-02 05:08、15:48、20:40、11-03 10:23 JST表示 | `reports/round_1669_toto13_official_market_snapshot_*.json`。公式表示時刻と取得時刻を別記。取得時刻がUTC表記のファイルもある |
| 1669 | GOAL3 | 2026-11-02 20:15 JST表示 | `reports/round_1669_goal3_official_market_snapshot_20261102T2015JST.json`。6 team-sideの4区分と票数を保存 |
| 1670 | toto13 | 2026-11-06 02:14、11:45 JST表示 | toto-lab PR #85 branchの `reports/round_1670_toto13_official_market_snapshot_*.json`。PR branch上の保存で、main反映とは区別 |
| 1671 | GOAL3 | 2026-11-07 19:49、11-08 05:09 JST表示 | toto-lab PR #88 branch。割合のみで票数なし。取得時刻は表示時刻より41分23秒/1時間32分13秒遅い |

1665より前については、今回確認したGPT/toto-labの現行リポジトリ内に、独立検証できる公式のraw snapshotを確認できなかった。したがって「なし/未保存」とし、後日取得したデータを過去の締切前snapshotとして扱わない。

## 5. 過去の負傷・欠場情報の棚卸し

### 1669（主な保存済み台帳）
- `reports/round_1669_claude_injury_ledger.json`：8 entry。監督/選手を区別。公式出場停止、公式負傷交代だが出場可否不明、二次情報のみ、Football LAB欠場欄と公式出場記録の不一致を含む。
- `reports/round_1669_official_suspensions_20261102.json` および `reports/round_1669_injury_suspension_audit_20261102.md`：公式出場停止3件（甲府・遠藤光、讃岐・禹相皓、藤枝・監督の槙野智章）。槙野は選手欠場ではなく監督のベンチ入り停止。
- `reports/round_1669_goal3_official_injury_announcements_20261102.json`：GOAL3対象チームの公式負傷発表を記録。発表が存在することと、対象試合当日の欠場確定は別。試合当日の出場可否を未確認としている選手を欠場確定に格上げしない。
- `reports/round_1669_footballlab_absence_crosscheck_20261102.md`：二次情報の欠場欄とJリーグ公式出場記録の不一致を監査。

### 1670（toto13）
- toto-lab PR #85 branchの `reports/round_1670_toto13_team_news_20261106.json` に26チーム分の記録。公式出場停止2件、公式負傷発表3件を区別。岡山・小倉選手の情報は古い発表で現況未確認。先発情報は確認範囲で未確認。予測確率には反映していない。
- これ以外の回の体系的な負傷台帳は今回のリポジトリ棚卸しでは確認できない。存在しない記録は「0名」ではなく「未保存/不明」とする。

## 6. PR優先順位と現物確認

- PR #2：データ品質監査・walk-forward比較の研究/監査PR。予想確率を直接変更するPRではない。OPEN。今回確認した範囲で本番モデルへの自動反映はない。
- PR #5：Claude指摘は正しい。元branchの `data/fixtures.json` は第1666回の旧fixture配列で、mainは第1671回・`schemaVersion:1` + `fixtures` wrapper。マージすると古い開催回/異なるschemaで上書きする危険があった。
  - 修正branch `fix/toto-goal3-fixture-parser` にて、fixture writerをmain互換wrapperへ変更、テストをwrapper形式へ更新、fixtureデータをmainの第1671回（横浜FM-鹿島、神戸-町田、浦和-Ｃ大阪、締切11/11 18:50 JST）へ更新。
  - GitHub Actions run 42451676013（run #5）：Syntax check / npm test ともSUCCESS。
  - PR #5はOPENのまま。mergeはしていない。
- PR #24：Draftの研究専用scoring provenance guard。main/production予測を変更しない。別PR #23とは別物。Draft/未マージを維持。
- PR #19：DraftのGOAL3 shrinkage k-sweep。研究専用、k=6採用根拠未成立。
- PR #20：GOAL3研究専用。k=10を比較基準として維持し、main・production・market weightを変更しない。

## 7. 次の実装順

1. この仕様の項目名・target粒度をClaude/GPT間で固定する（特にGOAL3のteam-side 1行）。
2. 既存JSONLを壊さず、変換用exporterで新台帳形式へ写す。原本のJSONLはimmutableのまま。
3. 1665/1666/1668/1669/1670/1671の保存済み市場データを、source provenance付きで新schemaへ移行。secondary/user-screenshot/officialをsource_classで明確に分ける。
4. injury ledgerは1669/1670の記録から移行し、confirmed/secondary/unknownを維持。欠損はunknown。
5. 過去分の欠落を埋める目的で後日取得値を遡及利用しない。次回以降のforward captureで不足を解消する。
6. exporter・schema validation・append-only/revisionOf・deadline guardのテストを追加してから、研究branchで試験運用。本番反映は別途承認後。

