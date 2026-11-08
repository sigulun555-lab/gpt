# 予想精度改善のためのデータ仕様 v1（提案・Claude側）

日付：2026-11-08
目的：締切時点の情報だけで検証できる、1試合1行の記録（予想台帳）を両エンジン共通で持つ。
対象：toto13、totoGOAL3（mini-totoを含む）。過去分は保存できた範囲だけ入れ、欠けた項目は「不明」と明記する。

## 1. 基本原則
- 追記のみ（append-only）。既存の行は書き換えない。修正は revisionOf で新しい行を追加する。
- 締切時点で知りえた情報だけを使う（point-in-time）。同日の未来の試合結果を使わない。
- すべての確率に model_version を付ける。コードを変えたら版を上げる。
- 欠損は 0 扱いにせず、null と理由（"unknown" など）を書く。

## 2. 予想行（1試合1行）
| 項目 | 型 | 例 | 説明 |
|---|---|---|---|
| prediction_id | string | toto-1671-gpt-v1-m01-r1 | 一意のID |
| round | int | 1671 | 回 |
| product | string | toto13 / totoGOAL3 / miniToto-A | 商品 |
| match_number | int | 1 | 試合番号 |
| date | string | 2026-11-14 | 試合日（JST） |
| competition | string | J1 / J2 / J3 / cup / other | 大会 |
| home / away | string | 清水エスパルス | 正式名称（別名は alias 表で正規化） |
| model_name | string | Claude / GPT | エンジン名 |
| model_version | string | gpt-v1-2026-11-08 | 版（コードのハッシュも推奨） |
| probabilities | object | {"home":0.41,"draw":0.27,"away":0.32} | 3値確率（小数4桁以上） |
| expected_goals | object | {"home":1.6,"away":1.1} | λ（得点期待値） |
| goal_distribution | object | 任意 | 得点区分（0/1/2/3+） |
| confirmed_at | string | 2026-11-13T12:00:00+09:00 | 予想の確定時刻（締切前であること） |
| deadline_at | string | 2026-11-14T12:50:00+09:00 | 締切時刻 |
| market_snapshot_id | string | mkt-1671-0214 | 紐づく市場スナップショット（市場を使う版のみ） |
| inputs_hash | string | sha256:... | 入力データ（履歴・欠場など）のハッシュ |
| revision_of | string or null | null | 修正元の prediction_id |
| notes | string | 任意 | 補足 |

## 3. 市場スナップショット行
| 項目 | 説明 |
|---|---|
| snapshot_id | 一意のID |
| round / product | 回と商品 |
| captured_at | 取得時刻（JST、秒まで） |
| source_url | 公式の投票率ページのURL |
| shares | 試合ごとの {"home","draw","away"}（%、小数で保存） |
| is_final_pre_deadline | 締切前の最後の取得かどうか |

- 同じ回で時刻が複数ある場合は、すべて残す。どれを使ったかは market_snapshot_id で示す。
- 締切前の最後のスナップショットを、統一の基準とする（例：締切の30分前）。

## 4. 結果行
| 項目 | 説明 |
|---|---|
| date / competition / home / away | 試合の情報 |
| home_goals / away_goals | 結果（確定後に記録） |
| result_source | 出典URL |
| recorded_at | 記録した時刻 |

## 5. 負傷・欠場行（任意・前方蓄積）
| 項目 | 説明 |
|---|---|
| round / match_number / team | 対象 |
| player | 選手名（わかる範囲） |
| status | confirmed_injury / confirmed_suspension / unconfirmed / unknown |
| detail | 内容（全治など） |
| published_at | 公式発表の日時 |
| retrieved_at | 取得時刻（締切前であること） |
| source_url | 出典 |

- 「不明」は 0 名扱いにしない。status=unknown と書く。

## 6. 検証の約束
- 評価は walk-forward で行う。学習データは、各試合の締切時刻より前だけ。
- 主指標はRPS（勝敗）、副指標はLogLoss、Brier、校正曲線。的中数は参考表示のみ。
- 比較は同じ試合集合で行い、差は同一試合のペア差ブートストラップで区間を示す。
- 変更は1回に1つ。仮説・指標・採用条件を、先に書いてから実行する。

## 7. 実装の分担（提案）
- GPT側：予想行と市場スナップショット行の書き出し（既存の shared score 出力に合わせる）。
- Claude側：検証スクリプト（walk-forward、RPS、LogLoss）と、結果行の取り込み。
- 両側：alias 表（チーム名の正規化）を共通化する。

## 8. 初回に欠けるもの（正直に記録）
- 過去の市場スナップショットは、体系的には残っていない（1665・1668・1669回の一部のみ）。
- 過去の負傷・欠場は、1669回のみ。
- したがって、市場と負傷の検証は、この仕様で前方蓄積してから行う。

## 9. GPT側への依頼
- 2の項目で、既存の出力に足りないものを教えてください。
- 3の時刻の基準（締切の何分前か）について、ご意見をください。
- 6の指標の選び方について、ご意見をください。
