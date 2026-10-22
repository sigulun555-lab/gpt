# Jリーグ toto 自動予想アプリ

iPhone Safari と ChatGPT から同じ最新予想を確認するための静的Webアプリです。データ更新は GitHub Actions が6時間ごとに実行します。

## 自動処理
1. Jリーグ公開日程CSVを取得
2. toto公式ページから開催回・販売締切・対象13試合を解析
3. 過去の試合結果を使って 1/0/2 の確率を計算
4. data/*.json を更新し、GitHub Pages に公開

## 状態と制限
- 公式totoページから対象13試合をすべて確認できない場合、予想を生成しません。
- 負傷・欠場情報はまだ自動取得していません。
- 予測モデルはベースラインで、的中を保証しません。
- JリーグCSVはコミュニティ管理の公開ミラーであり、公式totoデータではありません。元データの利用条件を確認してください。
- GitHub Actionsのスケジュールは遅延する場合があります。
- 初回のPages公開には GitHub リポジトリの Settings → Pages → Build and deployment → Source を「GitHub Actions」にする必要があります。

## 使い方
- iPhone: 公開後の GitHub Pages URL をSafariで開く
- 手動更新: Actions → Update J.League data and predictions → Run workflow
- GPT: ActionsのOpenAPI仕様は `GPT-ACTION-openapi.json`。公開後、GPTのActions設定に登録する。

## データ
- `data/status.json`: 最終試行・最終成功・エラー
- `data/predictions.json`: 開催回と予想
- `data/history.json`: 過去試合
- `data/backtest.json`: 時系列バックテスト

## 開発
Node.js 20+ で `npm test`。架空のデータでテストした結果は実精度を示しません。
