# Hint — 失敗パターン分析

検出された手がかり:

- **大きな相対誤差**: rel_diff > 1% の行が 41682 件存在。  → 数値精度ではなく `計算ロジックの違い` (発見 #60 の disc fallback 級の構造バグ) の疑い。  → `worst.csv` を見て、prod/ref 値の order of magnitude を確認。
- **時刻集中**: failing 行が 1 日に集中 (246973 件)。  → 週末跨ぎ・市場閉鎖・ニュース時刻等のイベント駆動の可能性。  → 失敗日付:  2026-04-01

## 次のステップ
1. `worst.csv` で個別行を確認 (大きい順)
2. `failing.parquet` を Polars/Pandas で読み、特徴量別・時刻別に grouping
3. 該当する production コードの該当 TF / 該当 engine module を再確認
4. 必要に応じて該当バーの input data (OHLCV deque) を再現してデバッグ

## 関連ドキュメント
- Train_Serve_Skew_Audit_Report.md (発見 #59〜#62 の Phase 9d セクション)
- Skew_Detection_Hardening_Plan.md (Layer 4 静的検出と組み合わせて trace)
