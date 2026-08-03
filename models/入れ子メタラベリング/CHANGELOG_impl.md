# 入れ子メタラベリング 実装 変更履歴（仕様書v2準拠）

対象: 既存9スクリプトの**中身改修**（案i・名前据置・新規スクリプトなし）。
blueprint.py / aggregate_daily_to_monthly.py は無改修（同梱せず）。
全ファイル `python -m py_compile` 済み。生産者↔消費者のカラム名整合を確認済み。
※ 構文・整合の検証のみ。実データ／16hラベリングは未実行。

---

## 1. create_proxy_labels_polars_patch_regime_Universal_Brain_V5.py（第2ラベルセット化）
- 旧 `RULE_LONG/SHORT` を削除し、上部に**手動設定パネル**を新設：
  - combos 早見表（コメント・人間用メモ）
  - `RULE_A_LONG/SHORT`（選抜脳M1用）, `RULE_B_LONG/SHORT`（本番発注M2用）。既定 = combo 1（A=B=1:5/30）。
- `_calculate_labels_for_batch`：A幾何・B幾何それぞれで PT/SL/t1_max を構築し、
  `_numba_find_hits_dual` を**2回呼ぶ**（シグネチャ不変）。出力に
  `label_A_long/short, duration_A_long/short, label_B_long/short, duration_B_long/short`。
- **A==B 配線検算**：`RULE_A==RULE_B` のとき初回パーティションで `label_A==label_B` を照合、
  不一致なら `RuntimeError` で停止（`__init__` に `_ab_identity_checked` フラグ追加）。
- run() ログ / max_lookahead / レポート関数を A/B へ更新（レポートは B幾何を表示）。
- アンカー(`close`=L+180)・SPREAD・numba 走査は不変。

## 2. sample_uniqueness_weighting_calculate.py（concurrency 4系統化）
- DuckDB クエリを `duration_A_long/short, duration_B_long/short` から
  `concurrency_A_long/short, concurrency_B_long/short` を計算する4系統に拡張。

## 3. sample_uniqueness_weighting_join.py（uniqueness 4系統化）
- `uniqueness_A_long/short, uniqueness_B_long/short` を計算（各 1/concurrency、null/0→0.0）。
- 中間 `concurrency_*`（4本）を除外。join キー `[timestamp, timeframe]` 不変。

## 4. update_feature_list_v5.py（除外リスト・未来リーク防止）
- `non_feature_cols` に A/B の label/duration/uniqueness/concurrency を個別追加。
- **接頭辞ガード** `("label_","duration_","uniqueness_","concurrency_")` を特徴量ループに追加（二重防御）。

## 5. split_features_first_orthogonal.py（除外リスト）
- `should_exclude` に同じ接頭辞ガードを追加。M1/M2 直交分割ルールは不変。

## 6. model_training_metalabeling_Ax2.py（M1 = A）
- 学習ラベルを `label_A_{dir}`、重みを `uniqueness_A_{dir}` に変更。
- exclude_exact に A/B 追加＋接頭辞ガード。CV(5,3,2)・scale_pos_weight・OOF出力は不変。

## 7. model_training_metalabeling_Bx2.py（接続＝本丸）
- `meta_label` を **`label_B_{dir}`** に、素の重みを **`uniqueness_B_{dir}`** に変更。
- **床を config 化**：module 定数 `M1_GATE_LOGIT`（既定 0.0=proba0.5、combo2以降は下げる旨をコメント）。
  旧ハードコード `THRESHOLD_LOGIT=0.0` を置換。
- **合成サンプル重み**：`uniqueness = uniqueness_B × h(q_A)`、
  `h(q_A)=2*sigmoid(logit)-1=tanh(logit/2)` を `clip(0,1)`。`uniqueness` カラムに上書き（Cx2 が無改修で消費）。
- exclude_exact に A/B 追加＋接頭辞ガード。`m1_pred_proba` は特徴量にしない（診断カラムとしては通過）。

## 8. model_training_metalabeling_Cx2.py（M2 = B）
- **q_A 特徴注入を撤廃**：`self.features_m2.append("m1_pred_proba")`（旧190-193行）を削除。
  → §9.4（q_A=g(x)は無益）＋ fold不一致リークの主経路を閉塞。q_A の影響はサンプル重み経由のみ。
- **M1 最終学習**を `label_A_{dir}` / `uniqueness_A_{dir}` に変更（scale_pos_weight_m1・学習・較正の全 M1 経路）。
- **M2** は `meta_label`（=B幾何ラベル）/ `uniqueness`（=合成重み）で不変。
- isotonic 較正（M1・M2）不変＝単調較正層。exclude_exact に A/B 追加＋接頭辞ガード。

## 9. backtest_simulator_cimera.py（発注 = B・手動）
- `BacktestConfig` バリア設定の直上に combos 早見表（B側）と手動運用の注記を追加。
- **コード値は未変更**（combo 1 の既定 1:5/30 が現行値と一致）。combo 2以降は pt/sl/td を手で B値へ、
  `m2_proba_threshold`(0.70)/`m2_delta_threshold`(0.30) は基底率≈50%で要再スイープ。

---

## 実行手順（1 combo）
1. `create_proxy_...V5.py` 上部の `RULE_A_*` / `RULE_B_*` を combo の値に書換。
2. combo 2以降のみ、`Bx2` の `M1_GATE_LOGIT` を必要なら下げる。
3. `bash run_phase_d3_pipeline.sh 14 22`（ラベリング〜Cx2）。
4. `backtest_simulator_cimera.py` の `BacktestConfig` の pt/sl/td を同 combo の B値に合わせて実行。
5. 順序: combo 1（電源確認・A==B検算ログ確認）→ 2（本命）→ 3（並走）→ 4以降。

## 初回ランで見るべきログ
- create_proxy: `[A==B CHECK] OK ...`（combo 1）。出れば2ラベル配線は正常。FAILEDなら停止＝バリア/numbaのバグ。
