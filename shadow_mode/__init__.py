"""
[Layer 1] Shadow Mode 差分テストフレームワーク

Production の RealtimeFeatureEngine を歴史的データで replay し、
学習側 S2 出力と数値完全一致を検証する。

エントリーポイント: run_shadow_test.py
モジュール構成:
    replay_bridge      - 歴史 M0.5 parquet を読み込み、test 期間バーを供給
    feature_capturer   - ShadowEngine (RealtimeFeatureEngine subclass) で
                         _calculate_base_features の戻り値を捕捉
    reference_builder  - S2_FEATURES_VALIDATED から比較用 long-format を構築
    diff_aggregator    - 捕捉値 vs リファレンスの数値比較
    diff_report        - 差分集計レポートを出力
    stress_injector    - (v2 拡張用) 合成ストレスシナリオ注入
    run_shadow_test    - 全体オーケストレーション (CLI)
"""
__version__ = "1.0.0"
