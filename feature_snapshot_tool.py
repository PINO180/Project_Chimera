# feature_snapshot_tool.py
# =====================================================================
# ツール①: 特徴量スナップショット出力ツール
# 
# 【目的】
#   シグナルが発生した瞬間の全特徴量値をCSVに記録する。
#   本番が生成している特徴量の値が学習時の分布と乖離していないかを
#   確認するための診断ツール。
#
# 【使い方】
#   main.pyのシグナル処理部分に以下を追記:
#
#   from feature_snapshot_tool import save_feature_snapshot
#   save_feature_snapshot(signal.feature_dict, p_long_m1_raw, p_short_m1_raw,
#                         p_long_m2_raw, p_short_m2_raw, atr_ratio)
#
# 【出力】
#   /workspace/data/diagnostics/feature_snapshots/
#     snapshot_YYYYMMDD_HHMMSS_L{m2l:.3f}_S{m2s:.3f}.csv
# =====================================================================

import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

OUTPUT_DIR = Path("/workspace/data/diagnostics/feature_snapshots")
logger = logging.getLogger("ProjectCimera.FeatureSnapshot")


def save_feature_snapshot(
    feature_dict: Dict[str, float],
    p_m1_long: float,
    p_m1_short: float,
    p_m2_long: float,
    p_m2_short: float,
    atr_ratio: float,
    max_snapshots: int = 500,
) -> None:
    """
    シグナル発生時の全特徴量値をCSVに保存する。

    Args:
        feature_dict: signal.feature_dict（全特徴量の辞書）
        p_m1_long:    M1 Long予測値
        p_m1_short:   M1 Short予測値
        p_m2_long:    M2 Long予測値
        p_m2_short:   M2 Short予測値
        atr_ratio:    ATR Ratio
        max_snapshots: 保持する最大ファイル数（古いものから削除）
    """
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        now = datetime.now(timezone.utc)
        fname = (
            f"snapshot_{now.strftime('%Y%m%d_%H%M%S')}"
            f"_L{p_m2_long:.3f}_S{p_m2_short:.3f}.csv"
        )
        fpath = OUTPUT_DIR / fname

        with open(fpath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # ヘッダー
            writer.writerow(["feature_name", "value"])
            # メタ情報を先頭に
            writer.writerow(["_timestamp_utc", now.isoformat()])
            writer.writerow(["_m1_long",  round(p_m1_long, 6)])
            writer.writerow(["_m1_short", round(p_m1_short, 6)])
            writer.writerow(["_m2_long",  round(p_m2_long, 6)])
            writer.writerow(["_m2_short", round(p_m2_short, 6)])
            writer.writerow(["_atr_ratio", round(atr_ratio, 6)])
            writer.writerow(["---", "---"])
            # 全特徴量（名前順でソート）
            for name, val in sorted(feature_dict.items()):
                writer.writerow([name, val])

        logger.info(f"📸 [Snapshot] 特徴量スナップショット保存: {fname} ({len(feature_dict)}件)")

        # 古いファイルを削除（max_snapshots超過分）
        existing = sorted(OUTPUT_DIR.glob("snapshot_*.csv"))
        if len(existing) > max_snapshots:
            for old_file in existing[: len(existing) - max_snapshots]:
                old_file.unlink()
                logger.debug(f"古いスナップショット削除: {old_file.name}")

    except Exception as e:
        logger.error(f"特徴量スナップショット保存失敗: {e}", exc_info=True)
