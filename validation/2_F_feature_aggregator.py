import sys
import logging
from pathlib import Path

# 動的パス解決（ルール7準拠）
sys.path.append(str(Path(__file__).resolve().parents[1]))

from blueprint import (
    S3_FINAL_FEATURE_TEAM,
    S3_SURVIVED_HF_FEATURES,
    S3_FEATURES_FOR_ALPHA_DECAY,
)

# [Phase 6 修正] ログ統一: 2_A/2_B と同じ logging 形式に揃える
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """
    第二防衛線を突破したLF特徴量とHF特徴量を統合し、
    最終的な特徴量セットを作成する。
    """
    logger.info("=" * 50)
    logger.info("### Chapter 2 Finale: Final Feature Aggregator ###")
    logger.info("=" * 50)

    if not S3_FINAL_FEATURE_TEAM.exists():
        raise FileNotFoundError(
            f"LF feature list (.txt) not found at: {S3_FINAL_FEATURE_TEAM}"
        )
    if not S3_SURVIVED_HF_FEATURES.exists():
        raise FileNotFoundError(
            f"HF feature list (.txt) not found at: {S3_SURVIVED_HF_FEATURES}"
        )

    logger.info(f"Loading LF features from: {S3_FINAL_FEATURE_TEAM}")
    with open(S3_FINAL_FEATURE_TEAM, "r") as f:
        # ヘッダーもインデックスもないため、全ての行をそのまま読み込む
        lf_features = [line.strip() for line in f if line.strip()]
    logger.info(f"-> Found {len(lf_features)} LF features.")

    logger.info(f"Loading HF features from: {S3_SURVIVED_HF_FEATURES}")
    with open(S3_SURVIVED_HF_FEATURES, "r") as f:
        hf_features = [line.strip() for line in f if line.strip()]
    logger.info(f"-> Found {len(hf_features)} HF features.")

    logger.info("\nCombining and deduplicating feature lists...")
    combined_features = lf_features + hf_features
    unique_features = sorted(list(set(combined_features)))

    logger.info(f"  - Total features before deduplication: {len(combined_features)}")
    logger.info(f"  - Total unique features after deduplication: {len(unique_features)}")

    # [Phase 6 修正] sample_weight 系を最終特徴量セットから除外
    # - HF 側: 2_E で EXCLUDE_SAMPLE_WEIGHT=1 採用済 → survived_hf_features.txt は 0 件
    # - LF 側: 2_C は sample_weight 除外を実装していないため final_feature_team.txt に
    #   sample_weight_M3 等が 6 件残る (D1/H12/H4/H6/MN/W1)
    # 過去 (Phase 5 以前) は variance フィルタで自動除外されていたため顕在化せず、
    # Phase 6 で volume = tick_count 補完によって sample_weight も有意な分散を持ち
    # 通過するようになった。本ステップで明示的に除外する。
    # 除外しないと S5 純化 → S6 ラベリング に sample_weight_neutralized_* が伝搬し、
    # update_feature_list_v5.py の 'sample_weight in col' フィルタで除外されるが、
    # 不要な OLS 純化計算 (約 6 列分の rolling regression) が走るため計算コストの無駄。
    sw_features = [f for f in unique_features if "sample_weight" in f]
    if sw_features:
        logger.info(f"\n[Phase 6 Filter] Excluding {len(sw_features)} sample_weight features:")
        for f in sw_features:
            logger.info(f"    - {f}")
        unique_features = [f for f in unique_features if "sample_weight" not in f]
        logger.info(f"  -> Filtered features: {len(unique_features)}")

    logger.info(f"\nSaving final feature set to: {S3_FEATURES_FOR_ALPHA_DECAY}")
    with open(S3_FEATURES_FOR_ALPHA_DECAY, "w") as f:
        for feature in unique_features:
            f.write(f"{feature}\n")

    logger.info("\n" + "=" * 50)
    logger.info("### Chapter 2 Finale Completed Successfully! ###")
    logger.info(
        f"The final army of {len(unique_features)} elite features has been assembled."
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
