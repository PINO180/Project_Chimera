import sys
from pathlib import Path

# blueprint.pyが/workspace直下にあることを想定
sys.path.append("/workspace")

from blueprint import (
    S3_FINAL_FEATURE_TEAM,
    S3_SURVIVED_HF_FEATURES,
    S3_FEATURES_FOR_ALPHA_DECAY,
)


def main():
    """
    第二防衛線を突破したLF特徴量とHF特徴量を統合し、
    最終的な特徴量セットを作成する。
    """
    print("=" * 50)
    print("### Chapter 2 Finale: Final Feature Aggregator ###")
    print("=" * 50)

    if not S3_FINAL_FEATURE_TEAM.exists():
        raise FileNotFoundError(
            f"LF feature list (.txt) not found at: {S3_FINAL_FEATURE_TEAM}"
        )
    if not S3_SURVIVED_HF_FEATURES.exists():
        raise FileNotFoundError(
            f"HF feature list (.txt) not found at: {S3_SURVIVED_HF_FEATURES}"
        )

    print(f"Loading LF features from: {S3_FINAL_FEATURE_TEAM}")
    with open(S3_FINAL_FEATURE_TEAM, "r") as f:
        # --- ★★★ 最後の修正箇所 ★★★ ---
        # ヘッダーもインデックスもないため、全ての行をそのまま読み込む
        lf_features = [line.strip() for line in f if line.strip()]
    print(f"-> Found {len(lf_features)} LF features.")

    print(f"Loading HF features from: {S3_SURVIVED_HF_FEATURES}")
    with open(S3_SURVIVED_HF_FEATURES, "r") as f:
        hf_features = [line.strip() for line in f if line.strip()]
    print(f"-> Found {len(hf_features)} HF features.")

    print("\nCombining and deduplicating feature lists...")
    combined_features = lf_features + hf_features
    unique_features = sorted(list(set(combined_features)))

    print(f"  - Total features before deduplication: {len(combined_features)}")
    print(f"  - Total unique features after deduplication: {len(unique_features)}")

    print(f"\nSaving final feature set to: {S3_FEATURES_FOR_ALPHA_DECAY}")
    with open(S3_FEATURES_FOR_ALPHA_DECAY, "w") as f:
        for feature in unique_features:
            f.write(f"{feature}\n")

    print("\n" + "=" * 50)
    print("### Chapter 2 Finale Completed Successfully! ###")
    print(
        f"The final army of {len(unique_features)} elite features has been assembled."
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
