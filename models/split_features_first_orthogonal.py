# /workspace/models/split_features_first_orthogonal.py
# 特徴量リストをM1/M2に直交分割するスクリプト（1周目用）
#
# 入力: S3_FEATURES_FOR_TRAINING_V5 (final_feature_set_v5.txt)
# 出力: S3_SELECTED_FEATURES_ORTHOGONAL_DIR/
#         m1_long_features.txt
#         m1_short_features.txt
#         m2_long_features.txt
#         m2_short_features.txt
#
# 分割ルール:
#   M1: e1b_, e1c_, e1d_ で始まるもの
#   M2: e1a_, e1e_, e1f_ で始まるもの + atr_ratio_M3
#   除外(共通): H4/H6/H12/D1/W1/MN を含む時間足 + sample_weight を含むもの
#   未分類特徴量が存在した場合: 警告を出力して終了（ファイル未生成）

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (
    S3_FEATURES_FOR_TRAINING_V5,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
)

# =================================================================
# 設定
# =================================================================

# 上位足サフィックス除外リスト（TO30システムのため H4以上は不要）(olsバッファ充填不測の為M30・H1も排除)
EXCLUDE_TIMEFRAME_SUFFIXES = ("_M30", "_H1", "_H4", "_H6", "_H12", "_D1", "_W1", "_MN")

# M1に割り当てるエンジンプレフィックス
M1_PREFIXES = ("e1b_", "e1c_", "e1d_")

# M2に割り当てるエンジンプレフィックス
M2_PREFIXES = ("e1a_", "e1e_", "e1f_")

# M2に個別追加する特徴量（プレフィックスで分類できないもの）
M2_EXACT = {"atr_ratio_M3"}


def should_exclude(col: str) -> bool:
    """上位足・sample_weight系・volume/tick_count系を除外する"""
    if "sample_weight" in col:
        return True
    # [Phase 6 修正] volume / tick_count 系を除外
    # Phase 6 で volume = tick_count 補完が学習側 s1_1_B に入ったため、
    # 過去には variance フィルタで自動除外されていた volume カラムが
    # 有意な値で 2_B を通過し下流に流れる可能性がある。
    # 学習特徴量としての volume / tick_count は除外し、engine_1_D が
    # 計算する e1d_volume_ratio / e1d_cmf_* / e1d_vwap_dist_* 等の
    # 派生特徴量を使う設計にする。
    base_name = col.split("_M")[0].split("_H")[0].split("_D")[0].split("_W")[0]
    if base_name in ("volume", "tick_count"):
        return True
    for suffix in EXCLUDE_TIMEFRAME_SUFFIXES:
        if suffix in col:
            return True
    return False


def classify(col: str):
    """
    特徴量をM1/M2に分類する。
    どちらにも属さない場合は None を返す。
    """
    if col in M2_EXACT:
        return "m2"
    if col.startswith(M1_PREFIXES):
        return "m1"
    if col.startswith(M2_PREFIXES):
        return "m2"
    return None


def main():
    print("=" * 60)
    print("  split_features_first_orthogonal.py")
    print("  M1/M2 直交分割 (1周目)")
    print("=" * 60)

    # --- 入力ファイル読み込み ---
    if not S3_FEATURES_FOR_TRAINING_V5.exists():
        print(f"❌ ERROR: 入力ファイルが見つかりません: {S3_FEATURES_FOR_TRAINING_V5}")
        sys.exit(1)

    with open(S3_FEATURES_FOR_TRAINING_V5, "r") as f:
        raw_features = [line.strip() for line in f if line.strip()]

    print(f"\n📄 入力特徴量数: {len(raw_features)}")

    # --- 分類処理 ---
    m1_features = []
    m2_features = []
    excluded = []
    unclassified = []

    for col in raw_features:
        if should_exclude(col):
            excluded.append(col)
            continue

        category = classify(col)
        if category == "m1":
            m1_features.append(col)
        elif category == "m2":
            m2_features.append(col)
        else:
            unclassified.append(col)

    # --- 除外サマリー ---
    print(f"🚫 除外 (上位足・sample_weight): {len(excluded)} 件")

    # --- 未分類チェック: 存在した場合は警告して終了 ---
    if unclassified:
        print("\n" + "=" * 60)
        print("❌ ERROR: 未分類の特徴量が存在します。ファイルを生成しません。")
        print("   以下の特徴量がM1/M2どちらのエンジンにも属しません:")
        print("=" * 60)
        for col in unclassified:
            print(f"   - {col}")
        print(
            "\n対処: blueprint の M1_PREFIXES / M2_PREFIXES / M2_EXACT を更新するか、"
        )
        print(
            "      update_feature_list_v5.py の non_feature_cols に追加してください。"
        )
        sys.exit(1)

    # --- 結果サマリー ---
    print(f"✅ M1 特徴量数: {len(m1_features)}")
    print(f"✅ M2 特徴量数: {len(m2_features)}")
    print(
        f"   (M1+M2+除外 = {len(m1_features) + len(m2_features) + len(excluded)} / 入力 {len(raw_features)})"
    )

    # --- 出力 ---
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR.mkdir(parents=True, exist_ok=True)

    output_files = {
        "m1_long_features.txt": m1_features,
        "m1_short_features.txt": m1_features,
        "m2_long_features.txt": m2_features,
        "m2_short_features.txt": m2_features,
    }

    for filename, features in output_files.items():
        out_path = S3_SELECTED_FEATURES_ORTHOGONAL_DIR / filename
        with open(out_path, "w") as f:
            for feat in features:
                f.write(f"{feat}\n")
        print(f"💾 {out_path.name} ({len(features)} 件)")

    print("\n✅ 直交分割完了")
    print(f"   -> {S3_SELECTED_FEATURES_ORTHOGONAL_DIR}")


if __name__ == "__main__":
    main()
