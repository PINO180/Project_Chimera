"""
split_features_orthogonal.py
============================
selected_features_v5 内の全特徴量ファイルから和集合を取り、
M1（方向予測）/ M2（環境認識）に直交分割して
selected_features_orthogonal_v5 に出力するスクリプト。

分割ルール:
  M1専用 : e1b・e1c・e1d カテゴリ（方向予測・プライスアクション・カルマン）
  M2専用 : e1a・e1e・e1f カテゴリ（統計モーメント・スペクトル・複雑系）
           + atr_ratio_M3（環境認識）
  除外   : m1_pred_proba（Bxが動的追加するため不要）
           sample_weight_neutralized_*（重みカラム・特徴量ではない）
  timeframe: Axコードが先頭に強制追加するためファイル管理対象外

使用方法:
  python split_features_orthogonal.py
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import S3_SELECTED_FEATURES_DIR, S3_SELECTED_FEATURES_ORTHOGONAL_DIR

# =========================================================
# 分割ルール定義
# =========================================================
M1_CATEGORIES = {"e1b", "e1c", "e1d"}
M2_CATEGORIES = {"e1a", "e1e", "e1f"}
M2_EXTRA      = {"atr_ratio_M3"}   # カテゴリ外だがM2専用
EXCLUDE       = {"m1_pred_proba"}  # Bxが動的追加するため除外

def get_category(feature: str) -> str | None:
    """特徴量名からe1x カテゴリを返す。該当なしはNone。"""
    for cat in ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]:
        if cat in feature:
            return cat
    return None

def classify(feature: str) -> str:
    """特徴量をm1/m2/common/excludeに分類して返す。"""
    if feature in EXCLUDE:
        return "exclude"
    if feature.startswith("sample_weight"):
        return "exclude"
    if feature in M2_EXTRA:
        return "m2"
    cat = get_category(feature)
    if cat in M1_CATEGORIES:
        return "m1"
    if cat in M2_CATEGORIES:
        return "m2"
    # e1x カテゴリに属さない特徴量（timeframe除く）はM1/M2両方に持たせる
    return "common"

def main():
    logging.info("=== 特徴量直交分割スクリプト ===")

    # --- 入力ファイルの確認（m1/m2の4ファイルのみ対象）---
    target_files = [
        "m1_long_features.txt",
        "m1_short_features.txt",
        "m2_long_features.txt",
        "m2_short_features.txt",
    ]
    source_files = []
    for fname in target_files:
        p = S3_SELECTED_FEATURES_DIR / fname
        if p.exists():
            source_files.append(p)
        else:
            logging.warning(f"  ファイルが見つかりません（スキップ）: {p}")

    if not source_files:
        logging.error(f"特徴量ファイルが見つかりません: {S3_SELECTED_FEATURES_DIR}")
        sys.exit(1)

    logging.info(f"入力ディレクトリ: {S3_SELECTED_FEATURES_DIR}")
    logging.info(f"対象ファイル: {[f.name for f in source_files]}")

    # --- 全ファイルの和集合を取得 ---
    all_features: set[str] = set()
    for path in source_files:
        with open(path) as f:
            feats = {line.strip() for line in f if line.strip()}
        all_features |= feats
        logging.info(f"  {path.name}: {len(feats)}特徴量")

    logging.info(f"和集合: {len(all_features)}特徴量")

    # --- 分類 ---
    common_features = sorted([f for f in all_features if classify(f) == "common"])
    m1_features     = sorted([f for f in all_features if classify(f) in ("m1", "common")])
    m2_features     = sorted([f for f in all_features if classify(f) in ("m2", "common")])
    excluded        = sorted([f for f in all_features if classify(f) == "exclude"])

    logging.info(f"\n分割結果:")
    logging.info(f"  M1専用（e1b/c/d）       : {len([f for f in all_features if classify(f) == 'm1'])}特徴量")
    logging.info(f"  M2専用（e1a/e/f + atr）  : {len([f for f in all_features if classify(f) == 'm2'])}特徴量")
    logging.info(f"  共通（other）            : {len(common_features)}特徴量")
    if common_features:
        for c in common_features:
            logging.info(f"    共通: {c}")
    logging.info(f"  除外（weight/pred_proba）: {len(excluded)}特徴量")
    if excluded:
        for e in excluded:
            logging.info(f"    除外: {e}")
    logging.info(f"  → M1合計: {len(m1_features)}特徴量 / M2合計: {len(m2_features)}特徴量")

    # --- 出力ディレクトリ作成 ---
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"\n出力ディレクトリ: {S3_SELECTED_FEATURES_ORTHOGONAL_DIR}")

    # --- ファイル出力 ---
    output_files = {
        "m1_long_features.txt":  m1_features,
        "m1_short_features.txt": m1_features,
        "m2_long_features.txt":  m2_features,
        "m2_short_features.txt": m2_features,
    }

    for filename, features in output_files.items():
        out_path = S3_SELECTED_FEATURES_ORTHOGONAL_DIR / filename
        with open(out_path, "w") as f:
            f.write("\n".join(features) + "\n")
        logging.info(f"  出力: {filename} ({len(features)}特徴量)")

    logging.info("\n✅ 特徴量直交分割完了")
    logging.info(f"   M1: e1b・e1c・e1d → {len(m1_features)}特徴量")
    logging.info(f"   M2: e1a・e1e・e1f + atr_ratio_M3 → {len(m2_features)}特徴量")
    logging.info(f"   ※ m1_pred_proba はBxが動的追加するため除外済み")
    logging.info(f"   ※ timeframe はAxコードが先頭に強制追加するため除外済み")


if __name__ == "__main__":
    main()
