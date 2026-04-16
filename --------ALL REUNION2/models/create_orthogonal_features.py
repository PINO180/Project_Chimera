"""
create_orthogonal_features.py

selected_features_orthogonal_v5 の4ファイルから
案5-B（二重直交）に基づいて不要時間足の特徴量を削除し、
同ディレクトリ内の <timestamp>/ サブフォルダに出力する。

■ 削除ルール
  m1_long / m1_short  : M15 / M30 / H1 を削除 → M0.5〜M8 のみ残す
  m2_long / m2_short  : M0.5 / M1 / M3 / M5 / M8 / M30 / H1 を削除 → M15 のみ残す

■ 特例（時間足フィルタ対象外・常に保持）
  atr_ratio_M3   : M2に必要な環境指標（_M3サフィックスだが削除しない）
  m1_pred_proba  : M2に必要なM1予測値（サフィックスなしだが削除しない）
"""

import re
from pathlib import Path
from datetime import datetime

# =================================================================
# パス設定
# =================================================================
BASE_DIR   = Path("/workspace/data/XAUUSD/stratum_3_artifacts/selected_features_orthogonal_v5")
SRC_DIR    = BASE_DIR                                          # 読み込み元
TIMESTAMP  = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / TIMESTAMP                             # 出力先サブフォルダ

# =================================================================
# フィルタ定義
# =================================================================
M1_REMOVE_SUFFIXES = {"_M15", "_M30", "_H1"}   # M1側: この時間足を削除
M2_KEEP_SUFFIX     = "_M15"                     # M2側: これだけ残す
ALWAYS_KEEP        = {"atr_ratio_M3", "m1_pred_proba"}  # 常に保持する特殊カラム


def get_timeframe_suffix(feature_name):
    m = re.search(r'_(M[\d.]+|H\d+)$', feature_name)
    return f"_{m.group(1)}" if m else None


def filter_features(lines, remove_suffixes=None, keep_suffix=None):
    kept = []
    original = 0

    for line in lines:
        stripped = line.strip()
        if not stripped:
            kept.append(line)
            continue

        original += 1

        if stripped in ALWAYS_KEEP:
            kept.append(line)
            continue

        suffix = get_timeframe_suffix(stripped)

        if remove_suffixes is not None:
            if suffix in remove_suffixes:
                continue

        if keep_suffix is not None:
            if suffix != keep_suffix:
                continue

        kept.append(line)

    remaining = sum(1 for l in kept if l.strip())
    return kept, original, remaining


def process(src_path, dst_path, remove_suffixes=None, keep_suffix=None):
    lines = src_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, original, remaining = filter_features(lines, remove_suffixes, keep_suffix)
    dst_path.write_text("".join(kept), encoding="utf-8")

    removed = original - remaining
    print(f"  {src_path.name}")
    print(f"    元件数: {original:>4}  →  残件数: {remaining:>4}  (削除: {removed})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n読み込み元: {SRC_DIR}")
    print(f"出力先    : {OUTPUT_DIR}\n")

    print("【M1モデル】M15 / M30 / H1 を削除")
    for fname in ("m1_long_features.txt", "m1_short_features.txt"):
        process(
            src_path        = SRC_DIR / fname,
            dst_path        = OUTPUT_DIR / fname,
            remove_suffixes = M1_REMOVE_SUFFIXES,
        )

    print()

    print("【M2モデル】M15 のみ残す（atr_ratio_M3 / m1_pred_proba は保持）")
    for fname in ("m2_long_features.txt", "m2_short_features.txt"):
        process(
            src_path    = SRC_DIR / fname,
            dst_path    = OUTPUT_DIR / fname,
            keep_suffix = M2_KEEP_SUFFIX,
        )

    print("\n✅ 完了")


if __name__ == "__main__":
    main()
