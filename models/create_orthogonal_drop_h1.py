"""
create_orthogonal_drop_h1.py

selected_features_orthogonal_v5 の4ファイルから
H1のみを削除し、同ディレクトリ内の orthogonal_drop_h1/ に出力する。

■ 削除ルール（全4ファイル共通）
  H1 を削除 → M0.5 / M1 / M3 / M5 / M8 / M15 / M30 を残す

■ 特例（常に保持）
  atr_ratio_M3 / m1_pred_proba
"""

import re
from pathlib import Path

BASE_DIR   = Path("/workspace/data/XAUUSD/stratum_3_artifacts/selected_features_orthogonal_v5")
SRC_DIR    = BASE_DIR
OUTPUT_DIR = BASE_DIR / "orthogonal_drop_h1"

REMOVE_SUFFIXES = {"_H1"}
ALWAYS_KEEP     = {"atr_ratio_M3", "m1_pred_proba"}


def get_timeframe_suffix(feature_name):
    m = re.search(r'_(M[\d.]+|H\d+)$', feature_name)
    return f"_{m.group(1)}" if m else None


def filter_features(lines, remove_suffixes):
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
        if suffix in remove_suffixes:
            continue
        kept.append(line)
    remaining = sum(1 for l in kept if l.strip())
    return kept, original, remaining


def process(src_path, dst_path):
    lines = src_path.read_text(encoding="utf-8").splitlines(keepends=True)
    kept, original, remaining = filter_features(lines, REMOVE_SUFFIXES)
    dst_path.write_text("".join(kept), encoding="utf-8")
    removed = original - remaining
    print(f"  {src_path.name}")
    print(f"    元件数: {original:>4}  →  残件数: {remaining:>4}  (削除: {removed})")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n読み込み元: {SRC_DIR}")
    print(f"出力先    : {OUTPUT_DIR}\n")
    print("【全モデル】H1 を削除")
    for fname in ("m1_long_features.txt", "m1_short_features.txt",
                  "m2_long_features.txt", "m2_short_features.txt"):
        process(SRC_DIR / fname, OUTPUT_DIR / fname)
    print("\n✅ 完了")


if __name__ == "__main__":
    main()
