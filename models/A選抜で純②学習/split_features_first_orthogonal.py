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
    """上位足・sample_weight系・volume/tick_count系・ラベル系を除外する"""
    # [入れ子メタラベリング] ラベル/duration/uniqueness/concurrency 系接頭辞ガード。
    # label_A_long, label_B_short, uniqueness_A_* 等の新カラムが M1/M2 特徴量に
    # 漏れ込む（＝未来情報リーク）のを、個別列挙に依存せず一括で塞ぐ最終防御線。
    if col.startswith(("label_", "duration_", "uniqueness_", "concurrency_")):
        return True
    # 生の価格/一時列（close_future=P(L+180)錨用の一時列など）は特徴量にしない
    if col in ("close_future", "close", "open", "high", "low"):
        return True
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
    print("  M1=e1b/e1c/e1d(定規) / M2=全特徴量(武器・純②) — 直交解除")
    print("=" * 60)

    # --- 入力ファイル読み込み ---
    if not S3_FEATURES_FOR_TRAINING_V5.exists():
        print(f"❌ ERROR: 入力ファイルが見つかりません: {S3_FEATURES_FOR_TRAINING_V5}")
        sys.exit(1)

    with open(S3_FEATURES_FOR_TRAINING_V5, "r") as f:
        raw_features = [line.strip() for line in f if line.strip()]

    print(f"\n📄 入力特徴量数: {len(raw_features)}")

    # --- 分類処理 ---
    # [入れ子メタラベリング] 直交分割の解除。
    #   旧Two-Brain（M1もM2も同一ラベル）では、多様化を「特徴量の直交」で作る必要があり
    #   M1=e1b/e1c/e1d と M2=e1a/e1e/e1f を排他分割してエコーチェンバーを物理排除していた。
    #   新設計では M1(A)=label_A(P(L)錨・下駄) と M2(B)=label_B(P(L+180)錨・純②) で
    #   ラベルそのものが異なる＝多様化はラベル側で達成済み。その上に特徴量直交を重ねるのは
    #   冗長で、しかも純②(alone 0.57)という難問を解く B を情報飢餓にする。
    #   さらに B のラベルは純②で、d を符号化した価格動態(e1b/e1c/e1d)では"カンニング"できない
    #   (事実B: ①特徴注入では純②は学習不能)ため、全特徴を与えても「エコーで水増し」は起きず、
    #   Bの初到達に効く(情報増)か効かない(中立)かのどちらか。
    #   → M1 は従来通り e1b/e1c/e1d のみ。M2 は全特徴量(M1の分を含む)。
    m1_features = []
    m2_features = []
    excluded = []
    unclassified = []

    for col in raw_features:
        if should_exclude(col):
            excluded.append(col)
            continue

        category = classify(col)
        if category is None:
            # どのエンジンにも属さない真の未分類（下の警告で停止）
            unclassified.append(col)
            continue

        # M1（定規）= e1b/e1c/e1d のみ（従来通り）
        if category == "m1":
            m1_features.append(col)
        # M2（武器・純②）= 全特徴量。分類済みの列は M1 由来も含めて全て M2 に入れる。
        m2_features.append(col)

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
    print(f"✅ M1 特徴量数: {len(m1_features)}  (e1b/e1c/e1d のみ・定規)")
    print(f"✅ M2 特徴量数: {len(m2_features)}  (全特徴量・武器/純②。M1の分を包含)")
    print(
        f"   (分類済み {len(m2_features)} + 除外 {len(excluded)} + 未分類 {len(unclassified)} "
        f"= {len(m2_features) + len(excluded) + len(unclassified)} / 入力 {len(raw_features)})"
    )
    print(
        f"   ※ M1 は M2 の部分集合（e1b/e1c/e1d ⊂ 全特徴量）。直交分割は解除。"
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
