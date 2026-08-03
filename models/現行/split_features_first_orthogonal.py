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
#   M2: e1a_, e1e_, e1f_ で始まるもの + atr_ratio_M3 / eff_ratio_* / d_atr_*
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

# [PROTO] トリガー時間足。ラベリングの TARGET_TIMEFRAMES と必ず一致させること。
#   ラベリングが出す追加特徴量 (atr_ratio_*/eff_ratio_*/d_atr_*) は
#   この TF のサフィックスを持つ。M3 -> M1 に変えると名前が全部変わるため、
#   ここをハードコードから TF 変数に置き換えている。
TRIGGER_TF = "M1"

# [PROTO] 効率比の窓 (ラベリングの ER_WINDOWS_BY_TF と一致させること)
#   M0.5:(120,150) / M1:(60,80) / M3:(20,25) / M5:(12,15) / M8:(8,10) / M15:(4,5)
ER_WINDOWS = (60, 80)

# M1に割り当てるエンジンプレフィックス
M1_PREFIXES = ("e1b_", "e1c_", "e1d_")

# M2に割り当てるエンジンプレフィックス
M2_PREFIXES = ("e1a_", "e1e_", "e1f_")

# M2に個別追加する特徴量（プレフィックスで分類できないもの）
# -----------------------------------------------------------------------------
# eff_ratio_* / d_atr_* は engine_1_* ではなくラベリング側（create_proxy_labels）で
# 計算して S6 に出しているため、e1x_ のプレフィックスを持たない。
# atr_ratio_M3 と同じ経路で作られる特徴量なので、同じく M2 に割り当てる。
#
# 【なぜ M2 か】
#   これらは「方向の材料」として追加したものであり、方向を学ぶのは M2 だから。
#   今回の構成では M1 はゲート off で選抜の役割を持たないため、M1 に入れると死ぬ。
#
#   さらに重要な点として、低ボラ帯と高ボラ帯では構造そのものが異なる:
#       atr_ratio<0.8 … 小さな動き(0.3-0.6 ATR)が15分かけて戻る（窓120分）
#       atr_ratio>=0.8… 大きな動き(1.3+ ATR)が1分で戻る（窓360分）
#   木がこの場合分けを学ぶには atr_ratio と eff_ratio / d_atr が
#   【同じモデルの中に揃っている】必要がある。別々のモデルに配ると分岐できない。
#   atr_ratio_M3 が既に M2 側にいるので、揃えるなら M2。
# -----------------------------------------------------------------------------
M2_EXACT = {
    f"atr_ratio_{TRIGGER_TF}",
    *(f"eff_ratio_{k}_{TRIGGER_TF}" for k in ER_WINDOWS),   # 効率比
    f"d_atr_{TRIGGER_TF}",          # (close − open) / ATR  符号つき1バー変位
}


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
    print(f"🎯 TRIGGER_TF = {TRIGGER_TF}  /  M2_EXACT = {sorted(M2_EXACT)}")

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
