import pandas as pd
from pathlib import Path

# ============================================================
# ベースディレクトリ
# ============================================================
BASE_DIR = Path("/workspace/data/XAUUSD/stratum_3_artifacts/optuna_results")

# ============================================================
# 処理対象ファイル一覧（Long 3本 + Short 3本）
# ============================================================
INPUT_FILES = [
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.36.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.5.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_spread_0.8.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.36.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.5.csv",
    BASE_DIR / "optuna_top100_pure_atr_results_short_spread_0.8.csv",
]

# ============================================================
# 時間足の順序定義
# ============================================================
TIMEFRAME_ORDER = ["M1", "M3", "M5", "M8", "M15", "H1", "H4", "H6", "H12", "mixed"]

# ============================================================
# カラム名の差異を吸収するマッピング
# ============================================================
COLUMN_MAP = {
    "atr_period": "trigger_ratio",
    "atr_threshold": "atr_filter",
    "Total_NP": "Total_Net_Profit",
}

# ============================================================
# 出力用ヘルパー
# ============================================================
def write_section(f, title, dataframe, write_header=True):
    f.write(title + "\n")
    dataframe.to_csv(f, index=False, header=write_header)
    f.write("\n")


# ============================================================
# 1ファイル分の変換処理
# ============================================================
def process_file(input_path: Path):
    # 出力パスを入力ファイル名から自動生成
    # 例: optuna_top100_pure_atr_results_short_spread_0.5.csv
    #   → optuna_topdata_pure_atr_results_short_spread_0.5.csv
    output_name = input_path.name.replace("top100", "topdata")
    output_path = input_path.parent / output_name

    print(f"\n{'='*60}")
    print(f"処理中: {input_path.name}")
    print(f"出力先: {output_path.name}")
    print(f"{'='*60}")

    if not input_path.exists():
        print(f"  [SKIP] ファイルが存在しません: {input_path}")
        return

    df = pd.read_csv(input_path)
    print("カラム名:", df.columns.tolist())

    df = df.rename(columns=COLUMN_MAP)
    print("リネーム後:", df.columns.tolist())

    if "Total_Net_Profit" not in df.columns:
        raise ValueError(
            f"'Total_Net_Profit'カラムが見つかりません。\n"
            f"実際のカラム名: {df.columns.tolist()}\n"
            f"COLUMN_MAPに対応を追加してください。"
        )
    if "Adjusted_PF" not in df.columns:
        raise ValueError(
            f"'Adjusted_PF'カラムが見つかりません。\n"
            f"実際のカラム名: {df.columns.tolist()}\n"
            f"COLUMN_MAPに対応を追加してください。"
        )

    # データクリーニング
    df["Total_Net_Profit"] = (
        df["Total_Net_Profit"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
    )
    if "Timeout_NP" in df.columns:
        df["Timeout_NP"] = (
            df["Timeout_NP"].astype(str).str.replace(r"[^\d\.\-]", "", regex=True)
        )

    df["Total_Net_Profit"] = pd.to_numeric(df["Total_Net_Profit"], errors="coerce").fillna(0.0)
    if "Timeout_NP" in df.columns:
        df["Timeout_NP"] = pd.to_numeric(df["Timeout_NP"], errors="coerce").fillna(0.0)

    # TNP > 0フィルタを削除（全件対象）
    valid_tf = df.copy()
    valid_tf["tf_order"] = valid_tf["timeframe"].apply(
        lambda x: TIMEFRAME_ORDER.index(x) if x in TIMEFRAME_ORDER else 999
    )
    valid_tf = valid_tf.sort_values("tf_order")

    # 出力1: 各timeframeのTNP上位5行
    top5_frames = []
    for tf in TIMEFRAME_ORDER:
        subset = valid_tf[valid_tf["timeframe"] == tf]
        if subset.empty:
            continue
        top5_frames.append(subset.nlargest(5, "Total_Net_Profit"))
    df_top5 = pd.concat(top5_frames).drop(columns=["tf_order"]) if top5_frames else pd.DataFrame()

    # 出力2: 各timeframeのTNP1位・APF1位
    out2_rows = []
    for tf in TIMEFRAME_ORDER:
        subset = valid_tf[valid_tf["timeframe"] == tf]
        if subset.empty:
            continue
        tnp1 = subset.nlargest(1, "Total_Net_Profit").copy()
        apf1 = subset.nlargest(1, "Adjusted_PF").copy()
        tnp1["timeframe"] = f"{tf}_TNP1"
        apf1["timeframe"] = f"{tf}_APF1"
        out2_rows.append(tnp1)
        out2_rows.append(apf1)
    df_out2 = pd.concat(out2_rows).drop(columns=["tf_order"]) if out2_rows else pd.DataFrame()

    # 出力3-A: 各timeframeのTNP1位一覧
    tnp1_rows = []
    for tf in TIMEFRAME_ORDER:
        subset = valid_tf[valid_tf["timeframe"] == tf]
        if subset.empty:
            continue
        tnp1_rows.append(subset.nlargest(1, "Total_Net_Profit"))
    df_tnp1 = pd.concat(tnp1_rows).drop(columns=["tf_order"]) if tnp1_rows else pd.DataFrame()

    # 出力3-B: 各timeframeのAPF1位一覧
    apf1_rows = []
    for tf in TIMEFRAME_ORDER:
        subset = valid_tf[valid_tf["timeframe"] == tf]
        if subset.empty:
            continue
        apf1_rows.append(subset.nlargest(1, "Adjusted_PF"))
    df_apf1 = pd.concat(apf1_rows).drop(columns=["tf_order"]) if apf1_rows else pd.DataFrame()

    # CSV書き出し
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        write_section(f, "Top5 Total_Net_Profit per timeframe", df_top5)
        write_section(f, "TNP1st and APF1st per timeframe", df_out2)
        write_section(f, "TNP1st list (ascending timeframe)", df_tnp1)
        write_section(f, "APF1st list (ascending timeframe)", df_apf1)

    print(f"  Done: {output_path}")


# ============================================================
# メイン
# ============================================================
if __name__ == "__main__":
    for input_path in INPUT_FILES:
        process_file(input_path)

    print("\n全ファイルの処理が完了しました。")
