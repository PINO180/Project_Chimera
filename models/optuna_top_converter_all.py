import pandas as pd

# ============================================================
# パス設定（必要に応じて書き換えてください）
# ============================================================
INPUT_PATH = "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_top100_pure_atr_results_short.csv"
OUTPUT_PATH = "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_topdata_pure_atr_results_short.csv"

# ============================================================
# 時間足の順序定義
# ============================================================
TIMEFRAME_ORDER = ["M1", "M3", "M5", "M8", "M15", "H1", "H4", "H6", "H12", "mixed"]

# ============================================================
# カラム名の差異を吸収するマッピング（どちらのファイルでも自動対応）
# 新カラム名 → 処理内部で使う統一名
# ============================================================
COLUMN_MAP = {
    "atr_period": "trigger_ratio",
    "atr_threshold": "atr_filter",
    "Total_NP": "Total_Net_Profit",
}

# ============================================================
# データ読み込み・カラム名統一
# ============================================================
df = pd.read_csv(INPUT_PATH)

# 実際のカラム名を表示（デバッグ用）
print("カラム名:", df.columns.tolist())

# カラム名を統一名にリネーム（該当しないカラムは無視）
df = df.rename(columns=COLUMN_MAP)

# リネーム後のカラム名を確認
print("リネーム後:", df.columns.tolist())

# ============================================================
# Total_Net_Profitカラムの存在チェック
# ============================================================
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

# ============================================================
# 以降は共通処理
# ============================================================

# D1など全行がTotal_Net_Profit=0のtimeframeを除外
valid_tf = df.groupby("timeframe").filter(lambda x: x["Total_Net_Profit"].max() > 0)

# 対象timeframeを順序通りに並べる
valid_tf = valid_tf.copy()
valid_tf["tf_order"] = valid_tf["timeframe"].apply(
    lambda x: TIMEFRAME_ORDER.index(x) if x in TIMEFRAME_ORDER else 999
)
valid_tf = valid_tf.sort_values("tf_order")


# ============================================================
# 出力用ヘルパー
# ============================================================
def write_section(f, title, dataframe, write_header=True):
    f.write(title + "\n")
    dataframe.to_csv(f, index=False, header=write_header)
    f.write("\n")


# ============================================================
# 出力1: 各timeframeのTNP上位5行
# ============================================================
top5_frames = []
for tf in TIMEFRAME_ORDER:
    subset = valid_tf[valid_tf["timeframe"] == tf]
    if subset.empty:
        continue
    top5 = subset.nlargest(5, "Total_Net_Profit")
    top5_frames.append(top5)

df_top5 = pd.concat(top5_frames).drop(columns=["tf_order"])

# ============================================================
# 出力2: 各timeframeのTNP1位・APF1位
# ============================================================
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

df_out2 = pd.concat(out2_rows).drop(columns=["tf_order"])

# ============================================================
# 出力3-A: 各timeframeのTNP1位一覧
# ============================================================
tnp1_rows = []
for tf in TIMEFRAME_ORDER:
    subset = valid_tf[valid_tf["timeframe"] == tf]
    if subset.empty:
        continue
    tnp1_rows.append(subset.nlargest(1, "Total_Net_Profit"))

df_tnp1 = pd.concat(tnp1_rows).drop(columns=["tf_order"])

# ============================================================
# 出力3-B: 各timeframeのAPF1位一覧
# ============================================================
apf1_rows = []
for tf in TIMEFRAME_ORDER:
    subset = valid_tf[valid_tf["timeframe"] == tf]
    if subset.empty:
        continue
    apf1_rows.append(subset.nlargest(1, "Adjusted_PF"))

df_apf1 = pd.concat(apf1_rows).drop(columns=["tf_order"])

# ============================================================
# CSVに書き出し
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8", newline="") as f:
    write_section(f, "Top5 Total_Net_Profit per timeframe", df_top5, write_header=True)
    write_section(f, "TNP1st and APF1st per timeframe", df_out2, write_header=True)
    write_section(f, "TNP1st list (ascending timeframe)", df_tnp1, write_header=True)
    write_section(f, "APF1st list (ascending timeframe)", df_apf1, write_header=True)

print(f"Done: {OUTPUT_PATH}")
