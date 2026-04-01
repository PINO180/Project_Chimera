import pandas as pd
from io import StringIO

# ============================================================
# パス設定（必要に応じて書き換えてください）
# ============================================================
INPUT_PATH = "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_topdata_pure_atr_results_sp36.csv"
OUTPUT_PATH = "/workspace/data/XAUUSD/stratum_7_models/1A_2B/optuna_topdata_pure_atr_results_sp36.md"

# ============================================================
# セクションタイトルとして扱う行を判定
# ============================================================
SECTION_TITLES = [
    "Top5 Total_Net_Profit per timeframe",
    "TNP1st and APF1st per timeframe",
    "TNP1st list (ascending timeframe)",
    "APF1st list (ascending timeframe)",
]

# ============================================================
# CSVを読み込んでセクションごとに分割
# ============================================================
with open(INPUT_PATH, "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

sections = []  # (title, [データ行]) のリスト
current_title = None
current_lines = []

for line in raw_lines:
    stripped = line.strip()

    # 空行はスキップ
    if not stripped:
        continue

    # セクションタイトル行の判定
    if stripped in SECTION_TITLES:
        # 前のセクションを保存
        if current_title and current_lines:
            sections.append((current_title, current_lines))
        current_title = stripped
        current_lines = []
    else:
        current_lines.append(line)

# 最後のセクションを保存
if current_title and current_lines:
    sections.append((current_title, current_lines))


# ============================================================
# DataFrameをMarkdownテーブルに変換するヘルパー
# ============================================================
def df_to_markdown(df):
    cols = df.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    separator = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                # 大きい数値は読みやすくカンマ区切り
                if abs(v) >= 1000:
                    cells.append(f"{v:,.2f}")
                else:
                    cells.append(str(v))
            elif isinstance(v, int):
                cells.append(f"{v:,}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator] + rows)


# ============================================================
# Markdownファイルに書き出し
# ============================================================
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write("# Trading Strategy Analysis Results\n\n")

    for title, lines in sections:
        f.write(f"## {title}\n\n")
        try:
            df = pd.read_csv(StringIO("".join(lines)))
            f.write(df_to_markdown(df))
        except Exception as e:
            f.write(f"(parse error: {e})\n")
        f.write("\n\n")

print(f"Done: {OUTPUT_PATH}")
