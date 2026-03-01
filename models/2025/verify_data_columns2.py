import polars as pl
from pathlib import Path

# パスは環境に合わせてください
BASE_DIR = Path(
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/meta_labeled_oof_partitioned_v2"
)

# 全てのparquetファイルを取得してソート
files = sorted(list(BASE_DIR.glob("**/*.parquet")))

if len(files) < 2:
    print("ファイルが足りません")
else:
    # 最初の日（Day 1）と、適当に離れた日（Day 50）を取得
    file_1 = files[0]
    file_2 = files[50]

    df1 = pl.read_parquet(file_1)
    df2 = pl.read_parquet(file_2)

    val1 = df1["trend_bias_25"][0]
    val2 = df2["trend_bias_25"][0]

    date1 = str(file_1).split("year=")[1].split("/")[0]  # 年だけ簡易取得
    date2 = str(file_2).split("year=")[1].split("/")[0]

    print(f"File 1 ({file_1.parent.name}): trend_bias_25 = {val1}")
    print(f"File 2 ({file_2.parent.name}): trend_bias_25 = {val2}")

    if val1 != val2:
        print(
            "\n✅ 成功！ 日付によって値が変動しています。LightGBMはこれを学習できます。"
        )
    else:
        print("\n❌ 失敗... 値が変化していません。")
