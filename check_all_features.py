import polars as pl
import sys

# ファイルパス
file_path = "/workspace/data/XAUUSD/stratum_7_models/1A_2B/context_features_v2.parquet"

print(f"読み込み中: {file_path}")

try:
    # Parquetファイルを読み込む
    df = pl.read_parquet(file_path)

    # 数値カラムのみを抽出（timestampなどを除外）
    numeric_cols = [
        col
        for col, dtype in df.schema.items()
        if dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32] and col != "timestamp"
    ]

    print(f"検証対象カラム数: {len(numeric_cols)}")

    # 各カラムについて統計とサンプルを表示
    for col_name in numeric_cols:
        print("\n" + "=" * 60)
        print(f"📌 カラム名: {col_name}")

        # 統計量を計算
        stats = df.select(
            [
                pl.col(col_name).min().alias("最小値"),
                pl.col(col_name).mean().alias("平均値"),
                pl.col(col_name).median().alias("中央値"),
                pl.col(col_name).max().alias("最大値"),
                # 0.0 の割合も確認（埋め合わせの確認のため）
                (pl.col(col_name) == 0.0).mean().alias("ゼロの割合"),
            ]
        )

        print("--- 統計情報 ---")
        print(stats)

        # 実際の値を5行表示（0以外の値が見たい場合はフィルタも可能ですが、まずはそのまま先頭と、ランダムサンプリングを表示）
        print("\n--- サンプルデータ (先頭5行) ---")
        print(df.select(["timestamp", col_name]).head(5))

        # もし中央値が0なら、0以外の有効な値も少し見てみる
        if stats["中央値"][0] == 0:
            print("\n--- (参考) 0以外の有効値サンプル (最大5行) ---")
            non_zero = (
                df.filter(pl.col(col_name) != 0.0)
                .select(["timestamp", col_name])
                .head(5)
            )
            if len(non_zero) > 0:
                print(non_zero)
            else:
                print("※ 有効な値（非ゼロ）は見つかりませんでした。")

except Exception as e:
    print(f"エラーが発生しました: {e}")
    import traceback

    traceback.print_exc()
