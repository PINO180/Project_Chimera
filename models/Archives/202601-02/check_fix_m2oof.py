import polars as pl

# 犯人の可能性が高いファイル
TARGET_FILE = (
    "/workspace/data/XAUUSD/stratum_7_models/1A_2B/m2_oof_predictions_v2.parquet"
)

try:
    df = pl.read_parquet(TARGET_FILE)
    print(f"検査対象: {TARGET_FILE}")
    print(f"総行数: {df.height}")

    # 重複チェック
    dup_check = df.group_by("timestamp").len().filter(pl.col("len") > 1)

    if dup_check.height > 0:
        print(f"❌ 【確定】このファイルが汚染されています！")
        print(f"   同一時刻に {dup_check['len'].max()} 行の重複があります。")
        print("   シミュレーターはこれを結合して爆発しています。")
    else:
        print(
            "✅ このファイルも正常です。（となると、シミュレーターの結合ロジック自体が疑われます）"
        )

except Exception as e:
    print(f"読込エラー: {e}")
