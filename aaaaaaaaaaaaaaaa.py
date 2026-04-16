import polars as pl
from blueprint import (
    S7_M1_OOF_PREDICTIONS_LONG,
    S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
)
import numpy as np

for name, m1_path, m2_path in [
    ("LONG", S7_M1_OOF_PREDICTIONS_LONG, S7_M2_OOF_PREDICTIONS_LONG),
    ("SHORT", S7_M1_OOF_PREDICTIONS_SHORT, S7_M2_OOF_PREDICTIONS_SHORT),
]:
    print(f"\n{'=' * 50}")
    print(f"=== {name} ===")

    m1 = pl.read_parquet(m1_path)
    m2 = pl.read_parquet(m2_path)

    print(f"M1 OOF 行数: {len(m1)}")
    print(f"M2 OOF 行数: {len(m2)}")

    # M1のlogit変換後の値（Bx2と同じ変換）
    raw = m1["prediction"].to_numpy()
    clipped = np.clip(raw, 1e-7, 1 - 1e-7)
    logits = np.clip(np.log(clipped / (1 - clipped)), -10.0, 10.0)

    m1_pass = (raw >= 0.50).sum()
    m1_fail = (raw < 0.50).sum()
    print(f"M1 proba >= 0.50: {m1_pass}行 ({m1_pass / len(m1) * 100:.1f}%)")
    print(f"M1 proba <  0.50: {m1_fail}行 ({m1_fail / len(m1) * 100:.1f}%)")

    # M2 OOFにM1 < 0.50のバーが紛れ込んでいないか確認
    # timestamp + timeframe でjoinして確認
    m1_fail_keys = m1.filter(pl.col("prediction") < 0.50).select(
        ["timestamp", "timeframe"]
    )
    m2_keys = m2.select(["timestamp", "timeframe"])

    leaked = m2_keys.join(m1_fail_keys, on=["timestamp", "timeframe"], how="inner")
    print(f"M2 OOFにM1<0.50のバーが混入: {len(leaked)}行")
    if len(leaked) > 0:
        print("  ⚠️  混入あり → シミュレーターと本番で挙動が異なる可能性")
    else:
        print("  ✅ 混入なし → 暗黙的依存は問題なし")
