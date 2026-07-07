#!/usr/bin/env python3
"""summarize_failing.py — shadow_mode の failing.parquet を多角的に集計。
pre-OLS 乖離が何に局在しているか確定する。"""
import sys, re
import pandas as pd
import numpy as np

def parse_engine(feat):
    m = re.match(r"(e1[a-f])_", feat); return m.group(1) if m else "other"

def main():
    df = pd.read_parquet(sys.argv[1])
    print(f"# failing 全{len(df)}行の構造分析\n")
    df["engine"] = df["feature_name"].map(parse_engine)

    print("## 1. engine 別\n")
    g = df.groupby("engine")
    s = pd.DataFrame({
        "count": g.size(), "abs_diff_max": g["abs_diff"].max(),
        "abs_diff_median": g["abs_diff"].median(), "rel_diff_max": g["rel_diff"].max(),
        "rel_diff_median": g["rel_diff"].median(),
    }).sort_values("count", ascending=False)
    print(s.to_markdown(floatfmt=".4f"))

    print("\n## 2. TF 別\n")
    g2 = df.groupby("timeframe")
    s2 = pd.DataFrame({
        "count": g2.size(), "abs_diff_max": g2["abs_diff"].max(),
        "rel_diff_median": g2["rel_diff"].median(),
    }).sort_values("count", ascending=False)
    print(s2.to_markdown(floatfmt=".4f"))

    print("\n## 3. feature 別 (全件、count降順)\n")
    g3 = df.groupby("feature_name")
    s3 = pd.DataFrame({
        "count": g3.size(), "abs_diff_max": g3["abs_diff"].max(),
        "rel_diff_max": g3["rel_diff"].max(),
        "TFs": g3["timeframe"].apply(lambda x: ",".join(sorted(set(x)))),
    }).sort_values("count", ascending=False)
    print(s3.to_markdown(floatfmt=".4f"))

    print("\n## 4. 計算カテゴリ別 (キーワード分類)\n")
    def categorize(feat):
        f = feat.lower()
        if any(k in f for k in ["kurtosis","variance","rolling_var","_var","var_"]): return "variance/kurtosis系"
        if any(k in f for k in ["force_index","volume","chaikin","accumulation","mfi","cmf"]): return "volume系"
        if any(k in f for k in ["musical","tonality","acoustic","spectral","aesthetic","rhythm"]): return "spectral/音響系"
        if any(k in f for k in ["atr","kalman","kpss","t_dist","iqr","stabilization","theil","adf"]): return "その他統計/平滑"
        return "他"
    df["cat"] = df["feature_name"].map(categorize)
    g4 = df.groupby("cat")
    s4 = pd.DataFrame({
        "count": g4.size(), "abs_diff_max": g4["abs_diff"].max(),
        "rel_diff_median": g4["rel_diff"].median(),
    }).sort_values("count", ascending=False)
    print(s4.to_markdown(floatfmt=".4f"))

    print("\n## 5. 時刻分布 (上位15時刻)\n")
    df["ts_min"] = pd.to_datetime(df["timestamp"]).dt.strftime("%H:%M")
    print(df["ts_min"].value_counts().head(15).to_markdown())

if __name__ == "__main__":
    main()
