#!/usr/bin/env python3
"""diagnose_skew_structure_v2.py — BC_rel(相対)の分母爆発を排除し、
絶対差 BC_abs と 残差std正規化指標 BC_abs_over_Bstd で真の乖離を見る。"""
import sys, re
import pandas as pd
import numpy as np

def parse_engine(feat):
    m = re.match(r"(e1[a-f])_", feat); return m.group(1) if m else "other"
def parse_tf(feat):
    m = re.search(r"_(M0\.5|M\d+|H\d+|D\d+|W\d+|MN)$", feat); return m.group(1) if m else "none"

def main():
    df = pd.read_parquet(sys.argv[1])
    df["engine"] = df["feature"].map(parse_engine)
    df["tf"] = df["feature"].map(parse_tf)
    df["BC_abs_over_Bstd"] = df["BC_abs_mean"] / (df["B_std"].abs() + 1e-10)

    print("# Skew 構造診断 v2 (絶対差ベース)\n")
    print(f"- 総 feature: {len(df)}")
    print(f"- BC_abs_mean: median={df['BC_abs_mean'].median():.4f}, mean={df['BC_abs_mean'].mean():.4f}, p90={df['BC_abs_mean'].quantile(0.9):.4f}")
    print(f"- BC_abs_over_Bstd: median={df['BC_abs_over_Bstd'].median():.4f} (1未満=残差スケール内ノイズ, 1超=真の乖離)")
    df["severe_abs"] = df["BC_abs_over_Bstd"] > 0.5
    print(f"- 真の重度乖離 (BC_abs > 0.5*B_std): {df['severe_abs'].sum()} ({df['severe_abs'].mean()*100:.1f}%)")
    print(f"  [比較] 旧 BC_rel>=0.1 基準: {(df['BC_rel']>=0.1).sum()} ({(df['BC_rel']>=0.1).mean()*100:.1f}%)")

    print("\n## TF × engine クロス (BC_abs_over_Bstd median)\n")
    piv = df.pivot_table(index="engine", columns="tf", values="BC_abs_over_Bstd", aggfunc="median")
    print(piv.to_markdown(floatfmt=".3f"))

    print("\n## engine 別 (絶対差ベース)\n")
    g = df.groupby("engine")
    summ = pd.DataFrame({
        "count": g.size(),
        "BC_abs_median": g["BC_abs_mean"].median(),
        "BC_abs_over_Bstd_med": g["BC_abs_over_Bstd"].median(),
        "BC_corr_median": g["BC_corr"].median(),
        "B_std_median": g["B_std"].median(),
        "Bnear0_pct": g.apply(lambda x: (x["B_mean"].abs()<0.01).mean()*100, include_groups=False),
    }).sort_values("BC_abs_over_Bstd_med", ascending=False)
    print(summ.to_markdown(floatfmt=".3f"))

    print("\n## 真の乖離 TOP20 (BC_abs_over_Bstd 降順)\n")
    top = df.nlargest(20,"BC_abs_over_Bstd")[["feature","n","BC_abs_mean","B_std","BC_abs_over_Bstd","BC_corr","BC_rel"]]
    print(top.to_markdown(index=False,floatfmt=".4f"))

    print("\n## 分母爆発の幻 (BC_rel>=0.5 だが BC_abs_over_Bstd<0.1) TOP10\n")
    cand = df[df["BC_rel"]>=0.5].nsmallest(10,"BC_abs_over_Bstd")[["feature","n","BC_rel","BC_abs_mean","B_std","BC_abs_over_Bstd"]]
    print(cand.to_markdown(index=False,floatfmt=".4f"))

if __name__ == "__main__":
    main()
