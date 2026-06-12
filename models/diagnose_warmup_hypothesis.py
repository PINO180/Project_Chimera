#!/usr/bin/env python3
"""diagnose_warmup_hypothesis.py — warmup/window 成熟度仮説の検証。
仮説: OLS窓が育ちきっていない(発火サンプル n が少ない)feature ほど乖離大。"""
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

    print("# warmup/window 成熟度 仮説検証\n")
    print("## 1. 発火数 n と 乖離 の関係 (warmup仮説の核心)\n")
    valid = df[df["B_std"].abs() > 0.01].copy()
    print(f"- 全 feature: {len(df)}, B_std>0.01 の有効 feature: {len(valid)}")
    print(f"  (B_std<=0.01 の {len(df)-len(valid)} 件は『定数残差』= 正規化爆発の幻として除外)\n")
    corr_all = valid["n"].corr(valid["BC_abs_over_Bstd"], method="spearman")
    print(f"- Spearman corr(n, BC_abs_over_Bstd) = {corr_all:.3f}")
    print(f"  → 負なら『n少ない=乖離大』= warmup仮説支持\n")
    valid["n_bucket"] = pd.cut(valid["n"], bins=[0,12,20,40,1000], labels=["n<=12","n13-20","n21-40","n>40"])
    g = valid.groupby("n_bucket", observed=True)
    summ = pd.DataFrame({
        "count": g.size(),
        "BC_abs_over_Bstd_med": g["BC_abs_over_Bstd"].median(),
        "BC_corr_med": g["BC_corr"].median(),
        "severe_pct(>0.5)": g.apply(lambda x:(x["BC_abs_over_Bstd"]>0.5).mean()*100),
    })
    print(summ.to_markdown(floatfmt=".3f"))
    print("\n## 2. 定数残差(B_std<=0.01)除外後の真の乖離\n")
    sev = (valid["BC_abs_over_Bstd"]>0.5).sum()
    print(f"- 有効 feature {len(valid)} 中、 真の重度乖離(>0.5*B_std): {sev} ({sev/len(valid)*100:.1f}%)")
    print("\n## 3. 定数残差除外後の engine 別乖離\n")
    g2 = valid.groupby("engine")
    s2 = pd.DataFrame({
        "count": g2.size(),
        "BC_abs_over_Bstd_med": g2["BC_abs_over_Bstd"].median(),
        "severe_pct": g2.apply(lambda x:(x["BC_abs_over_Bstd"]>0.5).mean()*100),
        "median_n": g2["n"].median(),
    }).sort_values("BC_abs_over_Bstd_med", ascending=False)
    print(s2.to_markdown(floatfmt=".3f"))
    print("\n## 4. n>40 (窓が最も育つ低TF) に限定した乖離\n")
    big = valid[valid["n"]>40]
    sev_big = (big["BC_abs_over_Bstd"]>0.5).sum()
    print(f"- n>40 feature {len(big)} 中、 真の重度乖離: {sev_big} ({sev_big/max(len(big),1)*100:.1f}%)")
    print(f"  → 低ければ『窓さえ育てば一致』= warmup仮説の決定的証拠")
    print(f"  → 高ければ warmup では説明できない別真因")

if __name__ == "__main__":
    main()
