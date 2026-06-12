#!/usr/bin/env python3
"""diagnose_skew_structure.py — metrics_per_feature.parquet を多角的に切って
BC 乖離の構造的真因を特定する。軸: TF / engine / n / C-over-B ratio。"""
import sys, re
import pandas as pd
import numpy as np

SEVERE = 0.1

def parse_engine(feat: str) -> str:
    m = re.match(r"(e1[a-f])_", feat)
    return m.group(1) if m else "other"

def parse_tf(feat: str) -> str:
    m = re.search(r"_(M0\.5|M\d+|H\d+|D\d+|W\d+|MN)$", feat)
    return m.group(1) if m else "none"

def summarize(df, group_col):
    g = df.groupby(group_col)
    out = pd.DataFrame({
        "count":          g.size(),
        "BC_rel_median":  g["BC_rel"].median(),
        "BC_rel_p90":     g["BC_rel"].quantile(0.90),
        "severe_pct":     g.apply(lambda x: (x["BC_rel"] >= SEVERE).mean() * 100),
        "BC_corr_median": g["BC_corr"].median(),
        "sign_flip_med":  g["BC_sign_flip_rate"].median(),
        "CoverB_median":  g["BC_ratio_C_over_B"].median(),
        "AB_abs_median":  g["AB_abs_mean"].median(),
        "AC_abs_median":  g["AC_abs_mean"].median(),
    })
    out["AC_over_AB"] = out["AC_abs_median"] / (out["AB_abs_median"] + 1e-12)
    return out.sort_values("BC_rel_median", ascending=False)

def md_table(df, title):
    print(f"\n## {title}\n")
    cols = df.columns.tolist()
    print("| " + " | ".join([df.index.name or "key"] + cols) + " |")
    print("|" + "---|" * (len(cols) + 1))
    for idx, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                vals.append(f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}")
            else:
                vals.append(str(int(v)) if isinstance(v, (int, np.integer)) else str(v))
        print(f"| {idx} | " + " | ".join(vals) + " |")

def main():
    path = sys.argv[1]
    df = pd.read_parquet(path)
    print(f"# Skew 構造診断: {path}")
    print(f"\n- 総 feature 数: {len(df)}")
    print(f"- 重度乖離 (BC_rel>={SEVERE}): {(df['BC_rel']>=SEVERE).sum()} ({(df['BC_rel']>=SEVERE).mean()*100:.1f}%)")
    df["engine"] = df["feature"].map(parse_engine)
    df["tf"] = df["feature"].map(parse_tf)
    df["n_bucket"] = pd.cut(df["n"], bins=[0,12,20,40,1000],
                            labels=["n<=12(高TF)","n13-20","n21-40","n>40(低TF)"])
    md_table(summarize(df,"tf"),"軸1: TF 別")
    md_table(summarize(df,"engine"),"軸2: engine 別")
    md_table(summarize(df,"n_bucket"),"軸3: サンプル数 n 別")
    print("\n## 軸4: TF × engine クロス (BC_rel median)\n")
    piv = df.pivot_table(index="engine",columns="tf",values="BC_rel",aggfunc="median")
    print(piv.to_markdown(floatfmt=".2f"))
    print("\n## C/B ratio 極端 TOP15 (本番OLSが学習と最もスケール乖離)\n")
    df["CoverB_abs"] = (df["BC_ratio_C_over_B"]-1.0).abs()
    top = df.nlargest(15,"CoverB_abs")[["feature","n","BC_rel","BC_corr","BC_ratio_C_over_B","AB_abs_mean","AC_abs_mean"]]
    print(top.to_markdown(index=False,floatfmt=".3f"))

if __name__ == "__main__":
    main()
