#!/usr/bin/env python3
"""
diagnose_roster_weight.py — 乖離族 (高次モーメント・volume) がモデルでどれだけ効くか (案 B)

Two-Brain の 4 モデル (long_m1/m2, short_m1/m2) の gain importance を取り、
feature を族に分類して「乖離族の合計 gain 割合」を出す。
  乖離族の gain がほぼ 0 → train-serve の小差は予測に効かない → 予測突合不要、deploy へ。
  乖離族の gain が大きい → 効くので予測レベル突合 (案 A) が必要。

使い方:
  python diagnose_roster_weight.py
"""
from __future__ import annotations
import sys, pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
import joblib
sys.path.insert(0, "/workspace")
import blueprint as config

try:
    import lightgbm as lgb
except Exception:
    lgb = None

MODELS = {
    "long_m1":  getattr(config, "S7_M1_MODEL_LONG_PKL", None),
    "long_m2":  getattr(config, "S7_M2_MODEL_LONG_PKL", None),
    "short_m1": getattr(config, "S7_M1_MODEL_SHORT_PKL", None),
    "short_m2": getattr(config, "S7_M2_MODEL_SHORT_PKL", None),
}

# 乖離族の分類 (substring マッチ)。 小文字で判定。
DIVERGE = {
    "高次モーメント (z^8増幅)": ["moment_5","moment_6","moment_7","moment_8",
                               "kurtosis","jarque_bera"],
    "volume 系 (~2%系統差)":  ["volume","obv","mfi","accumulation","force_index",
                              "chaikin","money_flow","vwap"],
    "(参考)CCI/aroon/williams/stoch_rsi": ["commodity_channel_index","aroon",
                                          "williams_r","stochastic_rsi"],
}


def classify(feat):
    f = feat.lower()
    for label, subs in DIVERGE.items():
        if any(s in f for s in subs):
            return label
    return None


def get_importance(m):
    """(names, gains) を返す。 Booster / sklearn wrapper / dict に対応。"""
    if lgb is not None and isinstance(m, lgb.Booster):
        return m.feature_name(), m.feature_importance(importance_type="gain")
    if hasattr(m, "booster_"):
        b = m.booster_
        return b.feature_name(), b.feature_importance(importance_type="gain")
    if isinstance(m, dict):
        for v in m.values():
            r = get_importance(v)
            if r: return r
    if hasattr(m, "feature_importances_"):
        names = (list(getattr(m, "feature_name_", []))
                 or list(getattr(m, "feature_names_in_", [])))
        return names, np.asarray(m.feature_importances_, float)
    return None


def main():
    print("="*80)
    print("  diagnose_roster_weight — 乖離族の gain importance 割合 (案 B)")
    print("="*80)

    grand = defaultdict(float)   # 族 -> 全モデル合計 gain
    grand_total = 0.0

    for name, path in MODELS.items():
        if path is None or not Path(path).exists():
            print(f"\n  [{name}] モデル無し: {path}")
            continue
        m = joblib.load(path)
        r = get_importance(m)
        if r is None:
            print(f"\n  [{name}] importance 取得不可 (型: {type(m)})")
            continue
        names, gains = r
        gains = np.asarray(gains, float)
        names = list(names)
        total = gains.sum()
        if total <= 0:
            print(f"\n  [{name}] gain 合計 0"); continue

        by_fam = defaultdict(float)
        for f, g in zip(names, gains):
            lab = classify(f)
            if lab:
                by_fam[lab] += g
                grand[lab] += g
        grand_total += total

        print(f"\n  [{name}]  使用特徴量 {len(names)} 個, 総 gain {total:.4g}")
        div_sum = sum(by_fam.values())
        print(f"    乖離族 合計 gain 割合: {100*div_sum/total:.2f}%")
        for lab in DIVERGE:
            g = by_fam.get(lab, 0.0)
            print(f"      {lab:<34}: {100*g/total:6.2f}%")
        # 乖離族の中で個別 TOP5
        rows = [(f, g) for f, g in zip(names, gains) if classify(f)]
        rows.sort(key=lambda x: x[1], reverse=True)
        if rows:
            print(f"    乖離族 個別 TOP5 (gain%):")
            for f, g in rows[:5]:
                print(f"      {f:<44}{100*g/total:6.2f}%")

    if grand_total > 0:
        print("\n" + "="*80)
        print(f"  【全モデル合算】 乖離族 合計 gain 割合: {100*sum(grand.values())/grand_total:.2f}%")
        for lab in DIVERGE:
            print(f"    {lab:<34}: {100*grand.get(lab,0.0)/grand_total:6.2f}%")
        print("="*80)
        print("  乖離族の gain が ~0% に近い → 小差は予測に効かない → 予測突合不要、 deploy へ。")
        print("  乖離族の gain が大きい       → 効くので案 A (予測レベル突合) が必要。")
        print("="*80)


if __name__ == "__main__":
    main()
