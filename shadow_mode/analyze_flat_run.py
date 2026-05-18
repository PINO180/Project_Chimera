#!/usr/bin/env python3
"""Phase 3: フラット run の結果分析

ReplayBridge SSoT alignment patch (§B.12.10.13) の効果を、
元の e1a Cluster A 70 cells (e1a_residual_70.csv) との 集合 diff で測定する。

実行:
  python3 analyze_flat_run.py \\
    --report   /workspace/shadow_mode/reports/flat_run_v1 \\
    --residual /workspace/e1a_residual_70.csv

出力 (stdout + report/analysis.md):
  - 原 70 cells の現状 (解消した数 / まだ failing の数)
  - 解消した cells の feature breakdown
  - まだ残ってる cells の feature × ts breakdown
  - 新規 failing (e1a / 他 engine 別)
  - 判定 (✅ 確証 / ⚠️ 部分 / ❌ 効果なし)
"""

from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--report", type=Path, required=True,
                   help="新 run の dir (failing.parquet を含む)")
    p.add_argument("--residual", type=Path, required=True,
                   help="e1a_residual_70.csv のパス")
    args = p.parse_args()

    failing_path = args.report / "failing.parquet"
    if not failing_path.exists():
        print(f"ERROR: {failing_path} not found", file=sys.stderr)
        return 1
    if not args.residual.exists():
        print(f"ERROR: {args.residual} not found", file=sys.stderr)
        return 1

    # --- load ---
    post = pd.read_parquet(failing_path)
    post["timestamp"] = pd.to_datetime(post["timestamp"], utc=True)
    post_keys = set(zip(post["timestamp"], post["timeframe"], post["feature_name"]))

    ref = pd.read_csv(args.residual)
    ref["timestamp"] = pd.to_datetime(ref["timestamp"], utc=True)
    ref_keys = set(zip(ref["timestamp"], ref["timeframe"], ref["feature_name"]))

    # --- 集合演算 ---
    still_failing = ref_keys & post_keys           # 70 cells のうちまだ失敗
    now_passing   = ref_keys - post_keys           # 70 cells のうち解消
    new_failing   = post_keys - ref_keys           # 70 cells になかった新規失敗
    new_e1a       = {k for k in new_failing if k[2].startswith("e1a")}
    new_other     = new_failing - new_e1a

    # --- engine 別 breakdown ---
    eng_post = Counter(k[2].split("_")[0] for k in post_keys)
    eng_new  = Counter(k[2].split("_")[0] for k in new_failing)

    # --- 判定ロジック ---
    pct_solved = len(now_passing) / 70 * 100 if ref_keys else 0
    if len(now_passing) == 70 and len(new_e1a) == 0:
        verdict = "✅ Cluster A 完全解消 (70/70、新規 e1a なし)"
        next_action = "→ ReplayBridge SSoT alignment が真因確証。Plan §B.12.10.13 commit。"
    elif len(now_passing) >= 60 and len(new_e1a) < 10:
        verdict = f"✅ Cluster A 大幅改善 ({len(now_passing)}/70 = {pct_solved:.1f}% 解消)"
        next_action = (f"→ 残った {len(still_failing)} cells は別原因の可能性。"
                       "Step 2 (ts 軸 trace) で詰める。")
    elif len(now_passing) >= 30:
        verdict = f"⚠️  Cluster A 部分解消 ({pct_solved:.1f}%)"
        next_action = "→ ReplayBridge patch は部分的に効果あり、残りに別真因が併存。"
    elif len(now_passing) == 0 and len(new_e1a) > 100:
        verdict = "❌ patch 逆効果 (Cluster A 改善ゼロ + 新規大量)"
        next_action = "→ revert、別仮説へ。"
    elif len(now_passing) == 0:
        verdict = "❌ Cluster A 全く改善せず"
        next_action = "→ ReplayBridge patch は的外れ、別仮説 / Step 2 へ。"
    else:
        verdict = f"⚠️  Cluster A 一部解消 ({pct_solved:.1f}%)"
        next_action = "→ パターン要観察。"

    # --- output ---
    lines = []
    L = lines.append
    L("=" * 72)
    L("Phase 3: ReplayBridge SSoT alignment patch — 効果測定")
    L("=" * 72)
    L("")
    L(f"## Run dir: {args.report}")
    L(f"## Residual ref: {args.residual}")
    L(f"## Total failing in new run: {len(post_keys):,}")
    L("")
    L("## 原 70 cells (e1a_residual_70.csv) の現状")
    L(f"   解消した:    {len(now_passing)} / 70  ({pct_solved:.1f}%)")
    L(f"   まだ failing: {len(still_failing)} / 70")
    L("")

    if now_passing:
        L("## 解消した cells (= patch 効果) — feature 別")
        for f, cnt in Counter(k[2] for k in now_passing).most_common():
            L(f"   {f}: {cnt}")
        L("")

    if still_failing:
        L("## まだ残ってる cells (= patch では未解決) — feature 別")
        for f, cnt in Counter(k[2] for k in still_failing).most_common():
            L(f"   {f}: {cnt}")
        L("")
        L("## まだ残ってる cells — ts 別 (先頭 15)")
        for ts, cnt in Counter(k[0] for k in still_failing).most_common(15):
            L(f"   {ts}: {cnt}")
        L("")

    L("## 新規 failing (70 cells になかった)")
    L(f"   e1a (= patch 副作用 / cache rebuild bug 等): {len(new_e1a)}")
    L(f"   他 engine: {len(new_other)}")
    if new_other:
        L("   engine 別 breakdown:")
        for prefix in ["e1b", "e1c", "e1d", "e1e", "e1f"]:
            cnt = eng_new.get(prefix, 0)
            if cnt > 0:
                L(f"     {prefix}: {cnt}")
    L("")

    L("## 全 engine 別 failure 数 (参考)")
    for prefix in ["e1a", "e1b", "e1c", "e1d", "e1e", "e1f"]:
        cnt = eng_post.get(prefix, 0)
        L(f"   {prefix}: {cnt}")
    L("")

    L("=" * 72)
    L(f"判定: {verdict}")
    L(next_action)
    L("=" * 72)

    md = "\n".join(lines)
    print(md)

    # report dir に保存
    out_md = args.report / "analysis_post_patch.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"\nSaved: {out_md}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
