#!/usr/bin/env python3
"""
inspect_qastate_seed.py — QAState seed artifact の 5σ クリップ境界を検査

[確定済み]
  - gap バー (disc=True) で raw=S2=7680 (学習は素通し)、 production=115
  - production の EWM は現在値を取り込む実装 → 生値 7680 なら境界 ≥1880 で 115 にならない
  - gap バーは signal の 2 日前 = dry-run warmup 中に処理 = skip_qa_update=True で
    QA EWM が artifact seed に凍結される

[本スクリプトの問い]
  S3_QA_STATES_DIR/qa_state_e1{a,d}.pkl の (tf, feature) seed の mean/var/n から
  5σ 境界を計算し、 115 (production clip 値) と 7680 (raw) のどちら側か:

    境界 ≈ ±115 (σ≈23, 狭い) → warmup が凍結された狭い seed で 7680 をクリップ
                                = (M2') warmup skip_update が真因。
                                修正: warmup で seed から「凍結」 でなく「継続 update」 させる
    境界 > 7680 (σ>1536, 広い) → seed は広い → 凍結でも 7680 を通すはず
                                → 115 は別経路 (M1: 生値自体が違う)

[使い方]
  python inspect_qastate_seed.py
  python inspect_qastate_seed.py --qa-dir /workspace/data/XAUUSD/stratum_3_artifacts/qa_states_v5
"""

from __future__ import annotations

import sys
import math
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, "/workspace")
import blueprint as config  # noqa: E402

# 検査対象 (engine, tf, feature)
TARGETS: List[Tuple[str, str, str]] = [
    ("e1a", "M5", "e1a_statistical_moment_8_50"),
    ("e1a", "M1", "e1a_statistical_moment_5_50"),
    ("e1d", "M3", "e1d_commodity_channel_index_20"),
    ("e1d", "M5", "e1d_commodity_channel_index_14"),
]


def alpha_for_tf(tf: str) -> float:
    """production self.alpha 相当 (half_life = timeframe_bars_per_day[tf] → α)。"""
    tbpd = getattr(config, "timeframe_bars_per_day", None)
    hl = None
    if isinstance(tbpd, dict):
        hl = tbpd.get(tf)
    if hl is None:
        # フォールバック: 分換算 (1日 = 1440 分)
        minutes = {"M0.5": 0.5, "M1": 1, "M3": 3, "M5": 5, "M8": 8, "M15": 15}.get(tf, 5)
        hl = 1440.0 / minutes
    return 1.0 - math.exp(-math.log(2.0) / float(hl)), float(hl)


def bias_corrected_std(var: float, n: int, alpha: float) -> float:
    """rfe_1A_statistics の bias 補正済 ewm_std を再現。"""
    if n <= 1 or var <= 0:
        return 0.0
    r2 = (1.0 - alpha) ** 2
    m = n - 1
    if r2 < 1.0 - 1e-15:
        sum_w2 = alpha * alpha * (1.0 - r2 ** m) / (1.0 - r2) + r2 ** m
    else:
        sum_w2 = 1.0
    if sum_w2 < 1.0 - 1e-15:
        return math.sqrt(max(var / (1.0 - sum_w2) * (1.0 - sum_w2), 0.0)) if False else \
               math.sqrt(max(var * (1.0 / (1.0 - sum_w2)), 0.0))
    return 0.0


def find_artifact_files(qa_dir: Path, engine: str) -> List[Path]:
    pats = [f"qa_state_{engine}.pkl", f"*{engine}*.pkl"]
    found = []
    for p in pats:
        found += list(qa_dir.glob(p))
    # 重複除去・順序維持
    seen = set(); uniq = []
    for f in found:
        if f not in seen:
            seen.add(f); uniq.append(f)
    return uniq


def deep_find(obj: Any, feature: str, tf: str, depth: int = 0
              ) -> Optional[Dict[str, Any]]:
    """artifact 構造が不明でも (tf, feature) に対応する {mean,var,n} を探す。"""
    if depth > 6 or obj is None:
        return None
    if isinstance(obj, dict):
        # 直接 feature キー
        for k in (feature, f"{tf}|{feature}", (tf, feature)):
            if k in obj and isinstance(obj[k], dict):
                cand = obj[k]
                if any(x in cand for x in ("mean", "ewm_mean", "var", "ewm_var")):
                    return cand
        # tf キー配下に feature
        if tf in obj and isinstance(obj[tf], dict):
            sub = obj[tf]
            if feature in sub and isinstance(sub[feature], dict):
                return sub[feature]
        # 再帰
        for v in obj.values():
            r = deep_find(v, feature, tf, depth + 1)
            if r is not None:
                return r
    return None


def extract_mvn(d: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[int]]:
    mean = d.get("mean", d.get("ewm_mean"))
    var = d.get("var", d.get("ewm_var"))
    n = d.get("n", d.get("ewm_n", d.get("count")))
    try:
        mean = float(mean) if mean is not None else None
        var = float(var) if var is not None else None
        n = int(n) if n is not None else None
    except Exception:
        pass
    return mean, var, n


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-dir", type=Path, default=config.S3_QA_STATES_DIR)
    args = parser.parse_args()

    print("=" * 72)
    print("  inspect_qastate_seed.py — QAState seed の 5σ クリップ境界")
    print("=" * 72)
    print(f"  qa-dir: {args.qa_dir}")
    if not args.qa_dir.exists():
        print("  [FATAL] qa-dir が存在しません"); return
    print(f"  files: {[p.name for p in sorted(args.qa_dir.glob('*.pkl'))]}")

    cache: Dict[str, Any] = {}
    for engine, tf, feat in TARGETS:
        if engine not in cache:
            files = find_artifact_files(args.qa_dir, engine)
            if not files:
                cache[engine] = None
                print(f"\n  [SKIP] {engine}: artifact pkl 無し")
                continue
            with open(files[0], "rb") as f:
                cache[engine] = pickle.load(f)
            # 初回だけ top-level 構造を表示
            obj = cache[engine]
            print(f"\n  [{engine}] {files[0].name} top-level: ", end="")
            if isinstance(obj, dict):
                ks = list(obj.keys())
                print(f"dict, {len(ks)} keys, sample={ks[:5]}")
            else:
                print(type(obj))

        obj = cache.get(engine)
        if obj is None:
            continue

        alpha, hl = alpha_for_tf(tf)
        entry = deep_find(obj, feat, tf)
        print(f"\n  ── {tf} : {feat}  (α={alpha:.5g}, half_life={hl:g}) ──")
        if entry is None:
            print("     seed エントリが見つかりません (artifact に該当 feature/tf 無し?)")
            continue
        mean, var, n = extract_mvn(entry)
        if mean is None or var is None:
            print(f"     mean/var を抽出できず。 raw entry keys = {list(entry.keys())}")
            continue
        std = bias_corrected_std(var, n or 2, alpha)
        std_naive = math.sqrt(max(var, 0.0))
        lo = mean - 5.0 * std
        hi = mean + 5.0 * std
        print(f"     seed: mean={mean:+.6g}, var={var:.6g}, n={n}")
        print(f"     σ(bias補正)={std:.6g}  σ(naive√var)={std_naive:.6g}")
        print(f"     5σ 境界: [{lo:+.6g}, {hi:+.6g}]")
        # 判定
        if hi < 1000:
            print(f"     → 境界が狭い (上限 {hi:.4g} << 7680)")
            print(f"        = この seed で warmup 中に 7680 はクリップされる")
            print(f"        ★ (M2') warmup skip_update 凍結が真因と整合")
        elif hi >= 7680:
            print(f"     → 境界が広い (上限 {hi:.4g} >= 7680) = seed では 7680 を通す")
            print(f"        → 115 は別経路 (M1: 生値が違う) を疑う")
        else:
            print(f"     → 中間 (上限 {hi:.4g})。 production clip 値 115 と要照合")

    print("\n" + "=" * 72)
    print("  参考: production clip 後=115, raw/学習=7680")
    print("=" * 72)


if __name__ == "__main__":
    main()
