#!/usr/bin/env python3
"""
[Layer 1] DiffAggregator smoke test

合成 captured + reference データで aggregator の挙動を検証:
  ケース 1: 全一致 (PASS)
  ケース 2: 1 行だけ rtol を超える (FAIL, 1 件)
  ケース 3: prod-only 行と ref-only 行が混在 (unpaired カウント検証)
  ケース 4: 全失敗 (>99% で hint heuristic が発火)
  ケース 5: 特定 TF だけが失敗
"""

import sys
from pathlib import Path
import tempfile
import shutil

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from shadow_mode.diff_aggregator import DiffAggregator, DiffStats
from shadow_mode.diff_report import DiffReport


def make_long(rows):
    """[(ts, tf, name, value), ...] → DataFrame"""
    return pd.DataFrame(
        rows, columns=["timestamp", "timeframe", "feature_name", "value"]
    )


def case_1_all_match():
    print("=" * 60)
    print("Case 1: 全一致 (PASS)")
    print("=" * 60)
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    captured = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_feat2", 2.5),
        (ts, "M5", "e1b_feat3", -0.7),
    ])
    reference = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_feat2", 2.5),
        (ts, "M5", "e1b_feat3", -0.7),
    ])
    agg = DiffAggregator(rtol=1e-7, atol=1e-12)
    result = agg.compare(captured, reference)
    s = result["stats"]
    print(f"  total={s.total}, passed={s.passed}, failed={s.failed}, "
          f"is_pass={s.is_pass()}")
    assert s.total == 3 and s.passed == 3 and s.failed == 0 and s.is_pass()
    print("  ✅ PASS")
    return True


def case_2_one_failure():
    print()
    print("=" * 60)
    print("Case 2: 1 行 rtol 超過 (FAIL, 1 件)")
    print("=" * 60)
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    captured = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_feat2", 2.50001),  # 4e-6 程度の差異 → rtol=1e-7 超え
        (ts, "M5", "e1b_feat3", -0.7),
    ])
    reference = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_feat2", 2.5),
        (ts, "M5", "e1b_feat3", -0.7),
    ])
    agg = DiffAggregator(rtol=1e-7, atol=1e-12)
    result = agg.compare(captured, reference)
    s = result["stats"]
    print(f"  total={s.total}, passed={s.passed}, failed={s.failed}")
    print(f"  fail_by_feature={s.fail_by_feature}")
    assert s.total == 3 and s.passed == 2 and s.failed == 1
    assert s.fail_by_feature.get("e1a_feat2") == 1
    print("  ✅ PASS")
    return True


def case_3_unpaired():
    print()
    print("=" * 60)
    print("Case 3: prod-only / ref-only 行が混在")
    print("=" * 60)
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    captured = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_only_in_prod", 3.0),  # prod-only
    ])
    reference = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_only_in_ref", 4.0),   # ref-only
    ])
    agg = DiffAggregator(rtol=1e-7)
    result = agg.compare(captured, reference)
    s = result["stats"]
    print(f"  paired={s.total}, captured-only={s.unpaired_captured}, "
          f"ref-only={s.unpaired_reference}")
    assert s.total == 1 and s.unpaired_captured == 1 and s.unpaired_reference == 1
    print("  ✅ PASS")
    return True


def case_4_all_fail_hint():
    print()
    print("=" * 60)
    print("Case 4: 全失敗 → hint で 'セットアップ違い' 検出")
    print("=" * 60)
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    captured = make_long([
        (ts, "M3", "e1a_feat1", 1.0),
        (ts, "M3", "e1a_feat2", 2.0),
        (ts, "M3", "e1a_feat3", 3.0),
    ])
    reference = make_long([
        (ts, "M3", "e1a_feat1", 99.0),
        (ts, "M3", "e1a_feat2", 99.0),
        (ts, "M3", "e1a_feat3", 99.0),
    ])
    agg = DiffAggregator(rtol=1e-7)
    result = agg.compare(captured, reference)
    s = result["stats"]
    print(f"  total={s.total}, failed={s.failed}, fail_rate={s.fail_rate}")
    assert s.failed == 3 and s.fail_rate == 1.0

    # write report and check hint.md mentions setup mismatch
    tmp = Path(tempfile.mkdtemp())
    try:
        rep = DiffReport(output_dir=tmp)
        rep.write_all(result, {
            "test_period": "2026-04-01:2026-04-02",
            "scenario": "continuous",
        })
        hint_text = (tmp / "hint.md").read_text()
        assert "全件失敗" in hint_text or "セットアップ違い" in hint_text
        print(f"  hint.md correctly flags setup mismatch")
    finally:
        shutil.rmtree(tmp)
    print("  ✅ PASS")
    return True


def case_5_tf_isolated():
    print()
    print("=" * 60)
    print("Case 5: M3 だけが失敗 → hint で 'TF 単独失敗' 検出")
    print("=" * 60)
    ts = pd.Timestamp("2026-04-01 12:00:00", tz="UTC")
    captured = make_long([
        (ts, "M3", "e1a_x", 1.0),
        (ts, "M3", "e1a_y", 2.0),  # 失敗
        (ts, "M5", "e1b_z", 3.0),
    ])
    reference = make_long([
        (ts, "M3", "e1a_x", 1.0),
        (ts, "M3", "e1a_y", 9.99),  # 失敗
        (ts, "M5", "e1b_z", 3.0),
    ])
    agg = DiffAggregator(rtol=1e-7)
    result = agg.compare(captured, reference)
    s = result["stats"]
    print(f"  total={s.total}, failed={s.failed}, fail_by_tf={s.fail_by_timeframe}")
    assert s.failed == 1 and "M3" in s.fail_by_timeframe

    tmp = Path(tempfile.mkdtemp())
    try:
        rep = DiffReport(output_dir=tmp)
        rep.write_all(result, {
            "test_period": "2026-04-01:2026-04-02",
            "scenario": "continuous",
        })
        hint_text = (tmp / "hint.md").read_text()
        # "M3 単独失敗" を含むことを確認
        assert "M3" in hint_text and "単独失敗" in hint_text
        print(f"  hint.md correctly flags M3-only failure")
    finally:
        shutil.rmtree(tmp)
    print("  ✅ PASS")
    return True


def main():
    print()
    print("=" * 60)
    print("Layer 1 DiffAggregator + DiffReport smoke test")
    print("=" * 60)
    print()

    tests = [
        ("Case 1 全一致", case_1_all_match),
        ("Case 2 1 行失敗", case_2_one_failure),
        ("Case 3 unpaired", case_3_unpaired),
        ("Case 4 全失敗 hint", case_4_all_fail_hint),
        ("Case 5 TF 単独失敗 hint", case_5_tf_isolated),
    ]
    results = []
    for name, fn in tests:
        try:
            results.append((name, fn()))
        except AssertionError as e:
            print(f"  ❌ {name} ASSERTION FAILED: {e}")
            results.append((name, False))
        except Exception as e:
            print(f"  ❌ {name} EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    print()
    print("=" * 60)
    print("サマリー")
    print("=" * 60)
    for name, passed in results:
        marker = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {marker}  {name}")

    if all(p for _, p in results):
        print()
        print("✅ 全 smoke test PASS")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
