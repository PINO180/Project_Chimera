"""
[Layer 1] DiffReport

DiffAggregator の出力を以下の形式でファイル化する:
  - summary.md       : 人が読む要約 (PASS/FAIL、内訳)
  - paired.parquet   : 全 inner-join 結果 (大規模、デバッグ用)
  - failing.parquet  : failing 行のみ (CI で artifact 保存)
  - worst.csv        : worst-K 行 (CSV、すぐ目視できる)
  - hint.md          : 失敗パターン分析と原因 hint (簡易ヒューリスティック)
"""

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .diff_aggregator import DiffStats

logger = logging.getLogger("shadow_mode.diff_report")


class DiffReport:
    """差分レポート出力。

    Args:
        output_dir: レポート出力先ディレクトリ
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_all(self, result: Dict[str, object], context: Dict[str, str]) -> None:
        """全レポートを書き出す。

        Args:
            result: DiffAggregator.compare() の戻り値
            context: 実行コンテキスト
                {
                    "test_period": "2026-04-01 → 2026-05-01",
                    "scenario": "continuous",
                    "rtol": "1e-7", "atol": "1e-12",
                    ... (任意)
                }
        """
        stats: DiffStats = result["stats"]
        paired: pd.DataFrame = result["paired"]
        failing: pd.DataFrame = result["failing"]
        worst: pd.DataFrame = result["worst"]

        self._write_summary_md(stats, context)
        self._write_paired_parquet(paired)
        self._write_failing_parquet(failing)
        self._write_worst_csv(worst)
        self._write_hint_md(stats, failing, worst, context)

        logger.info(f"Reports written to {self.output_dir}/")

    # -------------------------------------------------------------- writers

    def _write_summary_md(self, stats: DiffStats, ctx: Dict[str, str]) -> None:
        path = self.output_dir / "summary.md"
        verdict = "✅ **PASS**" if stats.is_pass() else "❌ **FAIL**"
        ctx_str = "\n".join(f"- **{k}**: {v}" for k, v in ctx.items())

        # 特徴量別 fail count (top 20)
        feat_lines: List[str] = []
        for feat, n in list(stats.fail_by_feature.items())[:20]:
            feat_lines.append(f"  - `{feat}`: {n}")
        if not feat_lines:
            feat_lines.append("  (なし)")

        # TF 別 fail count
        tf_lines: List[str] = []
        for tf, n in stats.fail_by_timeframe.items():
            tf_lines.append(f"  - `{tf}`: {n}")
        if not tf_lines:
            tf_lines.append("  (なし)")

        content = f"""# Shadow Mode Layer 1 — Diff Report

## Verdict: {verdict}

## Context
{ctx_str}

## Tolerance
- `rtol = {stats.rtol}`
- `atol = {stats.atol}`

## Counts
| Metric | Value |
|---|---|
| Total compared rows | {stats.total:,} |
| Passed | {stats.passed:,} |
| Failed | {stats.failed:,} |
| Fail rate | {stats.fail_rate * 100:.6f}% |
| Captured-only (no reference match) | {stats.unpaired_captured:,} |
| Reference-only (no captured match) | {stats.unpaired_reference:,} |

## Failing breakdown by timeframe
{chr(10).join(tf_lines)}

## Top failing features (top 20)
{chr(10).join(feat_lines)}

## Files
- `paired.parquet` — 全 inner-join 結果 (デバッグ用)
- `failing.parquet` — failing 行のみ (CI artifact)
- `worst.csv` — `abs_diff` top-K rows
- `hint.md` — 失敗パターン分析

## Pass criteria
- `failed = 0` で PASS
- `failed > 0` の場合、`hint.md` で原因の手がかりを参照
"""
        path.write_text(content, encoding="utf-8")
        logger.info(f"  summary.md: {path}")

    def _write_paired_parquet(self, paired: pd.DataFrame) -> None:
        path = self.output_dir / "paired.parquet"
        if paired.empty:
            # Write an empty marker file
            path.write_bytes(b"")
            logger.info(f"  paired.parquet: empty (no data)")
            return
        paired.to_parquet(path, compression="snappy", index=False)
        logger.info(f"  paired.parquet: {path} ({len(paired):,} rows)")

    def _write_failing_parquet(self, failing: pd.DataFrame) -> None:
        path = self.output_dir / "failing.parquet"
        if failing.empty:
            path.write_bytes(b"")
            logger.info(f"  failing.parquet: empty (= pass)")
            return
        failing.to_parquet(path, compression="snappy", index=False)
        logger.info(f"  failing.parquet: {path} ({len(failing):,} rows)")

    def _write_worst_csv(self, worst: pd.DataFrame) -> None:
        path = self.output_dir / "worst.csv"
        if worst.empty:
            path.write_text("", encoding="utf-8")
            return
        worst.to_csv(path, index=False)
        logger.info(f"  worst.csv: {path} ({len(worst):,} rows)")

    def _write_hint_md(
        self,
        stats: DiffStats,
        failing: pd.DataFrame,
        worst: pd.DataFrame,
        ctx: Dict[str, str],
    ) -> None:
        """失敗パターンを簡易ヒューリスティックで分析し、原因 hint を出す。"""
        path = self.output_dir / "hint.md"

        if stats.is_pass():
            content = """# Hint

✅ 全ての特徴量がリファレンスと一致しました (failed = 0)。
追加の解析は不要です。
"""
            path.write_text(content, encoding="utf-8")
            return

        # ヒューリスティック検出
        hints: List[str] = []

        # 1. 特定 TF だけが失敗 → TF 固有のロジックバグの可能性
        if len(stats.fail_by_timeframe) == 1:
            only_tf = next(iter(stats.fail_by_timeframe))
            hints.append(
                f"- **{only_tf} 単独失敗**: 失敗が `{only_tf}` のみに集中している。"
                f"  → 当該 TF の resample / lookback / OHLCV 集約に bug の可能性。"
                f"  特に `_resample_and_update_buffer` の bucket 判定や "
                f"`data_buffers['{only_tf}']` のサイズ確認を推奨。"
            )

        # 2. 特定の特徴量 prefix だけが失敗 → engine module 固有の bug
        if not failing.empty:
            failing_engines = failing["feature_name"].str.extract(
                r"^(e1[a-f])_"
            )[0].dropna().unique()
            if len(failing_engines) == 1:
                eng = failing_engines[0]
                hints.append(
                    f"- **{eng}_ engine 単独失敗**: 失敗が `{eng}_*` 特徴量のみ。"
                    f"  → `engine_1_{eng[-1].upper()}_a_vast_universe_of_features.py` "
                    f"または `realtime_feature_engine.FeatureModule1{eng[-1].upper()}` の"
                    f"  `_build_polars_pieces` の式不整合の可能性。"
                )

        # 3. 大きな相対誤差 (>1%) → 致命的な計算違い
        big_rel = failing[failing["rel_diff"] > 0.01]
        if not big_rel.empty:
            hints.append(
                f"- **大きな相対誤差**: rel_diff > 1% の行が {len(big_rel)} 件存在。"
                f"  → 数値精度ではなく `計算ロジックの違い` (発見 #60 の disc fallback 級の構造バグ) の疑い。"
                f"  → `worst.csv` を見て、prod/ref 値の order of magnitude を確認。"
            )

        # 4. timestamp 集中 → 特定の時刻に発生 (例: 週末跨ぎ、市場閉鎖)
        if not failing.empty:
            ts_series = failing["timestamp"]
            unique_dates = ts_series.dt.date.nunique()
            n_failing = len(failing)
            if unique_dates < 5 and n_failing > 100:
                hints.append(
                    f"- **時刻集中**: failing 行が {unique_dates} 日に集中 ({n_failing} 件)。"
                    f"  → 週末跨ぎ・市場閉鎖・ニュース時刻等のイベント駆動の可能性。"
                    f"  → 失敗日付:  "
                    + ", ".join(
                        sorted(set(d.strftime("%Y-%m-%d")
                                  for d in ts_series.dt.date))[:5]
                    )
                )

        # 5. 全件失敗 → セットアップ違い (warmup 不足、データ範囲ズレ等)
        if stats.fail_rate > 0.99:
            hints.append(
                "- **全件失敗 (>99%)**: ほぼ全行が不一致 → 通常の bug ではなく "
                "**セットアップ違い** の可能性が高い。"
                "  → reference の TF 識別、timestamp の TZ、feature_name の "
                "  TF suffix 剥がし、warmup_end_ts の境界、を確認。"
            )

        if not hints:
            hints.append(
                "- 自動検出可能なパターンは見つかりませんでした。"
                "`worst.csv` と `failing.parquet` を手動で精査してください。"
            )

        content = f"""# Hint — 失敗パターン分析

検出された手がかり:

{chr(10).join(hints)}

## 次のステップ
1. `worst.csv` で個別行を確認 (大きい順)
2. `failing.parquet` を Polars/Pandas で読み、特徴量別・時刻別に grouping
3. 該当する production コードの該当 TF / 該当 engine module を再確認
4. 必要に応じて該当バーの input data (OHLCV deque) を再現してデバッグ

## 関連ドキュメント
- Train_Serve_Skew_Audit_Report.md (発見 #59〜#62 の Phase 9d セクション)
- Skew_Detection_Hardening_Plan.md (Layer 4 静的検出と組み合わせて trace)
"""
        path.write_text(content, encoding="utf-8")
        logger.info(f"  hint.md: {path}")
