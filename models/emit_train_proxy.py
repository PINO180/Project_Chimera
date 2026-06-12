#!/usr/bin/env python3
"""
emit_train_proxy.py — 学習側 ground-truth proxy (= 修正後 2_G が実際に貼る proxy) を
TF 別に出力する。

[目的]
  verify_dump_deque_order.py の X 比較は、 学習側 proxy を旧式 (close/close.shift(1)-1)
  + merge_asof(allow_exact_matches=False) で再構成しており、 これは production 自己整合を
  測る装置になっていた (= deploy 判断に使えない)。
  本スクリプトは「修正後 2_G が実際に出力する proxy」 を、 2_G 本体の
  build_proxy_lazyframe を逐語流用して生成する (= 式ドリフトゼロ)。

[何を出すか]
  各 HF TF (M0.5/M1/M3/M5/M8/M15) について:
    - その TF の feature timestamp 軸 (= S2 e1a 代表、 label="left" = バー開始時刻)
    - そこに proxy_M5 を 2_G と同一の join_asof(backward, exact 許可) で貼り
    - 2_G と同じく fill_null(0.0)
  → train_proxy_<TF>.parquet (columns: timestamp, proxy)

[なぜ start-time 軸か]
  学習側 2_G は feature 行 (= S1 label="left" = バー開始時刻) に proxy を貼る。
  production は _close_ts_for で「開始 + TF幅 = バー終値時刻」 を search_ts にする。
  本 ground-truth は 学習側 (開始時刻貼り) を忠実に再現する。 これと production dump
  x_deque を位置対応 (bar-by-bar、 cadence は Phase B で一致済) で突合してはじめて
  close-vs-start lookup ズレ (= 仮説 I) の実害が数字で出る。

[実行前提]
  blueprint.S1_PROCESSED / S2_FEATURES_VALIDATED が dump の timestamp 範囲
  (5/26) をカバーしていること。 OLS 診断パイプライン (run_ols_diagnosis_pipeline.sh)
  で 5/2-5/26 を延長計算した S1/S2 を指していること。
  (proxy 値は close.shift(1)/shift(2) のローカル計算なので期間長は精度に無関係。
   timestamp 範囲カバーだけが条件。)

[使い方]
  python emit_train_proxy.py
  python emit_train_proxy.py --out-dir /workspace/data/diagnostics/train_proxy
  python emit_train_proxy.py --two-g /workspace/validation/2_G_alpha_neutralizer.py
"""

from __future__ import annotations

import sys
import argparse
import importlib.util
import types
from pathlib import Path
from typing import Optional

import polars as pl

# ── プロジェクトルート解決 + blueprint ─────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
import blueprint as config  # noqa: E402

# X 比較の代表エンジン (verify_dump_deque_order.py と一致させる)
REP_ENGINE = "e1a"
ENGINE_TO_UNIVERSE = {
    "e1a": "A", "e1b": "B", "e1c": "C", "e1d": "D", "e1e": "E", "e1f": "F",
}


def load_two_g(explicit_path: Optional[Path]) -> types.ModuleType:
    """2_G_alpha_neutralizer.py を file path から import する。

    ファイル名が数字始まりで通常 import 不可のため importlib を使う。
    import 時副作用は無い (main() は __main__ ガード下、 module level は
    関数定義 + blueprint import のみ)。
    """
    candidates = []
    if explicit_path is not None:
        candidates.append(explicit_path)
    candidates += [
        PROJECT_ROOT / "validation" / "2_G_alpha_neutralizer.py",
        PROJECT_ROOT / "2_G_alpha_neutralizer.py",
        Path(__file__).resolve().parent / "2_G_alpha_neutralizer.py",
    ]
    for p in candidates:
        if p and Path(p).exists():
            spec = importlib.util.spec_from_file_location("alpha_neutralizer_2g", str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
            print(f"[INFO] 2_G を import: {p}")
            return mod
    raise FileNotFoundError(
        "2_G_alpha_neutralizer.py が見つかりません。 --two-g で明示してください。"
    )


def s2_timestamp_axis(engine: str, tf: str) -> Optional[pl.LazyFrame]:
    """S2 e1a の TF 別 timestamp 軸 (= 学習側 feature 行 = label=left)。

    verify_dump_deque_order.load_learning_y と同一のパス規約。
    """
    universe = ENGINE_TO_UNIVERSE[engine]
    path = (
        Path(config.S2_FEATURES_VALIDATED)
        / f"feature_value_a_vast_universe{universe}"
        / f"features_{engine}_{tf}.parquet"
    )
    if not path.exists():
        print(f"[WARN] S2 不在 ({engine}/{tf}): {path}")
        return None
    return (
        pl.scan_parquet(str(path))
        .select(["timestamp"])
        # S1 は ns、 S2 は us。 build_proxy_lazyframe が us に揃えるので us に統一
        .with_columns(pl.col("timestamp").cast(pl.Datetime("us")))
        .sort("timestamp")
    )


def emit_for_tf(
    tf: str, proxy_lf: pl.LazyFrame, proxy_col: str, out_dir: Path
) -> Optional[int]:
    """1 TF 分: S2 start-time 軸に 2_G と同一の join_asof で proxy を貼って出力。"""
    axis = s2_timestamp_axis(REP_ENGINE, tf)
    if axis is None:
        return None

    # 2_G L540-545 と同一: join_asof(on="timestamp", strategy="backward")
    #   polars backward は exact match を含む (<=)。 2_G の挙動を忠実に再現する。
    #   (verify 旧実装の merge_asof(allow_exact_matches=False) = strict < とは異なる)
    joined = (
        axis.join_asof(proxy_lf, on="timestamp", strategy="backward")
        # 2_G は純化計算で proxy を rolling に投入。 join 直後 null になりうる先頭
        # (M5 最初の 2 本) は fill_null(0.0)。 検証窓 (5/26、 系列深部) では proxy は
        # 既に充填済のため no-op。
        .with_columns(pl.col(proxy_col).fill_null(0.0).alias("proxy"))
        .select(["timestamp", "proxy"])
        .sort("timestamp")
    )
    df = joined.collect()
    out_path = out_dir / f"train_proxy_{tf}.parquet"
    df.write_parquet(out_path, compression="zstd")
    print(f"[OK] {tf:6s}: {len(df):>9,} rows -> {out_path.name}")
    return len(df)


def main() -> None:
    parser = argparse.ArgumentParser(description="emit train-side ground-truth proxy")
    parser.add_argument(
        "--out-dir", type=Path,
        default=config.DATA_DIR / "diagnostics" / "train_proxy",
    )
    parser.add_argument(
        "--two-g", type=Path, default=None,
        help="2_G_alpha_neutralizer.py の明示パス",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("  emit_train_proxy.py — 学習側 ground-truth proxy 生成")
    print("=" * 72)
    print(f"  S1_PROCESSED:          {config.S1_PROCESSED}")
    print(f"  S2_FEATURES_VALIDATED: {config.S2_FEATURES_VALIDATED}")
    print(f"  out-dir:               {args.out_dir}")

    g2 = load_two_g(args.two_g)

    proxy_tf = config.NEUTRALIZATION_CONFIG["HF"]["proxy_tf"]  # "M5"
    proxy_col = f"proxy_{proxy_tf}"
    print(f"  proxy_tf:              {proxy_tf}  (column: {proxy_col})")

    # 修正後 2_G の build_proxy_lazyframe を逐語流用 (= 式ドリフトゼロ)
    proxy_lf = g2.build_proxy_lazyframe(proxy_tf)
    if proxy_lf is None:
        print(f"[FATAL] proxy LazyFrame を構築できません ({proxy_tf})。 S1_PROCESSED を確認。")
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for tf in config.HF_TIMEFRAMES:
        n = emit_for_tf(tf, proxy_lf, proxy_col, args.out_dir)
        if n:
            total += n

    print("-" * 72)
    print(f"  完了: {total:,} rows across {len(config.HF_TIMEFRAMES)} TFs")
    print(f"  -> {args.out_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
