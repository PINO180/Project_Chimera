# module_unit_test.py
# =====================================================================
# モジュール単体テスト：学習側 vs 本番側 直接比較
#
# 【目的】
#   同一OHLCVウィンドウを
#     学習側: engine_1_X (Polars LazyFrame) → 全行の最終値[-1]
#     本番側: realtime_feature_engine_1X (Numpy) → calculate_features()の返値
#   に食わせ、QAなし・純化なしのraw値を1対1で直接比較する。
#
#   これにより「計算ロジックの差」だけを純粋に検出できる。
#
# 【使い方】
#   python module_unit_test.py              # 全モジュール(1A〜1F)
#   python module_unit_test.py --module 1C  # 特定モジュールのみ
#   python module_unit_test.py --timeframe M3  # 特定時間足のみ
#   python module_unit_test.py --seed 42 --bars 300  # 乱数シードとバー数指定
#
# 【出力】
#   コンソールにサマリーを表示。
#   乖離が大きい特徴量を特定してログに出力する。
# =====================================================================

import sys
import re
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("UnitTest")

# パス設定
# スクリプト: /workspace/diagnostics/module_unit_test.py
# blueprint.py: /workspace/
# 本番側モジュール: /workspace/execution/
# 学習側モジュール: /workspace/features/
here      = Path(__file__).resolve().parent   # /workspace/diagnostics
workspace = here.parent                        # /workspace
execution = workspace / "execution"            # /workspace/execution
features  = workspace / "features"             # /workspace/features

for p in [str(workspace), str(execution), str(features)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import blueprint as config

core_dir = str(config.CORE_DIR)
if core_dir not in sys.path:
    sys.path.insert(0, core_dir)

from core_indicators import calculate_atr_wilder

# 本番側モジュール（/workspace/execution/）
from realtime_feature_engine_1A_statistics   import FeatureModule1A
from realtime_feature_engine_1B_timeseries   import FeatureModule1B
from realtime_feature_engine_1C_technical    import FeatureModule1C
from realtime_feature_engine_1D_volume       import FeatureModule1D
from realtime_feature_engine_1E_signal       import FeatureModule1E
from realtime_feature_engine_1F_experimental import FeatureModule1F

# 学習側モジュール（/workspace/features/）
from engine_1_A_a_vast_universe_of_features import CalculationEngine as CalcA, ProcessingConfig as CfgA
from engine_1_B_a_vast_universe_of_features import CalculationEngine as CalcB, ProcessingConfig as CfgB
from engine_1_C_a_vast_universe_of_features import CalculationEngine as CalcC, ProcessingConfig as CfgC
from engine_1_D_a_vast_universe_of_features import CalculationEngine as CalcD, ProcessingConfig as CfgD
from engine_1_E_a_vast_universe_of_features import CalculationEngine as CalcE, ProcessingConfig as CfgE
from engine_1_F_a_vast_universe_of_features import CalculationEngine as CalcF, ProcessingConfig as CfgF

# 時間足ごとの bars_per_day
TIMEFRAME_BARS_PER_DAY = {
    "M0.5": 2880, "M1": 1440, "M3": 480,
    "M5": 288,   "M8": 180,  "M15": 96,
}

# 乖離判定閾値
THRESH_OK   = 1e-4   # 0.01%以下 → 一致
THRESH_NEAR = 0.01   # 1%以下   → 近似
THRESH_WARN = 0.05   # 5%以下   → 要注意
                     # 5%超     → 乖離


# ===========================================================================
# S1 実データ読み込み
# ===========================================================================

DEFAULT_LOOKBACK_BARS = 2116  # 本番バッファサイズに合わせる (SAFE_MIN_LOOKBACK=2016+100)

def load_ohlcv_for_timestamp(
    tf_name: str,
    target_ts: pd.Timestamp,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> Optional[Dict[str, np.ndarray]]:
    """S1_MULTITIMEFRAME から指定タイムスタンプの直前 lookback_bars 本の OHLCV を取得"""
    # M0.5: S1にtimeframe=M0.5があれば直接読む、なければM1からリサンプリング
    if tf_name == "M0.5":
        m05_path = config.S1_MULTITIMEFRAME / "timeframe=M0.5"
        if m05_path.exists():
            source_tf = "M0.5"
        else:
            source_tf = "M1"
            logger.info("M0.5のS1データなし → M1からリサンプリングします")
    else:
        source_tf = tf_name

    tf_path = config.S1_MULTITIMEFRAME / f"timeframe={source_tf}"
    if not tf_path.exists():
        logger.warning(f"S1パスが存在しません: {tf_path}")
        return None
    try:
        lf = pl.scan_parquet(str(tf_path / "*.parquet"))
        target_utc = (
            target_ts.tz_convert("UTC") if target_ts.tzinfo else target_ts.tz_localize("UTC")
        )
        target_naive_ms = pl.lit(target_utc.replace(tzinfo=None)).cast(pl.Datetime("ms"))
        df = (
            lf.filter(pl.col("timestamp") <= target_naive_ms)
            .sort("timestamp")
            .collect()
            .tail(lookback_bars)
        )
        if len(df) < 10:
            logger.warning(f"{tf_name}: データ不足 ({len(df)}本)")
            return None
        if tf_name == "M0.5" and source_tf == "M1":
            # M1からリサンプリング
            pdf = df.to_pandas().set_index("timestamp")
            pdf.index = pd.to_datetime(pdf.index, utc=True)
            pdf_final = (
                pdf[["open", "high", "low", "close", "volume"]]
                .resample("30s", label="right", closed="right")
                .agg({"open": "first", "high": "max", "low": "min",
                      "close": "last", "volume": "sum"})
                .dropna()
                .tail(lookback_bars)
                .reset_index()
            )
        else:
            pdf_final = df.to_pandas()
        return {
            "open":        pdf_final["open"].values.astype(np.float64),
            "high":        pdf_final["high"].values.astype(np.float64),
            "low":         pdf_final["low"].values.astype(np.float64),
            "close":       pdf_final["close"].values.astype(np.float64),
            "volume":      pdf_final["volume"].values.astype(np.float64),
            "_timestamps": pdf_final.get("timestamp", pd.Series(dtype="object")).values,
        }
    except Exception as e:
        logger.error(f"S1読み込みエラー ({tf_name}): {e}")
        return None


def ohlcv_to_polars(ohlcv: Dict[str, np.ndarray]) -> pl.DataFrame:
    """numpy OHLCV dict を Polars DataFrame に変換する"""
    n = len(ohlcv["close"])
    ts_arr = ohlcv.get("_timestamps")
    if ts_arr is not None and len(ts_arr) == n:
        ts = pd.to_datetime(ts_arr, utc=True)
    else:
        ts = pd.date_range("2024-01-01", periods=n, freq="1min", tz="UTC")
    return pl.DataFrame({
        "timestamp": ts.astype("int64") // 10**6,
        "open":   ohlcv["open"],
        "high":   ohlcv["high"],
        "low":    ohlcv["low"],
        "close":  ohlcv["close"],
        "volume": ohlcv["volume"].astype(np.int64),
    }).with_columns(pl.col("timestamp").cast(pl.Datetime("ms")))


# ===========================================================================
# 学習側: Polars で全行計算し最終行[-1]を取り出す
# ===========================================================================

def _extract_last_row(result_lf: pl.LazyFrame, prefix: str) -> Dict[str, float]:
    """LazyFrameの最終行からprefixで始まる特徴量を取り出す"""
    df = result_lf.collect()
    if df.is_empty():
        return {}
    last_row = df.row(-1, named=True)
    out = {}
    for k, v in last_row.items():
        if not k.startswith(prefix):
            continue
        try:
            f = float(v)
            out[k] = f if np.isfinite(f) else float("nan")
        except (TypeError, ValueError):
            out[k] = float("nan")
    return out


def run_learning_side(
    module_tag: str,
    ohlcv: Dict[str, np.ndarray],
    tf_name: str,
) -> Dict[str, float]:
    """学習側エンジンで特徴量計算し最終バーの値を返す（QAなし・raw値）"""
    lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)

    cfg_map = {
        "1A": (CalcA, CfgA),
        "1B": (CalcB, CfgB),
        "1C": (CalcC, CfgC),
        "1D": (CalcD, CfgD),
        "1E": (CalcE, CfgE),
        "1F": (CalcF, CfgF),
    }
    CalcCls, CfgCls = cfg_map[module_tag]

    # ProcessingConfig のフィールド名がエンジンごとに異なる
    # 1A/1B/1D/1E/1F: input_path  1C: input_base_path
    cfg_kwargs = {"timeframes": [tf_name], "timeframe_bars_per_day": {tf_name: lb}}
    if module_tag == "1C":
        cfg_kwargs["input_base_path"] = str(config.S1_MULTITIMEFRAME)
    else:
        cfg_kwargs["input_path"] = str(config.S1_MULTITIMEFRAME)
    cfg = CfgCls(**cfg_kwargs)
    engine = CalcCls(cfg)
    lf = ohlcv_to_polars(ohlcv).lazy()
    prefix = engine.prefix  # e.g. "e1a_"

    # --- エンジンごとに正しいAPIを呼ぶ ---

    if module_tag == "1A":
        # 1A: _get_all_feature_expressions でQAをバイパスしraw値を取得
        # calculate_one_group はQA(EWM 5σ)を適用するため本番rawと不一致になる可能性がある。
        # _get_all_feature_expressions は __temp_atr_13 カラムを参照するため、
        # 事前に calculate_atr_wilder で注入してから呼ぶ。
        result_lf = lf.with_columns([
            (pl.struct(["high", "low", "close"]).map_batches(
                lambda s: pl.Series(calculate_atr_wilder(
                    s.struct.field("high").to_numpy(),
                    s.struct.field("low").to_numpy(),
                    s.struct.field("close").to_numpy(),
                    13,
                )),
                return_dtype=pl.Float64,
            ) + 1e-10).alias("__temp_atr_13")
        ])
        try:
            all_exprs = engine._get_all_feature_expressions(timeframe=tf_name)
            exprs_list = list(all_exprs.values())
            CHUNK = 200
            for i in range(0, len(exprs_list), CHUNK):
                try:
                    result_lf = result_lf.with_columns(exprs_list[i:i+CHUNK])
                except Exception as e:
                    logger.warning(f"1A expr chunk {i} エラー: {e}")
        except Exception as e:
            logger.warning(f"1A _get_all_feature_expressions エラー: {e}")
        return _extract_last_row(result_lf, prefix)

    elif module_tag == "1B":
        # 1B: _get_all_feature_expressions でQAをバイパスしraw値を取得
        # __temp_atr_safe を事前に inject_temp_atr で注入してから呼ぶ
        result_lf = engine.inject_temp_atr(lf)
        try:
            all_exprs = engine._get_all_feature_expressions(timeframe=tf_name)
            exprs_list = list(all_exprs.values())
            CHUNK = 200
            for i in range(0, len(exprs_list), CHUNK):
                try:
                    result_lf = result_lf.with_columns(exprs_list[i:i+CHUNK])
                except Exception as e:
                    logger.warning(f"1B expr chunk {i} エラー: {e}")
        except Exception as e:
            logger.warning(f"1B _get_all_feature_expressions エラー: {e}")
        return _extract_last_row(result_lf, prefix)
    elif module_tag == "1C":
        # 1C: _get_all_feature_expressions でQAをバイパスしraw値を取得
        # calculate_all_features はQA(EWM 5σ)を適用するため本番rawと不一致になる可能性がある
        result_lf = lf
        try:
            result_lf = engine._get_all_feature_expressions(lf, timeframe=tf_name)
        except Exception as e:
            logger.warning(f"1C _get_all_feature_expressions エラー: {e}")
        return _extract_last_row(result_lf, prefix)

    else:
        # 1D/1E/1F: _get_all_feature_expressions を直接使いQAをバイパス
        # calculate_one_group はQA(EWM 5σクリップ)を適用するため、
        # 本番側のraw値（qa_state=None）と比較するunit testでは不一致になる。
        # raw値同士を比較するためQA前の式を直接 with_columns に渡す。
        # 1Dは _get_all_feature_expressions(timeframe=...) あり
        # 1E/1F は引数なし
        result_lf = lf
        try:
            if module_tag == "1D":
                all_exprs = engine._get_all_feature_expressions(timeframe=tf_name)
            else:
                all_exprs = engine._get_all_feature_expressions()
            exprs_list = list(all_exprs.values())
            CHUNK = 200
            for i in range(0, len(exprs_list), CHUNK):
                try:
                    result_lf = result_lf.with_columns(exprs_list[i:i+CHUNK])
                except Exception as e:
                    logger.warning(f"{module_tag} expr chunk {i} エラー: {e}")
        except Exception as e:
            logger.warning(f"{module_tag} _get_all_feature_expressions エラー: {e}")
        return _extract_last_row(result_lf, prefix)


# ===========================================================================
# 本番側: Numpy で calculate_features を呼ぶ
# ===========================================================================

def run_realtime_side(
    module_tag: str,
    ohlcv: Dict[str, np.ndarray],
    tf_name: str,
) -> Dict[str, float]:
    """本番側モジュールで特徴量計算しraw値を返す（QAなし）"""
    lb = TIMEFRAME_BARS_PER_DAY.get(tf_name, 1440)
    mod_map = {
        "1A": FeatureModule1A,
        "1B": FeatureModule1B,
        "1C": FeatureModule1C,
        "1D": FeatureModule1D,
        "1E": FeatureModule1E,
        "1F": FeatureModule1F,
    }
    mod = mod_map[module_tag]
    try:
        return mod.calculate_features(ohlcv, lookback_bars=lb, qa_state=None)
    except Exception as e:
        logger.warning(f"本番側モジュール {module_tag} エラー: {e}")
        return {}


# ===========================================================================
# 1対1比較
# ===========================================================================

def compare_module(
    module_tag: str,
    tf_name: str,
    target_ts: pd.Timestamp,
    lookback_bars: int = DEFAULT_LOOKBACK_BARS,
) -> List[Dict]:
    ohlcv = load_ohlcv_for_timestamp(tf_name, target_ts, lookback_bars)
    if ohlcv is None:
        print(f"  [{module_tag}/{tf_name}] S1データなし → スキップ")
        return []

    learn = run_learning_side(module_tag, ohlcv, tf_name)
    rt    = run_realtime_side(module_tag, ohlcv, tf_name)

    if not learn:
        print(f"  [{module_tag}/{tf_name}] 学習側: 結果なし")
        return []
    if not rt:
        print(f"  [{module_tag}/{tf_name}] 本番側: 結果なし")
        return []

    records = []
    # e1e_sample_weight は学習側に存在しないため比較対象外
    all_keys = (set(learn.keys()) | set(rt.keys())) - {"e1e_sample_weight"}

    for key in sorted(all_keys):
        lv = learn.get(key, float("nan"))
        rv = rt.get(key, float("nan"))

        only_in = None
        if key not in learn:
            only_in = "rt_only"
        elif key not in rt:
            only_in = "learn_only"

        lv_f = lv if np.isfinite(lv) else 0.0
        rv_f = rv if np.isfinite(rv) else 0.0
        abs_diff = abs(lv_f - rv_f)
        scale    = max(abs(lv_f), abs(rv_f), 1e-8)
        rel_diff = abs_diff / scale

        records.append({
            "module":    module_tag,
            "timeframe": tf_name,
            "feature":   key,
            "learn_val": round(lv, 8),
            "rt_val":    round(rv, 8),
            "abs_diff":  round(abs_diff, 8),
            "rel_diff":  round(rel_diff, 6),
            "only_in":   only_in or "",
            "note":      _classify(rel_diff, only_in),
        })

    return records


def _classify(rel_diff: float, only_in: Optional[str]) -> str:
    if only_in == "learn_only":
        return "📘 学習側のみ"
    if only_in == "rt_only":
        return "📗 本番側のみ"
    if rel_diff < THRESH_OK:
        return "✅ 一致"
    elif rel_diff < THRESH_NEAR:
        return "🟡 近似(<1%)"
    elif rel_diff < THRESH_WARN:
        return "🟠 要注意(<5%)"
    else:
        return "🔴 乖離(5%+)"


# ===========================================================================
# メイン
# ===========================================================================

def run(
    modules: List[str],
    timeframes: List[str],
    timestamps: List[str],
    lookback_bars: int,
    top_n: int,
    show_ok: bool,
) -> None:
    all_records = []

    ts_list = [
        pd.Timestamp(ts).tz_localize("UTC") if "+" not in ts else pd.Timestamp(ts)
        for ts in timestamps
    ]

    for ts in ts_list:
        print(f"\n== タイムスタンプ: {ts} ==")
        for mod in modules:
            for tf in timeframes:
                print(f"  テスト中: {mod} / {tf}")
                try:
                    recs = compare_module(mod, tf, ts, lookback_bars)
                    all_records.extend(recs)
                    n_ok  = sum(1 for r in recs if r["note"] == "✅ 一致")
                    n_bad = sum(1 for r in recs if "乖離" in r["note"])
                    print(f"    → {len(recs)}件比較  一致={n_ok}  乖離={n_bad}")
                except Exception as e:
                    print(f"    → エラー: {e}")

    if not all_records:
        print("比較できたレコードが0件でした。")
        return

    df = pd.DataFrame(all_records)

    # 全体サマリー
    n_total     = len(df)
    note_counts = df["note"].value_counts()
    n_ok        = note_counts.get("✅ 一致", 0)
    n_near      = note_counts.get("🟡 近似(<1%)", 0)
    n_warn      = note_counts.get("🟠 要注意(<5%)", 0)
    n_bad       = note_counts.get("🔴 乖離(5%+)", 0)
    n_lonly     = note_counts.get("📘 学習側のみ", 0)
    n_ronly     = note_counts.get("📗 本番側のみ", 0)

    print("\n" + "=" * 70)
    print("  モジュール単体テスト結果（学習側 Polars vs 本番側 Numpy）")
    print(f"  タイムスタンプ: {timestamps}  lookback={lookback_bars}本")
    print("=" * 70)
    print(f"  比較件数: {n_total}")
    print(f"  ✅ 一致     (<0.01%): {n_ok:4d}件 ({n_ok/n_total*100:.1f}%)")
    print(f"  🟡 近似     (<1%)  : {n_near:4d}件 ({n_near/n_total*100:.1f}%)")
    print(f"  🟠 要注意   (<5%)  : {n_warn:4d}件 ({n_warn/n_total*100:.1f}%)")
    print(f"  🔴 乖離     (5%+)  : {n_bad:4d}件 ({n_bad/n_total*100:.1f}%)")
    print(f"  📘 学習側のみ      : {n_lonly:4d}件")
    print(f"  📗 本番側のみ      : {n_ronly:4d}件")

    # モジュール別
    grp = df[df["only_in"] == ""].groupby("module")["rel_diff"].agg(["mean", "max", "count"])
    print("\n--- モジュール別 平均相対乖離 ---")
    for mod, row in grp.sort_values("mean", ascending=False).iterrows():
        print(f"  {mod}: mean={row['mean']*100:6.3f}%  max={row['max']*100:8.3f}%  ({int(row['count'])}件)")

    # 時間足別
    grp_tf = df[df["only_in"] == ""].groupby("timeframe")["rel_diff"].agg(["mean", "max", "count"])
    print("\n--- 時間足別 平均相対乖離 ---")
    for tf, row in grp_tf.sort_values("mean", ascending=False).iterrows():
        print(f"  {tf:6s}: mean={row['mean']*100:6.3f}%  max={row['max']*100:8.3f}%  ({int(row['count'])}件)")

    # 乖離大 TOP N
    df_bad = df[df["only_in"] == ""].sort_values("rel_diff", ascending=False)
    print(f"\n--- 乖離大 TOP{top_n} ---")
    for _, row in df_bad.head(top_n).iterrows():
        print(
            f"  {row['note']} [{row['module']}/{row['timeframe']}] "
            f"{row['feature']:<50} "
            f"learn={row['learn_val']:>12.6f}  rt={row['rt_val']:>12.6f}  "
            f"rel={row['rel_diff']*100:>7.3f}%"
        )

    # 片側にしかない特徴量
    if n_lonly + n_ronly > 0:
        print(f"\n--- 片側のみ存在する特徴量 ---")
        for _, row in df[df["only_in"] != ""].iterrows():
            print(f"  {row['note']} [{row['module']}/{row['timeframe']}] {row['feature']}")

    print("=" * 70)

    # --- ファイル出力 ---
    output_dir = Path("/workspace/data/diagnostics/unit_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime as _dt
    now_str = _dt.now().strftime("%Y%m%d_%H%M%S")

    # CSV（全レコード）
    csv_path = output_dir / f"detail_{now_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n詳細CSV保存: {csv_path}")

    # TXT（サマリー）
    txt_path = output_dir / f"summary_{now_str}.txt"
    lines_for_file = [
        "=" * 70,
        "  モジュール単体テスト結果（学習側 Polars vs 本番側 Numpy）",
        f"  タイムスタンプ: {timestamps}  lookback={lookback_bars}本",
        "=" * 70,
        f"  比較件数: {n_total}",
        f"  ✅ 一致     (<0.01%): {n_ok:4d}件 ({n_ok/n_total*100:.1f}%)",
        f"  🟡 近似     (<1%)  : {n_near:4d}件 ({n_near/n_total*100:.1f}%)",
        f"  🟠 要注意   (<5%)  : {n_warn:4d}件 ({n_warn/n_total*100:.1f}%)",
        f"  🔴 乖離     (5%+)  : {n_bad:4d}件 ({n_bad/n_total*100:.1f}%)",
        f"  📘 学習側のみ      : {n_lonly:4d}件",
        f"  📗 本番側のみ      : {n_ronly:4d}件",
        "",
        "--- モジュール別 平均相対乖離 ---",
    ]
    for mod, row in grp.sort_values("mean", ascending=False).iterrows():
        lines_for_file.append(f"  {mod}: mean={row['mean']*100:6.3f}%  max={row['max']*100:8.3f}%  ({int(row['count'])}件)")
    lines_for_file += ["", "--- 時間足別 平均相対乖離 ---"]
    for tf, row in grp_tf.sort_values("mean", ascending=False).iterrows():
        lines_for_file.append(f"  {tf:6s}: mean={row['mean']*100:6.3f}%  max={row['max']*100:8.3f}%  ({int(row['count'])}件)")
    lines_for_file += ["", f"--- 乖離大 TOP{top_n} ---"]
    for _, row in df_bad.head(top_n).iterrows():
        lines_for_file.append(
            f"  {row['note']} [{row['module']}/{row['timeframe']}] "
            f"{row['feature']:<50} "
            f"learn={row['learn_val']:>12.6f}  rt={row['rt_val']:>12.6f}  "
            f"rel={row['rel_diff']*100:>7.3f}%"
        )
    if n_lonly + n_ronly > 0:
        lines_for_file += ["", "--- 片側のみ存在する特徴量 ---"]
        for _, row in df[df["only_in"] != ""].iterrows():
            lines_for_file.append(f"  {row['note']} [{row['module']}/{row['timeframe']}] {row['feature']}")
    lines_for_file.append("=" * 70)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines_for_file) + "\n")
    print(f"サマリーTXT保存: {txt_path}")


def main():
    ALL_MODULES = ["1A", "1B", "1C", "1D", "1E", "1F"]
    ALL_TFS     = ["M0.5", "M1", "M3", "M5", "M8", "M15"]
    # reproducibility_verifier の乖離大 TOP2 をデフォルトに使用
    DEFAULT_TIMESTAMPS = ["2024-05-13 01:18:00", "2022-06-01 10:54:00"]

    parser = argparse.ArgumentParser(description="モジュール単体テスト: 学習側 vs 本番側")
    parser.add_argument("--module",     nargs="+", default=ALL_MODULES,
                        choices=ALL_MODULES, help="テスト対象モジュール")
    parser.add_argument("--timeframe",  nargs="+", default=ALL_TFS,
                        help="テスト対象時間足")
    parser.add_argument("--timestamps", nargs="+", default=DEFAULT_TIMESTAMPS,
                        help="S1から切り出すタイムスタンプ（複数可）")
    parser.add_argument("--lookback",   default=DEFAULT_LOOKBACK_BARS, type=int,
                        help="タイムスタンプ直前の取得バー数")
    parser.add_argument("--top_n",      default=100, type=int)
    parser.add_argument("--show_ok",    action="store_true")
    args = parser.parse_args()

    run(
        modules=args.module,
        timeframes=args.timeframe,
        timestamps=args.timestamps,
        lookback_bars=args.lookback,
        top_n=args.top_n,
        show_ok=args.show_ok,
    )


if __name__ == "__main__":
    main()
