# reproducibility_verifier.py
# =====================================================================
# ツール③: 再現性検証ツール
#
# 【目的】
#   過去のS6データをrealtime_feature_engineに流し込んで
#   M1→Logit→M2の推論を再現し、OOFファイルのM2予測値と比較する。
#   「本番が学習時と同じ予測をしているか」を数値で確認する。
#
# 【使い方】
#   python reproducibility_verifier.py [--n_samples 50] [--direction long]
#                                       [--min_proba 0.70] [--output_dir ./verify_out]
#
# 【出力】
#   verify_out/
#     comparison_long.csv   # OOF予測値 vs 再現予測値の比較
#     comparison_short.csv
#     summary_report.txt    # 一致率・MAE・相関係数・乖離大トップN
# =====================================================================

import sys
import logging
import argparse
import tempfile
import warnings
import numpy as np
import pandas as pd
import polars as pl
import joblib
from pathlib import Path
from datetime import timezone
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Verifier")

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import blueprint as config
from blueprint import (
    S6_WEIGHTED_DATASET,
    S7_M1_OOF_PREDICTIONS_LONG, S7_M1_OOF_PREDICTIONS_SHORT,
    S7_M2_OOF_PREDICTIONS_LONG, S7_M2_OOF_PREDICTIONS_SHORT,
    S7_M1_MODEL_LONG_PKL, S7_M1_MODEL_SHORT_PKL,
    S7_M2_MODEL_LONG_PKL, S7_M2_MODEL_SHORT_PKL,
    S3_SELECTED_FEATURES_ORTHOGONAL_DIR,
    S1_PROCESSED,
    S7_MODELS,
)
from execution.realtime_feature_engine import RealtimeFeatureEngine


def load_feature_list(filepath: Path) -> List[str]:
    with open(filepath) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("is_trigger")]


def build_feature_list_union() -> List[str]:
    """main.pyと同じ: 4ファイルのunionを返す"""
    union, seen = [], set()
    for fp in [
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt",
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt",
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt",
        S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt",
    ]:
        for line in open(fp):
            name = line.strip()
            if name and name not in seen:
                seen.add(name)
                union.append(name)
    return union


def build_market_proxy_from_s1(timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """
    S1_PROCESSEDのM5データからmarket_proxy（1バー前比リターン）を構築する。
    main.pyのg_market_proxyと同じ計算式。
    """
    try:
        m5_path = S1_PROCESSED / "timeframe=M5"
        if not m5_path.exists():
            logger.warning("M5データが見つかりません。空のproxyを使用します。")
            return pd.DataFrame(
                columns=["market_proxy"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
            )

        # 必要な期間を読み込む
        ts_min = timestamps.min() - pd.Timedelta(days=10)
        ts_max = timestamps.max() + pd.Timedelta(days=1)

        lf = pl.scan_parquet(str(m5_path / "**/*.parquet"))
        df_m5 = (
            lf.filter(
                (pl.col("timestamp") >= ts_min.to_pydatetime()) &
                (pl.col("timestamp") <= ts_max.to_pydatetime())
            )
            .select(["timestamp", "close"])
            .sort("timestamp")
            .collect()
        )

        if df_m5.is_empty():
            logger.warning("M5データが空です。")
            return pd.DataFrame(
                columns=["market_proxy"],
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
            )

        # M5の1バー前比リターン（main.pyと同じ）
        closes = df_m5["close"].to_numpy()
        returns = np.concatenate([[np.nan], np.diff(closes) / (closes[:-1] + 1e-10)])
        ts_list = df_m5["timestamp"].to_list()

        proxy_df = pd.DataFrame(
            {"market_proxy": returns},
            index=pd.DatetimeIndex(ts_list, name="timestamp")
        )
        if proxy_df.index.tzinfo is None:
            proxy_df.index = proxy_df.index.tz_localize("UTC")
        else:
            proxy_df.index = proxy_df.index.tz_convert("UTC")

        proxy_df = proxy_df.dropna()
        logger.info(f"Market proxy構築完了: {len(proxy_df)}行")
        return proxy_df

    except Exception as e:
        logger.error(f"Market proxy構築失敗: {e}", exc_info=True)
        return pd.DataFrame(
            columns=["market_proxy"],
            index=pd.DatetimeIndex([], tz="UTC", name="timestamp")
        )


def load_ohlcv_for_timestamps(
    timestamps: List[pd.Timestamp],
    lookback_bars_m05: int = 8000,
) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    各検証タイムスタンプに対して、その時刻までのM0.5 OHLCVデータを返す。
    """
    m05_path = S1_PROCESSED / "timeframe=M0.5"
    if not m05_path.exists():
        raise FileNotFoundError(f"M0.5データが見つかりません: {m05_path}")

    ts_min = min(timestamps) - pd.Timedelta(minutes=0.5 * lookback_bars_m05)
    ts_max = max(timestamps) + pd.Timedelta(minutes=3)

    logger.info(f"M0.5データ読み込み中: {ts_min} 〜 {ts_max}")
    lf = pl.scan_parquet(str(m05_path / "**/*.parquet"))
    df_all = (
        lf.filter(
            (pl.col("timestamp") >= ts_min.to_pydatetime()) &
            (pl.col("timestamp") <= ts_max.to_pydatetime())
        )
        .select(["timestamp", "open", "high", "low", "close", "volume"])
        .sort("timestamp")
        .collect()
    )

    logger.info(f"M0.5データ読み込み完了: {len(df_all)}行")

    result = {}
    ts_arr = df_all["timestamp"].to_list()
    for target_ts in timestamps:
        # target_ts以前のデータを取得
        idx = next(
            (i for i, t in enumerate(reversed(ts_arr)) if t <= target_ts.to_pydatetime()),
            None
        )
        if idx is None:
            continue
        end_idx = len(ts_arr) - idx
        start_idx = max(0, end_idx - lookback_bars_m05)
        chunk = df_all.slice(start_idx, end_idx - start_idx)
        if not chunk.is_empty():
            result[target_ts] = chunk

    return result


def run_inference_on_bars(
    engine: RealtimeFeatureEngine,
    bars_df: pl.DataFrame,
    market_proxy: pd.DataFrame,
    models: Dict,
    feature_lists: Dict,
) -> Tuple[float, float, float, float, Dict]:
    """
    バーデータをエンジンに流し込んでM1→M2推論を実行。
    最後のM3確定時のシグナルを返す。
    """
    last_signal = None
    p_m1l = p_m1s = p_m2l = p_m2s = 0.0

    for row in bars_df.iter_rows(named=True):
        bar = {
            "timestamp": pd.Timestamp(row["timestamp"]).tz_localize("UTC")
                if pd.Timestamp(row["timestamp"]).tzinfo is None
                else pd.Timestamp(row["timestamp"]).tz_convert("UTC"),
            "open":   float(row["open"]),
            "high":   float(row["high"]),
            "low":    float(row["low"]),
            "close":  float(row["close"]),
            "volume": float(row["volume"]),
        }
        signals = engine.process_new_m05_bar(bar, market_proxy)
        if signals:
            last_signal = signals[-1]

    if last_signal is None:
        return 0.0, 0.0, 0.0, 0.0, {}

    feature_dict = last_signal.feature_dict or {}

    # M1推論
    X_long_m1 = np.array([[feature_dict.get(f, 0.0) for f in feature_lists["long_m1"]]])
    p_m1l = float(models["long_m1"].predict(X_long_m1)[0])

    X_short_m1 = np.array([[feature_dict.get(f, 0.0) for f in feature_lists["short_m1"]]])
    p_m1s = float(models["short_m1"].predict(X_short_m1)[0])

    # M2推論（M1 >= 0.50のみ）
    if p_m1l >= 0.50:
        fd_l = feature_dict.copy()
        p_l_clip = np.clip(p_m1l, 1e-7, 1 - 1e-7)
        fd_l["m1_pred_proba"] = float(np.clip(np.log(p_l_clip / (1 - p_l_clip)), -10.0, 10.0))
        X_long_m2 = np.array([[fd_l.get(f, 0.0) for f in feature_lists["long_m2"]]])
        p_m2l = float(models["long_m2"].predict(X_long_m2)[0])
    else:
        p_m2l = 0.0

    if p_m1s >= 0.50:
        fd_s = feature_dict.copy()
        p_s_clip = np.clip(p_m1s, 1e-7, 1 - 1e-7)
        fd_s["m1_pred_proba"] = float(np.clip(np.log(p_s_clip / (1 - p_s_clip)), -10.0, 10.0))
        X_short_m2 = np.array([[fd_s.get(f, 0.0) for f in feature_lists["short_m2"]]])
        p_m2s = float(models["short_m2"].predict(X_short_m2)[0])
    else:
        p_m2s = 0.0

    return p_m1l, p_m1s, p_m2l, p_m2s, feature_dict


def verify(
    direction: str = "long",
    n_samples: int = 50,
    min_proba: float = 0.70,
    output_dir: Path = Path("./verify_out"),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"=== 再現性検証開始: direction={direction}, n={n_samples}, min_proba={min_proba} ===")

    # --- モデル・特徴量リスト読み込み ---
    logger.info("モデル読み込み中...")
    models = {
        "long_m1":  joblib.load(S7_M1_MODEL_LONG_PKL),
        "long_m2":  joblib.load(S7_M2_MODEL_LONG_PKL),
        "short_m1": joblib.load(S7_M1_MODEL_SHORT_PKL),
        "short_m2": joblib.load(S7_M2_MODEL_SHORT_PKL),
    }

    feature_lists = {
        "long_m1":  load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_long_features.txt"),
        "long_m2":  load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_long_features.txt"),
        "short_m1": load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m1_short_features.txt"),
        "short_m2": load_feature_list(S3_SELECTED_FEATURES_ORTHOGONAL_DIR / "m2_short_features.txt"),
    }
    # m1_pred_probaを末尾に強制（main.pyと同じ）
    for key in ["long_m2", "short_m2"]:
        fl = feature_lists[key]
        if "m1_pred_proba" in fl:
            fl.remove("m1_pred_proba")
        fl.append("m1_pred_proba")

    # --- OOFファイルからサンプル選択 ---
    oof_long_path  = S7_M2_OOF_PREDICTIONS_LONG
    oof_short_path = S7_M2_OOF_PREDICTIONS_SHORT

    oof_path = oof_long_path if direction == "long" else oof_short_path
    logger.info(f"OOFファイル読み込み: {oof_path}")
    oof_df = pl.read_parquet(oof_path)
    logger.info(f"OOF総件数: {len(oof_df)}")

    # min_proba以上のサンプルからランダムにn件選択
    oof_high = oof_df.filter(pl.col("prediction") >= min_proba)
    logger.info(f"prediction >= {min_proba}: {len(oof_high)}件")

    if len(oof_high) == 0:
        logger.error("サンプルが0件です。min_probaを下げてください。")
        return

    n_actual = min(n_samples, len(oof_high))
    oof_sample = oof_high.sample(n_actual, seed=42, shuffle=True)
    logger.info(f"検証サンプル: {n_actual}件")

    # タイムスタンプリストを作成
    sample_timestamps = [
        pd.Timestamp(ts).tz_localize("UTC") if pd.Timestamp(ts).tzinfo is None
        else pd.Timestamp(ts).tz_convert("UTC")
        for ts in oof_sample["timestamp"].to_list()
    ]
    oof_predictions = oof_sample["prediction"].to_list()

    # --- Market proxy構築 ---
    logger.info("Market proxy構築中...")
    ts_index = pd.DatetimeIndex(sample_timestamps)
    market_proxy = build_market_proxy_from_s1(ts_index)

    # --- M0.5データ読み込み ---
    ohlcv_map = load_ohlcv_for_timestamps(sample_timestamps, lookback_bars_m05=8000)
    logger.info(f"OHLCVデータ取得: {len(ohlcv_map)}/{len(sample_timestamps)}件")

    # --- feature_listの一時ファイル作成 ---
    union = build_feature_list_union()
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
    tmp.write("\n".join(union))
    tmp.close()

    # --- 各サンプルで検証 ---
    results = []
    for i, (ts, oof_pred) in enumerate(zip(sample_timestamps, oof_predictions)):
        if ts not in ohlcv_map:
            logger.warning(f"[{i+1}/{n_actual}] OHLCVデータなし: {ts}")
            continue

        logger.info(f"[{i+1}/{n_actual}] 検証中: {ts} (OOF={oof_pred:.4f})")

        # エンジンを新規作成（各サンプル独立）
        try:
            engine = RealtimeFeatureEngine(feature_list_path=tmp.name)
        except Exception as e:
            logger.error(f"エンジン初期化失敗: {e}")
            continue

        bars_df = ohlcv_map[ts]
        try:
            p_m1l, p_m1s, p_m2l, p_m2s, feat_dict = run_inference_on_bars(
                engine, bars_df, market_proxy, models, feature_lists
            )
        except Exception as e:
            logger.error(f"推論失敗 ({ts}): {e}", exc_info=True)
            continue

        reprod_pred = p_m2l if direction == "long" else p_m2s
        diff = reprod_pred - oof_pred
        rel_diff = abs(diff) / (abs(oof_pred) + 1e-10)

        results.append({
            "timestamp":     ts.isoformat(),
            "oof_pred":      round(oof_pred, 6),
            "reprod_pred":   round(reprod_pred, 6),
            "diff":          round(diff, 6),
            "abs_diff":      round(abs(diff), 6),
            "rel_diff_%":    round(rel_diff * 100, 2),
            "m1_pred":       round(p_m1l if direction == "long" else p_m1s, 6),
            "n_features":    len(feat_dict),
        })

    if not results:
        logger.error("有効な結果が0件でした。")
        return

    # --- CSV出力 ---
    results_df = pd.DataFrame(results).sort_values("abs_diff", ascending=False)
    csv_path = output_dir / f"comparison_{direction}.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info(f"比較CSV保存: {csv_path}")

    # --- サマリーレポート ---
    abs_diffs = results_df["abs_diff"].values
    oof_vals  = results_df["oof_pred"].values
    rep_vals  = results_df["reprod_pred"].values

    mae = np.mean(abs_diffs)
    rmse = np.sqrt(np.mean((oof_vals - rep_vals) ** 2))
    corr = np.corrcoef(oof_vals, rep_vals)[0, 1] if len(results) > 1 else 0.0
    within_001 = np.mean(abs_diffs < 0.01) * 100
    within_005 = np.mean(abs_diffs < 0.05) * 100
    within_010 = np.mean(abs_diffs < 0.10) * 100

    report_lines = [
        "=" * 60,
        f"  再現性検証レポート ({direction.upper()})",
        "=" * 60,
        f"  検証サンプル数    : {len(results)}件",
        f"  min_proba閾値    : {min_proba}",
        "",
        "--- 一致精度 ---",
        f"  MAE              : {mae:.6f}",
        f"  RMSE             : {rmse:.6f}",
        f"  相関係数         : {corr:.6f}",
        f"  差分 < 0.01      : {within_001:.1f}%",
        f"  差分 < 0.05      : {within_005:.1f}%",
        f"  差分 < 0.10      : {within_010:.1f}%",
        "",
        "--- OOF予測値統計 ---",
        f"  mean={oof_vals.mean():.4f}  std={oof_vals.std():.4f}  min={oof_vals.min():.4f}  max={oof_vals.max():.4f}",
        "",
        "--- 再現予測値統計 ---",
        f"  mean={rep_vals.mean():.4f}  std={rep_vals.std():.4f}  min={rep_vals.min():.4f}  max={rep_vals.max():.4f}",
        "",
        "--- 乖離大 TOP10 ---",
    ]
    for _, row in results_df.head(10).iterrows():
        report_lines.append(
            f"  {row['timestamp'][:19]}  OOF={row['oof_pred']:.4f}  再現={row['reprod_pred']:.4f}"
            f"  diff={row['diff']:+.4f}  ({row['rel_diff_%']:.1f}%)"
        )
    report_lines.append("=" * 60)

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    report_path = output_dir / f"summary_report_{direction}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text + "\n")
    logger.info(f"サマリーレポート保存: {report_path}")


def main():
    parser = argparse.ArgumentParser(description="再現性検証ツール")
    parser.add_argument("--direction",  default="long",     choices=["long", "short"])
    parser.add_argument("--n_samples",  default=50,  type=int)
    parser.add_argument("--min_proba",  default=0.70, type=float)
    parser.add_argument("--output_dir", default=None, type=Path)
    args = parser.parse_args()

    verify(
        direction=args.direction,
        n_samples=args.n_samples,
        min_proba=args.min_proba,
        output_dir=args.output_dir or (config.S7_MODELS / "reproducibility_results"),
    )


if __name__ == "__main__":
    main()
