# /workspace/models/create_proxy_labels_polars_patch_regime.py
# [フェーズ3: 最終ラベリングスクリプト - V5 双方向ラベリング仕様]

import sys
from pathlib import Path
import warnings
import argparse
import shutil
from dataclasses import dataclass, field
import logging
from typing import List, Dict, Any, Optional, Tuple
import polars as pl
from tqdm import tqdm
import re
import gc
import datetime as dt
import numpy as np
import calendar

try:
    from numba import njit, prange
    from numba.core.errors import NumbaPerformanceWarning

    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
    NUMBA_AVAILABLE = True
except ImportError:
    logging.warning(
        "Numba not found. Labeling performance will be significantly degraded."
    )
    NUMBA_AVAILABLE = False

# --- Project Path Setup ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# --- Blueprint Imports ---
from blueprint import (
    S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_VALIDATED, S6_LABELED_DATASET,
    S1_RAW_TICK_PARTITIONED, S1_PROCESSED, BARRIER_ATR_PERIOD, ATR_BASELINE_DAYS
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(processName)s - %(message)s",
)
warnings.filterwarnings("ignore", category=UserWarning, module="polars")
try:
    from polars.exceptions import PolarsUsePyarrowWarning

    warnings.filterwarnings("ignore", category=PolarsUsePyarrowWarning)
except ImportError:
    pass

# --- ▼▼▼ V5 双方向ラベリング ルール定義 ▼▼▼ ---

# 対象タイムフレームとATRの指定
TARGET_TIMEFRAMES = ["M3"]  # Optunaの結論: M3単体・ratio0.8・TD30min
ATR_PERIOD = BARRIER_ATR_PERIOD  # blueprintから取得
ATR_RATIO_THRESHOLD = 0.8  # ATR Ratio閾値（絶対値ではなく相対比率）

# timeframeごとの1日あたりバー数（ATR Ratio計算のbaseline_period算出に使用）
timeframe_bars_per_day = {
    "M0.5": 2880, "M1": 1440, "M3": 480, "M5": 288,
    "M8": 180, "M15": 96, "M30": 48, "H1": 24,
    "H4": 6, "H6": 4, "H12": 2, "D1": 1, "W1": 1, "MN": 1
}

# ★スプレッドコストを定義（spread_pips=50.0 → XAUUSD: 1pip=0.01ドル → 50pips=0.50ドル）
SPREAD = 0.50

# ロング用ルール
RULE_LONG = {
    "pt_mult": 1.0,   # pt_multiplier_long
    "sl_mult": 5.0,   # sl_multiplier_long
    "td": "30m",      # td_minutes_long: 30
}

# ショート用ルール
RULE_SHORT = {
    "pt_mult": 1.0,   # pt_multiplier_short
    "sl_mult": 5.0,   # sl_multiplier_short
    "td": "30m",      # td_minutes_short: 30
}
# --- ▲▲▲ 改造ここまで ▲▲▲ ---


# --- Configuration ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, dual-labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S1_PROCESSED  # tick/ATRはS1から直接取得（S2_FEATURES_VALIDATEDは不使用）
    output_dir: Path = S6_LABELED_DATASET

    filter_mode: str = "year"  # 'year', 'month', 'all'
    filter_year: Optional[int] = 2023
    filter_month: Optional[int] = None
    resume: bool = True
    execution_start_time: str = field(
        default_factory=lambda: dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

    def get_filter_description(self) -> str:
        if self.filter_mode == "year" and self.filter_year is not None:
            return f"Year = {self.filter_year}"
        elif (
            self.filter_mode == "month"
            and self.filter_year is not None
            and self.filter_month is not None
        ):
            month_str = f"{self.filter_month:02d}"
            return f"Year/Month = {self.filter_year}/{month_str}"
        elif self.filter_mode == "all":
            return "All Time"
        else:
            return f"Invalid Filter ({self.filter_mode})"


# --- ▼▼▼ Numba 双方向走査エンジン ▼▼▼ ---
def _njit_if_available(func):
    if NUMBA_AVAILABLE:
        return njit(func, parallel=True, fastmath=True, cache=True)
    else:
        return func


@_njit_if_available
def _numba_find_hits_dual(
    bets_t0: np.ndarray,
    bets_t1_max_long: np.ndarray,
    bets_t1_max_short: np.ndarray,
    bets_pt_long: np.ndarray,
    bets_sl_long: np.ndarray,
    bets_pt_short: np.ndarray,
    bets_sl_short: np.ndarray,
    ticks_ts: np.ndarray,
    ticks_high: np.ndarray,
    ticks_low: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Numba JIT compiled function to find barrier hits for BOTH Long and Short simultaneously.
    """
    n_bets = len(bets_t0)
    n_ticks = len(ticks_ts)

    # ロング用とショート用の出力配列を準備（duration配列は削除）
    out_pt_long = np.zeros(n_bets, dtype=np.int64)
    out_sl_long = np.zeros(n_bets, dtype=np.int64)
    out_pt_short = np.zeros(n_bets, dtype=np.int64)
    out_sl_short = np.zeros(n_bets, dtype=np.int64)

    if n_ticks == 0:
        return out_pt_long, out_sl_long, out_pt_short, out_sl_short
    for i in prange(n_bets):
        t0 = bets_t0[i]

        # ロング用バリア・TD
        t1_l = bets_t1_max_long[i]
        pt_l = bets_pt_long[i]
        sl_l = bets_sl_long[i]

        # ショート用バリア・TD
        t1_s = bets_t1_max_short[i]
        pt_s = bets_pt_short[i]
        sl_s = bets_sl_short[i]

        # --- 修正前 ---
        # start_idx = np.searchsorted(ticks_ts, t0, side="left")

        # --- 修正後 ---
        start_idx = np.searchsorted(ticks_ts, t0, side="right")

        # ヒットしたタイムスタンプを記録する変数
        pt_l_found = np.int64(0)
        sl_l_found = np.int64(0)
        pt_s_found = np.int64(0)
        sl_s_found = np.int64(0)

        # 各方向の走査を継続するかどうかのフラグ
        long_active = True
        short_active = True

        for j in range(start_idx, n_ticks):
            tick_time = ticks_ts[j]
            tick_high = ticks_high[j]
            tick_low = ticks_low[j]

            # --- ロング判定 ---
            if long_active:
                if tick_time > t1_l:
                    long_active = False  # TD超過
                else:
                    if pt_l_found == 0 and tick_high >= pt_l:
                        pt_l_found = tick_time
                    if sl_l_found == 0 and tick_low <= sl_l:
                        sl_l_found = tick_time

                    if pt_l_found != 0 or sl_l_found != 0:
                        long_active = False  # どちらかのバリアに当たったら終了

            # --- ショート判定 ---
            if short_active:
                if tick_time > t1_s:
                    short_active = False  # TD超過
                else:
                    if pt_s_found == 0 and tick_low <= pt_s:  # ショートのPTは安値側
                        pt_s_found = tick_time
                    if sl_s_found == 0 and tick_high >= sl_s:  # ショートのSLは高値側
                        sl_s_found = tick_time

                    if pt_s_found != 0 or sl_s_found != 0:
                        short_active = False  # どちらかのバリアに当たったら終了

            # 早期退出: ロングもショートも決着がついていればループを抜ける
            # （durationのNumba内計算は削除し、breakのみを維持）
            if not long_active and not short_active:
                break

        out_pt_long[i] = pt_l_found
        out_sl_long[i] = sl_l_found
        out_pt_short[i] = pt_s_found
        out_sl_short[i] = sl_s_found

    # duration配列を削除し、純粋な到達時刻(マイクロ秒)の4つだけを返す
    return out_pt_long, out_sl_long, out_pt_short, out_sl_short


# --- ▲▲▲ Numba 双方向走査エンジンここまで ▲▲▲ ---

# =========================================================================
# ProxyLabelingEngine クラス前半 (初期化・データロード処理)
# =========================================================================


class ProxyLabelingEngine:
    """Engine to create a dual-labeled subset of data for proxy model training."""

    def __init__(self, config: ProxyLabelConfig):
        self.config = config
        logging.info(f"Using output directory: {self.config.output_dir}")

        # --- [修正] ロング・ショートそれぞれの集計用辞書を用意 ---
        self.label_counts_long: Dict[int, int] = {1: 0, 0: 0}
        self.label_counts_short: Dict[int, int] = {1: 0, 0: 0}

        self.report_data: List[Dict[str, Any]] = []
        self._validate_paths()
        self._validate_config()

    def _validate_paths(self):
        if not self.config.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found: {self.config.input_dir}"
            )
        if not self.config.price_data_source_dir.exists():
            raise FileNotFoundError(
                f"Price data source dir not found: {self.config.price_data_source_dir}"
            )

    def _validate_config(self):
        cfg = self.config
        if cfg.filter_mode == "year" and cfg.filter_year is None:
            raise ValueError(
                "Filter mode 'year' requires a specific year (--year YYYY)."
            )
        if cfg.filter_mode == "month":
            if cfg.filter_year is None:
                raise ValueError(
                    "Filter mode 'month' requires a specific year (--year YYYY or via YYYY/MM)."
                )
            if cfg.filter_month is None or not (1 <= cfg.filter_month <= 12):
                raise ValueError(
                    "Filter mode 'month' requires a valid month number (1-12, via YYYY/MM)."
                )
        if cfg.filter_mode not in ["year", "month", "all"]:
            raise ValueError(
                f"Invalid filter_mode: {cfg.filter_mode}. Choose 'year', 'month', or 'all'."
            )

    def _get_duration_in_minutes(self, duration_str: str) -> float:
        """Converts a duration string (e.g., '300m', '90s') to a float value in minutes."""
        match = re.match(r"^(\d+)([ms])$", duration_str)
        if not match:
            logging.warning(
                f"Unexpected duration format: {duration_str}. Trying to parse as minutes."
            )
            num_match = re.match(r"(\d+)", duration_str)
            if num_match:
                return float(num_match.group(1))
            return 0.0
        value, unit = match.groups()
        value_float = float(value)
        if unit == "m":
            return value_float
        elif unit == "s":
            return value_float / 60.0
        return 0.0

    # =========================================================================
    # S5/S2読み込み・ファイル探索 (M1限定最適化)
    # =========================================================================

    def _discover_feature_paths(self) -> List[Path]:
        logging.info(
            f"Recursively searching for feature paths in {self.config.input_dir}..."
        )
        discovered_paths = list(self.config.input_dir.rglob("features_*_neutralized*"))
        feature_paths = [
            p for p in discovered_paths if p.is_dir() or p.name.endswith(".parquet")
        ]
        if not feature_paths:
            raise FileNotFoundError(
                f"No feature paths found in {self.config.input_dir}."
            )
        logging.info(f"  -> Found {len(feature_paths)} feature paths.")
        return feature_paths

    def _build_unified_lazyframe(
        self, feature_paths: List[Path]
    ) -> Tuple[pl.LazyFrame, pl.LazyFrame]:
        """
        S5の特徴量データを取得。要件に基づき M1 以外のタイムフレームはスキップしメモリを節約。
        """
        all_lazy_frames_hive = []
        all_lazy_frames_file = []

        timeframe_pattern = re.compile(
            r"features_e\d+[a-z]?_([a-zA-Z0-9\.]+)(?:_neutralized)?"
        )
        cfg = self.config

        logging.info(
            f"  -> Separating Hive/Files for S5 (Filtering strictly for {TARGET_TIMEFRAMES})..."
        )
        for path in feature_paths:
            name_to_match = path.stem if path.is_file() else path.name
            match = timeframe_pattern.search(name_to_match)
            if not match:
                continue
            timeframe = match.group(1)

            timeframe_suffix = f"_{timeframe}"
            lf: Optional[pl.LazyFrame] = None

            # Hive partition scan
            if path.is_dir():
                scan_base_path = path
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    scan_base_path = scan_base_path / f"year={cfg.filter_year}"
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    scan_base_path = (
                        scan_base_path
                        / f"year={cfg.filter_year}/month={cfg.filter_month}"
                    )

                if scan_base_path.exists():
                    try:
                        lf = pl.scan_parquet(str(scan_base_path / "**/*.parquet"))
                    except Exception as e:
                        logging.warning(f"Failed to scan {scan_base_path}: {e}")
                        continue

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        all_lazy_frames_hive.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

            # Single file scan
            elif path.is_file():
                try:
                    lf_full = pl.scan_parquet(str(path))
                    if "timestamp" not in lf_full.collect_schema().names():
                        continue
                except Exception as e:
                    logging.warning(f"Failed to scan file {path}: {e}")
                    continue

                date_filter: Optional[pl.Expr] = None
                if cfg.filter_mode == "year" and cfg.filter_year is not None:
                    date_filter = pl.col("timestamp").dt.year() == cfg.filter_year
                elif (
                    cfg.filter_mode == "month"
                    and cfg.filter_year is not None
                    and cfg.filter_month is not None
                ):
                    date_filter = (pl.col("timestamp").dt.year() == cfg.filter_year) & (
                        pl.col("timestamp").dt.month() == cfg.filter_month
                    )

                lf = lf_full.filter(date_filter) if date_filter is not None else lf_full

                if lf is not None:
                    lf_renamed = self._rename_features(lf, timeframe_suffix)
                    if lf_renamed is not None:
                        all_lazy_frames_file.append(
                            lf_renamed.with_columns(
                                pl.lit(timeframe).alias("timeframe")
                            )
                        )

        unified_hive_lf = pl.LazyFrame()
        if all_lazy_frames_hive:
            unified_hive_lf = pl.concat(all_lazy_frames_hive, how="diagonal")
            logging.info(
                f"  -> Prepared S5 Hive LazyFrame ({len(all_lazy_frames_hive)} sources)."
            )
        else:
            logging.warning(f"No S5 Hive data found for {TARGET_TIMEFRAMES}.")

        unified_file_lf = pl.LazyFrame()
        if all_lazy_frames_file:
            unified_file_lf = pl.concat(all_lazy_frames_file, how="diagonal")
            logging.info(
                f"  -> Prepared S5 non-tick LazyFrame ({len(all_lazy_frames_file)} sources)."
            )
        else:
            logging.warning(f"No S5 non-tick data found for {TARGET_TIMEFRAMES}.")

        return unified_hive_lf, unified_file_lf

    def _rename_features(
        self, lf: pl.LazyFrame, timeframe_suffix: str
    ) -> Optional[pl.LazyFrame]:
        try:
            current_schema = lf.collect_schema()
            if not current_schema.names():
                return None
            feature_cols = [col for col in current_schema.names() if col != "timestamp"]
            rename_exprs = [
                pl.col(col).alias(f"{col}{timeframe_suffix}") for col in feature_cols
            ]
            select_exprs = [pl.col("timestamp").cast(pl.Datetime("us", "UTC"))]
            if rename_exprs:
                select_exprs.extend(rename_exprs)
            return lf.select(select_exprs)
        except Exception as e:
            logging.warning(f"Failed to rename features: {e}")
            return None

    def _load_all_price_data(self) -> Dict[str, Any]:
        """S1_PROCESSEDからWilder平滑化でATR絶対値・ATR Ratioを自前計算して返す。
        tick価格データは月次チャンクループ内で直接スキャンするためここでは扱わない。"""
        tick_dir = S1_RAW_TICK_PARTITIONED
        if not tick_dir.exists():
            raise FileNotFoundError(f"Master tick directory not found: {tick_dir}")
        logging.info(f"  -> Confirmed tick source: '{tick_dir}' (will be loaded per-month chunk).")

        # --- S1_PROCESSEDのOHLCVからWilder平滑化でATR絶対値を自前計算 ---
        # e1c_atr_13はATR/ATR_13の相対値（≈1.0）のため使用不可
        # atr_ratioも全期間データで事前計算する（日次ループ内での計算は精度・速度ともに問題あり）
        all_atr_lfs = []
        for tf in TARGET_TIMEFRAMES:
            price_dir_tf = S1_PROCESSED / f"timeframe={tf}"
            if not price_dir_tf.exists():
                logging.warning(f"  -> S1_PROCESSED/timeframe={tf} が見つかりません。スキップします。")
                continue
            target_atr_name = f"e1c_atr_{ATR_PERIOD}_{tf}"
            atr_ratio_name = f"atr_ratio_{tf}"
            baseline_period = timeframe_bars_per_day.get(tf, 1440) * ATR_BASELINE_DAYS

            # [DISC-FLAG 対応] s1_1_B が出力する disc 列を読み込み、
            #   不連続バー (disc=True) では前バーcloseを使わず H-L のみで TR を計算する。
            #   これにより週末跨ぎや祝日のギャップが ATR を異常値で汚染するのを防ぐ。
            #   本番側 core_indicators.calculate_barrier_atr と同じ思想 (Train-Serve Skew Free)。
            atr_lf = (
                pl.scan_parquet(str(price_dir_tf / "*.parquet"))
                .select(["timestamp", "high", "low", "close", "disc"])
                .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
                .sort("timestamp")
                .with_columns([
                    # disc=True の足は H-L のみ、それ以外は通常の True Range
                    pl.when(pl.col("disc"))
                    .then(pl.col("high") - pl.col("low"))
                    .otherwise(
                        pl.max_horizontal(
                            pl.col("high") - pl.col("low"),
                            (pl.col("high") - pl.col("close").shift(1)).abs(),
                            (pl.col("low") - pl.col("close").shift(1)).abs(),
                        )
                    )
                    .ewm_mean(alpha=1 / ATR_PERIOD, adjust=False)
                    .alias(target_atr_name)
                ])
                # ATR Ratioも全期間データで計算（日次ループ内での計算より精度・速度ともに優れる）
                .with_columns([
                    (
                        pl.col(target_atr_name) /
                        (pl.col(target_atr_name).rolling_mean(window_size=baseline_period, min_samples=1) + 1e-10)
                    ).alias(atr_ratio_name)
                ])
                .select(["timestamp", target_atr_name, atr_ratio_name])
            )
            all_atr_lfs.append(atr_lf)
            logging.info(
                f"  -> Prepared ATR blueprint: S1_PROCESSED/timeframe={tf} -> '{target_atr_name}' + '{atr_ratio_name}' (baseline={baseline_period}bars, disc-aware)"
            )

        if not all_atr_lfs:
            raise ValueError(
                f"FATAL: No valid ATR columns could be computed for {TARGET_TIMEFRAMES} from S1_PROCESSED."
            )

        return {"atr_lfs": all_atr_lfs}

    # =========================================================================
    # ヘルパー関数群 (パーティション探索・時間計算)
    # =========================================================================

    def _discover_partitions(self, unified_lf: pl.LazyFrame) -> pl.DataFrame:
        if unified_lf.collect_schema().names() == []:
            return pl.DataFrame({"date": []}).select(pl.col("date").cast(pl.Date))
        df_dates = (
            unified_lf.select(pl.col("timestamp").dt.date().alias("date"))
            .unique()
            .collect()
        )
        if df_dates.is_empty():
            return df_dates.select(pl.col("date").cast(pl.Date))
        return df_dates.sort("date")

    def _get_bar_duration_minutes(self, timeframe: str) -> float:
        if timeframe == "tick":
            return 0.0
        if timeframe == "MN":
            return 1.0 * 43200
        value_match = re.search(r"(\d*\.?\d+)", timeframe)
        unit_match = re.search(r"([A-Z])", timeframe)
        if not value_match or not unit_match:
            return 0.0
        try:
            value = float(value_match.group(1)) if value_match.group(1) else 1.0
        except ValueError:
            return 0.0
        unit = unit_match.group(1)
        if unit == "M":
            return value
        if unit == "H":
            return value * 60
        if unit == "D":
            return value * 1440
        if unit == "W":
            return value * 10080
        return 0.0

    # =========================================================================
    # メイン実行ループ (run)
    # =========================================================================

    def run(self):
        logging.info(f"### Phase 3: Final Labeling (V5 Dual-Directional Labeling) ###")
        logging.info(f"Applying filter: {self.config.get_filter_description()}")
        logging.info(f"Target Timeframes: {TARGET_TIMEFRAMES}")
        logging.info(
            f"ATR Ratio Filter: atr_ratio >= {ATR_RATIO_THRESHOLD} (Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day)"
        )
        logging.info(
            f"Long Rule : PT={RULE_LONG['pt_mult']}, SL={RULE_LONG['sl_mult']}, TD={RULE_LONG['td']}"
        )
        logging.info(
            f"Short Rule: PT={RULE_SHORT['pt_mult']}, SL={RULE_SHORT['sl_mult']}, TD={RULE_SHORT['td']}"
        )

        cfg = self.config

        if not cfg.resume and cfg.output_dir.exists():
            logging.info(
                f"Resume is disabled. Deleting existing directory: {cfg.output_dir}"
            )
            shutil.rmtree(cfg.output_dir)
        cfg.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logging.info("Step 1: Discovering feature paths...")
            all_feature_paths = self._discover_feature_paths()

            logging.info("Step 2: Building unified bet/feature frames (Lazy)...")
            unified_hive_lf, unified_file_lf = self._build_unified_lazyframe(
                all_feature_paths
            )

            if (
                unified_hive_lf.collect_schema().names() == []
                and unified_file_lf.collect_schema().names() == []
            ):
                logging.warning(
                    f"No data found for filter '{cfg.get_filter_description()}'. Exiting."
                )
                self._generate_report()
                return

            logging.info("Step 3: Preparing price data blueprints...")
            price_components = self._load_all_price_data()
            atr_lfs = price_components["atr_lfs"]

            # max_lookahead_minutesを先に定義（月チャンクのマージン計算に使用）
            max_lookahead_minutes = max(
                self._get_duration_in_minutes(RULE_LONG["td"]),
                self._get_duration_in_minutes(RULE_SHORT["td"]),
            )
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)
            # 月末に加算するルックアヘッドマージン（TD分 + 安全バッファ3日）
            lookahead_margin = dt.timedelta(minutes=max_lookahead_minutes) + dt.timedelta(days=3)

            # ATRは全期間・全時間足で1回だけ事前ロード（軽量なのでOK）
            logging.info(f"   -> Pre-loading {len(atr_lfs)} ATR files into memory...")
            atr_dfs = [lf.collect().sort("timestamp") for lf in atr_lfs]

            logging.info("Step 4: Discovering daily partitions for processing...")
            partitions_df_hive = self._discover_partitions(unified_hive_lf)
            partitions_df_file = self._discover_partitions(unified_file_lf)
            partitions_df = (
                pl.concat([partitions_df_hive, partitions_df_file])
                .unique()
                .sort("date")
            )

            logging.info(f"   -> Found {len(partitions_df)} daily partitions.")
            if partitions_df.is_empty():
                logging.warning("No partitions found. Exiting.")
                self._generate_report()
                return

            # 処理対象の年月一覧を作成（外側ループ用）
            months_df = (
                partitions_df.select(
                    pl.col("date").dt.year().alias("year"),
                    pl.col("date").dt.month().alias("month"),
                )
                .unique()
                .sort(["year", "month"])
            )

            logging.info(
                f"Step 5: Starting monthly chunked processing loop... "
                f"({len(months_df)} months / {len(partitions_df)} days)"
            )

            # =========================================================
            # 外側ループ：月単位でtickをロード→処理→破棄
            # =========================================================
            for month_row in tqdm(
                months_df.iter_rows(named=True),
                total=len(months_df),
                desc="Processing Months",
            ):
                y, m = month_row["year"], month_row["month"]
                _, last_day = calendar.monthrange(y, m)

                # その月のtick範囲（ルックアヘッドマージン付き）
                month_start = dt.datetime(y, m, 1, tzinfo=dt.timezone.utc)
                month_end = (
                    dt.datetime(y, m, last_day, tzinfo=dt.timezone.utc)
                    + lookahead_margin
                )

                # ★ その月のtickデータだけをメモリに載せる
                # hive_partitioning=Trueで述語プッシュダウンを確実に有効化
                logging.debug(f"Loading tick chunk for {y}-{m:02d}...")
                try:
                    base_price_chunk_df = (
                        pl.scan_parquet(
                            str(S1_RAW_TICK_PARTITIONED / "**/*.parquet"),
                            hive_partitioning=True,
                        )
                        .rename({"datetime": "timestamp"})
                        .select("timestamp", "mid_price")
                        .with_columns(
                            pl.col("timestamp").cast(pl.Datetime("us", "UTC")),
                            pl.col("mid_price").alias("close"),
                            pl.col("mid_price").alias("high"),
                            pl.col("mid_price").alias("low"),
                        )
                        .select("timestamp", "close", "high", "low")
                        .filter(pl.col("timestamp").is_between(month_start, month_end))
                        .collect()
                        .unique("timestamp", keep="first")
                        .sort("timestamp")
                    )
                except Exception as e:
                    logging.warning(f"Failed to load tick chunk for {y}-{m:02d}: {e}. Skipping month.")
                    continue

                if base_price_chunk_df.is_empty():
                    logging.warning(f"No tick data found for {y}-{m:02d}. Skipping month.")
                    continue

                logging.debug(
                    f"  -> Tick chunk {y}-{m:02d}: {len(base_price_chunk_df):,} rows loaded."
                )

                # その月に含まれる日のリストを取得
                days_in_month = partitions_df.filter(
                    (pl.col("date").dt.year() == y) &
                    (pl.col("date").dt.month() == m)
                )

                # =====================================================
                # 内側ループ：日次処理（既存ロジックをそのまま維持）
                # =====================================================
                for row in days_in_month.iter_rows(named=True):
                    current_date = row["date"]
                    year_d, month_d, day_d = (
                        current_date.year,
                        current_date.month,
                        current_date.day,
                    )

                    output_partition_dir = (
                        cfg.output_dir / f"year={year_d}/month={month_d}/day={day_d}"
                    )
                    if cfg.resume and output_partition_dir.exists():
                        logging.debug(
                            f"Resuming: Output exists for {current_date}. Skipping."
                        )
                        continue

                    if "timestamp" in unified_hive_lf.collect_schema().names():
                        daily_bets_hive_df = unified_hive_lf.filter(
                            pl.col("timestamp").dt.date() == current_date
                        ).collect()
                    else:
                        daily_bets_hive_df = pl.DataFrame()

                    if "timestamp" in unified_file_lf.collect_schema().names():
                        daily_bets_file_df = unified_file_lf.filter(
                            pl.col("timestamp").dt.date() == current_date
                        ).collect()
                    else:
                        daily_bets_file_df = pl.DataFrame()

                    raw_bets_df = pl.concat(
                        [daily_bets_hive_df, daily_bets_file_df], how="diagonal"
                    )

                    if raw_bets_df.is_empty():
                        continue

                    # 1. 各タイムスタンプで有効なターゲット時間足を抽出（重複排除込み）
                    valid_targets_df = (
                        raw_bets_df.filter(pl.col("timeframe").is_in(TARGET_TIMEFRAMES))
                        .select(["timestamp", "timeframe"])
                        .unique()
                    )

                    # 2. 全時間足の特徴量をtimestampごとに1行に集約
                    # （forward_fillは未来情報リークのリスクがあるため削除、group_by+aggの重複排除は維持）
                    master_features_df = (
                        raw_bets_df.drop("timeframe")
                        .group_by("timestamp")
                        .agg(pl.all().drop_nulls().first())
                        .sort("timestamp")
                    )

                    # 3. 集約済み特徴量を各ターゲット時間足行に結合
                    daily_bets_df = valid_targets_df.join(
                        master_features_df, on="timestamp", how="left"
                    ).sort(["timeframe", "timestamp"])

                    if daily_bets_df.is_empty():
                        continue

                    min_ts_req = daily_bets_df["timestamp"].min()
                    if min_ts_req is None:
                        continue

                    # ロング/ショートの最長TD + 余裕分(2日)で窓を切り出す
                    max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)

                    # ★ base_price_chunk_df（月チャンク）から窓を切り出す
                    price_window_df = (
                        base_price_chunk_df.filter(
                            pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                        )
                        .sort("timestamp")
                    )

                    if price_window_df.is_empty():
                        continue

                    for atr_df in atr_dfs:
                        atr_df_small = atr_df.filter(
                            pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                        )
                        if not atr_df_small.is_empty():
                            price_window_df = price_window_df.join_asof(
                                atr_df_small, on="timestamp"
                            )

                    price_window_df = price_window_df.fill_null(strategy="forward")

                    daily_labeled_df = self._calculate_labels_for_batch(
                        daily_bets_df, price_window_df
                    )

                    if daily_labeled_df is not None and not daily_labeled_df.is_empty():
                        self._update_label_counts_dual(daily_labeled_df)
                        self._collect_report_data_dual(daily_labeled_df, current_date)
                        output_partition_dir.mkdir(parents=True, exist_ok=True)
                        daily_labeled_df.write_parquet(
                            output_partition_dir / "data.parquet", compression="zstd"
                        )

                    # 日次メモリ解放
                    del daily_bets_df, price_window_df, daily_labeled_df
                    del daily_bets_hive_df, daily_bets_file_df
                    gc.collect()

                # ★ 月チャンク終了: tickデータをメモリから破棄
                del base_price_chunk_df
                gc.collect()
                logging.debug(f"  -> Tick chunk {y}-{m:02d} released from memory.")

            self._log_final_summary()
            self._generate_report()

        except Exception as e:
            logging.error(f"A critical error occurred: {e}", exc_info=True)
            raise

    # =========================================================================
    # コアロジック: ATR収縮フィルター適用と双方向ラベルの一括計算
    # =========================================================================

    def _calculate_labels_for_batch(
        self, daily_bets_df: pl.DataFrame, price_window_df: pl.DataFrame
    ) -> pl.DataFrame | None:
        if daily_bets_df.is_empty():
            return None

        if not NUMBA_AVAILABLE:
            logging.error("Numba is required for labeling but not found. Skipping.")
            return None

        try:
            ticks_df_np = price_window_df.select(
                pl.col("timestamp").cast(pl.Int64).alias("ticks_ts"),
                pl.col("high").alias("ticks_high"),
                pl.col("low").alias("ticks_low"),
            )
            ticks_ts_np = ticks_df_np["ticks_ts"].to_numpy()
            ticks_high_np = ticks_df_np["ticks_high"].to_numpy()
            ticks_low_np = ticks_df_np["ticks_low"].to_numpy()
        except Exception as e:
            logging.error(f"Failed to convert tick data to Numpy arrays: {e}")
            return None

        labeled_chunks = []
        for timeframe_tuple, group_df in daily_bets_df.group_by("timeframe"):
            timeframe = timeframe_tuple[0]
            if timeframe is None or group_df.is_empty():
                continue

            # 対象外のデータが混入した場合は安全のためスキップ
            if timeframe not in TARGET_TIMEFRAMES:
                logging.debug(
                    f"Skipping timeframe {timeframe} (Targets are {TARGET_TIMEFRAMES})."
                )
                continue

            # ロング/ショートそれぞれのタイムアウト（TD）を計算
            td_long_minutes = self._get_duration_in_minutes(RULE_LONG["td"])
            td_short_minutes = self._get_duration_in_minutes(RULE_SHORT["td"])

            t1_max_long_expr = pl.col("timestamp") + pl.duration(
                minutes=td_long_minutes
            )
            t1_max_short_expr = pl.col("timestamp") + pl.duration(
                minutes=td_short_minutes
            )

            atr_col_name = f"e1c_atr_{ATR_PERIOD}_{timeframe}"
            atr_ratio_col_name = f"atr_ratio_{timeframe}"  # 事前計算済みカラム名
            if atr_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR column '{atr_col_name}' not found. Skipping."
                )
                continue
            if atr_ratio_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR ratio column '{atr_ratio_col_name}' not found. Skipping."
                )
                continue

            original_cols = [
                c for c in group_df.columns if c not in ["timestamp", "close"]
            ]

            bets_with_price_df = group_df.join_asof(
                price_window_df.select(
                    ["timestamp", "close", "high", "low", atr_col_name, atr_ratio_col_name]
                ),
                on="timestamp",
            ).filter(pl.col(atr_col_name).is_not_null())

            if bets_with_price_df.is_empty():
                continue

            # atr_value・atr_ratioともに事前計算済みカラムをそのまま使用（日次ループ内での再計算なし）
            bets_df_with_atr = bets_with_price_df.with_columns(
                pl.col(atr_col_name).alias("atr_value"),
                pl.col(atr_ratio_col_name).alias("atr_ratio"),
            )

            # --- 全行を保持しつつ、ATR Ratioでエントリー起点に is_trigger=1 を付与 ---
            bets_df_all = bets_df_with_atr.with_columns(
                pl.when(pl.col("atr_ratio") >= ATR_RATIO_THRESHOLD)
                .then(1)
                .otherwise(0)
                .cast(pl.Int8)
                .alias("is_trigger")
            )

            # Numbaに渡して重い計算をするのは is_trigger==1 の行だけにする
            bets_df_filtered = bets_df_all.filter(pl.col("is_trigger") == 1)
            if bets_df_filtered.is_empty():
                continue

            # バリアの一括計算 (ロングとショート両方)
            bets_df = bets_df_filtered.select(
                pl.col("timestamp").alias("t0"),
                # ★修正: ロング用バリア（Askエントリー想定: PTは遠く、SLは近く）
                (
                    pl.col("close")
                    + pl.col("atr_value") * RULE_LONG["pt_mult"]
                    + SPREAD
                ).alias("pt_long"),
                (
                    pl.col("close")
                    - pl.col("atr_value") * RULE_LONG["sl_mult"]
                    + SPREAD
                ).alias("sl_long"),
                t1_max_long_expr.alias("t1_max_long"),
                # ★修正: ショート用バリア（Bidエントリー/Ask決済想定: PTは遠く、SLは近く）
                (
                    pl.col("close")
                    - pl.col("atr_value") * RULE_SHORT["pt_mult"]
                    - SPREAD
                ).alias("pt_short"),
                (
                    pl.col("close")
                    + pl.col("atr_value") * RULE_SHORT["sl_mult"]
                    - SPREAD
                ).alias("sl_short"),
                t1_max_short_expr.alias("t1_max_short"),
                pl.col("atr_value"),
                pl.col("atr_ratio"),  # ★ S6出力に含める（バックテストシミュレーターが再計算不要になる）
                pl.col("close"),
                pl.col(original_cols),
            )

            if bets_df.is_empty():
                continue

            try:
                bets_t0_np = bets_df["t0"].cast(pl.Int64).to_numpy()
                bets_t1_max_l_np = bets_df["t1_max_long"].cast(pl.Int64).to_numpy()
                bets_t1_max_s_np = bets_df["t1_max_short"].cast(pl.Int64).to_numpy()

                bets_pt_l_np = bets_df["pt_long"].to_numpy(writable=True)
                bets_sl_l_np = bets_df["sl_long"].to_numpy(writable=True)
                bets_pt_s_np = bets_df["pt_short"].to_numpy(writable=True)
                bets_sl_s_np = bets_df["sl_short"].to_numpy(writable=True)
            except Exception as e:
                logging.error(f"Failed to convert bets data to Numpy arrays: {e}")
                continue

            # Numbaによる双方向同時判定
            out_pt_l, out_sl_l, out_pt_s, out_sl_s = _numba_find_hits_dual(
                bets_t0_np,
                bets_t1_max_l_np,
                bets_t1_max_s_np,
                bets_pt_l_np,
                bets_sl_l_np,
                bets_pt_s_np,
                bets_sl_s_np,
                ticks_ts_np,
                ticks_high_np,
                ticks_low_np,
            )

            # 計算済みトリガー行の DataFrame 作成
            calculated_df = (
                bets_df.with_columns(
                    pl.Series("pt_l_time", out_pt_l),
                    pl.Series("sl_l_time", out_sl_l),
                    pl.Series("pt_s_time", out_pt_s),
                    pl.Series("sl_s_time", out_sl_s),
                )
                .with_columns(
                    # ロング用ラベルと「正確な決済時刻(マイクロ秒)」の特定
                    label_long=pl.when(
                        (pl.col("pt_l_time") > 0)
                        & (
                            (pl.col("sl_l_time") == 0)
                            | (pl.col("pt_l_time") < pl.col("sl_l_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_l=pl.when(
                        (pl.col("pt_l_time") > 0)
                        & (
                            (pl.col("sl_l_time") == 0)
                            | (pl.col("pt_l_time") < pl.col("sl_l_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.col("pt_l_time"))
                    .when(pl.col("sl_l_time") > 0)
                    .then(pl.col("sl_l_time"))
                    .otherwise(
                        pl.col("t1_max_long").cast(pl.Int64)
                    ),  # ←変更: Int64にキャスト
                    # ショート用ラベルと「正確な決済時刻(マイクロ秒)」の特定
                    label_short=pl.when(
                        (pl.col("pt_s_time") > 0)
                        & (
                            (pl.col("sl_s_time") == 0)
                            | (pl.col("pt_s_time") < pl.col("sl_s_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_s=pl.when(
                        (pl.col("pt_s_time") > 0)
                        & (
                            (pl.col("sl_s_time") == 0)
                            | (pl.col("pt_s_time") < pl.col("sl_s_time"))  # 🟢 修正
                        )
                    )
                    .then(pl.col("pt_s_time"))
                    .when(pl.col("sl_s_time") > 0)
                    .then(pl.col("sl_s_time"))
                    .otherwise(
                        pl.col("t1_max_short").cast(pl.Int64)
                    ),  # ←変更: Int64にキャスト
                )
                .with_columns(
                    # マイクロ秒の差分から「実経過時間（分）」を算出して Float32 で保持
                    duration_long=(
                        (pl.col("end_l") - pl.col("t0").cast(pl.Int64))
                        / 1_000_000
                        / 60.0  # ←変更: t0をInt64にキャスト
                    ).cast(pl.Float32),
                    duration_short=(
                        (pl.col("end_s") - pl.col("t0").cast(pl.Int64))
                        / 1_000_000
                        / 60.0  # ←変更: t0をInt64にキャスト
                    ).cast(pl.Float32),
                )
                .select(
                    [
                        "t0",
                        "label_long",
                        "label_short",
                        "duration_long",
                        "duration_short",
                    ]
                )
            )

            # 全データ（M1のすべての足）へ Left Join (非トリガー行は自動的にNullになる)
            final_group_df = (
                bets_df_all.rename({"timestamp": "t0"})
                .join(calculated_df, on="t0", how="left")
                .rename({"t0": "timestamp"})
            )

            # atr_valueはシミュレーターで必須になるためドロップせずに保持する
            # 【重要】シミュレーター用の close と atr_value は残しつつ、不要なゴミを捨てる

            # 存在するカラムのみを削除対象にする（安全なdrop）
            drop_candidates = ["open", "high", "low", atr_col_name]
            actual_drops = [c for c in drop_candidates if c in final_group_df.columns]

            labeled_chunks.append(final_group_df.drop(actual_drops))
        if not labeled_chunks:
            return None
        return pl.concat(labeled_chunks).sort("timestamp")

    # =========================================================================
    # 集計・レポート機能 (双方向対応)
    # =========================================================================

    def _update_label_counts_dual(self, df: pl.DataFrame):
        # ロングの集計
        long_counts = df.group_by("label_long").len()
        for row in long_counts.iter_rows(named=True):
            if row["label_long"] in self.label_counts_long:
                self.label_counts_long[row["label_long"]] += row["len"]

        # ショートの集計
        short_counts = df.group_by("label_short").len()
        for row in short_counts.iter_rows(named=True):
            if row["label_short"] in self.label_counts_short:
                self.label_counts_short[row["label_short"]] += row["len"]

    def _collect_report_data_dual(self, df: pl.DataFrame, current_date: dt.date):
        try:
            required_cols = [
                "timestamp",
                "timeframe",
                "is_trigger",
                "label_long",
                "label_short",
                "duration_long",
                "duration_short",
            ]
            if not all(col in df.columns for col in required_cols):
                return

            # --- [修正] OOM回避: トリガーが発火した行(is_trigger==1)だけをレポート用に抽出 ---
            report_df = (
                df.filter(pl.col("is_trigger") == 1)
                .with_columns(pl.lit(current_date).alias("date"))
                .select(
                    [
                        "timeframe",
                        "label_long",
                        "label_short",
                        "duration_long",
                        "duration_short",
                        "date",
                    ]
                )
            )
            self.report_data.extend(report_df.to_dicts())
        except Exception as e:
            logging.warning(f"Error collecting report data for {current_date}: {e}")

    def _log_final_summary(self):
        total_samples = sum(self.label_counts_long.values())
        if total_samples == 0:
            logging.warning("No samples were processed.")
            return

        long_win = self.label_counts_long.get(1, 0)
        short_win = self.label_counts_short.get(1, 0)

        long_loss = total_samples - long_win
        short_loss = total_samples - short_win

        scale_pos_weight_long = long_loss / long_win if long_win > 0 else 1.0
        scale_pos_weight_short = short_loss / short_win if short_win > 0 else 1.0

        summary = (
            "\n" + "=" * 60 + "\n"
            f"### V5 Dual-Directional Labeling COMPLETED ###\n"
            f"Output Dir: {self.config.output_dir}\n"
            f"  - Total Labeled Triggers (is_trigger=1): {total_samples}\n"
            f"  - Long Win (1): {long_win} / Loss (0): {long_loss}  => `scale_pos_weight_long`: {scale_pos_weight_long:.4f}\n"
            f"  - Short Win (1): {short_win} / Loss (0): {short_loss} => `scale_pos_weight_short`: {scale_pos_weight_short:.4f}\n"
            + "="
            * 60
        )
        logging.info(summary)

    def _generate_report(self):
        logging.info("Generating detailed execution report...")
        cfg = self.config
        report_path = cfg.output_dir / "execution_report_v5_dual.md"

        if not self.report_data:
            report_path.write_text("# Execution Report\n\nNo data generated.")
            return

        try:
            df = pl.from_dicts(self.report_data)
            total = len(df)
            l_win = df.filter(pl.col("label_long") == 1).height
            s_win = df.filter(pl.col("label_short") == 1).height

            # --- Duration統計: 勝ち/負け別に分解 ---
            df_l_win  = df.filter(pl.col("label_long") == 1)["duration_long"]
            df_l_loss = df.filter(pl.col("label_long") == 0)["duration_long"]
            df_s_win  = df.filter(pl.col("label_short") == 1)["duration_short"]
            df_s_loss = df.filter(pl.col("label_short") == 0)["duration_short"]

            avg_dur_l_win  = df_l_win.mean()  or 0.0
            med_dur_l_win  = df_l_win.median() or 0.0
            avg_dur_l_loss = df_l_loss.mean()  or 0.0
            med_dur_l_loss = df_l_loss.median() or 0.0
            avg_dur_s_win  = df_s_win.mean()  or 0.0
            med_dur_s_win  = df_s_win.median() or 0.0
            avg_dur_s_loss = df_s_loss.mean()  or 0.0
            med_dur_s_loss = df_s_loss.median() or 0.0

            daily_activity = (
                df.group_by("date").len().sort("len", descending=True).limit(10)
            )
            daily_table = "| Date | Valid Setup Samples |\n|:---|---:|\n"
            for row in daily_activity.to_dicts():
                daily_table += f"| `{row['date']}` | `{row['len']:,}` |\n"

            report_content = f"""
# Proxy Labeling Engine - Execution Report (V5 Dual-Directional) ⚔️

### 1. Execution Summary
| Item | Value |
|:---|:---|
| **Filter Applied** | `{cfg.get_filter_description()}` |
| **Target Timeframes** | `{TARGET_TIMEFRAMES}` |
| **ATR Filter** | `atr_ratio >= {ATR_RATIO_THRESHOLD}` (ATR Period: {ATR_PERIOD}, Baseline: {ATR_BASELINE_DAYS} day) |
| **Long Rule** | `PT: {RULE_LONG["pt_mult"]}, SL: {RULE_LONG["sl_mult"]}, TD: {RULE_LONG["td"]}` |
| **Short Rule** | `PT: {RULE_SHORT["pt_mult"]}, SL: {RULE_SHORT["sl_mult"]}, TD: {RULE_SHORT["td"]}` |

### 2. Overall Performance
| Metric | Count | Win Rate |
|:---|---:|---:|
| **Total Setups (ATR Ratio >= threshold)** | `{total:,}` | - |
| **Long Profit-Take** | `{l_win:,}` | `{l_win / total:.2%}` |
| **Short Profit-Take** | `{s_win:,}` | `{s_win / total:.2%}` |

### 3. Event Duration Breakdown (Win vs Loss)
| Direction | Outcome | Avg Duration | Median Duration |
|:---|:---|---:|---:|
| **Long** | Win (PT hit) | `{avg_dur_l_win:.1f} min` | `{med_dur_l_win:.1f} min` |
| **Long** | Loss (SL/TD) | `{avg_dur_l_loss:.1f} min` | `{med_dur_l_loss:.1f} min` |
| **Short** | Win (PT hit) | `{avg_dur_s_win:.1f} min` | `{med_dur_s_win:.1f} min` |
| **Short** | Loss (SL/TD) | `{avg_dur_s_loss:.1f} min` | `{med_dur_s_loss:.1f} min` |

### 4. Top 10 Busiest Days (Setups)
{daily_table.strip()}
"""
            report_path.write_text(report_content.strip())
            logging.info(f"Report saved to: {report_path}")
        except Exception as e:
            logging.error(f"Failed to generate report: {e}", exc_info=True)


# =========================================================================
# CLI エントリーポイント
# =========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="[Phase 3] Create Final V5 Dual-Directional Labels."
    )
    parser.add_argument(
        "--filter-mode", type=str, default=None, choices=["year", "month", "all"]
    )
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--year-month", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    filter_year_arg = args.year
    filter_month_arg = None
    filter_mode_arg = args.filter_mode

    # --year-month が指定された場合、自動的に month モードとして処理
    if args.year_month:
        match = re.match(r"(\d{4})/(\d{1,2})$", args.year_month)
        if match:
            filter_year_arg, filter_month_arg = map(int, match.groups())
            filter_mode_arg = "month"

    # --year のみ指定された場合、自動的に year モードとして処理
    elif args.year:
        if not filter_mode_arg:
            filter_mode_arg = "year"

    # どちらも指定がない、かつ明示的な指定もなければ all にフォールバック
    if not filter_mode_arg:
        filter_mode_arg = "all"

    config = ProxyLabelConfig(
        filter_mode=filter_mode_arg,
        filter_year=filter_year_arg,
        filter_month=filter_month_arg,
        resume=not args.no_resume,
    )
    try:
        temp_engine_for_validation = ProxyLabelingEngine(config)
    except ValueError as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    print("\nStarting V5 Dual-Directional Labeling Engine...")
    engine = ProxyLabelingEngine(config)
    engine.run()
