# /workspace/models/create_proxy_labels_polars_patch_regime.py
# [フェーズ3: 最終ラベリングスクリプト - V5 双方向ラベリング仕様]
# - 対象: M1（1分足）限定
# - ATRフィルター: ATR(13) <= 0.5 （極小収縮）
# - ロング/ショート同時判定（Numba 1パス処理）

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
from blueprint import S5_NEUTRALIZED_ALPHA_SET, S2_FEATURES_FIXED, S6_LABELED_DATASET

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
TARGET_TIMEFRAME = "M1"
TARGET_ATR_COLUMN = "e1c_atr_13_M1"  # ATR13を使用
ATR_SHRINK_THRESHOLD = 0.5  # ATR <= 0.5 を評価対象とする

# ロング用ルール
RULE_LONG = {
    "pt_mult": 1.0,
    "sl_mult": 5.0,
    "td": "15m",
}

# ショート用ルール
RULE_SHORT = {
    "pt_mult": 1.0,  # 下落方向への利幅
    "sl_mult": 5.0,  # 上昇方向への損切幅
    "td": "5m",
}
# --- ▲▲▲ 改造ここまで ▲▲▲ ---


# --- Configuration ---
@dataclass
class ProxyLabelConfig:
    """Config for creating a context-adaptive, dual-labeled subset for proxy model training."""

    input_dir: Path = S5_NEUTRALIZED_ALPHA_SET
    price_data_source_dir: Path = S2_FEATURES_FIXED
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

        start_idx = np.searchsorted(ticks_ts, t0, side="left")

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
            f"  -> Separating Hive/Files for S5 (Filtering strictly for {TARGET_TIMEFRAME})..."
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
            logging.warning(f"No S5 Hive data found for {TARGET_TIMEFRAME}.")

        unified_file_lf = pl.LazyFrame()
        if all_lazy_frames_file:
            unified_file_lf = pl.concat(all_lazy_frames_file, how="diagonal")
            logging.info(
                f"  -> Prepared S5 non-tick LazyFrame ({len(all_lazy_frames_file)} sources)."
            )
        else:
            logging.warning(f"No S5 non-tick data found for {TARGET_TIMEFRAME}.")

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
        """S2の価格データ(Tick)と、ATR(13)を読み込む。"""
        price_dir = self.config.price_data_source_dir / "feature_value_a_vast_universeC"
        tick_dir = price_dir / "features_e1c_tick"
        if not tick_dir.exists():
            raise FileNotFoundError(f"Master price directory not found: {tick_dir}")
        logging.info(f"  -> Scanning '{tick_dir}' as master price source (S2 Tick).")

        base_lf = (
            pl.scan_parquet(str(tick_dir / "**/*.parquet"))
            .select("timestamp", "close", "high", "low")
            .with_columns(pl.col("timestamp").cast(pl.Datetime("us", "UTC")))
            .unique("timestamp", keep="first", maintain_order=True)
        )

        atr_files = list(price_dir.glob("features_e1c_*.parquet"))
        timeframe_pattern = re.compile(
            r"features_e1c_([a-zA-Z0-9\.]+)(?:_atr_only_tick_fixed)?\.parquet"
        )
        all_atr_lfs = []

        # --- [修正] ATR13 を指定 ---
        SOURCE_ATR_COLUMN_NAME = "e1c_atr_13"

        for f_path in atr_files:
            match = timeframe_pattern.search(f_path.name)
            if not match:
                continue
            timeframe = match.group(1)

            # --- [重要] M1のATRのみ読み込む ---
            if timeframe != TARGET_TIMEFRAME:
                continue

            target_atr_name = f"e1c_atr_13_{timeframe}"  # M1なら e1c_atr_13_M1 になる
            lf_original = pl.scan_parquet(str(f_path))
            schema_names = lf_original.collect_schema().names()

            lf = lf_original.with_columns(
                pl.col("timestamp").cast(pl.Datetime("us", "UTC"))
            )
            if SOURCE_ATR_COLUMN_NAME in schema_names:
                all_atr_lfs.append(
                    lf.select(["timestamp", SOURCE_ATR_COLUMN_NAME]).rename(
                        {SOURCE_ATR_COLUMN_NAME: target_atr_name}
                    )
                )
                logging.info(
                    f"  -> Prepared ATR blueprint: Loaded '{SOURCE_ATR_COLUMN_NAME}' -> '{target_atr_name}'"
                )
            else:
                logging.warning(
                    f"  -> Required ATR column '{SOURCE_ATR_COLUMN_NAME}' not found in {f_path.name}."
                )

        if not all_atr_lfs:
            raise ValueError(
                f"FATAL: No valid {SOURCE_ATR_COLUMN_NAME} columns were extracted for {TARGET_TIMEFRAME}."
            )

        return {"base_lf": base_lf, "atr_lfs": all_atr_lfs}

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
        logging.info(f"Target Timeframe: {TARGET_TIMEFRAME}")
        logging.info(
            f"ATR Shrink Filter: {TARGET_ATR_COLUMN} <= {ATR_SHRINK_THRESHOLD}"
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
            base_price_lf = price_components["base_lf"]
            atr_lfs = price_components["atr_lfs"]

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

            logging.info("Step 5: Starting daily processing loop...")

            # --- [重要] ロング・ショートのうち長い方のTDをルックアヘッドとして採用 ---
            max_lookahead_minutes = max(
                self._get_duration_in_minutes(RULE_LONG["td"]),
                self._get_duration_in_minutes(RULE_SHORT["td"]),
            )
            max_lookahead_delta = dt.timedelta(minutes=max_lookahead_minutes)

            total_partitions = len(partitions_df)
            for row in tqdm(
                partitions_df.iter_rows(named=True),
                total=total_partitions,
                desc="Processing Partitions",
            ):
                current_date = row["date"]
                year, month, day = (
                    current_date.year,
                    current_date.month,
                    current_date.day,
                )

                output_partition_dir = (
                    cfg.output_dir / f"year={year}/month={month}/day={day}"
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

                # --- [修正] 行増殖バグ対策 & 上位足データ結合(Forward Fill)対応 ---
                raw_bets_df = pl.concat(
                    [daily_bets_hive_df, daily_bets_file_df], how="diagonal"
                )

                if raw_bets_df.is_empty():
                    continue

                daily_bets_df = (
                    raw_bets_df.with_columns(
                        # M1（ターゲット）の行が存在するかどうかのフラグを作成
                        pl.when(pl.col("timeframe") == TARGET_TIMEFRAME)
                        .then(pl.lit(1))
                        .otherwise(pl.lit(0))
                        .alias("is_target")
                    )
                    .group_by("timestamp")
                    .agg(
                        # 同一タイムスタンプの特徴量を1行にマージ
                        pl.col("is_target").max(),
                        pl.exclude("timestamp", "timeframe", "is_target")
                        .drop_nulls()
                        .first(),
                    )
                    .sort("timestamp")
                    # 上位足（H1など）の直近確定値を前方補完し、M1の足に伝播させる
                    .with_columns(pl.exclude("timestamp", "is_target").forward_fill())
                    # M1データが存在するタイムスタンプのみを抽出
                    .filter(pl.col("is_target") == 1)
                    # 下流の処理のために timeframe 列を TARGET_TIMEFRAME (M1) で復元
                    .with_columns(pl.lit(TARGET_TIMEFRAME).alias("timeframe"))
                    .drop("is_target")
                )

                if daily_bets_df.is_empty():
                    continue

                min_ts_req = daily_bets_df["timestamp"].min()
                if min_ts_req is None:
                    continue

                # ロング/ショートの最長TD + 余裕分(2日)で窓を切り出す
                max_ts_req = min_ts_req + max_lookahead_delta + dt.timedelta(days=2)

                price_window_df = (
                    base_price_lf.filter(
                        pl.col("timestamp").is_between(min_ts_req, max_ts_req)
                    )
                    .collect()
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

                price_window_df = price_window_df.fill_null(
                    strategy="forward"
                ).fill_null(strategy="backward")

                # ラベル計算（第4回で定義）
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

                # メモリ解放
                del daily_bets_df, price_window_df, daily_labeled_df
                del daily_bets_hive_df, daily_bets_file_df
                gc.collect()

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

            # M1以外のデータが混入した場合は安全のためスキップ
            if timeframe != TARGET_TIMEFRAME:
                logging.debug(
                    f"Skipping timeframe {timeframe} (Target is {TARGET_TIMEFRAME} only)."
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

            atr_col_name = TARGET_ATR_COLUMN
            if atr_col_name not in price_window_df.columns:
                logging.warning(
                    f"Required ATR column '{atr_col_name}' not found. Skipping."
                )
                continue

            original_cols = [
                c for c in group_df.columns if c not in ["timestamp", "close"]
            ]

            bets_with_price_df = group_df.join_asof(
                price_window_df.select(
                    ["timestamp", "close", "high", "low", atr_col_name]
                ),
                on="timestamp",
            ).filter(pl.col(atr_col_name).is_not_null())

            if bets_with_price_df.is_empty():
                continue

            bets_df_with_atr = bets_with_price_df.with_columns(
                pl.col(atr_col_name).alias("atr_value")
            )
            # --- [変更] 全行を保持しつつ、エントリー起点に is_trigger=1 を付与 ---
            bets_df_all = bets_df_with_atr.with_columns(
                pl.when(pl.col("atr_value") <= ATR_SHRINK_THRESHOLD)
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
                # ロング用バリア
                (pl.col("close") + pl.col("atr_value") * RULE_LONG["pt_mult"]).alias(
                    "pt_long"
                ),
                (pl.col("close") - pl.col("atr_value") * RULE_LONG["sl_mult"]).alias(
                    "sl_long"
                ),
                t1_max_long_expr.alias("t1_max_long"),
                # ショート用バリア
                (pl.col("close") - pl.col("atr_value") * RULE_SHORT["pt_mult"]).alias(
                    "pt_short"
                ),
                (pl.col("close") + pl.col("atr_value") * RULE_SHORT["sl_mult"]).alias(
                    "sl_short"
                ),
                t1_max_short_expr.alias("t1_max_short"),
                pl.col("atr_value"),
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
                            | (pl.col("pt_l_time") <= pl.col("sl_l_time"))
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_l=pl.when(
                        (pl.col("pt_l_time") > 0)
                        & (
                            (pl.col("sl_l_time") == 0)
                            | (pl.col("pt_l_time") <= pl.col("sl_l_time"))
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
                            | (pl.col("pt_s_time") <= pl.col("sl_s_time"))
                        )
                    )
                    .then(pl.lit(1, dtype=pl.Int8))
                    .otherwise(pl.lit(0, dtype=pl.Int8)),
                    end_s=pl.when(
                        (pl.col("pt_s_time") > 0)
                        & (
                            (pl.col("sl_s_time") == 0)
                            | (pl.col("pt_s_time") <= pl.col("sl_s_time"))
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
            drop_candidates = ["open", "high", "low", "e1c_atr_13_M1"]
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

            # --- [追加] Duration（決済までの時間）の統計を計算 ---
            avg_duration_l = df["duration_long"].mean()
            med_duration_l = df["duration_long"].median()
            avg_duration_s = df["duration_short"].mean()
            med_duration_s = df["duration_short"].median()

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
| **Target Timeframe** | `{TARGET_TIMEFRAME}` |
| **ATR Filter** | `{TARGET_ATR_COLUMN} <= {ATR_SHRINK_THRESHOLD}` |
| **Long Rule** | `PT: {RULE_LONG["pt_mult"]}, SL: {RULE_LONG["sl_mult"]}, TD: {RULE_LONG["td"]}` |
| **Short Rule** | `PT: {RULE_SHORT["pt_mult"]}, SL: {RULE_SHORT["sl_mult"]}, TD: {RULE_SHORT["td"]}` |

### 2. Overall Performance & Event Duration
| Metric | Count | Win Rate | Avg Duration | Median Duration |
|:---|---:|---:|---:|---:|
| **Total Setups (M1 ATR shrink)** | `{total:,}` | - | - | - |
| **Long Profit-Take** | `{l_win:,}` | `{l_win / total:.2%}` | `{avg_duration_l:.1f} min` | `{med_duration_l:.1f} min` |
| **Short Profit-Take** | `{s_win:,}` | `{s_win / total:.2%}` | `{avg_duration_s:.1f} min` | `{med_duration_s:.1f} min` |

### 3. Top 10 Busiest Days (Setups)
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
