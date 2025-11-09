# /workspace/execution/extreme_risk_engine.py
"""
極限リスク管理エンジン 2.0
ケリー基準、確率キャリブレーション、状態管理、市場レジーム適応を統合

[V4移植版]
- backtest_simulator_minimumlotset.py の最新ロット計算ロジックを移植
- 動的レバレッジ制限 (Exness仕様)
- JST時間帯ロット制限 (Exness仕様)
- 証拠金ベースのロット上限
- 最小ロット(0.01)への切り上げ処理
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import blueprint as config
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple, TypedDict
import logging
import joblib
from sklearn.calibration import CalibratedClassifierCV
from collections import deque
from decimal import Decimal, getcontext
import zoneinfo

# --- Decimal の精度を設定 (シミュレーターと一致) ---
getcontext().prec = 5000

# --- 定数 (シミュレーターから移植) ---
CONTRACT_SIZE = Decimal("100")  # 1 lot = 100 oz (XAUUSD)
JST = zoneinfo.ZoneInfo("Asia/Tokyo")  # JST時間帯


# 独自モジュール
# (state_manager, market_regime_detector は main.py から渡される)
from state_manager import StateManager, SystemState, Position, EventType
from market_regime_detector import MarketRegimeDetector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# 型定義
class MarketInfo(TypedDict):
    """
    市場情報の型定義 (V4戦略のS6データ項目を追加)
    """

    current_price: float
    atr: float
    predicted_time: Optional[float]
    volatility_ratio: Optional[float]  # GARCH予測 / 平均ATR
    # --- V4 (S6) データ ---
    sl_multiplier: float
    pt_multiplier: float
    direction: int
    payoff_ratio: float


class ExtremeRiskEngineV2:
    """
    極限リスク管理エンジン 2.0
    数学的原則に基づいた最適資本配分と動的市場適応
    """

    def __init__(
        self,
        config_path: str = str(config.CONFIG_RISK),
        state_manager: Optional[StateManager] = None,
        regime_detector: Optional[MarketRegimeDetector] = None,
        m1_model_path: Optional[str] = str(
            config.S7_M1_CALIBRATED
        ),  # デフォルトパス追加
        m2_model_path: Optional[str] = str(config.S7_M2_CALIBRATED),
    ):  # デフォルトパス追加
        """
        Args:
            config_path: リスク管理設定ファイル
            state_manager: 状態管理マネージャー
            regime_detector: 市場レジーム検知器
            m1_model_path: M1較正済みモデルのパス
            m2_model_path: M2較正済みモデルのパス
        """
        self.config = self._load_config(config_path)

        # 状態管理
        self.state_manager = state_manager or StateManager()

        # 市場レジーム検知
        self.regime_detector = regime_detector

        # 較正済みモデル
        self.m1_calibrated: Optional[CalibratedClassifierCV] = None
        self.m2_calibrated: Optional[CalibratedClassifierCV] = None

        # M1性能履歴（ローリング統計用）
        self.m1_precision_history: deque = deque(maxlen=20)
        self.m1_f1_history: deque = deque(maxlen=20)

        if m1_model_path:
            self.load_calibrated_model(m1_model_path, model_type="M1")
        if m2_model_path:
            self.load_calibrated_model(m2_model_path, model_type="M2")

        logger.info("ExtremeRiskEngineV2を初期化しました。")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """リスク管理設定を読み込む"""
        config_file = Path(config_path)

        if not config_file.exists():
            logger.warning(
                f"設定ファイル '{config_path}' が見つかりません。デフォルト設定を使用します。"
            )
            default_config = {
                "base_risk_percent": 0.02,
                "max_drawdown": 0.20,
                "drawdown_reduction_threshold": 0.15,
                "base_confidence_threshold": 0.60,
                "min_risk_reward_ratio": 2.0,
                "base_atr_sl_multiplier": 1.0,
                "base_atr_tp_multiplier": 2.0,
                "kelly_fraction": 0.5,  # ハーフケリー
                "max_risk_per_trade": 0.05,  # 1取引あたり最大5%
                "max_positions": 1,  # ★ 1ポジルール
                "contract_size": 100.0,  # (Decimal(CONTRACT_SIZE) と一致させる)
                "pip_value_per_lot": 10.0,  # (XAUUSDの1pip=$10)
                "pip_multiplier": 100.0,  # 価格差→pips変換係数（USD/JPY: 100, XAU/USD: 100, EUR/USD: 10000）
                "min_lot_size": 0.01,
                "base_leverage": 2000.0,
                "time_filters": {
                    "blocked_hours_before_news": 0.5,
                    "blocked_hours_after_news": 0.5,
                    "blocked_hours_weekend_close": 4,
                    "blocked_hours_weekend_open": 2,
                },
            }

            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, "w") as f:
                json.dump(default_config, f, indent=2)

            return default_config

        with open(config_file, "r") as f:
            return json.load(f)

    # ========== モデル管理 (変更なし) ==========

    def load_calibrated_model(self, model_path: str, model_type: str) -> None:
        """
        較正済みモデルを読み込む

        Args:
            model_path: モデルファイルパス
            model_type: 'M1' or 'M2'
        """
        try:
            model = joblib.load(model_path)

            if model_type == "M1":
                self.m1_calibrated = model
                logger.info(f"✓ M1較正済みモデル読み込み: {model_path}")
            elif model_type == "M2":
                self.m2_calibrated = model
                logger.info(f"✓ M2較正済みモデル読み込み: {model_path}")
            else:
                raise ValueError(f"無効なモデルタイプ: {model_type}")

        except Exception as e:
            logger.error(f"✗ モデル読み込み失敗 ({model_type}): {e}")

    def predict_with_confidence(
        self, features: np.ndarray, market_info: MarketInfo
    ) -> Tuple[float, float]:
        """
        較正済みモデルで予測と確信度を取得 (M2文脈特徴量を使用)

        Args:
            features: 基本特徴量ベクトル (S3_FEATURES_FOR_TRAINING)
            market_info: 市場情報 (V4文脈特徴量を含む)

        Returns:
            (M1予測確率, M2成功確率)
        """
        if self.m1_calibrated is None or self.m2_calibrated is None:
            raise ValueError("較正済みモデルが読み込まれていません。")

        # M1予測（利食い確率）
        p_m1 = self.m1_calibrated.predict_proba(features)[0, 1]

        # M2用拡張特徴量の構築 (model_training_metalabeling_C_contextplus.py と一致させる)
        # 1. 基本特徴量 (features[0])
        # 2. M1出力 (p_m1)
        # 3. 文脈特徴量 (market_info から取得)

        context_features_m2_names = [
            "hmm_prob_0",
            "hmm_prob_1",
            "atr",
            "e1a_statistical_kurtosis_50",
            "e1c_adx_21",
            "e2a_mfdfa_hurst_mean_1000",
            "e2a_kolmogorov_complexity_1000",
        ]

        extended_features_list = list(features[0])  # 1. 基本特徴量
        extended_features_list.append(p_m1)  # 2. M1出力

        # 3. 文脈特徴量
        try:
            for feature_name in context_features_m2_names:
                # market_info から文脈特徴量を取得
                # (main.py が S7_CONTEXT_FEATURES から取得して market_info に詰める前提)
                value = market_info.get(feature_name, 0.0)
                extended_features_list.append(value)
        except Exception as e:
            logger.error(f"M2文脈特徴量の構築に失敗: {e}。market_info: {market_info}")
            raise ValueError("M2文脈特徴量の構築エラー")

        # 配列に変換
        extended_features = np.array([extended_features_list])

        # M2予測（M1シグナルの成功確率）
        p_m2 = self.m2_calibrated.predict_proba(extended_features)[0, 1]

        return float(p_m1), float(p_m2)

    def update_m1_performance(self, precision: float, f1_score: float) -> None:
        """
        M1モデルの性能統計を更新

        Args:
            precision: 最新のプレシジョン
            f1_score: 最新のF1スコア
        """
        self.m1_precision_history.append(precision)
        self.m1_f1_history.append(f1_score)
        logger.debug(f"M1性能更新: Precision={precision:.4f}, F1={f1_score:.4f}")

    def calculate_volatility_adjustment(
        self, current_volatility: float, historical_avg: float
    ) -> float:
        """
        ボラティリティ比率による調整係数を計算（GARCH適応）
        (V4では未使用だが、将来の拡張のため残置)

        Args:
            current_volatility: 現在のボラティリティ（GARCH予測値）
            historical_avg: 過去平均ATR

        Returns:
            調整係数（0.7〜1.2）
        """
        if historical_avg <= 0:
            logger.warning("過去平均ATRが無効です。調整係数1.0を返します。")
            return 1.0

        volatility_ratio = current_volatility / historical_avg

        # 設定から閾値を取得（既存実装との互換性）
        vol_config = {
            "high_threshold": 1.5,
            "high_multiplier": 0.7,
            "low_threshold": 0.5,
            "low_multiplier": 1.2,
        }

        if volatility_ratio > vol_config["high_threshold"]:
            adjustment = vol_config["high_multiplier"]
            logger.info(
                f"高ボラティリティ検出（比率: {volatility_ratio:.2f}）。"
                f"ロット調整: {adjustment}"
            )
        elif volatility_ratio < vol_config["low_threshold"]:
            adjustment = vol_config["low_multiplier"]
            logger.info(
                f"低ボラティリティ検出（比率: {volatility_ratio:.2f}）。"
                f"ロット調整: {adjustment}"
            )
        else:
            adjustment = 1.0

        return adjustment

    def calculate_kelly_fraction(
        self, p_win: float, win_loss_ratio: float, kelly_fraction: float = 0.5
    ) -> float:
        """
        ケリー基準で最適な資本配分比率を計算 (変更なし)

        Args:
            p_win: 勝利確率（較正済み）
            win_loss_ratio: 勝敗比率（利食い幅 / 損切り幅）
            kelly_fraction: ケリー分数（0.5でハーフケリー、0.25でクォーターケリー）

        Returns:
            最適資本配分比率（0-1）
        """
        # ケリー基準の公式: f* = (b*p - q) / b
        # b = win_loss_ratio, p = p_win, q = 1 - p_win

        if win_loss_ratio <= 0 or p_win <= 0 or p_win >= 1:
            logger.warning("ケリー基準の計算に無効なパラメータ")
            return 0.0

        q_lose = 1.0 - p_win

        # ケリー基準
        kelly_f = (win_loss_ratio * p_win - q_lose) / win_loss_ratio

        # 負の値（期待値マイナス）の場合は0
        if kelly_f <= 0:
            logger.info(
                f"期待値マイナスのシグナル（Kelly={kelly_f:.4f}）。エントリー不可。"
            )
            return 0.0

        # 分数ケリー（リスク軽減）
        fractional_kelly = kelly_f * kelly_fraction

        logger.debug(
            f"Kelly計算: p_win={p_win:.4f}, b={win_loss_ratio:.2f}, "
            f"f*={kelly_f:.4f}, fractional={fractional_kelly:.4f}"
        )

        return fractional_kelly

    # --- ▼▼▼ [V4移植] シミュレーターからのヘルパー関数 ▼▼▼ ---
    def _get_effective_leverage(self, equity: Decimal) -> Decimal:
        """有効証拠金に基づいてExnessのレバレッジ制限を適用"""
        base_leverage_dec = Decimal(str(self.config.get("base_leverage", 2000.0)))
        # Exnessの証拠金レベル (USD)
        if equity < Decimal("5000"):
            limit_leverage = base_leverage_dec  # ベース設定を使用
        elif equity < Decimal("30000"):
            limit_leverage = Decimal("2000")  # 最大2000倍
        elif equity < Decimal("100000"):
            limit_leverage = Decimal("1000")  # 最大1000倍
        else:
            limit_leverage = Decimal("500")  # 最大500倍

        # 設定した基本レバレッジと証拠金による上限のうち、小さい方を適用
        return base_leverage_dec.min(limit_leverage)

    def _get_max_lot_allowed(self, timestamp_utc: datetime) -> Decimal:
        """JST時間帯に基づいてExnessの最大ロット数を返す"""
        timestamp_jst = timestamp_utc.astimezone(JST)
        hour_jst = timestamp_jst.hour

        # 日本時間 午前6:00 ～ 午後3:59 (15:59) -> 20ロット
        if 6 <= hour_jst < 16:
            return Decimal("20")
        # 日本時間 午後4:00 (16:00) ～ 午前5:59 -> 200ロット
        else:
            return Decimal("200")

    # --- ▲▲▲ [V4移植] ヘルパー関数ここまで ▲▲▲ ---

    # --- ▼▼▼ [V4移植] シミュレーターのロット計算ロジックに完全置換 ▼▼▼ ---
    def calculate_position_size_kelly(
        self,
        account_balance: float,
        p_m2: float,
        market_info: MarketInfo,
        current_drawdown: float,
        timestamp_utc: datetime,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        ケリー基準でポジションサイズを計算 (V4シミュレーターロジック)

        Args:
            account_balance: 口座残高 (float)
            p_m2: M2の成功確率（較正済み, float）
            market_info: 市場情報 (V4 S6データを含む)
            current_drawdown: 現在のドローダウン（0-1）
            timestamp_utc: 現在時刻 (datetime)

        Returns:
            (ロット数, 計算詳細)
        """
        # --- 0. 定数とDecimal型への変換 ---
        current_capital = Decimal(str(account_balance))
        DECIMAL_ZERO = Decimal("0.0")
        DECIMAL_ONE = Decimal("1.0")
        DECIMAL_MAX_RISK = Decimal(str(self.config["max_risk_per_trade"]))
        DECIMAL_KELLY_FRACTION = Decimal(str(self.config["kelly_fraction"]))
        DECIMAL_MIN_LOT_SIZE = Decimal(str(self.config["min_lot_size"]))
        DECIMAL_CONTRACT_SIZE = Decimal(str(self.config["contract_size"]))

        # ドローダウンチェック
        if current_drawdown >= self.config["max_drawdown"]:
            logger.warning(
                f"最大ドローダウン超過（{current_drawdown:.2%}）。エントリー不可。"
            )
            return 0.0, {"reason": "max_drawdown_exceeded"}

        if current_drawdown >= self.config["drawdown_reduction_threshold"]:
            DECIMAL_KELLY_FRACTION *= Decimal("0.5")
            logger.info(
                f"ドローダウン警戒レベル（{current_drawdown:.2%}）。Kelly分数を50%削減。"
            )

        # --- 1. ケリー推奨のベット割合を計算 ---
        p_decimal = Decimal(str(p_m2))
        win_loss_ratio_float = market_info["payoff_ratio"]
        b = Decimal(str(win_loss_ratio_float))

        # 既存の `calculate_kelly_fraction` を呼び出す
        kelly_f_fractional = self.calculate_kelly_fraction(
            p_m2, win_loss_ratio_float, float(DECIMAL_KELLY_FRACTION)
        )
        kelly_f_fractional_decimal = Decimal(str(kelly_f_fractional))

        if kelly_f_fractional_decimal <= DECIMAL_ZERO:
            return 0.0, {"reason": "negative_expected_value (Kelly <= 0)"}

        # --- 2. リスク額を決定 (最大リスク制約適用) ---
        effective_bet_fraction = kelly_f_fractional_decimal.min(DECIMAL_MAX_RISK)
        risk_amount_decimal = current_capital * effective_bet_fraction

        # --- 3. リスクベースの希望ロットを計算 ---
        atr_value_float = market_info["atr"]
        sl_multiplier_float = market_info["sl_multiplier"]

        if atr_value_float <= 0 or sl_multiplier_float <= 0:
            logger.warning(
                f"Invalid ATR ({atr_value_float}) or SL mult ({sl_multiplier_float}) at {timestamp_utc}, cannot calculate lot size."
            )
            return 0.0, {"reason": "invalid_atr_or_sl_multiplier"}

        # 3a. 動的SL幅 (価格単位)
        dynamic_sl_PRICE_decimal = Decimal(str(atr_value_float)) * Decimal(
            str(sl_multiplier_float)
        )

        # 3b. 1ロットあたりのストップロス価値（通貨）
        # (価格単位SL * 契約サイズ)
        stop_loss_currency_per_lot = dynamic_sl_PRICE_decimal * DECIMAL_CONTRACT_SIZE

        # 3c. リスクベースの希望ロットサイズ
        if stop_loss_currency_per_lot > DECIMAL_ZERO:
            desired_lot_size_decimal = risk_amount_decimal / stop_loss_currency_per_lot
        else:
            desired_lot_size_decimal = DECIMAL_ZERO

        if desired_lot_size_decimal <= DECIMAL_ZERO:
            return 0.0, {"reason": "desired_lot_size_is_zero"}

        # --- 4. 3つの制約を適用 ---

        # 4a. 実効レバレッジ
        effective_leverage_decimal = self._get_effective_leverage(current_capital)

        # 4b. 証拠金ベースの最大許容ロット
        current_price_decimal = Decimal(str(market_info["current_price"]))
        if (
            effective_leverage_decimal > DECIMAL_ZERO
            and current_price_decimal > DECIMAL_ZERO
        ):
            max_lot_by_margin = (current_capital * effective_leverage_decimal) / (
                current_price_decimal * DECIMAL_CONTRACT_SIZE
            )
            max_lot_by_margin = max_lot_by_margin.max(DECIMAL_ZERO)
        else:
            max_lot_by_margin = DECIMAL_ZERO

        # 4c. 時間帯ベースの最大許容ロット
        max_lot_allowed_by_broker = self._get_max_lot_allowed(timestamp_utc)

        # --- 5. 最終ロットサイズを決定 ---
        final_lot_size_decimal = desired_lot_size_decimal.min(max_lot_by_margin).min(
            max_lot_allowed_by_broker
        )
        final_lot_size_decimal = final_lot_size_decimal.max(DECIMAL_ZERO)

        # --- 6. 最小ロット(0.01)の切り上げチェック ---
        if (final_lot_size_decimal > DECIMAL_ZERO) and (
            final_lot_size_decimal < DECIMAL_MIN_LOT_SIZE
        ):
            final_lot_size_decimal = DECIMAL_MIN_LOT_SIZE

        # 0.01単位に丸める (MT5仕様)
        final_lot_size_float = float(final_lot_size_decimal.quantize(Decimal("0.01")))

        if final_lot_size_float < 0.01:
            logger.info(f"最終ロットサイズが最小単位未満: {final_lot_size_float:.4f}")
            return 0.0, {"reason": "lots_below_minimum_after_constraints"}

        details = {
            "kelly_f_fractional": float(kelly_f_fractional_decimal),
            "effective_bet_fraction": float(effective_bet_fraction),
            "risk_amount": float(risk_amount_decimal),
            "win_loss_ratio": float(b),
            "stop_loss_currency_per_lot": float(stop_loss_currency_per_lot),
            "desired_lot_size": float(desired_lot_size_decimal),
            "max_lot_by_margin": float(max_lot_by_margin),
            "max_lot_by_broker": float(max_lot_allowed_by_broker),
            "final_lot_size": final_lot_size_float,
            "reason": "success",
        }

        logger.info(
            f"V4 Kelly最適ロット: {final_lot_size_float:.2f} (リスク: {effective_bet_fraction:.2%}, "
            f"希望: {desired_lot_size_decimal:.2f}, 証拠金上限: {max_lot_by_margin:.2f}, ブローカー上限: {max_lot_allowed_by_broker:.2f})"
        )

        return final_lot_size_float, details

    # --- ▲▲▲ [V4移植] ロット計算置換ここまで ▲▲▲ ---

    # --- ▼▼▼ [V4対応] シグネチャ変更: sl/tp multiplier を引数で受け取れるようにする ▼▼▼ ---
    def calculate_sl_tp(
        self,
        entry_price: float,
        atr: float,
        direction: str,
        sl_multiplier: Optional[float] = None,
        tp_multiplier: Optional[float] = None,
        regime_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        損切り・利食いラインを計算 (V4対応)

        Args:
            entry_price: エントリー価格
            atr: 現在のATR値
            direction: 取引方向（'BUY' or 'SELL'）
            sl_multiplier: [V4] S6から渡されるSL係数
            tp_multiplier: [V4] S6から渡されるTP係数
            regime_params: 市場レジーム別パラメータ (V4ではフォールバックとしてのみ使用)

        Returns:
            {'stop_loss': float, 'take_profit': float}
        """
        # 1. V4戦略のmultiplier (S6由来) を最優先
        sl_mult = sl_multiplier
        tp_mult = tp_multiplier

        # 2. V4 multiplier が無い場合、レジームパラメータを使用
        if sl_mult is None and regime_params:
            sl_mult = regime_params["atr_multiplier_sl"]
        if tp_mult is None and regime_params:
            tp_mult = regime_params["atr_multiplier_tp"]

        # 3. それも無い場合、configのデフォルト値を使用
        if sl_mult is None:
            sl_mult = self.config["base_atr_sl_multiplier"]
        if tp_mult is None:
            tp_mult = self.config["base_atr_tp_multiplier"]

        # 4. 予測時間に基づく動的調整 (V4では使用しないが、ロジックは残す)
        # if predicted_time is not None:
        #     if predicted_time < 30:
        #         tp_mult *= 1.25
        #     elif predicted_time > 90:
        #         tp_mult *= 0.75

        # 5. 計算
        if direction == 1:  # 'BUY' (S6のdirectionは 1)
            stop_loss = entry_price - sl_mult * atr
            take_profit = entry_price + tp_mult * atr
        elif direction == -1:  # 'SELL' (S6のdirectionは -1)
            stop_loss = entry_price + sl_mult * atr
            take_profit = entry_price - tp_mult * atr
        else:
            raise ValueError(
                f"無効な取引方向: {direction} (1 or -1 である必要があります)"
            )

        # MT5/ブローカーの価格精度に合わせて丸める (例: XAUUSDは小数点以下2桁または3桁)
        # ここでは 3桁 に丸めておく
        return {"stop_loss": round(stop_loss, 3), "take_profit": round(take_profit, 3)}

    # --- ▲▲▲ [V4対応] シグネチャ変更ここまで ▲▲▲ ---

    # ========== エントリー条件チェック (変更なし) ==========

    def check_entry_conditions(
        self,
        p_m2: float,
        current_positions: int,
        current_time: Optional[datetime] = None,
        news_times: Optional[List[datetime]] = None,
        regime_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        エントリー条件をチェック

        Args:
            p_m2: M2成功確率
            current_positions: 現在の保有ポジション数
            current_time: 現在時刻
            news_times: 経済指標発表時刻リスト
            regime_params: 市場レジーム別パラメータ

        Returns:
            {'allowed': bool, 'reason': str}
        """
        # 確信度チェック（レジーム適応）
        if regime_params:
            threshold = regime_params["confidence_threshold"]
        else:
            threshold = self.config["base_confidence_threshold"]

        if p_m2 < threshold:
            return {
                "allowed": False,
                "reason": f"確信度不足（{p_m2:.2%} < {threshold:.2%}）",
            }

        # ポジション数チェック
        if current_positions >= self.config["max_positions"]:
            return {
                "allowed": False,
                "reason": f"最大ポジション数到達（{current_positions}/{self.config['max_positions']}）",
            }

        # 時間帯フィルター
        if current_time is not None:
            # ニュース前後
            if news_times is not None:
                for news_time in news_times:
                    time_diff = abs((current_time - news_time).total_seconds() / 3600)
                    blocked_hours = self.config["time_filters"][
                        "blocked_hours_before_news"
                    ]
                    if time_diff < blocked_hours:
                        return {
                            "allowed": False,
                            "reason": f"経済指標発表前後の取引禁止時間帯（{time_diff:.1f}時間前）",
                        }

            # 週末前後
            if current_time.weekday() == 4 and current_time.hour >= 21:
                return {"allowed": False, "reason": "週末クローズ前の取引禁止時間帯"}

            if current_time.weekday() == 0 and current_time.hour < 9:
                return {"allowed": False, "reason": "週明けオープン後の取引禁止時間帯"}

        return {"allowed": True, "reason": "すべての条件をクリア"}

    # ========== 統合取引コマンド生成 (V4ロジック呼び出し) ==========

    def generate_trade_command(
        self,
        features: np.ndarray,
        market_info: MarketInfo,
        market_data_for_regime: Optional[Any] = None,
        current_time: Optional[datetime] = None,
        news_times: Optional[List[datetime]] = None,
    ) -> Dict[str, Any]:
        """
        取引コマンドを生成（統合メイン関数）

        Args:
            features: AI予測用の基本特徴量
            market_info: 市場情報 (V4の S6データを含む)
            market_data_for_regime: 市場レジーム検知用のデータ（pd.DataFrame）
            current_time: 現在時刻 (UTC)
            news_times: 経済指標発表時刻リスト

        Returns:
            取引コマンド辞書
        """
        if current_time is None:
            current_time = datetime.now(datetime.timezone.utc)

        # 状態の取得
        if self.state_manager.current_state is None:
            logger.error("システム状態が初期化されていません。")
            return self._generate_hold_command("state_not_initialized")

        state = self.state_manager.current_state

        # 市場レジームの検知
        regime_params = None
        if self.regime_detector and market_data_for_regime is not None:
            try:
                # (V4では主にM2モデルに文脈が組み込まれたが、
                #  念のため旧レジームロジックも残し、確信度閾値などに使用する)
                regime_info = self.regime_detector.detect_current_regime(
                    market_data_for_regime
                )
                regime_params = regime_info["risk_params"]
                logger.info(f"現在の市場レジーム: {regime_info['regime_name']}")
            except Exception as e:
                logger.warning(f"レジーム検知失敗: {e}。デフォルトパラメータを使用。")

        # AI予測（較正済み確率）
        try:
            p_m1, p_m2 = self.predict_with_confidence(features, market_info)
            logger.info(f"AI予測: P(M1)={p_m1:.4f}, P(M2)={p_m2:.4f}")
        except Exception as e:
            logger.error(f"AI予測失敗: {e}")
            return self._generate_hold_command("prediction_failed")

        # エントリー条件チェック
        entry_check = self.check_entry_conditions(
            p_m2=p_m2,
            current_positions=len(state.open_positions),
            current_time=current_time,
            news_times=news_times,
            regime_params=regime_params,
        )

        if not entry_check["allowed"]:
            logger.info(f"エントリー不可: {entry_check['reason']}")
            return self._generate_hold_command(entry_check["reason"])

        # 取引方向の決定 (V4: S6の 'direction' を使用)
        direction_int = market_info["direction"]
        direction_str = "BUY" if direction_int == 1 else "SELL"

        # SL/TPの計算 (V4: S6の 'atr', 'sl_multiplier', 'pt_multiplier' を使用)
        sl_tp = self.calculate_sl_tp(
            entry_price=market_info["current_price"],
            atr=market_info["atr"],
            direction=direction_int,
            sl_multiplier=market_info["sl_multiplier"],
            tp_multiplier=market_info["pt_multiplier"],
            regime_params=regime_params,  # フォールバック
        )

        # ポジションサイズの計算（V4: ケリー基準）
        lots, calc_details = self.calculate_position_size_kelly(
            account_balance=state.current_balance,
            p_m2=p_m2,
            market_info=market_info,  # V4: S6データ(atr, multipliers, payoff_ratio)を渡す
            current_drawdown=state.current_drawdown,
            timestamp_utc=current_time,  # V4: 時間帯制限のため
        )

        if lots <= 0:
            logger.info(
                f"ポジションサイズ計算でエントリー不可: {calc_details.get('reason')}"
            )
            return self._generate_hold_command(
                calc_details.get("reason", "zero_position_size")
            )

        # 取引コマンドの生成
        trade_command = {
            "action": direction_str,
            "lots": lots,
            "entry_price": market_info["current_price"],
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "confidence_m2": p_m2,
            "confidence_m1": p_m1,
            "predicted_time": market_info.get("predicted_time", 0),
            "reason": (
                f"M2確信度{p_m2:.2%}, V4 Kelly={calc_details['kelly_f_fractional']:.2%}, "
                f"リスク{calc_details['effective_bet_fraction']:.2%}, {direction_str}シグナル"
            ),
            "risk_amount": calc_details.get("risk_amount", 0.0),
            "win_loss_ratio": calc_details.get("win_loss_ratio", 0.0),
            "kelly_fraction": calc_details.get("kelly_f_fractional", 0.0),
            "volatility_adjustment": calc_details.get("volatility_adjustment", 1.0),
            "timestamp": (
                current_time.isoformat() if current_time else datetime.now().isoformat()
            ),
        }

        logger.info(
            f"✓ 取引コマンド生成: {direction_str} {lots}ロット @ {market_info['current_price']:.3f}"
        )
        logger.info(f"  SL: {sl_tp['stop_loss']:.3f}, TP: {sl_tp['take_profit']:.3f}")
        logger.info(
            f"  M2確信度: {p_m2:.2%}, Kelly(V4): {calc_details['kelly_f_fractional']:.2%}"
        )

        # イベント記録
        if self.state_manager.use_event_sourcing:
            self.state_manager.append_event(
                EventType.TRADE_SIGNAL_SENT, {"command": trade_command}
            )

        return trade_command

    def _generate_hold_command(self, reason: str) -> Dict[str, Any]:
        """HOLDコマンドを生成"""
        return {
            "action": "HOLD",
            "lots": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "confidence_m2": 0.0,
            "confidence_m1": 0.0,
            "predicted_time": 0,
            "reason": reason,
            "risk_amount": 0.0,
            "timestamp": datetime.now().isoformat(),
        }


# 使用例 (変更なし)
if __name__ == "__main__":
    import pandas as pd

    # 状態管理の初期化
    state_manager = StateManager(use_event_sourcing=True)

    # 初期状態の設定
    initial_state = SystemState(
        timestamp=datetime.now().isoformat(),
        current_equity=1000000.0,
        current_balance=1000000.0,
        current_drawdown=0.0,
        open_positions={},
        m1_rolling_precision=[],
        m2_rolling_auc=[],
        recent_trades_count=0,
    )
    state_manager.current_state = initial_state
    state_manager.save_checkpoint(initial_state)

    # リスクエンジンの初期化
    engine = ExtremeRiskEngineV2(state_manager=state_manager)

    # サンプルデータ
    sample_features = np.random.randn(1, 50)  # ダミー特徴量

    # [V4対応] MarketInfo に S6由来のデータを追加
    market_info_typed: MarketInfo = {
        "current_price": 2350.50,
        "atr": 1.85,  # V4 (R4) では 5.0 以上だが、テストのため仮の値
        "predicted_time": 45.0,
        "volatility_ratio": 1.2,
        # --- V4 (S6) データ ---
        "sl_multiplier": 5.0,  # (R4ルール)
        "pt_multiplier": 1.0,  # (R4ルール)
        "direction": 1,  # (BUY)
        "payoff_ratio": 1.0 / 5.0,  # (R4ルール: pt/sl)
        # --- V4 (M2文脈) データ ---
        "hmm_prob_0": 0.8,
        "hmm_prob_1": 0.2,
        "e1a_statistical_kurtosis_50": 3.5,
        "e1c_adx_21": 22.0,
        "e2a_mfdfa_hurst_mean_1000": 0.55,
        "e2a_kolmogorov_complexity_1000": 0.9,
    }

    # 注: 実際の使用では較正済みモデルとレジーム検知器が必要
    print("=" * 60)
    print("取引コマンド生成テスト (V4ロジック)")
    print("=" * 60)
    print("\n注: このテストは較正済みモデルなしで実行されます。")
    print("実運用では M1/M2 較正済みモデルが必須です。\n")

    # モデルなしではHOLDになる
    command = engine.generate_trade_command(
        features=sample_features,
        market_info=market_info_typed,
        current_time=datetime.now(datetime.timezone.utc),
    )

    print(f"\n生成されたコマンド:")
    print(json.dumps(command, indent=2, ensure_ascii=False))

    print("\n✓ テスト完了")
