# /workspace/execution/extreme_risk_engine.py
"""
極限リスク管理エンジン 5.0 (Project Cimera V5仕様)
Two-Brainアーキテクチャ対応: AI推論を外部化し、純粋な資金管理と動的バリア計算に特化

[V5リファクタリング内容]
- 廃止: ケリー基準、モデルの内部ロード・直列推論、時間/曜日フィルター
- 新造: 資金比例固定ロット (Auto Lot) と Exness動的レバレッジ制限
- 新造: ATRベースの純粋なSL/TP計算
"""

import sys
import json
import logging
from pathlib import Path
from decimal import Decimal, getcontext
from typing import Dict, Any, Optional
from datetime import datetime

# --- 上位ディレクトリにある blueprint.py を読み込むための設定 ---
sys.path.append(str(Path(__file__).resolve().parents[1]))
import blueprint as config

# Decimal の精度を設定
getcontext().prec = 50

logger = logging.getLogger(__name__)


class ExtremeRiskEngineV5:
    """
    極限リスク管理エンジン 5.0 (Project Cimera V5仕様)
    外部から受け取った確率(p_long/p_short)と純粋な数式に基づく執行管理
    """

    def __init__(self, config_path: str = str(config.CONFIG_RISK)):
        """
        blueprint.py に定義された risk_config.json を読み込む
        """
        self.config = self._load_config(config_path)
        logger.info("ExtremeRiskEngineV5 (Two-Brain Mode) を初期化しました。")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """リスク管理設定を読み込む。なければV5デフォルトを返す。"""
        try:
            with open(config_path, "r") as f:
                loaded_config = json.load(f)
                logger.info(f"設定ファイル '{config_path}' を読み込みました。")
                return loaded_config
        except Exception as e:
            logger.warning(
                f"設定の読み込みに失敗しました。V5デフォルトを使用します: {e}"
            )
            # エラー時のフォールバックも新しいV5仕様に合わせておきます
            return {
                "base_capital": 10000.0,
                "lot_per_base": 0.1,
                "max_lot_absolute": 200.0,
                "contract_size": 100.0,
                "min_lot_size": 0.01,
                "base_leverage": 2000.0,
                "base_atr_sl_multiplier": 5.0,
                "base_atr_tp_multiplier": 1.0,
            }

    def _get_exness_leverage(self, equity: Decimal) -> Decimal:
        """Exnessの証拠金レベルに基づく動的レバレッジ制限"""
        if equity < Decimal("30000"):
            return Decimal("2000")
        elif equity < Decimal("100000"):
            return Decimal("1000")
        else:
            return Decimal("500")

    def calculate_auto_lot(
        self,
        equity: float,
        current_price: float,
        base_capital: Optional[float] = None,
        lot_per_base: Optional[float] = None,
        effective_leverage_cap: Optional[float] = None,
    ) -> float:
        """
        資金比例固定ロット (Auto Lot) を計算し、レバレッジ上限で安全にキャップする
        """
        eq_dec = Decimal(str(max(equity, 0.0)))
        price_dec = Decimal(str(current_price))
        b_cap_dec = Decimal(str(base_capital or self.config["base_capital"]))
        l_base_dec = Decimal(str(lot_per_base or self.config["lot_per_base"]))
        contract_dec = Decimal(str(self.config["contract_size"]))
        min_lot_dec = Decimal(str(self.config["min_lot_size"]))
        max_lot_abs = Decimal(str(self.config["max_lot_absolute"]))

        if eq_dec <= 0 or price_dec <= 0 or b_cap_dec <= 0:
            return 0.0

        # 1. 複利ベースの基本ロット算出: (equity / base_capital) * lot_per_base
        base_lot = (eq_dec / b_cap_dec) * l_base_dec

        # 2. レバレッジに基づく証拠金上限ロット算出
        if effective_leverage_cap is not None:
            leverage_dec = Decimal(str(effective_leverage_cap))
        else:
            leverage_dec = self._get_exness_leverage(eq_dec)

        # Max Lot = (Equity * Leverage) / (Price * Contract Size)
        max_lot_margin = (eq_dec * leverage_dec) / (price_dec * contract_dec)

        # 3. 最小値をとってキャップをかける (基本ロット vs 証拠金限界 vs 絶対上限200)
        final_lot = min(base_lot, max_lot_margin, max_lot_abs)

        # 4. 最小ロット単位(0.01)で切り捨てて丸める
        final_lot_quantized = float((final_lot // min_lot_dec) * min_lot_dec)

        logger.debug(
            f"Auto Lot計算: 基本={base_lot:.2f}, 証拠金上限({leverage_dec}倍)={max_lot_margin:.2f} -> 最終={final_lot_quantized:.2f}"
        )
        return final_lot_quantized

    def calculate_sl_tp(
        self,
        entry_price: float,
        atr: float,
        direction: int,
        sl_multiplier: float,
        tp_multiplier: float,
    ) -> Dict[str, float]:
        """
        ATR値を絶対の定規とした純粋なSL/TP計算
        direction: 1 (Long), -1 (Short)
        """
        if direction == 1:
            sl = entry_price - (atr * sl_multiplier)
            tp = entry_price + (atr * tp_multiplier)
        elif direction == -1:
            sl = entry_price + (atr * sl_multiplier)
            tp = entry_price - (atr * tp_multiplier)
        else:
            raise ValueError(
                f"無効な取引方向です: {direction} (1 または -1 を指定してください)"
            )

        # 価格精度に合わせて丸める (XAUUSD等を想定し小数点第3位まで)
        return {"stop_loss": round(sl, 3), "take_profit": round(tp, 3)}

    def generate_trade_command(
        self,
        action: str,
        p_long: float,
        p_short: float,
        current_price: float,
        atr: float,
        equity: float,
        sl_multiplier: float,
        tp_multiplier: float,
        base_capital: Optional[float] = None,
        lot_per_base: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        外部(司令塔)から受け取った確率とパラメータに基づき、最終的な執行コマンドを生成する
        """
        if action == "HOLD":
            return self._generate_hold_command("司令塔からのHOLD指示", p_long, p_short)

        direction = 1 if action == "BUY" else -1

        # Auto Lot計算
        lots = self.calculate_auto_lot(
            equity=equity,
            current_price=current_price,
            base_capital=base_capital,
            lot_per_base=lot_per_base,
        )

        if lots < self.config["min_lot_size"]:
            return self._generate_hold_command(
                "資金不足による最小ロット未達", p_long, p_short
            )

        # SL/TP計算
        sl_tp = self.calculate_sl_tp(
            entry_price=current_price,
            atr=atr,
            direction=direction,
            sl_multiplier=sl_multiplier,
            tp_multiplier=tp_multiplier,
        )

        trade_command = {
            "action": action,
            "lots": lots,
            "entry_price": current_price,
            "stop_loss": sl_tp["stop_loss"],
            "take_profit": sl_tp["take_profit"],
            "p_long": p_long,
            "p_short": p_short,
            "reason": f"TwoBrain Signal: {action} (p_long={p_long:.2f}, p_short={p_short:.2f})",
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✓ コマンド生成: {action} {lots}Lot @ {current_price:.3f} (SL: {sl_tp['stop_loss']:.3f}, TP: {sl_tp['take_profit']:.3f})"
        )
        return trade_command

    def _generate_hold_command(
        self, reason: str, p_long: float, p_short: float
    ) -> Dict[str, Any]:
        """HOLDコマンドの生成"""
        return {
            "action": "HOLD",
            "lots": 0.0,
            "entry_price": 0.0,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "p_long": p_long,
            "p_short": p_short,
            "reason": reason,
            "timestamp": datetime.now().isoformat(),
        }


if __name__ == "__main__":
    # 単体テスト用モック
    logging.basicConfig(level=logging.DEBUG)
    engine = ExtremeRiskEngineV5()

    # 残高5万ドルの場合のシミュレーション
    print(
        engine.generate_trade_command(
            action="BUY",
            p_long=0.85,
            p_short=0.15,
            current_price=2350.50,
            atr=5.2,
            equity=50000.0,
            sl_multiplier=2.0,
            tp_multiplier=3.0,
        )
    )
