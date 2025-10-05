"""
Project Forge 統合実行スクリプト
全コンポーネントを統合し、リアルタイム取引システムを起動
"""
import blueprint as config
import numpy as np
import pandas as pd
import time
import signal
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import logging
import joblib

# プロジェクトモジュール
from state_manager import StateManager, SystemState, EventType
from market_regime_detector import MarketRegimeDetector
from extreme_risk_engine import ExtremeRiskEngineV2, MarketInfo
from mql5_bridge_publisher import MQL5BridgePublisherV2, BridgeConfig, FileBasedBridge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/forge_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ForgeSystem:
    """
    Project Forge 統合システム
    全コンポーネントを統合し、リアルタイム取引を実行
    """
    
    def __init__(self, config_path: str = str(config.CONFIG_SYSTEM)):
        """
        Args:
            config_path: システム設定ファイル
        """
        self.config_path = Path(config_path)
        self.is_running = False

        # コンポーネント
        self.state_manager: Optional[StateManager] = None
        self.regime_detector: Optional[MarketRegimeDetector] = None
        self.risk_engine: Optional[ExtremeRiskEngineV2] = None
        self.bridge: Optional[MQL5BridgePublisherV2] = None
        self.fallback_bridge: Optional[FileBasedBridge] = None

        # 較正済みモデル
        self.m1_calibrated: Optional[Any] = None
        self.m2_calibrated: Optional[Any] = None

        logger.info("ForgeSystemを初期化しました。")
    
    def initialize(self) -> bool:
        """
        システムの初期化と起動準備

        Returns:
            初期化成功の場合True
        """
        try:
            logger.info("=" * 60)
            logger.info("Project Forge システム初期化開始")
            logger.info("=" * 60)

            # 1. 状態管理マネージャーの初期化
            logger.info("\n[1/5] 状態管理マネージャーを初期化中...")
            self.state_manager = StateManager(
                checkpoint_dir=str(config.STATE_CHECKPOINT_DIR),
                event_log_path=str(config.STATE_EVENT_LOG),
                use_event_sourcing=True
            )

            # チェックポイントから状態を復元
            loaded_state = self.state_manager.load_checkpoint()

            if loaded_state is None:
                logger.info("チェックポイントが見つかりません。新規状態を作成します。")
                initial_state = SystemState(
                    timestamp=datetime.now().isoformat(),
                    current_equity=1000000.0,  # 初期残高: 100万円
                    current_balance=1000000.0,
                    current_drawdown=0.0,
                    open_positions={},
                    m1_rolling_precision=[],
                    m2_rolling_auc=[],
                    recent_trades_count=0
                )
                self.state_manager.current_state = initial_state
                self.state_manager.save_checkpoint(initial_state)

            logger.info("✓ 状態管理マネージャー初期化完了")

            # 2. 市場レジーム検知器の初期化
            logger.info("\n[2/5] 市場レジーム検知器を初期化中...")
            regime_model_path = config.S7_REGIME_MODEL

            if regime_model_path.exists():
                self.regime_detector = MarketRegimeDetector(
                    method='hmm',
                    n_regimes=4,
                    model_path=str(regime_model_path)
                )
                logger.info("✓ 市場レジーム検知器初期化完了（学習済みモデル使用）")
            else:
                logger.warning("市場レジーム検知器モデルが見つかりません。")
                logger.warning("レジーム適応機能は無効化されます。")
                self.regime_detector = None

            # 3. 較正済みモデルの読み込み
            logger.info("\n[3/5] 較正済みAIモデルを読み込み中...")
            m1_path = config.S7_M1_CALIBRATED
            m2_path = config.S7_M2_CALIBRATED

            if m1_path.exists() and m2_path.exists():
                self.m1_calibrated = joblib.load(m1_path)
                self.m2_calibrated = joblib.load(m2_path)
                logger.info("✓ 較正済みモデル読み込み完了")
            else:
                logger.error("較正済みモデルが見つかりません。")
                logger.error("システムを起動できません。")
                return False

            # 4. リスク管理エンジンの初期化
            logger.info("\n[4/5] リスク管理エンジンを初期化中...")
            self.risk_engine = ExtremeRiskEngineV2(
                config_path=str(config.CONFIG_RISK),
                state_manager=self.state_manager,
                regime_detector=self.regime_detector,
                m1_model_path=str(m1_path),
                m2_model_path=str(m2_path)
            )
            logger.info("✓ リスク管理エンジン初期化完了")

            # 5. MQL5ブリッジの初期化
            logger.info("\n[5/5] MQL5ブリッジを初期化中...")
            bridge_config_obj = BridgeConfig(
                trade_endpoint=config.ZMQ["trade_endpoint"],
                heartbeat_endpoint=config.ZMQ["heartbeat_endpoint"],
                log_dir=str(config.LOGS_ZMQ_BRIDGE)
            )

            self.bridge = MQL5BridgePublisherV2(config=bridge_config_obj)

            # コールバック設定
            self.bridge.on_connection_lost = self._on_connection_lost
            self.bridge.on_connection_restored = self._on_connection_restored

            if self.bridge.connect():
                logger.info("✓ MQL5ブリッジ初期化完了")
            else:
                logger.warning("MQL5ブリッジの接続に失敗しました。")
                logger.warning("フォールバックモードで継続します。")

            # フォールバックブリッジ
            self.fallback_bridge = FileBasedBridge()

            # ブローカー整合性検証
            logger.info("\n[整合性検証] ブローカー状態を確認中...")
            self._perform_broker_reconciliation()

            logger.info("\n" + "=" * 60)
            logger.info("✓ Project Forge システム初期化完了")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"✗ システム初期化失敗: {e}", exc_info=True)
            return False
    
    def _perform_broker_reconciliation(self) -> None:
        """ブローカー整合性検証を実行"""
        if self.bridge and self.bridge.is_connected:
            broker_state = self.bridge.request_broker_state()
            
            if broker_state:
                reconciliation = self.state_manager.reconcile_with_broker(broker_state)
                
                if not reconciliation['is_consistent']:
                    logger.warning("ブローカー状態との不整合を検出しました。")
                    for disc in reconciliation['discrepancies']:
                        logger.warning(f"  - {disc}")
                    logger.info("ローカル状態をブローカーに同期しました。")
            else:
                logger.warning("ブローカー状態を取得できませんでした。")
        else:
            logger.warning("MQL5ブリッジが接続されていません。整合性検証をスキップします。")
    
    def _on_connection_lost(self) -> None:
        """接続喪失時のコールバック"""
        logger.error("⚠ MQL5 EAとの接続が失われました。")
        logger.info("システムは停止し、接続復旧を待ちます。")
        
        # イベント記録
        self.state_manager.append_event(
            EventType.SYSTEM_STOPPED,
            {'reason': 'connection_lost'}
        )
    
    def _on_connection_restored(self) -> None:
        """接続復旧時のコールバック"""
        logger.info("✓ MQL5 EAとの接続が復旧しました。")
        
        # ブローカー整合性検証を再実行
        self._perform_broker_reconciliation()
        
        # イベント記録
        self.state_manager.append_event(
            EventType.SYSTEM_STARTED,
            {'reason': 'connection_restored'}
        )
    
    def run_trading_loop(self, 
                        interval_seconds: int = 60,
                        demo_mode: bool = True) -> None:
        """
        取引ループを実行
        
        Args:
            interval_seconds: シグナル評価間隔（秒）
            demo_mode: デモモード（Trueの場合は実際の発注なし）
        """
        logger.info("\n" + "=" * 60)
        logger.info("取引ループ開始")
        logger.info("=" * 60)
        logger.info(f"評価間隔: {interval_seconds}秒")
        logger.info(f"デモモード: {'有効' if demo_mode else '無効'}")
        logger.info("Ctrl+Cで停止")
        
        self.is_running = True
        iteration = 0
        
        try:
            while self.is_running:
                iteration += 1
                logger.info(f"\n{'=' * 60}")
                logger.info(f"イテレーション #{iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'=' * 60}")
                
                # 1. 市場データの取得（実装: 実際にはTickデータストリームやAPIから取得）
                market_data = self._fetch_market_data()
                
                if market_data is None:
                    logger.warning("市場データの取得に失敗しました。次のイテレーションまで待機します。")
                    time.sleep(interval_seconds)
                    continue
                
                # 2. 特徴量の抽出（実装: 実際には特徴量生成パイプラインを実行）
                features = self._extract_features(market_data)
                
                # 3. 市場情報の構築（型安全）
                market_info_typed: MarketInfo = {
                    'current_price': market_data['close'],
                    'atr': market_data['atr'],
                    'predicted_time': None,  # AI予測から取得
                    'volatility_ratio': market_data.get('volatility_ratio', 1.0)
                }
                
                # 4. 取引コマンドの生成
                trade_command = self.risk_engine.generate_trade_command(
                    features=features,
                    market_info=market_info_typed,
                    market_data_for_regime=market_data.get('history'),
                    current_time=datetime.now(),
                    news_times=None  # 経済指標カレンダーから取得
                )
                
                logger.info(f"\n生成された取引コマンド: {trade_command['action']}")
                
                # 4. 取引コマンドの送信
                if trade_command['action'] != 'HOLD':
                    if not demo_mode:
                        self._send_trade_command(trade_command)
                    else:
                        logger.info("【デモモード】実際の発注はスキップされました。")
                        logger.info(f"  アクション: {trade_command['action']}")
                        logger.info(f"  ロット: {trade_command['lots']:.2f}")
                        logger.info(f"  エントリー: {trade_command['entry_price']:.2f}")
                        logger.info(f"  SL: {trade_command['stop_loss']:.2f}")
                        logger.info(f"  TP: {trade_command['take_profit']:.2f}")
                        logger.info(f"  確信度: {trade_command['confidence_m2']:.2%}")
                logger.info(f"  理由: {trade_command['reason']}")
                
                # 5. 状態の更新とチェックポイント保存
                if iteration % 10 == 0:  # 10イテレーションごとに保存
                    self.state_manager.save_checkpoint(self.state_manager.current_state)
                
                # 6. 統計情報の表示
                if iteration % 5 == 0:
                    self._display_statistics()
                
                # 次のイテレーションまで待機
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("\n\n✓ 停止シグナルを受信しました。")
            self.shutdown()
        
        except Exception as e:
            logger.error(f"\n✗ 取引ループエラー: {e}", exc_info=True)
            self.shutdown()
    
    def _fetch_market_data(self) -> Optional[Dict[str, Any]]:
        """
        市場データを取得（ダミー実装）
        実際にはTickデータストリーム、MT5 API、または外部APIから取得
        """
        # ダミーデータ
        return {
            'close': 150.25,
            'atr': 0.85,
            'volatility_ratio': 1.2,  # GARCH予測 / 平均ATR
            'history': pd.DataFrame({
                'close': np.random.randn(100) * 0.5 + 150,
                'high': np.random.randn(100) * 0.5 + 150.5,
                'low': np.random.randn(100) * 0.5 + 149.5,
                'open': np.random.randn(100) * 0.5 + 150,
                'volume': np.random.randint(100000, 500000, 100)
            })
        }
    
    def _extract_features(self, market_data: Dict[str, Any]) -> np.ndarray:
        """
        特徴量を抽出（ダミー実装）
        実際には第1章・第2章の特徴量生成パイプラインを実行
        """
        # ダミー特徴量（50次元）
        return np.random.randn(1, 50)
    
    def _send_trade_command(self, trade_command: Dict[str, Any]) -> bool:
        """取引コマンドを送信"""
        success = False
        
        # ZeroMQで送信
        if self.bridge and self.bridge.is_connected:
            success = self.bridge.send_trade_command(trade_command)
        
        # 失敗時はフォールバック
        if not success:
            logger.warning("ZeroMQ送信失敗。フォールバックモードに切り替えます。")
            success = self.fallback_bridge.write_command(trade_command)
        
        # イベント記録
        if success:
            self.state_manager.append_event(
                EventType.TRADE_SIGNAL_SENT,
                {'command': trade_command}
            )
        
        return success
    
    def _display_statistics(self) -> None:
        """統計情報を表示"""
        logger.info("\n" + "-" * 60)
        logger.info("システム統計")
        logger.info("-" * 60)
        
        # 状態情報
        if self.state_manager.current_state:
            state = self.state_manager.current_state
            logger.info(f"エクイティ: {state.current_equity:,.2f} 円")
            logger.info(f"残高: {state.current_balance:,.2f} 円")
            logger.info(f"ドローダウン: {state.current_drawdown:.2%}")
            logger.info(f"保有ポジション: {len(state.open_positions)}個")
            logger.info(f"総取引回数: {state.recent_trades_count}回")
        
        # 通信統計
        if self.bridge:
            stats = self.bridge.get_statistics()
            logger.info(f"\n通信統計:")
            logger.info(f"  送信成功: {stats['messages_acked']}回")
            logger.info(f"  送信失敗: {stats['messages_failed']}回")
            logger.info(f"  成功率: {stats['success_rate']:.2%}")
            logger.info(f"  ハートビート受信: {stats['heartbeats_received']}回")
        
        logger.info("-" * 60)
    
    def shutdown(self) -> None:
        """システムのシャットダウン"""
        logger.info("\n" + "=" * 60)
        logger.info("システムシャットダウン開始")
        logger.info("=" * 60)
        
        self.is_running = False
        
        # 最終状態の保存
        if self.state_manager and self.state_manager.current_state:
            logger.info("最終状態を保存中...")
            self.state_manager.save_checkpoint(self.state_manager.current_state)
            self.state_manager.append_event(
                EventType.SYSTEM_STOPPED,
                {'reason': 'manual_shutdown'}
            )
        
        # ブリッジの切断
        if self.bridge:
            logger.info("MQL5ブリッジを切断中...")
            self.bridge.disconnect()
        
        logger.info("=" * 60)
        logger.info("✓ システムシャットダウン完了")
        logger.info("=" * 60)


def signal_handler(sig, frame):
    """シグナルハンドラー（Ctrl+C）"""
    logger.info("\n\n停止シグナルを受信しました。")
    sys.exit(0)


if __name__ == '__main__':
    # シグナルハンドラーの設定
    signal.signal(signal.SIGINT, signal_handler)
    
    # ログディレクトリの作成
    Path('logs').mkdir(exist_ok=True)
    
    print("=" * 60)
    print("Project Forge - 金融知性体システム")
    print("=" * 60)
    print()
    
    # システムの初期化
    forge = ForgeSystem()
    
    if forge.initialize():
        print("\n取引ループを開始します...")
        print("※ 現在はデモモードで動作します。")
        print("※ Ctrl+Cで停止できます。\n")
        
        # 取引ループ実行（デモモード）
        forge.run_trading_loop(
            interval_seconds=60,  # 60秒ごとに評価
            demo_mode=True  # デモモード有効
        )
    else:
        logger.error("システム初期化に失敗しました。")
        sys.exit(1)