# /workspace/backtest_simulator/stress_test_spread.py
import sys
import shutil
from pathlib import Path
import logging
import optuna

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backtest_simulator_cimera_margin_level_optuna import (
    BacktestConfig,
    BacktestSimulator,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [STRESS TEST] %(message)s"
)

global_preloaded_data = None


def objective(trial: optuna.Trial):
    # 悪魔の極悪スプレッド3パターン
    spread = trial.suggest_categorical("spread_pips", [80.0, 100.0, 150.0])

    # Trial 0 (Risk 5%) の最強パラメータで固定
    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=0.05,
        m2_proba_threshold=0.50,
        max_positions=100,
        max_consecutive_sl=2,
        cooldown_minutes_after_sl=30,
        # ★ここを動的に変更
        spread_pips=spread,
        test_limit_partitions=0,
        oof_mode=True,
        min_atr_threshold=2.0,
        sl_multiplier_long=5.0,
        pt_multiplier_long=1.0,
        sl_multiplier_short=5.0,
        pt_multiplier_short=1.0,
        td_minutes_long=60.0,
        td_minutes_short=60.0,
        margin_call_percent=100.0,
        stop_out_percent=20.0,
    )

    logging.info(f"--- 💀 Starting Stress Test: Spread {spread} pips ---")

    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)  # シミュレーターの標準ログをミュート

    try:
        simulator = BacktestSimulator(config)
        report_data = simulator.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(original_level)
        logging.error(f"Test failed with exception: {e}")
        raise optuna.exceptions.TrialPruned()

    logger.setLevel(original_level)

    # 結果ファイルの退避
    src_dir = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B")
    folder_name = f"StressTest_Spread_{int(spread)}"
    dst_dir = Path(f"/workspace/backtest_simulator/stress_results/{folder_name}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    files_to_copy = [
        "final_backtest_report_v5.json",
        "final_backtest_report_v5.txt",
        "equity_curve_v5.png",
        "detailed_trade_log_v5.csv",
    ]
    for fname in files_to_copy:
        src_file = src_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)

    # 簡易結果を記録
    max_dd = report_data.get("max_drawdown_pct", 0.0)
    profit_factor = report_data.get("profit_factor", 0.0)
    net_profit = report_data.get("final_capital", 0.0) - report_data.get(
        "initial_capital", 0.0
    )

    trial.set_user_attr("Total_Net_Profit", net_profit)
    trial.set_user_attr("Max_DD_Pct", max_dd)
    trial.set_user_attr("Profit_Factor", profit_factor)

    return float(net_profit)


if __name__ == "__main__":
    logging.info("Initializing pre-load sequence (This takes 1-2 mins)...")
    dummy_config = BacktestConfig(test_limit_partitions=0)
    dummy_simulator = BacktestSimulator(dummy_config)
    global_preloaded_data = dummy_simulator.preload_data()
    logging.info("Data preload complete! Let the nightmare begin...")

    study = optuna.create_study(
        study_name="v5_extreme_stress_test",
        sampler=optuna.samplers.GridSampler({"spread_pips": [80.0, 100.0, 150.0]}),
        direction="maximize",
    )
    study.optimize(objective, n_trials=3)

    print("\n" + "=" * 60)
    print(" 💀 EXTREME STRESS TEST FINISHED!")
    print("=" * 60)
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            print(
                f" 🩸 Spread {t.params['spread_pips']:>5} pips -> Profit: ${t.user_attrs.get('Total_Net_Profit', 0):>15,.2f} | DD: {t.user_attrs.get('Max_DD_Pct', 0):>5.2f}%"
            )
    print("=" * 60)
    print(
        " Check detailed charts & logs in: /workspace/backtest_simulator/stress_results/"
    )
