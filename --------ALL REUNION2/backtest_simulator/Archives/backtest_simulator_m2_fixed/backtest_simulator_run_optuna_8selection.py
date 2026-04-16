# /workspace/backtest_simulator/backtest_simulator_run_optuna.py
import sys
import shutil
from pathlib import Path
import logging
import optuna
import pandas as pd

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from backtest_simulator_cimera_margin_level_optuna import (
    BacktestConfig,
    BacktestSimulator,
)  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [OPTUNA] %(levelname)s - %(message)s"
)

OPTUNA_TEST_LIMIT = 0
MAX_ALLOWED_DRAWDOWN = 30.0

global_preloaded_data = None


def objective(trial: optuna.Trial):
    # =========================================================
    # 1. 運命の8パターンのための選択肢（GridSamplerと連動）
    # =========================================================
    fixed_risk_pct = trial.suggest_categorical("fixed_risk_pct", [0.01, 0.05])
    m2_th = trial.suggest_categorical("m2_th", [0.5])
    max_pos = trial.suggest_categorical("max_positions", [100])
    max_cons_sl = trial.suggest_categorical("max_consecutive_sl", [2, 3])
    cooldown = trial.suggest_categorical("cooldown_minutes_after_sl", [15, 30])

    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=fixed_risk_pct,
        m2_proba_threshold=m2_th,
        max_positions=max_pos,
        max_consecutive_sl=max_cons_sl,
        cooldown_minutes_after_sl=cooldown,
        test_limit_partitions=OPTUNA_TEST_LIMIT,
        oof_mode=True,
        # --- 固定パラメータ ---
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

    logging.info(f"--- Starting Final Validation Trial {trial.number} ---")
    logging.info(
        f"Params: Risk={fixed_risk_pct * 100:.0f}%, M2={m2_th}, MaxPos={max_pos}, MaxSL={max_cons_sl}, Cooldown={cooldown}m"
    )

    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)

    try:
        simulator = BacktestSimulator(config)
        report_data = simulator.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(original_level)
        logging.error(f"Trial {trial.number} failed with exception: {e}")
        raise optuna.exceptions.TrialPruned()

    logger.setLevel(original_level)

    # =========================================================
    # ★追加: 出力された4つのファイルを専用フォルダに退避させる
    # =========================================================
    # シミュレーターがファイルを出力する元の場所
    src_dir = Path("/workspace/data/XAUUSD/stratum_7_models/1A_2B")

    # 今回のTrial用の専用フォルダ名をパラメータから作成（例: Trial_0_Risk1_SL2_CD15）
    folder_name = f"Trial_{trial.number}_Risk{fixed_risk_pct * 100:.0f}_SL{max_cons_sl}_CD{cooldown}"
    dst_dir = Path(f"/workspace/backtest_simulator/grid_results/{folder_name}")
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
    # =========================================================

    max_dd = report_data.get("max_drawdown_pct", 0.0)
    sharpe = report_data.get("sharpe_ratio_annual", 0.0)
    profit_factor = report_data.get("profit_factor", 0.0)
    trades = report_data.get("total_trades", 0)
    min_margin = report_data.get("min_margin_level_pct", 9999.0)
    stop_outs = report_data.get("stop_out_count", 0)

    trial.set_user_attr(
        "Total_Net_Profit",
        report_data.get("final_capital", 0.0) - report_data.get("initial_capital", 0.0),
    )
    trial.set_user_attr("Max_DD_Pct", max_dd)
    trial.set_user_attr("Sharpe_Ratio", sharpe)
    trial.set_user_attr("Profit_Factor", profit_factor)
    trial.set_user_attr("Total_Trades", trades)
    trial.set_user_attr(
        "Min_Margin_Pct", float(min_margin) if min_margin != 9999.0 else None
    )

    if trades < 50 or stop_outs > 0 or max_dd > MAX_ALLOWED_DRAWDOWN:
        return -9999.0

    if min_margin < 150.0:
        sharpe -= (150.0 - min_margin) * 10

    score = sharpe + (profit_factor * 0.1)
    return float(score)


def export_rankings_to_csv(study: optuna.Study):
    logging.info("Exporting rankings to CSV...")
    trials_data = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"Trial_ID": t.number, "Score": t.value, **t.params, **t.user_attrs}
        trials_data.append(row)

    if not trials_data:
        return

    df = pd.DataFrame(trials_data)
    df = df[df["Score"] != -9999.0]

    df_sharpe = df.sort_values(by="Sharpe_Ratio", ascending=False).head(100)
    df_sharpe.to_csv(
        Path("/workspace/backtest_simulator/optuna_ranking_top100_sharpe.csv"),
        index=False,
    )

    df_np = df.sort_values(by="Total_Net_Profit", ascending=False).head(100)
    df_np.to_csv(
        Path("/workspace/backtest_simulator/optuna_ranking_top100_net_profit.csv"),
        index=False,
    )


def save_csv_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    export_rankings_to_csv(study)


if __name__ == "__main__":
    logging.info("Initializing pre-load sequence for Ultra-Fast Optuna...")
    dummy_config = BacktestConfig(test_limit_partitions=OPTUNA_TEST_LIMIT)
    dummy_simulator = BacktestSimulator(dummy_config)

    global_preloaded_data = dummy_simulator.preload_data()
    logging.info("Data preload complete! Starting FINAL 8 Candidates Grid Search...")

    db_path = "sqlite:///optuna_v5_tuning.db"
    study_name = "v5_final_8_candidates_grid"

    search_space = {
        "fixed_risk_pct": [0.01, 0.05],
        "m2_th": [0.5],
        "max_positions": [100],
        "max_consecutive_sl": [2, 3],
        "cooldown_minutes_after_sl": [15, 30],
    }

    study = optuna.create_study(
        study_name=study_name,
        storage=db_path,
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    logging.info(f"Starting Grid Search. Study: {study_name}")

    study.optimize(objective, n_trials=8, callbacks=[save_csv_callback])

    print("\n" + "=" * 50)
    print(" 🏆 FINAL 8 VALIDATION FINISHED!")
    print("=" * 50)
    print(f" Best Score: {study.best_value:.4f}")
    print(" Best Params: ")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 50)
    print(" Check your detailed files in: /workspace/backtest_simulator/grid_results/")
