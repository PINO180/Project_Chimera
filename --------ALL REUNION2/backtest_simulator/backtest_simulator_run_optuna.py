# /workspace/backtest_simulator/backtest_simulator_run_optuna.py

import sys
import shutil
import pickle
from pathlib import Path
import logging
import optuna
import pandas as pd

# --- プロジェクトのルートディレクトリをPythonの検索パスに追加 ---
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from blueprint import (  # noqa: E402
    S7_MODELS,
    S7_BACKTEST_OPTUNA_RESULTS,
    S7_BACKTEST_CACHE,
    S7_M2_OOF_PREDICTIONS_LONG,
    S7_M2_OOF_PREDICTIONS_SHORT,
)

from backtest_simulator_cimera_margin_level_optuna import (
    BacktestConfig,
    BacktestSimulator,
)  # noqa: E402

# --- Optuna用ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - [OPTUNA] %(levelname)s - %(message)s"
)

# ==========================================
# オプティマイザーの全体設定
# ==========================================
OPTUNA_TEST_LIMIT = 0  # 探索にかけるパーティション数（0=全期間）
MAX_ALLOWED_DRAWDOWN = 0.0  # 許容する最大ドローダウン（%）
MIN_TRADES_THRESHOLD = 100  # これ未満のトレード数はPruned扱い
# ==========================================

# オンメモリ展開したデータを保持するためのグローバル変数
global_preloaded_data = None


def objective(trial: optuna.Trial):
    """
    探索対象: m2_proba_threshold / m2_delta_threshold / fixed_risk_percent
    固定値  : min_atr_threshold（ラベリング前Optunaで決定済みの前提条件）
              sl/pt/td（トリプルバリア設計値）
    """
    # =========================================================
    # 1. 探索パラメータ
    # =========================================================
    m2_th = trial.suggest_categorical(
        "m2_th",
        [0.30, 0.50, 0.70, 0.80, 0.90, 0.95],
    )

    m2_delta = trial.suggest_categorical(
        "m2_delta",
        [0.00, 0.30, 0.50],
    )

    fixed_risk_pct = trial.suggest_categorical(
        "fixed_risk_pct",
        [0.02, 0.05, 0.10],
    )

    # =========================================================
    # 2. BacktestConfigの構築
    #    min_atr_threshold は固定（ラベリング前Optunaで決定済み）
    # =========================================================
    config = BacktestConfig(
        use_fixed_risk=True,
        fixed_risk_percent=fixed_risk_pct,
        m2_proba_threshold=m2_th,
        m2_delta_threshold=m2_delta,
        min_atr_threshold=0.8,  # 固定: ラベリング前Optunaで確定した前提条件
        max_positions=100,
        max_consecutive_sl=2,
        cooldown_minutes_after_sl=30,
        test_limit_partitions=OPTUNA_TEST_LIMIT,
        oof_mode=True,
        sl_multiplier_long=5.0,
        pt_multiplier_long=1.0,
        sl_multiplier_short=5.0,
        pt_multiplier_short=1.0,
        td_minutes_long=30.0,
        td_minutes_short=30.0,
        margin_call_percent=0.0,
        stop_out_percent=0.0,
    )

    logging.info(f"--- Starting Trial {trial.number:03d} ---")
    logging.info(
        f"Params: M2_th={m2_th}, Delta={m2_delta}, Risk={fixed_risk_pct * 100:.0f}%"
    )

    # Trial実行中は余計なログを抑制
    logger = logging.getLogger()
    original_level = logger.level
    logger.setLevel(logging.WARNING)

    try:
        simulator = BacktestSimulator(config)
        report_data = simulator.run(preloaded_data=global_preloaded_data)
    except Exception as e:
        logger.setLevel(original_level)
        logging.error(f"Trial {trial.number:03d} failed: {e}")
        raise optuna.exceptions.TrialPruned()

    logger.setLevel(original_level)

    # =========================================================
    # 3. 出力ファイルをTrial専用フォルダに退避
    #    出力元: S7_MODELS（blueprint定義）
    #    出力先: S7_BACKTEST_OPTUNA_RESULTS / Trial_XXX_...
    # =========================================================
    folder_name = (
        f"Trial_{trial.number:03d}_Th{m2_th}_D{m2_delta}_R{fixed_risk_pct * 100:.0f}"
    )
    dst_dir = S7_BACKTEST_OPTUNA_RESULTS / folder_name
    dst_dir.mkdir(parents=True, exist_ok=True)

    for fname in [
        "final_backtest_report_v5.json",
        "final_backtest_report_v5.txt",
        "equity_curve_v5.png",
        "detailed_trade_log_v5.csv",
    ]:
        src_file = S7_MODELS / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_dir / fname)

    # =========================================================
    # 4. 評価指標の抽出 & Trial属性セット
    # =========================================================
    max_dd = report_data.get("max_drawdown_pct", 0.0) or 0.0
    sharpe = report_data.get("sharpe_ratio_annual", 0.0) or 0.0
    sortino = report_data.get("sortino_ratio_annual", 0.0) or 0.0
    profit_factor = report_data.get("profit_factor", 0.0) or 0.0
    trades = report_data.get("total_trades", 0) or 0
    win_rate = report_data.get("win_rate_pct", 0.0) or 0.0
    min_margin = report_data.get("min_margin_level_pct", 9999.0)
    stop_outs = report_data.get("stop_out_count", 0) or 0
    net_profit = report_data.get("final_capital", 0.0) - report_data.get(
        "initial_capital", 0.0
    )

    trial.set_user_attr("Total_Net_Profit", net_profit)
    trial.set_user_attr("Max_DD_Pct", max_dd)
    trial.set_user_attr("Sharpe_Ratio", sharpe)
    trial.set_user_attr("Sortino_Ratio", sortino)
    trial.set_user_attr("Profit_Factor", profit_factor)
    trial.set_user_attr("Win_Rate_Pct", win_rate)
    trial.set_user_attr("Total_Trades", trades)
    trial.set_user_attr(
        "Min_Margin_Pct",
        float(min_margin) if min_margin != 9999.0 else None,
    )

    # =========================================================
    # 5. ペナルティ判定 → スコア算出
    # =========================================================
    if trades < MIN_TRADES_THRESHOLD:
        logging.info(
            f"Trial {trial.number:03d} Pruned: trades={trades} < {MIN_TRADES_THRESHOLD}"
        )
        raise optuna.exceptions.TrialPruned()

    if stop_outs > 0:
        logging.info(f"Trial {trial.number:03d} Penalized: stop_outs={stop_outs}")
        return -9999.0

    if max_dd > MAX_ALLOWED_DRAWDOWN:
        logging.info(
            f"Trial {trial.number:03d} Penalized: max_dd={max_dd:.1f}% > {MAX_ALLOWED_DRAWDOWN}%"
        )
        return -9999.0

    if min_margin < 150.0:
        sharpe -= (150.0 - min_margin) * 10

    score = sharpe + (profit_factor * 0.1)
    logging.info(
        f"Trial {trial.number:03d} Done: Score={score:.4f} "
        f"(Sharpe={sharpe:.2f}, PF={profit_factor:.2f}, DD={max_dd:.1f}%, Trades={trades})"
    )
    return float(score)


# =========================================================
# ランキング出力
# =========================================================
def export_rankings(study: optuna.Study):
    """全Trial終了後・各Trial後にT_NP順のランキングを全件CSV & コンソール出力する"""

    trials_data = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        if t.value is None or t.value == -9999.0:
            continue
        row = {
            "Trial_ID": t.number,
            "Score": t.value,
            "m2_th": t.params.get("m2_th"),
            "m2_delta": t.params.get("m2_delta"),
            "fixed_risk_pct": t.params.get("fixed_risk_pct"),
            "Sharpe_Ratio": t.user_attrs.get("Sharpe_Ratio"),
            "Sortino_Ratio": t.user_attrs.get("Sortino_Ratio"),
            "Profit_Factor": t.user_attrs.get("Profit_Factor"),
            "Win_Rate_Pct": t.user_attrs.get("Win_Rate_Pct"),
            "Max_DD_Pct": t.user_attrs.get("Max_DD_Pct"),
            "Total_Net_Profit": t.user_attrs.get("Total_Net_Profit"),
            "Total_Trades": t.user_attrs.get("Total_Trades"),
            "Min_Margin_Pct": t.user_attrs.get("Min_Margin_Pct"),
        }
        trials_data.append(row)

    if not trials_data:
        logging.warning("No complete trials to export.")
        return

    df = pd.DataFrame(trials_data)
    out_base = S7_BACKTEST_OPTUNA_RESULTS
    out_base.mkdir(parents=True, exist_ok=True)

    # Total_Net_Profitの降順で全件ソート
    df_ranked = df.sort_values(by="Total_Net_Profit", ascending=False)

    # CSV保存
    out_path = out_base / "ranking_Total_Net_Profit.csv"
    df_ranked.to_csv(out_path, index=False)

    # コンソール出力
    print(f"\n{'=' * 78}")
    print("  🏆 Ranking by Total Net Profit (All Complete Trials)")
    print(f"{'=' * 78}")
    print(
        f"{'#':<4} {'Trial':>6} {'m2_th':>6} {'delta':>6} {'risk%':>6} "
        f"{'T_NP':>10} {'Max_DD':>8} {'Trades':>7} {'Win_R':>7} {'Min_M':>8}"
    )
    print("-" * 78)
    for rank, (_, row) in enumerate(df_ranked.iterrows(), 1):
        # Min_Margin_Pct の欠損値対応
        min_margin = row["Min_Margin_Pct"]
        min_m_str = f"{min_margin:>8.1f}" if pd.notnull(min_margin) else f"{'---':>8}"

        print(
            f"{rank:<4} {int(row['Trial_ID']):>6} {row['m2_th']:>6} "
            f"{row['m2_delta']:>6} {row['fixed_risk_pct'] * 100:>5.0f}% "
            f"{row['Total_Net_Profit']:>10.1f} {row['Max_DD_Pct']:>8.2f} "
            f"{int(row['Total_Trades']):>7} {row['Win_Rate_Pct']:>7.1f} "
            f"{min_m_str}"
        )


def save_csv_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    """1Trial終了ごとにランキングをリアルタイム更新"""
    export_rankings(study)


if __name__ == "__main__":
    # =========================================================
    # 超速化: Optuna開始前にデータを1回だけメモリに全ロード
    # =========================================================
    S7_BACKTEST_OPTUNA_RESULTS.mkdir(parents=True, exist_ok=True)

    # =========================================================
    # キャッシュ管理: あれば読む・古ければ警告・なければ生成して保存
    # =========================================================
    def load_or_generate_cache() -> tuple:
        if S7_BACKTEST_CACHE.exists():
            # タイムスタンプチェック（OOFより古いキャッシュは警告）
            cache_mtime = S7_BACKTEST_CACHE.stat().st_mtime
            oof_mtime = max(
                S7_M2_OOF_PREDICTIONS_LONG.stat().st_mtime,
                S7_M2_OOF_PREDICTIONS_SHORT.stat().st_mtime,
            )
            stale = cache_mtime < oof_mtime

            print(f"\nキャッシュが存在します: {S7_BACKTEST_CACHE}")
            if stale:
                print(
                    "  ⚠️  キャッシュがOOFより古い可能性があります。再生成を推奨します。"
                )
            print("  [y] このまま使用する")
            print("  [r] 削除して再生成する")
            ans = input("選択 [y/r]: ").strip().lower()

            if ans == "r":
                S7_BACKTEST_CACHE.unlink()
                logging.info("キャッシュを削除しました。再生成します...")
            else:
                logging.info("キャッシュを読み込んでいます...")
                with open(S7_BACKTEST_CACHE, "rb") as f:
                    data = pickle.load(f)
                logging.info("キャッシュ読み込み完了。")
                return data

        # 生成
        logging.info("Initializing pre-load sequence for Ultra-Fast Optuna...")
        dummy_config = BacktestConfig(test_limit_partitions=OPTUNA_TEST_LIMIT)
        dummy_simulator = BacktestSimulator(dummy_config)
        data = dummy_simulator.preload_data()
        logging.info("データ生成完了。キャッシュに保存しています...")
        with open(S7_BACKTEST_CACHE, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"キャッシュ保存完了: {S7_BACKTEST_CACHE}")
        return data

    global_preloaded_data = load_or_generate_cache()
    logging.info("Data preload complete! Starting Optuna trials...")

    # =========================================================
    # Optuna最適化セッション
    # =========================================================
    db_path = str(S7_BACKTEST_OPTUNA_RESULTS / "optuna_v5_backtest.db")
    search_space = {
        "m2_th": [0.30, 0.50, 0.70, 0.80, 0.90, 0.95],
        "m2_delta": [0.00, 0.30, 0.50],
        "fixed_risk_pct": [0.02, 0.05, 0.10],
    }
    # 前回の修正に則り、選択肢を変更した場合は新規名称に変更する
    study_name = "v5_grid_full_125_v2"

    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        load_if_exists=True,
        direction="maximize",
        sampler=optuna.samplers.GridSampler(search_space),
    )

    n_trials = 54

    logging.info(f"Study: {study_name} | n_trials: {n_trials}")
    study.optimize(objective, n_trials=n_trials, callbacks=[save_csv_callback])

    # =========================================================
    # 最終ランキング出力
    # =========================================================
    print("\n\n" + "=" * 78)
    print("  📊 FINAL OPTIMIZATION RESULTS")
    print("=" * 78)
    export_rankings(study)

    print("\n" + "=" * 78)
    print(f"  Best Trial : #{study.best_trial.number:03d}")
    print(f"  Best Score : {study.best_value:.4f}")
    print("  Best Params:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    print("=" * 78)
    print(f"\n  Results saved to: {S7_BACKTEST_OPTUNA_RESULTS}")
