# add_decorator_selective.py - 適切な関数のみに@handle_zero_stdデコレータを追加
import re

INPUT_FILE = 'independent_features_clean.py'
OUTPUT_FILE = 'independent_features_fixed.py'

# デコレータを適用すべき関数のみを厳選（rolling().apply()から呼ばれる関数のみ）
TARGET_FUNCTIONS = {
    # 統計・数学計算系（rolling().apply()から呼ばれる）
    '_raw_moment_vectorized',
    '_median_abs_deviation_vectorized', 
    '_winsorized_mean_vectorized',
    '_extreme_value_ratio_vectorized',
    
    # 分布推定系
    '_estimate_t_df_vectorized',
    '_estimate_gamma_shape_vectorized', 
    '_estimate_beta_alpha_vectorized',
    
    # 信号処理系（rolling().apply()から呼ばれる）
    '_hilbert_amplitude_vectorized',
    '_hilbert_phase_vectorized',
    '_instantaneous_frequency_vectorized',
    '_spectral_centroid_vectorized',
    '_spectral_bandwidth_vectorized', 
    '_spectral_rolloff_vectorized',
    '_spectral_flux_vectorized',
    '_zero_crossing_rate_vectorized',
    
    # FFT系
    '_fft_power_mean_vectorized',
    '_fft_power_std_vectorized',
    '_fft_dominant_frequency_vectorized',
    '_fft_bandwidth_vectorized', 
    '_fft_phase_coherence_vectorized',
    
    # ウェーブレット系
    '_cwt_energy_vectorized',
    '_cwt_entropy_vectorized',
    
    # エントロピー・情報理論系
    '_shannon_entropy_vectorized',
    '_conditional_entropy_vectorized',
    '_lempel_ziv_complexity_vectorized',
    '_kolmogorov_complexity_approx_vectorized',
    '_encoding_length_vectorized',
    '_transfer_entropy_vectorized',
    
    # 相関・統計分析系（最重要）
    '_autocorrelation_vectorized',
    '_cross_correlation_vectorized',
    '_coherence_measure_vectorized', 
    '_adf_statistic',
    '_test_arch_effect',
    '_test_market_efficiency',
    '_volatility_clustering',
    
    # フラクタル・カオス系（rolling().apply()から呼ばれる）
    '_estimate_lyapunov_exponent',
    '_measure_chaos_degree',
    '_calculate_vorticity',
    '_measure_energy_cascade',
    '_estimate_kolmogorov_scale',
    '_estimate_taylor_scale',
    
    # 生物・医学系（rolling().apply()から呼ばれる）
    '_measure_trend_persistence',
    '_calculate_attention_metric', 
    '_measure_recovery_rate',
    '_analyze_gait_pattern',
    
    # 社会・経済系（rolling().apply()から呼ばれる）
    '_calculate_prisoners_dilemma_index',
    '_cooperation_index', 
    '_social_inertia',
    '_calculate_anchoring_effect',
    '_calculate_learning_effect',
    
    # その他の数値計算系
    '_analyze_harmony',
    '_safe_sample_entropy',
    '_safe_approx_entropy',
    '_safe_polyfit_slope',
    
    # 新規追加（相関計算対策）
    '_mutual_information_safe_vectorized',
    '_granger_causality_safe_test'
}

def get_function_name(line):
    """行から関数名を取得する"""
    match = re.search(r'def\s+([a-zA-Z0-9_]+)\s*\(', line)
    if match:
        return match.group(1)
    return None

def main():
    print(f"'{INPUT_FILE}' から適切な関数のみにデコレータを追加しています...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"エラー: '{INPUT_FILE}' が見つかりません。")
        return

    modified_count = 0
    print(f"\n対象関数数: {len(TARGET_FUNCTIONS)}")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for i, line in enumerate(lines):
            func_name = get_function_name(line.strip())
            is_target = func_name and func_name in TARGET_FUNCTIONS
            
            if is_target:
                # デコレータを追加
                indentation = re.match(r'(\s*)', line).group(1)
                decorator_line = f"{indentation}@handle_zero_std\n"
                f_out.write(decorator_line)
                print(f"  -> {func_name} にデコレータを追加しました。")
                modified_count += 1
            
            f_out.write(line)

    print(f"\n処理が完了しました！ {modified_count}箇所にデコレータを追加し、結果を'{OUTPUT_FILE}'に保存しました。")
    
    # 適用されなかった関数をチェック
    applied_functions = set()
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if '@handle_zero_std' in line and i+1 < len(lines):
                func_name = get_function_name(lines[i+1].strip())
                if func_name:
                    applied_functions.add(func_name)
    
    not_found = TARGET_FUNCTIONS - applied_functions
    if not_found:
        print(f"\n警告: 以下の関数が見つかりませんでした:")
        for func in sorted(not_found):
            print(f"  - {func}")

if __name__ == '__main__':
    main()