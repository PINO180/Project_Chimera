# add_decorator_selective.py - 適切な関数のみに@handle_zero_stdデコレータを追加
import re

INPUT_FILE = 'independent_features.py'
OUTPUT_FILE = 'independent_features_fixed.py'

# デコレータを適用すべき関数のみを厳選（rolling().apply()から呼ばれる関数など）
# このリストは、デコレータが必要な全てのヘルパー関数を網羅しています。
TARGET_FUNCTIONS = {
    # 統計・信号処理・数学系ヘルパー
    '_raw_moment_vectorized',
    '_median_abs_deviation_vectorized',
    '_winsorized_mean_vectorized',
    '_extreme_value_ratio_vectorized',
    '_estimate_t_df_vectorized',
    '_estimate_gamma_shape_vectorized',
    '_estimate_beta_alpha_vectorized',
    '_hilbert_amplitude_vectorized',
    '_hilbert_phase_vectorized',
    '_instantaneous_frequency_vectorized',
    '_autocorrelation_vectorized',
    '_cross_correlation_vectorized',
    '_spectral_centroid_vectorized',
    '_spectral_bandwidth_vectorized',
    '_spectral_rolloff_vectorized',
    '_spectral_flux_vectorized',
    '_zero_crossing_rate_vectorized',
    '_coherence_measure_vectorized',

    # フーリエ・ウェーブレット系ヘルパー
    '_fft_power_mean_vectorized',
    '_fft_power_std_vectorized',
    '_fft_dominant_frequency_vectorized',
    '_fft_bandwidth_vectorized',
    '_fft_phase_coherence_vectorized',
    '_cwt_energy_vectorized',
    '_cwt_entropy_vectorized',

    # 情報理論・エントロピー系ヘルパー
    '_shannon_entropy_vectorized',
    '_conditional_entropy_vectorized',
    '_lempel_ziv_complexity_vectorized',
    '_kolmogorov_complexity_approx_vectorized',
    '_encoding_length_vectorized',
    '_mutual_information_safe_vectorized',
    '_transfer_entropy_vectorized',
    '_safe_sample_entropy',
    '_safe_approx_entropy',

    # 生物学・心理学・行動経済学系ヘルパー
    '_safe_polyfit_slope',
    '_estimate_dominant_period',
    '_measure_trend_persistence',
    '_calculate_anchoring_effect',
    '_calculate_attention_metric',
    '_calculate_learning_effect',
    '_volatility_clustering',
    '_social_inertia',
    '_cooperation_index',
    '_calculate_prisoners_dilemma_index',
    
    # フラクタル・カオス・物理学系ヘルパー
    '_calculate_hurst_exponent',
    '_calculate_fractal_dimension',
    '_measure_self_similarity',
    '_calculate_dfa_alpha',
    '_calculate_multifractal_width',
    '_calculate_fractal_efficiency',
    '_estimate_lyapunov_exponent',
    '_calculate_correlation_dimension',
    '_measure_chaos_degree',
    '_detect_strange_attractor',
    '_analyze_poincare_section',
    '_measure_nonlinear_dynamics',
    '_calculate_vorticity',
    '_measure_energy_cascade',
    '_estimate_kolmogorov_scale',
    '_estimate_taylor_scale',

    # 計量経済学・金融工学系ヘルパー
    '_adf_statistic',
    '_cointegration_test',
    '_test_arch_effect',
    '_granger_causality_safe_test',
    '_test_market_efficiency',
    '_estimate_mean_reversion_speed',
    '_test_long_memory',

    # ネットワーク科学・社会物理学系ヘルパー
    '_calculate_network_density',
    '_calculate_centrality',
    '_calculate_clustering_coefficient',
    '_calculate_betweenness_centrality',
    '_calculate_eigenvector_centrality',

    # 美学・音楽理論系ヘルパー
    '_measure_golden_ratio_adherence',
    '_measure_symmetry',
    '_measure_aesthetic_harmony',
    '_measure_proportional_beauty',
    '_measure_visual_balance',
    '_analyze_harmony',

    # 天文学・宇宙論系ヘルパー
    '_calculate_orbital_indicator',
    '_detect_gravitational_wave_pattern',
    '_analyze_stellar_pulsation',
    '_analyze_planetary_motion',
    '_estimate_dark_energy',
    '_detect_big_bang_echo',
    '_analyze_cmb_pattern',
    '_measure_cosmic_inflation',

    # 生体力学・パフォーマンス科学系ヘルパー
    '_analyze_gait_pattern',
    '_measure_recovery_rate'
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
        i = 0
        while i < len(lines):
            line = lines[i]
            func_name = get_function_name(line.strip())
            is_target = func_name and func_name in TARGET_FUNCTIONS

            if is_target:
                # 既にデコレータが付いているかチェック
                has_decorator = False
                if i > 0 and '@handle_zero_std' in lines[i-1]:
                    has_decorator = True
                
                if not has_decorator:
                    # デコレータを追加
                    indentation = re.match(r'(\s*)', line).group(1)
                    decorator_line = f"{indentation}@handle_zero_std\n"
                    f_out.write(decorator_line)
                    print(f"  -> {func_name} にデコレータを追加しました。")
                    modified_count += 1
                else:
                    print(f"  -- {func_name} には既にデコレータがあります。（スキップ）")

            f_out.write(line)
            i += 1

    print(f"\n処理が完了しました！ {modified_count}箇所にデコレータを新規追加し、結果を'{OUTPUT_FILE}'に保存しました。")

    # 適用されなかった関数をチェック
    applied_functions = set()
    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        output_lines = f.readlines()
        for i, line in enumerate(output_lines):
            if '@handle_zero_std' in line and i + 1 < len(output_lines):
                func_name = get_function_name(output_lines[i + 1].strip())
                if func_name:
                    applied_functions.add(func_name)

    not_found = TARGET_FUNCTIONS - applied_functions
    if not_found:
        print(f"\n警告: 以下の対象関数がコード内で見つからず、デコレータを適用できませんでした:")
        for func in sorted(not_found):
            print(f"  - {func}")

if __name__ == '__main__':
    main()