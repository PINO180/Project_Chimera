2個のコマンドを実行しました, ファイルを表示しました2個のコマンドを実行しました, ファイルを表示しました学習側Eの全特徴量を整理します：学習側Eの特徴量数学習側Eの特徴量数学習側E: 62個の特徴量。次に本番側を確認します。4個のコマンドを実行しました4個のコマンドを実行しました基底名は学習側と一致 ✅（sample_weight が追加されているのは他エンジン同様）
本番側の calculate_features全体を見ます：本番側1E前半本番側1E前半重要：本番側Eは core_indicators からほぼ全UDFをimportしています。これは Single Source of Truth (SSOT) であり、学習側と本番側が完全に同じ関数を共有する設計。理論上、UDF実装の乖離は発生しない ✅
次に学習側Eが core_indicators の関数を使っているか確認：学習側Eのcore_indicators使用学習側Eのcore_indicators使用学習側Eも core_indicators から全UDFをimport ✅。両者が完全に同じ関数を使用しているため、UDF実装は完全一致が保証されている。
次は学習側の式と本番側の呼び出し方を比較します。各特徴量グループを順に確認します。
グループ1: Spectral系学習側のSpectral系学習側のSpectral系重要な発見： 学習側では map_batches を使って 配列全体に UDF を適用しています。rolling_map ではなく map_batches です。これは UDF が配列全体を受け取り、配列全体を返すことを意味します。
注: ある UDF が (close_pct_full, window) を受け取って、内部で末尾 window 本に対する計算を行うが、出力は配列全体である、という設計の可能性が高い。
実際に core_indicators の spectral_centroid_udf を確認します：コマンドを実行しました, ファイルを表示しましたコマンドを実行しました, ファイルを表示しました重要：UDFは内部でローリング処理を実装している：

入力は配列全体 signal
出力も配列全体（インデックス window_size - 1 以降が有効値）
各位置 i で signal[i - window_size + 1 : i + 1] のFFTを計算

つまり学習側 map_batches(spectral_centroid_udf, w) は配列全体を返し、Polarsはそれを各バーの値として保持。最終バーtでの値は result[t] = signal[t-w+1 : t+1] のスペクトル重心。
本番側を確認：2個のファイルを表示しました2個のファイルを表示しました本番側の構造を把握しました。次に学習側の各特徴量と比較していきます。
各特徴量の詳細比較
重要なポイント：spectral_flux
学習側： pl.col("close").pct_change().map_batches(spectral_flux_udf, w)

配列全体に UDF を適用、最終バーtでの値は result[t]
spectral_flux_udfが内部でどう計算しているか確認が必要
Spectral_flux_udfの実装Spectral_flux_udfの実装spectral_flux_udf は window_size * 2 本のデータが必要 で、隣接2フレーム分のスペクトル差を計算する。
学習側：pl.col("close").pct_change().map_batches(spectral_flux_udf, w)

全配列に対して計算、最終バーでの値は 直近 2*w 本のデータから計算された値

本番側：spectral_flux_udf(_window(close_pct, window * 2), window)

末尾 2*w 本だけを渡す。これでも UDF 内部で i = window_size*2-1 以降の計算が行われ、配列の最後の値だけが有効値となる。

ここで _window(close_pct, window * 2) で渡される配列は長さ 2*w なので、signal[0:w] が prev_buffer 用、signal[w:2*w] が curr_buffer 用となる。最終バーt = 2*w - 1。
学習側のバーtでの値：signal[t-2*w+1:t-w+1] を prev_buffer、signal[t-w+1:t+1] を curr_buffer として計算した結果。
これは実は同じ計算になる。本番側の _window(close_pct, window * 2) は close_pct[-2*w:] で、これは学習側のバーtでの signal[t-2*w+1:t+1]（実際には close_pct[-2*w:]）と一致 ✅
spectral_centroid 等（window 1個分のUDF）
学習側：pl.col("close").pct_change().map_batches(spectral_centroid_udf, w)

全配列を渡す、最終バーtでは signal[t-w+1:t+1] で計算

本番側：spectral_centroid_udf(_window(close_pct, window), window)

末尾 w 本を渡す。UDF 内で i=w-1 で計算。これは signal[0:w] のスペクトル重心 = 学習側のバーtと完全一致 ✅

しかし**map_batches と本番側の動作の違いに注意**：
学習側 map_batches(udf, w)：

Polarsは pct_change の全配列（学習データ全体、何百万本）を UDF に渡す
UDF は内部でローリング（window単位）で計算
結果は配列、最終バーtでの値は学習データ全体から計算した値

本番側：

_window(close_pct, window) で末尾 w 本だけを渡す
結果は1個（最終バーの値）

重要な疑問：spectral_centroid_udf の中で「signal[t-w+1:t+1]」を計算する際、 学習側と本番側で渡される signal が異なるが、最終バーでの計算結果は同じか？
検査：

学習側：signal = close_pct (全配列、長さ = 何百万)、計算は signal[i-w+1:i+1] for i = w-1, w, ..., n-1
本番側：signal = close_pct[-w:]（長さ w）、計算は signal[i-w+1:i+1] for i = w-1 （つまり1回だけ）

本番側の i = w-1 のとき、signal[0:w] = close_pct[-w:] → 学習側の最終バーtの値 close_pct[t-w+1:t+1] = close_pct[-w:] と完全一致 ✅
これは pct_change を渡すので、UDF に渡される signal は valid な値（学習データの先頭NaN除く）。両者で最終バーの値は完全一致する。
spectral_energy
学習側：(pct_change ** 2).rolling_sum(window)

Polars rolling_sum(window) のデフォルト min_samples = window
ウィンドウ内に NaN があれば NaN を返す

本番側：np.sum(w_e ** 2) （len(w_e) >= window のとき）

NaN があれば NaN

乖離あり ⚠️ ?
np.sum は NaN を含む配列を sum すると NaN を返す（どんな1個のNaNでも全体がNaN）：Np.sumとPolars rolling_sumのNaN挙動Np.sumとPolars rolling_sumのNaN挙動両者とも NaN 伝搬する ✅
spectral_peak_freq_128
学習側：pct_change().rolling_max(128) / (pct_change().rolling_std(128, ddof=1) + 1e-10)

最終バーtでの値：max(pct_change[t-127:t+1]) / (std(pct_change[t-127:t+1], ddof=1) + 1e-10)

本番側：float(np.max(w_pf)) / (float(np.std(w_pf, ddof=1)) + 1e-10) （len(w_pf) >= 128 のとき）

w_pf = close_pct[-128:] = pct_change[t-127:t+1] (最終バーt時点)

完全一致 ✅
ただし、ウィンドウ内NaN挙動を確認：

学習側 rolling_max rolling_std のNaN挙動
Rolling_maxとnp.maxのNaN挙動Rolling_maxとnp.maxのNaN挙動両者ともNaN伝搬。完全一致 ✅
wavelet系
学習側：学習側のwavelet系学習側のwavelet系wavelet_energy
学習側：map_batches(wavelet_energy_udf, w)
本番側：wavelet_energy_udf(_window(close_pct, window), window) → _last(...)
最終バー値完全一致 ✅
wavelet_mean
学習側：pct_change().rolling_mean(window)

最終バーtでの値：mean(pct_change[t-window+1:t+1])、Polars rolling_mean min_samples=window

本番側：np.mean(w_wv) (len(w_wv) >= window のとき)

w_wv = close_pct[-window:] = pct_change[t-window+1:t+1]
np.mean(arr) は NaN 伝搬

最終バー値完全一致 ✅（NaN伝搬挙動も両者一致）
wavelet_std
学習側：pct_change().rolling_std(window, ddof=1)

最終バーtでの値：std(pct_change[t-window+1:t+1], ddof=1)、Polars NaN伝搬

本番側：np.std(w_wv, ddof=1) (len(w_wv) >= window のとき)

np.stdはNaN伝搬

完全一致 ✅
wavelet_entropy_64
学習側：map_batches(wavelet_entropy_udf, 64)

全配列の wavelet_entropy_udf 計算

本番側：wavelet_entropy_udf(_window(close_pct, 64), 64) → _last(...)
UDFの実装を確認：Wavelet系UDFWavelet系UDFwavelet系UDF はSSOTで共有されているので実装一致 ✅
ヒルベルト系学習側のヒルベルト系学習側のヒルベルト系hilbert_amplitude_{w}
学習側：map_batches(hilbert_amplitude_udf, w) 全配列に適用
本番側：hilbert_amplitude_udf(_window(close_pct, window), window) → _last(...)
UDFはSSOT共有なので最終バー値完全一致 ✅
hilbert_amp_mean_100
学習側：pct_change().abs().rolling_mean(100)

最終バーtでの値：mean(|pct_change[t-99:t+1]|)、Polars rolling_mean min_samples=100、NaN伝搬

本番側：np.mean(w_amp) (w_amp = np.abs(close_pct)[-100:])

np.mean は NaN 伝搬

完全一致 ✅
hilbert_amp_std_100
学習側：pct_change().abs().rolling_std(100, ddof=1)
本番側：np.std(w_amp, ddof=1) (w_amp = np.abs(close_pct)[-100:])
完全一致 ✅
hilbert_amp_cv_100
学習側：abs.rolling_std(100, ddof=1) / (abs.rolling_mean(100) + 1e-10)
本番側：amp_std / (amp_mean + 1e-10)（mean が finite であれば）
完全一致 ✅
hilbert_phase_var_50, hilbert_phase_stability_50
UDF を使用、SSOT共有 ✅
hilbert_freq_mean_100, hilbert_freq_std_100
UDF を使用、SSOT共有 ✅
hilbert_freq_energy_ratio_100
学習側：
pythonatr_13_pct = atr_13_wilder / (close + 1e-10)
result = (pct_change ** 2).rolling_sum(100) / ((atr_13_pct ** 2) * 100 + 1e-10)
最終バーtでの値：

分子：sum(pct_change[t-99:t+1] ** 2)
分母：(atr_13[t] / (close[t] + 1e-10)) ** 2 * 100 + 1e-10

本番側：
pythonpct_energy = float(np.sum(w_fe ** 2)) if len(w_fe) >= 100 else np.nan
atr13_pct = atr13_last / (close_last + 1e-10)
features["..."] = pct_energy / (atr13_pct ** 2 * 100 + 1e-10) if (...)
完全一致 ✅
音響系学習側のacoustic系学習側のacoustic系acoustic系
学習側：UDF経由
本番側：UDF経由（SSOT）✅
signal_rms_50
学習側：(pct_change ** 2).rolling_mean(50).sqrt()

最終バーtでの値：sqrt(mean(pct_change[t-49:t+1] ** 2))、Polars rolling_mean min_samples=50

本番側：np.sqrt(np.mean(w_rms_50 ** 2)) (w_rms_50 = close_pct[-50:])

np.mean(arr ** 2) は NaN 伝搬

完全一致 ✅
signal_peak_to_peak_100
学習側：(close.rolling_max(100) - close.rolling_min(100)) / (atr_100 + 1e-10)

最終バーtでの値：(max(close[t-99:t+1]) - min(close[t-99:t+1])) / (atr_100[t] + 1e-10)

本番側：
pythonw_sig_100 = _window(close_arr, 100)
atr_100_full = calculate_atr_wilder(high, low, close, 100)  # 全配列計算
atr_last_val = float(atr_100_full[-1])
result = (max(w_sig_100) - min(w_sig_100)) / (atr_last_val + 1e-10)
完全一致 ✅（atr_100は両者とも全系列から計算した最終値）
signal_crest_factor_50
学習側：
pythonpct_change.rolling_max(50).abs() / ((pct_change ** 2).rolling_mean(50).sqrt() + 1e-10)
最終バーtでの値：

分子：abs(max(pct_change[t-49:t+1]))
分母：sqrt(mean(pct_change[t-49:t+1] ** 2)) + 1e-10

本番側：
pythonabs(float(np.max(w_cf))) / (rms_cf + 1e-10)

w_cf = close_pct[-50:]
rms_cf = sqrt(mean(w_cf ** 2))

完全一致 ✅
全体のサマリー：問題なさそう？
ここまでの精査で、ほぼ全てが完全一致しているように見えます。本番側Eは core_indicators の SSOT を活用して学習側と整合性を保っている設計が秀逸 ✅
残りの懸念点を再確認
_pct_change の挙動
本番側：本番側1Eの_pct_change本番側1Eの_pct_change本番側1Eの_pct_changeはnumpyの算術除算を使うので、prev=0のときの挙動はnumpy挙動（cur>0 → +inf, cur<0 → -inf, cur=0 → NaN）。学習側Polarsと自然に一致 ✅
エンジンDのpct_change_numbaでは明示的なif分岐で勝手にnp.infを返していたため修正が必要だったが、エンジンEはnumpy算術除算なのでそのまま正しい挙動になります。
QAState本番側1EのQAState本番側1EのQAStateQAState は他エンジンと同じ実装。起動時のシード差はウォームアップで緩和という設計（エンジンA同様）。
ただしエンジンAでは選択肢3（EWMスナップショット方式）のコメントを残しました。エンジンEでも同じコメントを残すべきか確認します。
エンジンA以降のエンジン（B,C,D）にはこのコメントを残していないかもしれません。確認します：他エンジンに選択肢コメントがあるか確認他エンジンに選択肢コメントがあるか確認エンジンA以外には選択肢3のコメントは残していません。これで一貫性があるので、エンジンEもこの方針に従います（コメントを追加しない）。
もしユーザーが希望する場合は、後でエンジンB/C/D/E/Fすべてに同様のコメントを一括で追加できます。
残りの厳密チェック
学習側にあるが本番側にない・または乖離している点を念入りに確認します。
重要な確認ポイント：spectral_flux で渡されるデータ長
学習側 map_batches(spectral_flux_udf, w) は配列全体を渡す。本番側は _window(close_pct, window * 2) で末尾 2*w 本を渡す。
学習側の最終バーtでの計算過程（spectral_flux_udf 内）：

i = t (最終バー)
prev_buffer: signal[i-2*w+1 : i-w+1] = signal[t-2*w+1 : t-w+1]
curr_buffer: signal[i-w+1 : i+1] = signal[t-w+1 : t+1]

本番側の計算過程：

signal = close_pct[-2*w:]、長さ 2*w
i = 2*w - 1 (signal の最終インデックス)
prev_buffer: signal[2*w-1-2*w+1 : 2*w-1-w+1] = signal[0 : w] = close_pct[-2*w:-w]
curr_buffer: signal[2*w-1-w+1 : 2*w] = signal[w : 2*w] = close_pct[-w:]

学習側のバーt最終値：

prev_buffer = close_pct[t-2*w+1 : t-w+1]
curr_buffer = close_pct[t-w+1 : t+1]

close_pct[-2*w:-w] と close_pct[t-2*w+1 : t-w+1] は同じか？
close_pct[-2*w:] の最初 w 本は close_pct[-2*w:-w] で、これは close_pct[len-2*w : len-w]。
t = len-1 なので t-2*w+1 = len-2*w と t-w+1 = len-w。
→ 完全一致 ✅
念のため acoustic_power_udf と acoustic_frequency_udf も同じ window 1個分の UDF か確認：Acoustic UDFのデータ長要件Acoustic UDFのデータ長要件両UDFとも window 1個分 OK。本番側の _window(close_pct, window) で末尾 w 本渡し → i = w-1 で計算 ✅
hilbert_freq_mean / std UDFHilbert系UDFのwindow要件Hilbert系UDFのwindow要件すべて window 1個分。本番側の _window(close_pct, window) で十分 ✅
全体の最終確認
エンジンEは：

全62特徴量の base name が両側で完全一致 ✅
UDFは core_indicators で SSOT 共有 ✅
map_batches パターンは本番側でも _window(close_pct, window) + _last(...) で正しく等価
spectral_flux のみ window * 2 を渡す特殊ケースだが正しく実装されている
spectral_energy / wavelet_mean / wavelet_std / hilbert_amp_* / signal_rms など Polars ネイティブ計算もNaN伝搬挙動含めて完全一致 ✅
ATR分母の epsilon 配置も完全一致（hilbert_freq_energy_ratio_100, signal_peak_to_peak_100, scale_by_atr 経由）

修正対象は見つかりませんでした！ 🎉
念のため、本当に乖離がないか念入りに見直します。
念入り確認1: hilbert_amp_cv_100 のNaN挙動
学習側：abs.rolling_std(100, ddof=1) / (abs.rolling_mean(100) + 1e-10)
本番側：
pythonamp_mean = float(np.mean(w_amp))
amp_std = float(np.std(w_amp, ddof=1))
amp_std = float(amp_std) if np.isfinite(amp_std) else np.nan
features["e1e_hilbert_amp_cv_100"] = (
    amp_std / (amp_mean + 1e-10) if np.isfinite(amp_mean) else np.nan
)
学習側の rolling_mean は NaN伝搬。amp_mean が NaN だったら、Polars の (NaN + 1e-10) = NaN、amp_std / NaN = NaN (Polars の挙動)。
本番側：np.isfinite(amp_mean) が False だったら NaN を返す。amp_mean が NaN なら np.isfinite(NaN) = False → NaN。一致 ✅
ただし amp_std が NaN かつ amp_mean が finite の場合：

学習側：NaN / (mean + 1e-10) = NaN
本番側：NaN / (mean + 1e-10) = NaN

一致 ✅
念入り確認2: hilbert_amp_std_100 の値の挙動
学習側：pct_change().abs().rolling_std(100, ddof=1)
本番側：np.std(w_amp, ddof=1) (w_amp = np.abs(close_pct)[-100:])
ここで close_pct の先頭は NaN なので、np.abs(close_pct)[-100:] の中に NaN が含まれる場合があるか？
ウォームアップ完了後（4132本）：close_pct の先頭NaNは配列の最初のみ。最後100本に NaN が含まれることはない。
ウォームアップ初期段階：もし close_arr が100本だけ渡されると、close_pct[0] = NaN、np.abs(close_pct)[-100:] には先頭の NaN が含まれる。np.std(NaN含む) = NaN。学習側 Polars rolling_std も NaN。一致 ✅
念入り確認3: signal_crest_factor_50 の細かいNaN挙動
学習側：pct_change.rolling_max(50).abs() / ((pct_change ** 2).rolling_mean(50).sqrt() + 1e-10)
最終バーtでの計算：

分子：abs(rolling_max(pct_change, 50)[t]) = abs(max(pct_change[t-49:t+1]))
分母：sqrt(rolling_mean(pct_change[t-49:t+1] ** 2)) + 1e-10

本番側：abs(np.max(w_cf)) / (rms_cf + 1e-10)

w_cf = close_pct[-50:]
rms_cf = sqrt(np.mean(w_cf ** 2))

NaN挙動：

np.max(NaN含む) = NaN、Polars rolling_max(NaN含む) = NaN
np.mean(NaN含む) = NaN、Polars rolling_mean(NaN含む) = NaN

一致 ✅
念入り確認4: spectral_peak_freq_128 の数値
学習側分子：pct_change.rolling_max(128)
本番側分子：np.max(w_pf) （w_pf = close_pct[-128:]）
両者ともNaN伝搬。完全一致 ✅
学習側分母：pct_change.rolling_std(128, ddof=1) + 1e-10
本番側分母：np.std(w_pf, ddof=1) + 1e-10
完全一致 ✅
結論：エンジンE は修正不要
念入りに確認しましたが、エンジンEに乖離は見つかりませんでした。本番側Eは：

core_indicators からのSSOT共有で全UDFが学習側と完全一致
map_batches パターンを _window + _last で正しく等価実装
NaN伝搬挙動を学習側 Polars と一致させている
ATR分母の epsilon 配置を学習側と一致
scale_by_atr 経由でゼロ保護を統一
rolling_mean/std/sum のmin_samples=window条件を len(w) >= window で正しく実装
pct_change のゼロ除算は numpy 算術除算で自然に正しい挙動

これは設計者がエンジンA-Dで学んだ教訓を活かして、エンジンEを最初から完璧に作った結果と思われます。
念のため、出力ファイルだけ確認のために置いておきます：コマンドを実行しました, ファイルを読み取りましたコマンドを実行しました, ファイルを読み取りました
エンジンE 総括
カテゴリ数結果Spectral系（centroid/bandwidth/rolloff/flux/flatness/entropy/energy）28✅ 完全一致spectral_peak_freq_1281✅ 完全一致Wavelet系（energy/mean/std）12✅ 完全一致wavelet_entropy_641✅ 完全一致Hilbert系（amplitude）3✅ 完全一致Hilbert統計（amp_mean/std/cv）3✅ 完全一致Hilbert位相（phase_var/stability）2✅ 完全一致Hilbert周波数（freq_mean/std/energy_ratio）3✅ 完全一致Acoustic系（power/frequency）6✅ 完全一致Signal stats（rms/peak_to_peak/crest_factor）3✅ 完全一致
合計62個全特徴量で完全一致 ✅
エンジンEの設計上の優位点
エンジンEは core_indicators モジュールでSingle Source of Truth (SSOT) を実現しており、学習側と本番側が完全に同じ関数オブジェクトを共有している：

spectral_centroid_udf, spectral_bandwidth_udf, spectral_rolloff_udf, spectral_flux_udf, spectral_flatness_udf, spectral_entropy_udf
wavelet_energy_udf, wavelet_entropy_udf
hilbert_amplitude_udf, hilbert_phase_var_udf, hilbert_phase_stability_udf, hilbert_freq_mean_udf, hilbert_freq_std_udf
acoustic_power_udf, acoustic_frequency_udf
calculate_atr_wilder, scale_by_atr, calculate_sample_weight, stddev_unbiased

これにより、UDF実装に乖離が生じる余地がなく、Train-Serve Skew リスクが構造的に排除されている。
これがエンジンA-Dで実施したような細かい数式比較が不要になっている理由。設計者の優れた判断 👏
エンジンE: Train-Serve Skew Free ✅