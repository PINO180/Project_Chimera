エンジンDの徹底比較を開始します。3個のコマンドを実行しました3個のコマンドを実行しましたwindow_sizesを確認：学習側Dのwindow_sizes学習側Dのwindow_sizesvolatility = [10, 20, 30, 50]、volume = [13, 21, 34]、general = [10, 20, 50, 100]
学習側Dの全特徴量を整理：学習側Dの全特徴量リスト学習側Dの全特徴量リスト本番側を確認します：本番側Dの全特徴量出力本番側Dの全特徴量出力基底名は学習側と一致 ✅。sample_weight は本番側でも生成されている（が AI推論では使われないのでOK）。
次に各UDFの実装を比較します。まず本番側Dの構造を確認：本番1Dの構造本番1Dの構造本番1Dの calculate_features を見ます：3個のファイルを表示しました3個のファイルを表示しました続けます。学習側Dの各特徴量の計算式を順に確認していきます。
グループ1: HV（Historical Volatility）系学習側HV計算式学習側HV計算式hv_annual_252の重要な乖離発見
学習側： pct_change().rolling_std(252, ddof=1) * sqrt(252) ← rolling_stdを直接使用
本番側： hv_standard_udf(pct_252) * np.sqrt(252) ← hv_standard_udfを使用
学習側の hv_standard_udf と rolling_std(ddof=1) は同じ値を返すか確認します：学習側のhv_standard_udf学習側のhv_standard_udf重要な乖離：
hv_standard_udf vs Polars rolling_std(ddof=1):
学習側 hv_standard_udf：

finite_returns = returns[np.isfinite(returns)] ← NaN/infを除外
len < 5 で NaN return
sqrt(sum((x - mean)² ) / (n - 1))

学習側 rolling_std(252, ddof=1)：

Polars rolling_std はウィンドウ内に NaN が1つでもあれば NaN を返す（NaN伝搬）

これは挙動が異なる！PolarsのNaN挙動確認PolarsのNaN挙動確認やはり Polars rolling_std は ウィンドウ内に NaN があれば NaN を返す。
一方、学習側 hv_standard_udf は finite フィルタで NaN を除外して計算する。これは学習側内部での実装乖離（rolling_stdとUDFで挙動が異なる）。
しかし学習側のpct_changeは「先頭のみNaN」（1本目の差分は計算不能）。実用上は252本目以降では NaN が含まれないので、両者の値は同じになる。
つまり：

学習側の hv_annual_252 = rolling_std(252, ddof=1) * sqrt(252) （Polarsベース、ウィンドウ内NaNあれば結果NaN）
本番側の hv_annual_252 = hv_standard_udf(pct_252) * sqrt(252)（UDFベース、NaN除外して計算）

ウォームアップ4132本以上では実害は無いが、厳密には実装が異なる。これは修正対象です。
hv_robust_annual_252の挙動確認Hv_robust_udf確認Hv_robust_udf確認hv_standard_udfとhv_robust_udfは学習側と本番側で完全一致 ✅
ただ問題は 学習側 hv_annual_252 が rolling_std を使っており、UDFを使っていない。本番側では hv_standard_udf を使っている。
修正対象1: hv_annual_252
学習側rolling_std(252, ddof=1):

ウィンドウ全252本に NaN が含まれれば → NaN
そうでなければ ddof=1 のstd

本番側 hv_standard_udf:

finite フィルタで NaN を除外（ウィンドウ内に NaN があっても計算）
finite < 5 なら NaN

ウォームアップ後（pct_change の先頭NaN以降）は両者一致。しかし厳密には実装が異なるため、本番側を学習側に合わせて修正します。
グループ2: Chaikin Volatility, Mass IndexChaikin VolatilityChaikin VolatilityChaikin Volatility完全一致 ✅Mass IndexMass IndexMass Index完全一致 ✅
グループ3: CMF, MFI, VWAPCMFCMFCMF完全一致 ✅MFIMFIMFI完全一致 ✅VWAPVWAPVWAP完全一致 ✅
vwap_distの計算式比較
学習側：学習側のvwap_dist学習側のvwap_dist重要発見：
学習側 vwap_dist：(close - vwap) / (atr_13 + 1e-10) ← 直接 +1e-10
本番側 vwap_dist：scale_by_atr(close - vwap, atr13)[-1] ← scale_by_atr 経由（内部で + 1e-10）
scale_by_atr の実装：
pythonout[i] = target_arr[i] / (atr_arr[i] + epsilon)  # epsilon=1e-10
これは学習側と完全同等 ✅
vol_ma1440 の比較
学習側： vol_ma1440 = pl.col("volume").rolling_mean(lookback_bars) + 1e-10

バーごとに rolling_mean(1440) を計算
末尾に + 1e-10

本番側： vol_ma1440 = float(np.mean(_window(volume_arr, lookback_bars)))

最終バー時点の固定値


1e-10 なし（使用箇所で個別に + 1e-10）



これはエンジンBで修正したのと同じパターンの乖離です ⚠️
エンジンBでは「各バーごとに rel_volume を計算」する必要がありました。エンジンDでも同じ問題があります。
obv_rel と accumulation_distribution_rel
学習側：
pythonobv_rel = obv_raw.diff() / vol_ma1440  # vol_ma1440 = rolling_mean(1440) + 1e-10、各バーで再計算
本番側：
pythonobv_diff = np.diff(obv_arr, prepend=np.nan)
features["e1d_obv_rel"] = float(obv_diff[-1] / (vol_ma1440 + 1e-10))  # 最終バー固定
乖離あり ⚠️ ただし、obv_rel = obv.diff() / vol_ma1440 で、現バー時点での obv.diff()[-1] = obv[-1] - obv[-2] と vol_ma1440[-1] のみが必要なので、最終バーの値だけなら本番側の計算は学習側のバー[-1]と完全一致。
具体的に：

学習側のバー t での値：(obv[t] - obv[t-1]) / (rolling_mean_1440[t] + 1e-10)
本番側の値：(obv[-1] - obv[-2]) / (np.mean(volume[-1440:]) + 1e-10)

rolling_mean_1440[t] = np.mean(volume[t-1439:t+1]) で、volume_arr の最終バーが t に対応する場合、本番側の np.mean(volume[-1440:]) = np.mean(volume[t-1439:t+1]) と完全一致。
→ obv_rel と accumulation_distribution_rel は 最終バーの値だけ取り出すなら完全一致 ✅
force_index_norm の比較
学習側：学習側のforce_index_norm学習側のforce_index_normforce_index_norm
学習側：force_raw / (atr_13_internal_expr * vol_ma1440 + 1e-10)

vol_ma1440 = rolling_mean(1440) + 1e-10（既に+1e-10含む）
分母 = atr * (rolling_mean(1440) + 1e-10) + 1e-10

本番側：force_raw / (atr13_last * vol_ma1440 + 1e-10)

vol_ma1440 = np.mean(volume[-1440:])（+ 1e-10なし）
分母 = atr * np.mean(volume[-1440:]) + 1e-10

乖離発見 ⚠️：
学習側分母：atr * (mean + 1e-10) + 1e-10 = atr*mean + atr*1e-10 + 1e-10
本番側分母：atr * mean + 1e-10
差は atr * 1e-10。XAU/USDの atr ≈ 5 なら 5 * 1e-10 = 5e-10 の差。でもこれが実害観測不能ではダメで、思想として揃える必要あり。
volume_ma20_rel と volume_ratio
学習側：

volume_ma20_rel = rolling_mean(volume, 20) / vol_ma1440
ここで vol_ma1440 = rolling_mean(1440) + 1e-10 （各バーで計算）
volume_ratio = volume / rolling_mean(volume, 20) （各バーで計算）

本番側：

volume_ma20_rel = vol_ma20 / (vol_ma1440 + 1e-10)
ここで vol_ma20 = np.mean(volume[-20:])、vol_ma1440 = np.mean(volume[-1440:]) （最終バーのみ）
volume_ratio = volume[-1] / (vol_ma20 + 1e-10)

この場合、最終バーtでの値が学習側と一致するか確認：
学習側の最終バーtでの値：

rolling_mean(volume, 20)[t] = mean(volume[t-19:t+1])
rolling_mean(volume, 1440)[t] = mean(volume[t-1439:t+1])
vol_ma1440[t] = mean(volume[t-1439:t+1]) + 1e-10
volume_ma20_rel[t] = mean(volume[t-19:t+1]) / (mean(volume[t-1439:t+1]) + 1e-10)

本番側：

vol_ma20 = np.mean(volume[-20:]) = mean(volume[t-19:t+1]) ✅
vol_ma1440 = np.mean(volume[-1440:]) = mean(volume[t-1439:t+1]) ✅
volume_ma20_rel = mean(volume[-20:]) / (mean(volume[-1440:]) + 1e-10)

最終バーの値だけなら完全一致 ✅
volume_ratio も同様：

学習側： volume[t] / rolling_mean(volume, 20)[t] = volume[t] / mean(volume[t-19:t+1])
本番側： volume[-1] / (np.mean(volume[-20:]) + 1e-10)

ただし学習側はゼロ保護なし、本番側は + 1e-10 あり ⚠️
volume_price_trend_norm
学習側：(close.pct_change() * volume).rolling_mean(10) / vol_ma1440

vol_ma1440 = rolling_mean(1440) + 1e-10（各バーで計算）

本番側：np.mean(close_pct * volume[-10:]) / (vol_ma1440 + 1e-10)

vol_ma1440 = np.mean(volume[-1440:])（最終バーのみ）

学習側のバーtでの値：

(close.pct_change() * volume).rolling_mean(10)[t] = (1/10) * Σ_{i=t-9}^{t} (pct_change[i] * volume[i])
分母：mean(volume[t-1439:t+1]) + 1e-10

本番側：

分子：np.mean(close_pct * volume[-10:])
分母：mean(volume[-1440:]) + 1e-10

ここで本番側の分子は close_pct[-10:] * volume[-10:] の平均 ですが、close_pct は全体配列から計算された pct_change なので、close_pct[t-9:t+1] の値が学習側と一致する。
つまり最終バーの値だけ取り出せば一致 ✅
詳細：最終バー計算における等価性まとめ
ここまでの解析で：

obv_rel、accumulation_distribution_rel：最終バー値は完全一致 ✅
volume_ma20_rel、volume_ratio、volume_price_trend_norm：最終バー値は完全一致 ✅
force_index_norm：わずかに乖離（分母の epsilon 配置） ⚠️

Bとは違う点
エンジンBでは volume_ma20 が rel_volume[i].rolling_mean(20) で、各バーの rel_volume を計算してから20本平均する必要があった。
エンジンDでは volume_ma20_rel = rolling_mean(volume, 20) / vol_ma1440 で、volumeの20本平均をvol_ma1440で割るだけ。最終バーで rolling_mean(volume, 20)[-1] = mean(volume[-20:]) なので、本番側の実装で正しい。
→ エンジンBの修正パターンとは異なる ✅
残りの確認
Donchian / Price Channel学習側のDonchian Channel学習側のDonchian Channel重要：学習側でも price_channel = donchian と完全同一の式（max/minを再計算しているだけ）。本番側のコメントに「学習側でも同値」と記載されている通り ✅
Pivot
学習側：
pythonprev_high = pl.col("high").rolling_max(20).shift(1)
prev_low = pl.col("low").rolling_min(20).shift(1)
prev_close = pl.col("close").shift(1)
pivot = (prev_high + prev_low + prev_close) / 3.0
最終バーtでの pivot[t]：

rolling_max(high, 20).shift(1)[t] = max(high[t-20:t]) （shift(1)で1バー前の値）
rolling_min(low, 20).shift(1)[t] = min(low[t-20:t])
close.shift(1)[t] = close[t-1]

本番側：
pythonprev_high_20 = float(np.max(high_arr[-21:-1]))   # high[t-20:t]
prev_low_20 = float(np.min(low_arr[-21:-1]))     # low[t-20:t]
prev_close_1 = float(close_arr[-2])               # close[t-1]
ここで high_arr[-21:-1] は インデックス [-21, -20, ..., -2] の20個 = high[t-20:t]。
学習側 rolling_max(20)[t-1] = max(high[t-20:t]) = high の長さ20のウィンドウ（high[t-20], ..., high[t-1]）。
→ 完全一致 ✅2個のファイルを表示しました2個のファイルを表示しましたすべてのprice action特徴量で学習側は + 1e-10 のゼロ保護を持つ。
本番側を比較：本番側のprice action詳細再確認本番側のprice action詳細再確認すべて完全一致 ✅
Donchian/Price Channel本番側 ATR分母の確認本番側Donchian分母本番側Donchian分母Donchian/Price Channel 完全一致 ✅
CCICCICCICCI完全一致 ✅
OBVOBVOBVOBV完全一致 ✅
Accumulation/DistributionA/D LineA/D LineA/D Line完全一致 ✅
Force Index UDFForce Index UDF学習側Force Index UDF学習側重要発見：
学習側の force_index_udf は配列を返しますが、本番側では：
python# Force Index normalized
if _atr_valid and len(close_arr) >= 2:
    price_change = close_arr[-1] - close_arr[-2]
    force_raw = price_change * float(volume_arr[-1])
    features["e1d_force_index_norm"] = float(
        force_raw / (atr13_last * vol_ma1440 + 1e-10)
    )
本番側は force_index_udf を呼ばず、直接 (close[-1] - close[-2]) * volume[-1] を計算。
学習側 force_index_udf は：

i = 0: result[0] = 0.0
i >= 1: (close[i] - close[i-1]) * volume[i]

そして学習側 expr：force_raw / (atr_13 * vol_ma1440 + 1e-10)
本番側の最終バー値：

(close[-1] - close[-2]) * volume[-1] / (atr * vol_ma1440 + 1e-10)

これは学習側のバーtでの値：

(close[t] - close[t-1]) * volume[t] / (atr[t] * vol_ma1440[t] + 1e-10)

最終バーでの値は計算式上は同じですが、vol_ma1440の扱いが異なる：
学習側： vol_ma1440 = pl.col("volume").rolling_mean(1440) + 1e-10

バー t での vol_ma1440[t] = mean(volume[t-1439:t+1]) + 1e-10
学習側 expr の分母： atr[t] * (mean(volume[t-1439:t+1]) + 1e-10) + 1e-10

本番側：vol_ma1440 = np.mean(volume[-1440:])

分母： atr * np.mean(volume[-1440:]) + 1e-10

乖離：

学習側分母：atr * mean(volume_1440) + atr * 1e-10 + 1e-10
本番側分母：atr * mean(volume_1440) + 1e-10

差は atr * 1e-10。XAU/USDの atr ≈ 5 なら 5e-10 ≈ epsilonの僅差。修正対象 ⚠️
Fibonacci Levels UDFFibonacci UDFFibonacci UDFFibonacci完全一致 ✅
Candlestick PatternsCandlestickCandlestickCandlestick完全一致 ✅
修正対象まとめ
エンジンDで発見した乖離をまとめます：
乖離1: hv_annual_252の実装
学習側： pct_change().rolling_std(252, ddof=1) * sqrt(252) ← Polars rolling_std使用
本番側： hv_standard_udf(pct_252) * sqrt(252) ← UDF使用
両者の違い：

学習側 Polars rolling_std はウィンドウ内NaN→NaN
本番側 hv_standard_udf は finite フィルタで NaN除外

乖離2: force_index_norm の分母 epsilon配置
学習側： force_raw / (atr * (vol_ma1440 + 1e-10) + 1e-10)

vol_ma1440自体が rolling_mean(1440) + 1e-10
学習側分母 = atr*mean(vol_1440) + atr*1e-10 + 1e-10

本番側： force_raw / (atr * vol_ma1440 + 1e-10)

vol_ma1440 = np.mean(volume[-1440:])（+1e-10なし）
本番側分母 = atr*mean(vol_1440) + 1e-10

差は atr * 1e-10。
乖離3: volume_ratio のゼロ保護
学習側： volume / rolling_mean(volume, 20) （ゼロ保護なし）
本番側： volume[-1] / (vol_ma20 + 1e-10) （ゼロ保護あり）
学習側はゼロ保護なし。Polarsは0除算で inf を返す。本番側は + 1e-10 でゼロ保護。本番側を学習側に合わせるべきです。
乖離4 (sample_weight): 学習側の生成位置
学習側では sample_weight を engine_1_D 内で生成。本番側でも生成しているが、これは AI 推論には使われない（DataFrameのサンプル重みのみ）。
実装上の差分なし ✅
修正開始本番側1D修正対象部分の再確認本番側1D修正対象部分の再確認修正1: hv_annual_252を学習側 rolling_std(252, ddof=1) 相当に変更Hv_annual_252を学習側 Polars rolling_std(252, ddof=1) 相当の挙動に変更Hv_annual_252を学習側 Polars rolling_std(252, ddof=1) 相当の挙動に変更修正2: force_index_norm の分母 epsilon配置Force_index_normの分母を学習側と完全一致させるForce_index_normの分母を学習側と完全一致させる修正3: volume_ratio のゼロ保護を学習側と一致させるVolume_ratioのゼロ保護を学習側と完全一致させる（保護なし）Volume_ratioのゼロ保護を学習側と完全一致させる（保護なし）確認：sma + 1e-10系のゼロ保護も学習側に揃っているか
obv_rel と accumulation_distribution_rel のゼロ保護を再確認します：
学習側： vol_ma1440 = rolling_mean(1440) + 1e-10

obv_rel = obv_raw.diff() / vol_ma1440 = obv_raw.diff() / (rolling_mean(1440) + 1e-10)

本番側：
pythonvol_ma1440 = float(np.mean(_window(volume_arr, lookback_bars)))  # +1e-10なし
features["e1d_obv_rel"] = float(obv_diff[-1] / (vol_ma1440 + 1e-10))
完全一致 ✅（vol_ma1440 + 1e-10 = mean + 1e-10）
同様にaccumulation_distribution_rel、volume_ma20_rel、volume_price_trend_normも：
学習側 volume_ma20_rel： pl.col("volume").rolling_mean(20) / vol_ma1440 = mean(vol_20) / (mean(vol_1440) + 1e-10)
本番側： vol_ma20 / (vol_ma1440 + 1e-10) = mean(vol_20) / (mean(vol_1440) + 1e-10) ✅
学習側 volume_price_trend_norm： (close.pct_change() * volume).rolling_mean(10) / vol_ma1440

分母 = mean(vol_1440) + 1e-10 ✅

本番側：
pythonvpt_window = _window(close_pct * volume_arr, 10)
features["e1d_volume_price_trend_norm"] = float(np.mean(vpt_window) / (vol_ma1440 + 1e-10))

分母 = mean(vol_1440) + 1e-10 ✅

完全一致 ✅
確認：Donchian / Pivot 系の分母 epsilon
学習側：(donchian_upper - close) / (atr_13_internal_expr + 1e-10)
本番側：(don_upper - close[-1]) / (atr13_last + 1e-10)
完全一致 ✅
確認：close_pct のゼロ保護
学習側：pl.col("close").pct_change() （内部で 0/0 → NaN/inf）
本番側 pct_change_numba：本番側pct_change_numba本番側pct_change_numba本番側 pct_change_numba は prev == 0 なら np.inf を返す。これは符号を考慮していない：

arr[i] > 0 かつ prev = 0 → 学習側 Polars: (x - 0) / 0 = +inf
arr[i] < 0 かつ prev = 0 → 学習側 Polars: (x - 0) / 0 = -inf
arr[i] = 0 かつ prev = 0 → 学習側 Polars: 0 / 0 = NaN

XAU/USDの価格は常に正なので arr[i] > 0、prev も 0 にならないため実害はない。思想として揃えるためには修正したほうがよいでしょう。
修正4: pct_change_numba を学習側 Polars と完全一致にPct_change_numbaを学習側Polarsと完全一致させるPct_change_numbaを学習側Polarsと完全一致させる確認：hv_regime_50
学習側の hv_regime_50：
pythonhv_50 = pl.col("close").pct_change().rolling_std(50, ddof=1)
q80_roll = hv_50.rolling_quantile(0.8, window_size=1440)
q60_roll = hv_50.rolling_quantile(0.6, window_size=1440)
((hv_50 > q80_roll).cast(pl.Int8) + (hv_50 > q60_roll).cast(pl.Int8))
つまり：

hv_50[t] = std(pct_change[t-49:t+1], ddof=1) （Polars rolling_std、NaNあれば NaN）
q80_roll[t] = quantile(hv_50[t-1439:t+1], 0.8) （Polars rolling_quantile）
結果： (hv_50[t] > q80_roll[t]) + (hv_50[t] > q60_roll[t])

本番側：
pythoncur_hv50 = hv_standard_udf(_window(close_pct, 50))  # ← UDFは finite除外
n_needed = 1440 + 50
if len(close_pct) >= n_needed and np.isfinite(cur_hv50):
    hist_pct = close_pct[-n_needed:]
    hv50_hist = np.full(n_needed, np.nan, dtype=np.float64)
    for _i in range(50 - 1, n_needed):
        hv50_hist[_i] = hv_standard_udf(hist_pct[_i - 49 : _i + 1])
    hv50_window = hv50_hist[-1440:]
    hv50_finite = hv50_window[np.isfinite(hv50_window)]
    if len(hv50_finite) >= 10:
        q80 = float(np.percentile(hv50_finite, 80, method="nearest"))
        q60 = float(np.percentile(hv50_finite, 60, method="nearest"))
        features["e1d_hv_regime_50"] = float(int(cur_hv50 > q80) + int(cur_hv50 > q60))
乖離発見 ⚠️：

cur_hv50 の計算：hv_standard_udf は finite フィルタで NaN除外。学習側 Polars rolling_std(50, ddof=1) はウィンドウ内に NaN があれば NaN。
hv50_hist の計算：各バーで hv_standard_udf(...) を使用。これも同じ問題。
rolling_quantile vs np.percentile：

学習側 Polars rolling_quantile(0.8, window_size=1440) は デフォルトで interpolation="nearest" （ドキュメント上は確認必要）
本番側 np.percentile(..., 80, method="nearest") ← method="nearest" 指定



これは確認必要：Polars rolling_quantileのデフォルトPolars rolling_quantileのデフォルト確認：Polars rolling_quantile のデフォルトは interpolation='nearest'。本番側の np.percentile(..., method="nearest") と一致 ✅
ただし重要な違い：

Polars は window内にNaNがあるとNaN？それともスキップ？ 確認が必要
PolarsのrollingquantileのNaN挙動PolarsのrollingquantileのNaN挙動重要：Polars rolling_quantile はウィンドウ内のNaNを「スキップ」（インデックス4でNaN含むウィンドウでも 3.0 を返している）。
本番側もNaN除外しているので 挙動一致 ✅
さらに：本番側の cur_hv50 = hv_standard_udf(_window(close_pct, 50)) と学習側 hv_50[t] = rolling_std(close_pct, 50, ddof=1)[t] の差を考える必要があります：

本番側 hv_standard_udf: ウィンドウ内 NaN を除外して std 計算（finite_returns）
学習側 rolling_std: ウィンドウ内 NaN があれば結果 NaN

ウォームアップ後（pct_changeの先頭NaN以降）はどちらも同じ値（NaNがウィンドウ内に含まれない）。実害なし。
ただし思想として揃える：本番側を学習側 Polars rolling_std と完全一致させるべき。
修正5: hv_regime_50 の cur_hv50 と hv50_hist 計算を学習側 rolling_std と一致させるファイルを表示しました, ファイルを編集しましたファイルを表示しました, ファイルを編集しました確認：hv_standard_{10,20,30,50} と hv_robust_{10,20,30,50}
学習側：
pythonexpressions[f"{p}hv_standard_{window}"] = (
    pl.col("close")
    .pct_change()
    .rolling_map(
        lambda s: hv_standard_udf(s.to_numpy()),
        window_size=window,
        min_samples=window,
    )
)
これは rolling_map で UDF を適用しています。だから hv_standard_udf を使うのは正しい！
pythonexpressions[f"{p}hv_robust_{window}"] = (
    pl.col("close")
    .pct_change()
    .rolling_map(
        lambda s: hv_robust_udf(s.to_numpy()),
        window_size=window,
        min_samples=window,
    )
)
これも hv_robust_udf を使う。
つまり hv_standard_{w} と hv_robust_{w} は本番側で UDF を使うのが正解 ✅
しかし hv_annual_252 は rolling_std を使うので UDF ではない。これが今回の修正点。
hv_robust_annual_252 は rolling_map(hv_robust_udf, window=252) なので UDF を使うのが正解 ✅（変更不要）
構文チェックと出力2個のコマンドを実行しました, ファイルを読み取りました2個のコマンドを実行しました, ファイルを読み取りました
修正内容まとめ
修正1: hv_annual_252 の実装変更
修正前：hv_standard_udf(pct_252) * np.sqrt(252)

UDFは finite フィルタで NaN除外して計算

修正後：np.std(pct_252, ddof=1) * np.sqrt(252) （ウィンドウ全体が有限値の場合のみ）

学習側 pl.col("close").pct_change().rolling_std(252, ddof=1) と完全一致
ウィンドウ内に1本でもNaN/infあれば NaN（Polars 挙動）

修正2: force_index_norm の分母 epsilon配置
修正前：force_raw / (atr13_last * vol_ma1440 + 1e-10)

分母 = atr * mean(vol_1440) + 1e-10

修正後：force_raw / (atr13_last * (vol_ma1440 + 1e-10) + 1e-10)

学習側 force_raw / (atr_13 * (rolling_mean(1440) + 1e-10) + 1e-10) と完全一致
分母 = atr * (mean(vol_1440) + 1e-10) + 1e-10 = atr*mean + atr*1e-10 + 1e-10

修正3: volume_ratio のゼロ保護削除
修正前：volume[-1] / (vol_ma20 + 1e-10)
修正後：volume[-1] / vol_ma20 （vol_ma20=0 のとき符号付き infを返す）

学習側 volume / rolling_mean(volume, 20) と完全一致（Polarsはゼロ除算で符号付き inf）

修正4: pct_change_numba のゼロ除算挙動
修正前：prev == 0 のとき常に np.inf
修正後：prev == 0 のとき：

cur > 0 → +inf
cur < 0 → -inf
cur == 0 → NaN

学習側 Polars/numpy のゼロ除算と完全一致（符号保持）。
修正5: hv_regime_50 の HV50 計算を rolling_std と一致
修正前：hv_standard_udf(...) 使用 ← finite除外で計算
修正後：_rolling_std_50_at(...) 使用 ← ウィンドウ内NaNあれば NaN
学習側 pl.col("close").pct_change().rolling_std(50, ddof=1) と完全一致。

エンジンD総括
カテゴリ結果全66特徴量（+ sample_weight）✅ 完全一致HV系（hv_standard/hv_robust × 4 windows + annual + robust_annual + regime）✅ 完全一致（修正後）Chaikin/Mass✅ 完全一致CMF/MFI/VWAP系✅ 完全一致OBV/AD/Force Index✅ 完全一致（修正後）Volume MA/Ratio/VPT系✅ 完全一致（修正後）Donchian/Price Channel✅ 完全一致CCI✅ 完全一致Pivot/Fib✅ 完全一致Candlestick✅ 完全一致Price Action系✅ 完全一致pct_change_numba✅ 完全一致（修正後）
エンジンD全66特徴量で Train-Serve Skew Free ✅