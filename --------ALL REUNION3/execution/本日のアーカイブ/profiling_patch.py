# profiling_patch.py
# =====================================================================
# 【使い方】
#   realtime_feature_engine.py の RealtimeFeatureEngine クラスに
#   このファイルの2つのメソッドをそのまま貼り替える。
#
#   変更箇所は「★ PERF」コメントで全てマークしてある。
#   本番に戻すときは PROFILING_ENABLED = False にするだけでログが止まる。
# =====================================================================

import time
import logging
import numpy as np

PROFILING_ENABLED = True  # False にすれば無音・ほぼゼロオーバーヘッド

# ======================================================================
# ① process_new_m1_bar  （RealtimeFeatureEngine のメソッドとして貼り替え）
# ======================================================================


def process_new_m1_bar(self, m1_bar, market_proxy_cache):
    """
    [メインループ・計測版]
    各ステップの所要時間を perf_counter で計測し INFO ログに出力する。
    """
    signal_list = []
    _t = time.perf_counter
    log = self.logger

    try:
        m1_timestamp = m1_bar["timestamp"]

        # ------------------------------------------------------------------
        # STEP 1: M1バッファ更新
        # ------------------------------------------------------------------
        t0 = _t()
        self.m1_dataframe.append(m1_bar)
        m1_bar_df = __import__("pandas").DataFrame([m1_bar]).set_index("timestamp")
        self._append_bar_to_buffer("M1", m1_bar_df, market_proxy_cache)
        t1 = _t()
        if PROFILING_ENABLED:
            log.info(
                f"[PERF] STEP1 M1バッファ更新:           {(t1 - t0) * 1000:7.1f} ms"
            )

        # ------------------------------------------------------------------
        # STEP 2: 全時間足リサンプリング
        # ------------------------------------------------------------------
        t2 = _t()
        newly_closed_timeframes = {}
        for tf_name, rule in self.TF_RESAMPLE_RULES.items():
            if tf_name not in self.data_buffers:
                continue
            tf_t0 = _t()
            new_timestamps = self._resample_and_update_buffer(
                tf_name, rule, market_proxy_cache
            )
            if PROFILING_ENABLED:
                log.info(
                    f"[PERF]   resample {tf_name:<5}: {(_t() - tf_t0) * 1000:6.1f} ms"
                    f"  (new_bars={len(new_timestamps)})"
                )
            if new_timestamps:
                newly_closed_timeframes[tf_name] = new_timestamps
        t3 = _t()
        if PROFILING_ENABLED:
            log.info(
                f"[PERF] STEP2 全TFリサンプリング合計:    {(t3 - t2) * 1000:7.1f} ms"
            )

        newly_closed_timeframes["M1"] = [m1_timestamp]

        # ------------------------------------------------------------------
        # STEP 3〜7: M3確定時のみ実行
        # ------------------------------------------------------------------
        if "M3" not in newly_closed_timeframes:
            return signal_list

        m3_timestamp = newly_closed_timeframes["M3"][-1]
        t_step3_total = 0.0
        t_step4_total = 0.0
        t_step5_total = 0.0

        if PROFILING_ENABLED:
            log.info(f"[PERF] *** M3確定 @ {m3_timestamp} 全時間足強制再計算開始 ***")

        # M3確定時：全時間足を強制再計算（学習側と一致）
        for tf_name in self.ALL_TIMEFRAMES.keys():
            if not self.is_buffer_filled.get(tf_name, False):
                continue

            try:
                data = {
                    col: np.array(self.data_buffers[tf_name][col], dtype=np.float64)
                    for col in self.OHLCV_COLS
                }

                # --- STEP 3: _calculate_base_features (1A〜1F) ---
                ts3a = _t()
                base_features = self._calculate_base_features(data, tf_name)
                ts3b = _t()
                t_step3_total += ts3b - ts3a
                if PROFILING_ENABLED:
                    log.info(
                        f"[PERF] STEP3 base_features [{tf_name}]:"
                        f" {(ts3b - ts3a) * 1000:7.1f} ms"
                        f"  (n_features={len(base_features)})"
                    )

                # --- STEP 4: _update_incremental_ols ---
                ts4a = _t()
                self._update_incremental_ols(
                    tf_name, base_features, market_proxy_cache, m3_timestamp
                )
                ts4b = _t()
                t_step4_total += ts4b - ts4a
                if PROFILING_ENABLED:
                    log.info(
                        f"[PERF] STEP4 update_ols     [{tf_name}]:"
                        f" {(ts4b - ts4a) * 1000:7.1f} ms"
                    )

                # --- STEP 5: _calculate_neutralized_features ---
                ts5a = _t()
                neutralized = self._calculate_neutralized_features(
                    base_features, tf_name, m3_timestamp, market_proxy_cache
                )
                ts5b = _t()
                t_step5_total += ts5b - ts5a
                if PROFILING_ENABLED:
                    log.info(
                        f"[PERF] STEP5 neutralize_ols [{tf_name}]:"
                        f" {(ts5b - ts5a) * 1000:7.1f} ms"
                        f"  (n_neutralized={len(neutralized)})"
                    )

                self.latest_features_cache[tf_name] = neutralized

            except Exception as e:
                self.logger.warning(f"{tf_name} 特徴量キャッシュ更新失敗: {e}")

        if PROFILING_ENABLED:
            log.info(
                f"[PERF] ========== STEP3〜5 合計 =========="
                f"\n[PERF]   STEP3 base_features  合計: {t_step3_total * 1000:7.1f} ms"
                f"\n[PERF]   STEP4 update_ols     合計: {t_step4_total * 1000:7.1f} ms"
                f"\n[PERF]   STEP5 neutralize_ols 合計: {t_step5_total * 1000:7.1f} ms"
            )

        # --- STEP 6: シグナルチェック（M3のみ）---
        ts6a = _t()
        V5_check_result = self._check_for_signal("M3", m3_timestamp)
        ts6b = _t()
        if PROFILING_ENABLED:
            log.info(
                f"[PERF] STEP6 check_signal   [M3]:"
                f" {(ts6b - ts6a) * 1000:7.1f} ms"
                f"  is_V5={V5_check_result.get('is_V5')}"
            )

        if V5_check_result["is_V5"]:
            # --- STEP 7: calculate_feature_vector ---
            ts7a = _t()
            feature_vector = self.calculate_feature_vector(
                "M3", m3_timestamp, market_proxy_cache
            )
            ts7b = _t()
            if PROFILING_ENABLED:
                log.info(
                    f"[PERF] STEP7 feature_vector [M3]: {(ts7b - ts7a) * 1000:7.1f} ms"
                )

            if feature_vector is not None:
                combined_features = dict(zip(self.feature_list, feature_vector[0]))
                signal = __import__(
                    "execution.realtime_feature_engine", fromlist=["Signal"]
                ).Signal(
                    features=feature_vector,
                    timestamp=m3_timestamp,
                    timeframe="M3",
                    market_info=V5_check_result["market_info"],
                    atr_value=V5_check_result["market_info"].get("atr_value", 0.0),
                    close_price=V5_check_result["market_info"].get(
                        "current_price", 0.0
                    ),
                    feature_dict=combined_features,
                )
                signal_list.append(signal)

        return signal_list

    except Exception as e:
        self.logger.error(f"process_new_m1_bar でエラー: {e}", exc_info=True)
        return []


# ======================================================================
# ② _calculate_base_features  （各モジュールを個別計測する版）
# ======================================================================


def _calculate_base_features(self, data, tf_name):
    """
    [特徴量ルーター・計測版]
    1A〜1F 各モジュールの処理時間を個別に計測する。
    """
    from execution.realtime_feature_engine_1A_statistics import FeatureModule1A
    from execution.realtime_feature_engine_1B_timeseries import FeatureModule1B
    from execution.realtime_feature_engine_1C_technical import FeatureModule1C
    from execution.realtime_feature_engine_1D_volume import FeatureModule1D
    from execution.realtime_feature_engine_1E_signal import FeatureModule1E
    from execution.realtime_feature_engine_1F_experimental import FeatureModule1F

    log = self.logger
    _t = time.perf_counter
    features = {}

    MODULES = [
        ("1A", FeatureModule1A),
        ("1B", FeatureModule1B),
        ("1C", FeatureModule1C),
        ("1D", FeatureModule1D),
        ("1E", FeatureModule1E),
        ("1F", FeatureModule1F),
    ]

    for tag, module in MODULES:
        try:
            ta = _t()
            result = module.calculate_features(data)
            tb = _t()
            features.update(result)
            if PROFILING_ENABLED:
                log.info(
                    f"[PERF]   Module {tag} [{tf_name}]:"
                    f" {(tb - ta) * 1000:6.1f} ms  (n={len(result)})"
                )
        except Exception as e:
            log.error(
                f"ベース特徴量の計算中にエラーが発生しました ({tf_name}/{tag}): {e}",
                exc_info=True,
            )

    # --- プロキシ特徴量（元のまま） ---
    def _window(arr, window):
        return arr[-window:] if len(arr) >= window else arr

    def _pct(arr):
        if len(arr) < 2:
            return np.full_like(arr, np.nan)
        arr_safe = arr[:-1].copy()
        arr_safe[arr_safe == 0] = 1e-10
        return np.concatenate(([np.nan], np.diff(arr) / arr_safe))

    close_pct = _pct(data["close"])
    high, low, close = data["high"], data["low"], data["close"]

    from core_indicators import calculate_atr_wilder

    if len(close) > 1:
        atr_arr = calculate_atr_wilder(
            high.astype(np.float64),
            low.astype(np.float64),
            close.astype(np.float64),
            self.ATR_CALC_PERIOD,
        )
        atr_last = float(atr_arr[-1]) if len(atr_arr) > 0 else np.nan
        features["atr"] = atr_last if np.isfinite(atr_last) else 0.0
    else:
        features["atr"] = 0.0

    features["log_return"] = (
        np.log((data["close"][-1] + 1e-10) / (data["close"][-2] + 1e-10))
        if len(data["close"]) > 1
        else 0.0
    )
    features["price_momentum"] = (
        data["close"][-1] - data["close"][-11] if len(data["close"]) > 10 else np.nan
    )
    features["rolling_volatility"] = (
        np.std(_window(close_pct, 20)) if len(close_pct) >= 20 else np.nan
    )
    features["volume_ratio"] = (
        data["volume"][-1] / (np.mean(_window(data["volume"], 20)) + 1e-10)
        if len(data["volume"]) > 0
        else np.nan
    )

    return features
