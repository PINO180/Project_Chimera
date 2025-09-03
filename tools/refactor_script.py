import re
import textwrap

def repair_final(file_path="refactored_independent_features.py"):
    """
    IndentationErrorを根治するため、インデントを自動計算して
    破損した関数ブロックを完全に修復するスクリプト。
    """
    print(f"--- 破損した '{file_path}' の最終修復を開始します ---")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            broken_code = f.read()
    except FileNotFoundError:
        print(f"❌ エラー: 修復対象のファイル '{file_path}' が見つかりません。")
        return

    # 正しい関数のコードブロックを、インデントなしの状態で定義
    correct_functions_source = {
        "_calculate_volume_price_trend": textwrap.dedent("""
        def _calculate_volume_price_trend(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
            \"\"\"出来高価格トレンド系指標（マルチタイムフレーム対応版）\"\"\"
            features = {}
            
            for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
                try:
                    ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                    close_series = ohlcv['close']
                    high_series = ohlcv['high']
                    low_series = ohlcv['low']
                    volume_series = ohlcv['volume']
                    suffix = f'_{tf}' if self.multi_timeframe else ''
                    
                    # Volume Weighted Average Price (VWAP)
                    vwap = self._compute_vwap_vectorized(high_series, low_series, close_series, volume_series)
                    features[f'vwap{suffix}'] = vwap.values
                    features[f'vwap_ratio{suffix}'] = (close_series / (vwap + 1e-8)).values
                    
                    # Time Weighted Average Price (TWAP)
                    twap = close_series.rolling(window=20).mean()
                    features[f'twap{suffix}'] = twap.values
                    features[f'twap_ratio{suffix}'] = (close_series / (twap + 1e-8)).values
                    
                    # Volume Rate of Change
                    for period in self.params['volume_roc_periods']:
                        vol_roc = ((volume_series / volume_series.shift(period)) - 1) * 100
                        features[f'volume_roc_{period}{suffix}'] = vol_roc.values
                    
                    # Ease of Movement (EOM)
                    eom = self._compute_ease_of_movement_vectorized(high_series, low_series, volume_series)
                    features[f'ease_of_movement{suffix}'] = eom.values
                    eom_ma_14 = eom.rolling(window=14).mean()
                    features[f'eom_ma_14{suffix}'] = eom_ma_14.values
                    
                    # Volume Oscillator
                    volume_osc = self._compute_volume_oscillator_vectorized(volume_series)
                    features[f'volume_oscillator{suffix}'] = volume_osc.values
                    
                    # Price Volume Trend Oscillator
                    pvt_osc = self._compute_pvt_oscillator_vectorized(close_series, volume_series)
                    features[f'pvt_oscillator{suffix}'] = pvt_osc.values
                    
                except Exception as e:
                    logger.warning(f"出来高価格トレンド計算エラー - 時間軸 {tf}: {e}")
                    continue
            
            return features
        """),
        "_calculate_moving_averages_advanced": textwrap.dedent("""
        def _calculate_moving_averages_advanced(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
            \"\"\"高度移動平均系指標（マルチタイムフレーム対応版）\"\"\"
            features = {}
            
            for tf in self.available_timeframes if self.multi_timeframe else ['1T']:
                try:
                    ohlcv = self._extract_ohlcv_for_timeframe(df, tf)
                    close_series = ohlcv['close']
                    suffix = f'_{tf}' if self.multi_timeframe else ''
                    
                    # Triangular Moving Average
                    for period in self.params['ma_advanced_periods']:
                        tma = self._triangular_ma_vectorized(close_series, period)
                        features[f'tma_{period}{suffix}'] = tma.values
                    
                    # Kaufman's Adaptive Moving Average (KAMA)
                    kama = self._compute_kama_vectorized(close_series)
                    features[f'kama{suffix}'] = kama.values
                    kama_slope = kama.diff()
                    features[f'kama_slope{suffix}'] = kama_slope.values
                    
                    # Zero Lag EMA
                    for period in self.params['zlema_periods']:
                        zlema = self._compute_zlema_vectorized(close_series, period)
                        features[f'zlema_{period}{suffix}'] = zlema.values
                    
                    # Double Exponential Moving Average (DEMA)
                    for period in self.params['dema_periods']:
                        dema = self._compute_dema_vectorized(close_series, period)
                        features[f'dema_{period}{suffix}'] = dema.values
                    
                    # Triple Exponential Moving Average (TEMA)
                    for period in self.params['tema_periods']:
                        tema = self._compute_tema_vectorized(close_series, period)
                        features[f'tema_{period}{suffix}'] = tema.values
                    
                    # Variable Moving Average (VMA)
                    vma = self._compute_vma_vectorized(close_series)
                    features[f'vma{suffix}'] = vma.values
                    
                except Exception as e:
                    logger.warning(f"高度移動平均計算エラー - 時間軸 {tf}: {e}")
                    continue
            
            return features
        """)
    }

    repaired_code = broken_code
    for func_name, correct_body_str in correct_functions_source.items():
        # 1. 破損したコードから、修復対象関数の正しいインデントを取得
        indent_pattern = re.compile(fr'(^\s*)def {func_name}\(self, df: pd.DataFrame\)', re.MULTILINE)
        match = indent_pattern.search(repaired_code)
        
        if not match:
            print(f"⚠️ 警告: 修復対象 '{func_name}' が見つかりませんでした。スキップします。")
            continue
        indentation = match.group(1)
        
        # 2. 正しいコードブロックの各行に、取得したインデントを適用
        lines = correct_body_str.strip().split('\n')
        indented_body = '\n'.join([f"{indentation}{line}" for line in lines])
        
        # 3. 破損した関数ブロック全体を、インデント済みの正しいブロックで置換
        block_pattern = re.compile(fr'^\s*def {func_name}\(self, df: pd.DataFrame\).*?return features', re.DOTALL | re.MULTILINE)
        if block_pattern.search(repaired_code):
            repaired_code = block_pattern.sub(indented_body, repaired_code, 1)
            print(f"✅ 関数 '{func_name}' を正常なコードに完全修復しました。")
        else:
            print(f"⚠️ 警告: '{func_name}' の置換に失敗しました。")

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(repaired_code)
    
    print(f"--- ✅ 最終修復完了。'{file_path}' は正常な状態に上書きされました。 ---")

if __name__ == "__main__":
    repair_final()