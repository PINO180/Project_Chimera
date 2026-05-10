# Shadow Mode Layer 1 — Diff Report

## Verdict: ❌ **FAIL**

## Context
- **test_period**: 2026-04-01 00:00:00+00:00 → 2026-04-02 00:00:00+00:00
- **warmup_days**: 30
- **scenario**: continuous
- **n_bars_processed**: 2723
- **n_captures**: 2724
- **rtol**: 1e-07
- **atol**: 1e-12
- **timeframes**: M0.5,M1,M3,M5,M8,M15
- **warmup_cache**: saved
- **git_status**: (not captured)

## Tolerance
- `rtol = 1e-07`
- `atol = 1e-12`

## Counts
| Metric | Value |
|---|---|
| Total compared rows | 944,706 |
| Passed | 697,733 |
| Failed | 246,973 |
| Fail rate | 26.142842% |
| Captured-only (no reference match) | 13,279 |
| Reference-only (no captured match) | 1,585,632 |

## Failing breakdown by timeframe
  - `M3`: 58596
  - `M5`: 54178
  - `M8`: 53533
  - `M1`: 34207
  - `M15`: 32360
  - `M0.5`: 14099

## Top failing features (top 20)
  - `e1a_statistical_skewness_50`: 1865
  - `e1a_statistical_skewness_20`: 1863
  - `e1a_fast_volume_mean_5`: 1444
  - `e1d_volume_price_trend_norm`: 1444
  - `e1d_force_index_norm`: 1444
  - `e1b_volume_ma20`: 1444
  - `e1d_volume_ma20_rel`: 1444
  - `e1a_fast_volume_mean_20`: 1444
  - `e1b_volume_price_trend`: 1444
  - `e1a_fast_volume_mean_10`: 1444
  - `e1a_fast_volume_mean_50`: 1443
  - `e1a_fast_volume_mean_100`: 1443
  - `e1d_accumulation_distribution_rel`: 1443
  - `e1d_obv_rel`: 1442
  - `e1c_ema_deviation_200`: 1408
  - `e1c_ema_200`: 1406
  - `e1e_signal_peak_to_peak_100`: 1384
  - `e1e_spectral_flux_512`: 1329
  - `e1c_atr_volatility_55`: 1230
  - `e1c_atr_trend_55`: 1225

## Files
- `paired.parquet` — 全 inner-join 結果 (デバッグ用)
- `failing.parquet` — failing 行のみ (CI artifact)
- `worst.csv` — `abs_diff` top-K rows
- `hint.md` — 失敗パターン分析

## Pass criteria
- `failed = 0` で PASS
- `failed > 0` の場合、`hint.md` で原因の手がかりを参照
