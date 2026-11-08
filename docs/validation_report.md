# Validation report: UK GDP nowcasting models

*Independent review of the [model development document](model_development_document.md). Checks can be rerun with `python scripts/validate.py` and `pytest`.*

## Summary

| Area | Rating | Finding |
|---|---|---|
| Data and real-time design | 🟢 Green | No look-ahead in 85,082 feature values; first estimates match ONS headlines |
| Implementation | 🟢 Green | From-scratch group lasso satisfies the optimality (KKT) conditions; partial penalisation matches direct estimation |
| Evaluation code | 🟡 Amber → 🟢 fixed | Diebold–Mariano variance was mis-specified; corrected before the results were written |
| Bridge, Bridge + factors, combination | 🟢 Green | Significantly better than AR at quarter end; stable across 2006–09, 2010s and post-COVID |
| Pure regularised and factor models | 🟡 Amber | Better than unrestricted OLS, but not significantly better than AR early in the quarter, and significantly worse than the bridge once output data arrive |
| Uncertainty intervals | 🟡 Amber | Well calibrated at horizons −1 to 2; too narrow for bridge-type models in the month after the quarter (57% coverage) |

**Conclusion.** The bridge-anchored combination is fit for use as a quarter-end nowcast, reported alongside the bridge. The pure regularised models are fit for use only as an early-quarter cross-check, not as a headline number.

## 1. Data and real-time design

**1.1 No look-ahead.** For each of the 361 backtest origins, the design row was rebuilt from the indicator panel *truncated at the origin month*, and compared with the row used in the backtest. All 85,082 non-missing feature values are identical (maximum absolute difference 0), and the missing-value pattern is identical. Real-time output and GDP values come from the vintage in force at the origin (unit tests `test_realtime_lookup_uses_latest_vintage_at_origin` and `test_gdp_known_at_respects_publication`).

**1.2 Training targets.** At none of the 361 origins does the training set include a quarter whose first estimate was published after the origin, or the target quarter itself.

**1.3 Target data.** The first estimates from the real-time database, rounded to one decimal place, match ONS's headline first estimates for every quarter checked:

| Quarter | Real-time database | ONS headline |
|---|---|---|
| 2008Q4 | −1.5% | −1.5% |
| 2009Q1 | −1.9% | −1.9% |
| 2016Q3 | +0.5% | +0.5% |
| 2022Q3 | −0.2% | −0.2% |
| 2024Q4 | +0.1% | +0.1% |
| 2025Q1 | +0.7% | +0.7% |
| 2025Q2 | +0.3% | +0.3% |
| 2025Q3 | +0.1% | +0.1% |

**1.4 Residual look-ahead (limitation).** Retail, trade, labour-market and price indicators are latest vintage. The publication calendar is respected, but later revisions to these series are not. The look-ahead experiment (section 4) measures how much revised *output* data flatter a backtest; the effect of the smaller revisions to the other indicators is not measured.

## 2. Implementation

| Check | Result |
|---|---|
| Group lasso with one feature per group equals scikit-learn's lasso | passes (`test_group_lasso_singletons_equals_lasso`) |
| Group lasso zeroes whole groups | passes |
| Group lasso KKT conditions on the real design (2019Q4, three penalties) | max relative stationarity gap 1e−7, no zero group violates its bound |
| Production solver settings (800 iterations) vs exact solution | fitted values within 1.1% over the whole path; selected groups identical in 2019Q4 and 2025Q4, two groups differ in 2008Q4 |
| Partial penalisation via residualisation equals direct partial penalty | passes (`test_bridge_anchored_matches_direct_partial_penalty`) |
| Partially penalised model with infinite penalty equals bridge OLS | passes |
| Standardisation uses training statistics only; missing → 0 | passes |
| Time-series CV folds never validate on the past | passes |

## 3. Evaluation code: a defect found and fixed

The Diebold–Mariano test originally added an unweighted lag-1 autocovariance of the loss differential. For one-step nowcasts that term should not be there, and when loss differences are negatively autocorrelated it shrinks the variance estimate. In the first run this produced a spurious "significant" result (group lasso vs AR at the end of month 1: t = 4.9, p < 0.001, for a 4% RMSE gain) and an undefined statistic for the lasso in the month after the quarter.

**Fix.** The test now uses no autocovariance terms by default (correct for one-step forecasts) and Bartlett weights if a lag is requested, which keeps the variance positive. After the fix, the group lasso result is t = 0.37, p = 0.71. All results in the development document use the corrected test. `test_dm_sign_and_symmetry` still passes.

## 4. Performance

**4.1 Against benchmarks.** At the end of the quarter, the bridge (RMSE 0.364 pp), Bridge + factors (0.359) and the combination (0.392) beat AR by 22–29% (Diebold–Mariano p = 0.025, 0.023 and 0.065). Every pure regularised model is significantly worse than the bridge at this horizon (p ≤ 0.04). Before output data arrive, no model is significantly better than AR.

**4.2 Overfitting control.** Ridge beats unrestricted minimum-norm OLS at every horizon, by 1–14% (10% at the end of the quarter). This is the expected effect of regularisation when p > n. The margin is smaller than textbook examples, because minimum-norm OLS is itself implicitly regularised (it picks the smallest-norm solution that interpolates the data).

**4.3 Stability over time.** The bridge is the best or within 0.07 pp of the best in all three sub-periods. Ridge's gain over AR is concentrated in 2006–09 (22%), against 3% in 2010–19.

**4.4 Benchmark to the data's own noise.** The root mean square revision from first to latest estimate is 0.365 pp (76 quarters). The best end-of-quarter nowcast error against the first estimate is the same size. Improvements below about 0.05 pp in RMSE should not be treated as meaningful: they are within the sampling uncertainty of 76 quarters, and the target itself is noisy.

**4.5 Robustness variants.**

Each variant re-runs the full 361-origin backtest (`outputs/tables/robustness.csv`, `lookahead.csv`, [notebook 06](../notebooks/06_robustness_and_lookahead.ipynb)). Changes are relative to the main design's RMSE against the first estimate.

| Variant | Effect |
|---|---|
| Early indicators only (no output or trade data) | Ridge's error falls only from 0.55 to 0.47 over the quarter, against 0.59 to 0.36 for the bridge. Late-quarter accuracy comes from official output data. |
| COVID quarters in training | Pure regularised models get much worse early in the quarter (lasso +56%, group lasso +55% at the end of month 1). Bridge-type models improve once output data arrive (bridge −9% at quarter end, −15% in the month after), because 2020's swings pin down the output–GDP mapping. |
| Rolling 40-quarter window | AR and ridge gain up to 9%. The bridge loses 4–29%, and the anchored models up to 81%, once the window drops 2008–09. |
| One-standard-error rule | Bridge-anchored models improve 6–17%: Bridge + ridge 0.364 (equal to the bridge) and Bridge + group lasso 0.353 (3% better than the bridge, not tested for significance) at quarter end. The pure regularised models mostly get worse. |
| Revised output data, judged against the first estimate | Bridge error *rises* 23% at quarter end and 8% in the month after. ONS builds its first estimate from first-published output, so revised data are the wrong input. |
| Conventional backtest (revised data as inputs, target and yardstick) | Reports a month-after-quarter bridge error of 0.169 pp, against 0.320 pp achievable in real time, and a 77% gain over AR instead of 32%. |

**Conclusions.** In every variant that includes output data, the most accurate end-of-quarter model is the bridge or a bridge-anchored model, and the best pure regularised model trails it by 6–47%. The one variant where regularisation adds to the bridge (heavier penalties, Bridge + group lasso) is a candidate for the next model version. It is not adopted here, because it was found by looking at the evaluation results. The look-ahead experiment shows why real-time data matter: a conventional backtest would roughly halve the reported late-quarter error.

## 5. Uncertainty

Empirical 80% intervals cover the first estimate 71–87% of the time at horizons −1 to 2 (target 80%). In the month after the quarter, coverage is 57% for the bridge and Bridge + factors: only 27 quarters are available, and intervals start around 2022. **Recommendation:** widen month-after intervals, for example by using the end-of-month-3 error distribution, until 40 quarters of errors exist.

## 6. Limitations and recommendations

1. **Add surveys.** The early-quarter nowcasts barely beat the mean. PMIs and the BoE Agents' scores are the obvious addition if a licence is available.
2. **Real-time vintages for the other indicators.** ONS publishes revisions triangles for retail sales. Adding them would close the residual look-ahead in section 1.4.
3. **Interpret selection with care.** The lasso's choice of Bank Rate in every fit is driven by 2008–09. Selected indicators are predictive associations in a 76-quarter sample, not causal drivers.
4. **Monitor.** Re-estimate if the combination's error falls outside its 80% interval in three consecutive quarters. Track the 2026Q3 nowcast against ONS's first estimate on 12 November 2026.
