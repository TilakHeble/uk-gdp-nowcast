# UK GDP nowcast: regularised regression in real time

[![tests](https://github.com/TilakHeble/uk-gdp-nowcast/actions/workflows/tests.yml/badge.svg)](https://github.com/TilakHeble/uk-gdp-nowcast/actions/workflows/tests.yml)

**Live dashboard:** https://uk-gdp-nowcast.streamlit.app/

How accurately can UK quarterly GDP growth be estimated *before* the Office for National Statistics publishes it, and does regularised regression (ridge, lasso, elastic net, adaptive and group lasso) on a wide panel of monthly indicators beat the bridge equations that central banks use?

This project answers with a **true real-time backtest**. Every one of 361 nowcasts (76 quarters × 5 points in the data cycle, 2006–2026) uses only the data published by its date, with GDP and output figures *as first published*. Those vintages are reconstructed from ONS's real-time database and revisions triangles, not taken from today's revised data.

## Key results

| | Before quarter | End of month 1 | End of quarter | 2 weeks before ONS |
|---|---|---|---|---|
| Historical mean | 0.556 | 0.550 | 0.549 | 0.378 |
| AR benchmark | 0.629 | 0.495 | 0.504 | 0.468 |
| OLS on all 236 features | 0.601 | 0.550 | 0.503 | 0.388 |
| Ridge | 0.559 | 0.477 | 0.453 | 0.383 |
| Group lasso | 0.568 | 0.475 | 0.474 | 0.354 |
| Factor model | 0.549 | **0.460** | 0.463 | 0.447 |
| **Bridge equation** | 0.590 | 0.476 | **0.364** | **0.320** |
| Bridge-anchored combination | 0.586 | 0.532 | 0.392 | 0.350 |

*RMSE against ONS's first estimate, percentage points of quarterly growth, COVID quarters excluded. [Full table](outputs/tables/rmse_first_estimate.csv).*

* **Regularisation tames overfitting.** With more features (236) than quarters (31–107), OLS is no better than the mean until the quarter ends. Ridge beats it at every horizon.
* **Official output data dominate.** Once the monthly Index of Services and Index of Production arrive, a two-indicator bridge equation is the most accurate model (28% below AR, Diebold–Mariano p = 0.025). Adding 33 more indicators through any penalty makes it 13–17% *worse*.
* **Look-ahead bias is large.** A conventional backtest on today's revised data reports a month-after-quarter error of 0.17 pp. The real-time error is 0.32 pp.
* **The nowcast is as good as the target allows.** The end-of-quarter error (0.36 pp) equals the size of ONS's own later revisions to its first estimate (0.37 pp).
* **Live nowcast for 2026Q3: +0.62%** (80% interval +0.15% to +0.88%), made with data to the end of October 2026. ONS publishes its first estimate on 12 November 2026.

Read the [executive summary](docs/executive_summary.md), the [model development document](docs/model_development_document.md) and the [validation report](docs/validation_report.md).

## What makes it different

* **Real-time data, not revised data.** GDP and output come from ONS vintages. A [look-ahead experiment](notebooks/06_robustness_and_lookahead.ipynb) re-runs everything the conventional way to measure how much revised data flatter a backtest: the late-quarter error is roughly halved.
* **The ragged edge.** Each indicator enters with its own publication lag, using six monthly lags per indicator (unrestricted MIDAS).
* **Penalties that respect structure.** Group lasso keeps or drops all lags of an indicator together. It is implemented from scratch (FISTA) and checked against the KKT conditions.
* **Partial penalisation.** "Bridge + X" models leave the bridge regressors unpenalised and regularise only the other indicators, via the Frisch–Waugh–Lovell theorem. This answers the practical question: does anything add to the bridge?
* **Honest evaluation.** Blocked time-series cross-validation, Diebold–Mariano tests, sub-period stability, interval coverage and six robustness variants. The validation report documents a bug found and fixed in the significance test.

## Repository

```
src/nowcast/        data download and parsing, features, models, backtest, evaluation
scripts/            pipeline, backtest, reports, validation, notebook builders
notebooks/          01-07 analysis: data and revisions -> live nowcast
learn/              01-08 regularised regression from basics to advanced, on this data
app/                Streamlit dashboard (reads outputs/tables, no fitting at runtime)
docs/               executive summary, model development document, validation report
outputs/            tables and figures; backtest predictions
tests/              look-ahead, model and evaluation unit tests
```

## Learn regularised regression

The [`learn/`](learn/README.md) folder goes from why OLS fails when p ≈ n, through ridge, lasso and elastic net with from-scratch implementations, to choosing λ for time series, adaptive and group lasso, the link to factor models, and the pitfalls (selection instability, post-selection inference, leakage, revisions). Every notebook uses this project's data.

## Reproduce

```bash
pip install -r requirements.txt
python scripts/run_pipeline.py --data        # download ONS and Bank of England data (~5 min)
python scripts/run_pipeline.py --backtest    # backtest, live nowcast, robustness (~90 min on 2 cores)
python scripts/run_pipeline.py --reports     # tables and figures
python scripts/run_pipeline.py --notebooks   # execute notebooks/ and learn/
python scripts/validate.py                   # independent validation checks
pytest
streamlit run app/streamlit_app.py
```

## Data

ONS: GDP real-time database (ABMI), revisions triangles for the Index of Services, Production and Manufacturing, and monthly GDP, Index of Production, retail sales, trade, labour market and consumer price datasets. Bank of England database: Bank Rate, gilt yields, sterling, mortgage rates and approvals, M4. All under the [Open Government Licence v3.0](https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Related projects

* [UK credit stress engine](https://github.com/TilakHeble/uk-credit-stress-engine): linear regression for bank credit losses under the Bank of England stress scenario.
* [NHS readmission risk](https://github.com/TilakHeble/nhs-readmission-risk): logistic regression for 30-day emergency readmissions.
