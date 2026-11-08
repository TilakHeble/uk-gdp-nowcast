"""Information-set tests: no feature may use data published after the nowcast origin."""
import numpy as np
import pandas as pd
import pytest

from nowcast.config import INDICATORS, N_MONTH_LAGS, PROCESSED
from nowcast.features import (GDPVintages, RealTimeIndicators, feature_names, group_of,
                              main_spec, origin_month)

needs_data = pytest.mark.skipif(not (PROCESSED / "gdp_vintages_long.csv").exists(),
                                reason="run the data pipeline first")


def test_origin_month():
    q = pd.Period("2024Q3", "Q")
    assert origin_month(q, -1) == pd.Period("2024-06", "M")
    assert origin_month(q, 0) == pd.Period("2024-07", "M")
    assert origin_month(q, 3) == pd.Period("2024-10", "M")


def test_feature_names_and_main_spec():
    names = feature_names()
    assert len(names) == len(INDICATORS) * N_MONTH_LAGS + 2
    main = [n for n in names if main_spec(n)]
    assert not any(group_of(n).endswith("(latest vintage)") for n in main)
    assert any(n.startswith("RT_IOS") for n in main)


def test_realtime_lookup_uses_latest_vintage_at_origin():
    long = pd.DataFrame({
        "series": ["S"] * 3,
        "vintage": pd.PeriodIndex(["2020-03", "2020-04", "2020-05"], freq="M"),
        "month": pd.PeriodIndex(["2020-01"] * 3, freq="M"),
        "value": [1.0, 2.0, 3.0],
    })
    rt = RealTimeIndicators(long)
    m = pd.Period("2020-01", "M")
    assert np.isnan(rt.value("S", pd.Period("2020-02", "M"), m))   # not yet published
    assert rt.value("S", pd.Period("2020-03", "M"), m) == 1.0
    assert rt.value("S", pd.Period("2020-04", "M"), m) == 2.0      # revised
    assert rt.value("S", pd.Period("2021-01", "M"), m) == 3.0


@needs_data
def test_gdp_known_at_respects_publication():
    from nowcast import data
    gdp = GDPVintages(data.load_gdp_vintages())
    q = pd.Period("2024Q2", "Q")
    fv = gdp.first_vintage[q]
    assert q not in gdp.known_at(fv - 1).index
    assert q in gdp.known_at(fv).index
    assert np.isclose(gdp.known_at(fv)[q], gdp.first[q])


@needs_data
def test_design_row_uses_only_published_months():
    from nowcast.features import build_all
    B = build_all([1], end_quarter=pd.Period("2024Q4", "Q"))
    X = B["X"][1]
    q = pd.Period("2024Q2", "Q")
    T = origin_month(q, 1)  # end of May 2024
    for ind in INDICATORS:
        if ind.source == "rt":
            continue
        latest_used = T - ind.lag
        expected = B["panel_t"][ind.code].get(latest_used, np.nan)
        got = X.loc[q, f"{ind.code}_l0"]
        assert (np.isnan(expected) and np.isnan(got)) or np.isclose(expected, got)
    # GDP for 2024Q1 was first published in May 2024 (two months after the quarter)
    assert np.isfinite(X.loc[q, "GDP_q1"])
