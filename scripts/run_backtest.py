"""Run the main pseudo-real-time backtest, robustness variants and the live nowcast.

    python scripts/run_backtest.py            # everything
    python scripts/run_backtest.py --main     # main backtest only
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nowcast.backtest import evaluation_tasks, live_tasks, run_backtest, run_origin  # noqa: E402
from nowcast.config import HORIZONS, LIVE_ORIGIN, OUTPUTS  # noqa: E402
from nowcast.features import build_all, lookahead_spec, soft_spec  # noqa: E402
from nowcast.models import (BRIDGE_PREFIXES, BridgeAnchored, ElasticNetModel,  # noqa: E402
                            GroupLassoModel, LassoModel, RidgeModel, SubsetOLS)


def ROBUST_MODELS():
    return [SubsetOLS("AR", ("GDP_",)), SubsetOLS("Bridge", BRIDGE_PREFIXES),
            RidgeModel(), LassoModel(), ElasticNetModel(), GroupLassoModel(),
            BridgeAnchored(RidgeModel()), BridgeAnchored(GroupLassoModel())]


def ONE_SE_MODELS():
    return [SubsetOLS("AR", ("GDP_",)), SubsetOLS("Bridge", BRIDGE_PREFIXES),
            RidgeModel(one_se=True), LassoModel(one_se=True), ElasticNetModel(one_se=True),
            GroupLassoModel(one_se=True), BridgeAnchored(RidgeModel(one_se=True)),
            BridgeAnchored(GroupLassoModel(one_se=True))]


def EARLY_MODELS():
    # no output data at all, so no bridge
    return [SubsetOLS("AR", ("GDP_",)), RidgeModel(), LassoModel(), ElasticNetModel(),
            GroupLassoModel()]


def LOOKAHEAD_MODELS():
    # the bridge uses latest-vintage monthly GDP here: the classic look-ahead set-up
    core = ("GDP_",) + tuple(f"ECY2_l{j}" for j in range(3))
    return [SubsetOLS("AR", ("GDP_",)), SubsetOLS("Bridge", core),
            RidgeModel(), LassoModel(), ElasticNetModel(), GroupLassoModel(),
            BridgeAnchored(RidgeModel(), core=core), BridgeAnchored(GroupLassoModel(), core=core)]


def CONVENTIONAL_MODELS():
    core = ("GDP_",) + tuple(f"ECY2_l{j}" for j in range(3))
    return [SubsetOLS("AR", ("GDP_",)), SubsetOLS("Bridge", core), RidgeModel(), LassoModel(),
            BridgeAnchored(RidgeModel(), core=core)]


def latest_vintage_B(B):
    """Copy of B whose GDP-lag features use today's vintage (where a value was available at T)."""
    latest = B["latest"]
    X2 = {}
    for h, X in B["X"].items():
        X = X.copy()
        for col, k in (("GDP_q2", 2), ("GDP_q1", 1)):
            vals = latest.reindex(X.index - k).values
            X[col] = np.where(X[col].notna(), vals, np.nan)
        X2[h] = X
    return dict(B, X=X2)


def main(only_main=False, only_variants=None):
    OUTPUTS.mkdir(exist_ok=True)
    B = build_all(HORIZONS)
    tasks = evaluation_tasks(B)
    if only_variants:
        return run_variants(B, tasks, only_variants)
    t0 = time.time()
    preds, coefs = run_backtest(B, tasks)
    preds.to_csv(OUTPUTS / "backtest_predictions.csv", index=False)
    with open(OUTPUTS / "backtest_coefficients.pkl", "wb") as f:
        pickle.dump(coefs, f)
    print(f"main backtest: {len(tasks)} origins in {time.time()-t0:.0f}s", flush=True)

    # live nowcast
    live_origin = pd.Period(LIVE_ORIGIN, "M")
    rows, live_coefs = [], []
    for q, h in live_tasks(B, live_origin):
        out, c = run_origin(B, q, h)
        rows += out
        live_coefs.append(c)
    pd.DataFrame(rows).to_csv(OUTPUTS / "live_nowcast.csv", index=False)
    with open(OUTPUTS / "live_coefficients.pkl", "wb") as f:
        pickle.dump(live_coefs, f)
    print("live nowcast written", flush=True)
    if only_main:
        return
    run_variants(B, tasks)


def run_variants(B, tasks, only=None):
    variants = {
        "lookahead_latest_vintage_output": (LOOKAHEAD_MODELS, dict(feature_filter=lookahead_spec)),
        "early_indicators_only": (EARLY_MODELS, dict(feature_filter=soft_spec)),
        "covid_in_training": (ROBUST_MODELS, dict(exclude_covid=False)),
        "rolling_window_40q": (ROBUST_MODELS, dict(window=40)),
        "one_se_rule": (ONE_SE_MODELS, dict()),
        # The conventional backtest: today's revised data as features, training target and
        # yardstick. Compare with the real-time results to measure look-ahead flattery.
        "conventional_latest_vintage": (CONVENTIONAL_MODELS,
                                        dict(feature_filter=lookahead_spec, target="latest")),
    }
    for name, (factory, kw) in variants.items():
        if only and name not in only:
            continue
        t0 = time.time()
        BB = latest_vintage_B(B) if name == "conventional_latest_vintage" else B
        p, _ = run_backtest(BB, tasks, models_factory=factory, **kw)
        p["variant"] = name
        p.to_csv(OUTPUTS / f"robust_{name}.csv", index=False)
        print(f"{name}: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", action="store_true")
    ap.add_argument("--variants", nargs="*", help="run only these robustness variants")
    a = ap.parse_args()
    main(a.main, a.variants)
