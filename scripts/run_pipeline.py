"""End-to-end pipeline.

    python scripts/run_pipeline.py --data        # download ONS + BoE data and build data/processed
    python scripts/run_pipeline.py --backtest    # backtest, live nowcast, robustness (~60 min on 2 cores)
    python scripts/run_pipeline.py --reports     # tables and figures for docs and dashboard
    python scripts/run_pipeline.py --notebooks   # execute notebooks/ and learn/
    python scripts/run_pipeline.py --all
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args], check=True)


def main():
    ap = argparse.ArgumentParser()
    for flag in ["data", "backtest", "reports", "notebooks", "all"]:
        ap.add_argument(f"--{flag}", action="store_true")
    a = ap.parse_args()
    if a.data or a.all:
        from nowcast import data
        data.download_all()
        data.process_all()
    if a.backtest or a.all:
        run("run_backtest.py")
    if a.reports or a.all:
        run("make_reports.py")
    if a.notebooks or a.all:
        run("build_notebooks.py")
        run("build_learn.py")


if __name__ == "__main__":
    main()
