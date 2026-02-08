# src/eval/run_mvp1.py
from __future__ import annotations
import json
from pathlib import Path

import pandas as pd

from src.baselines.zscore import zscore_flags
from src.baselines.iqr import iqr_flags
from src.eval.metrics import summarize_run, plot_fp_timeline


def main():
    reports = Path("reports")
    df = pd.read_csv(reports / "mvp1_synth_univariate.csv")

    value_col = "value"
    y_true_col = "y_true"

    # "unpleasantly honest" baselines
    df["pred_z3"] = zscore_flags(df[value_col], k=3.0)
    df["pred_iqr15"] = iqr_flags(df[value_col], k=1.5)

    # summaries
    out = {
        "zscore_k3": summarize_run(df, value_col, y_true_col, "pred_z3"),
        "iqr_k15": summarize_run(df, value_col, y_true_col, "pred_iqr15"),
    }

    (reports / "mvp1_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # plots
    plot_fp_timeline(df, value_col, y_true_col, "pred_z3", str(reports / "mvp1_fp_timeline_zscore.png"))
    plot_fp_timeline(df, value_col, y_true_col, "pred_iqr15", str(reports / "mvp1_fp_timeline_iqr.png"))

    print("[OK] Wrote reports/mvp1_results.json and FP timeline plots.")


if __name__ == "__main__":
    main()