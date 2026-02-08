# src/eval/metrics.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    fpr: float


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return tp, fp, tn, fn


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Metrics:
    tp, fp, tn, fn = confusion(y_true, y_pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return Metrics(tp, fp, tn, fn, precision, recall, f1, fpr)


def alerts_per_day(df: pd.DataFrame, y_pred_col: str, freq: str = "1min") -> float:
    # for 1min: 1440 points/day; for other frequencies, estimate via pandas
    if freq == "1min":
        points_per_day = 24 * 60
    else:
        # fallback: infer from timestamp diffs
        ts = pd.to_datetime(df["ts"])
        delta = (ts.iloc[1] - ts.iloc[0]).total_seconds()
        points_per_day = int(round(24 * 3600 / max(delta, 1)))
    total_days = len(df) / points_per_day
    total_alerts = float(df[y_pred_col].fillna(0).sum())
    return total_alerts / max(total_days, 1e-9)


def plot_fp_timeline(
    df: pd.DataFrame,
    value_col: str,
    y_true_col: str,
    y_pred_col: str,
    out_path: str,
) -> None:
    ts = pd.to_datetime(df["ts"])
    v = df[value_col].astype(float)
    y_true = df[y_true_col].astype(int)
    y_pred = df[y_pred_col].astype(int)

    fp_mask = (y_true == 0) & (y_pred == 1)

    # cumulative false positives
    fp_cum = fp_mask.cumsum()

    fig = plt.figure()
    plt.plot(ts, v, linewidth=1.0)
    plt.scatter(ts[fp_mask], v[fp_mask], s=12)  # intentionally no custom colors

    plt.twinx()
    plt.plot(ts, fp_cum, linewidth=1.0)

    plt.title("False Positives over Time (Model-only)")
    plt.xlabel("Time")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def summarize_run(df: pd.DataFrame, value_col: str, y_true_col: str, y_pred_col: str, freq: str = "1min") -> Dict:
    y_true = df[y_true_col].astype(int).to_numpy()
    y_pred = df[y_pred_col].astype(int).to_numpy()

    m = compute_metrics(y_true, y_pred)
    apd = alerts_per_day(df, y_pred_col, freq=freq)

    return {
        "confusion": {"tp": m.tp, "fp": m.fp, "tn": m.tn, "fn": m.fn},
        "precision": m.precision,
        "recall": m.recall,
        "f1": m.f1,
        "fpr": m.fpr,
        "alerts_per_day": apd,
    }