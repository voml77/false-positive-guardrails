# src/baselines/iqr.py
from __future__ import annotations
import numpy as np
import pandas as pd


def iqr_flags(
    s: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """
    Global IQR fence baseline.
    Intentionally simple: no rolling windows, no smoothing, no guardrails.
    Returns 1 for anomaly, 0 for normal; NaNs stay 0 (no prediction on missing).
    """
    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    flags = np.zeros_like(x, dtype=int)
    if mask.sum() < 10:
        return pd.Series(flags, index=s.index)

    q1 = np.quantile(x[mask], 0.25)
    q3 = np.quantile(x[mask], 0.75)
    iqr = q3 - q1

    # honest baseline: if iqr ~ 0, no anomalies
    if iqr <= 1e-12:
        return pd.Series(flags, index=s.index)

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    flags[mask] = ((x[mask] < lo) | (x[mask] > hi)).astype(int)
    return pd.Series(flags, index=s.index)


def iqr_scores(
    s: pd.Series,
    k: float = 1.5,
) -> pd.Series:
    """
    Optional: distance beyond fence as a score (0 inside, positive outside).
    NaN for missing.
    """
    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    scores = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 10:
        return pd.Series(scores, index=s.index)

    q1 = np.quantile(x[mask], 0.25)
    q3 = np.quantile(x[mask], 0.75)
    iqr = q3 - q1
    if iqr <= 1e-12:
        scores[mask] = 0.0
        return pd.Series(scores, index=s.index)

    lo = q1 - k * iqr
    hi = q3 + k * iqr

    # distance outside bounds (inside => 0)
    v = x[mask]
    dist = np.zeros_like(v, dtype=float)
    dist[v < lo] = lo - v[v < lo]
    dist[v > hi] = v[v > hi] - hi
    scores[mask] = dist
    return pd.Series(scores, index=s.index)