# src/baselines/zscore.py
from __future__ import annotations
import numpy as np
import pandas as pd


def zscore_flags(
    s: pd.Series,
    k: float = 3.0,
) -> pd.Series:
    """
    Global Z-Score baseline.
    Intentionally simple: no rolling windows, no smoothing, no guardrails.
    Returns 1 for anomaly, 0 for normal; NaNs stay 0 (no prediction on missing).
    """
    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    flags = np.zeros_like(x, dtype=int)
    if mask.sum() < 5:
        return pd.Series(flags, index=s.index)

    mu = np.mean(x[mask])
    sigma = np.std(x[mask])
    # honest baseline: if sigma ~ 0, no anomalies
    if sigma <= 1e-12:
        return pd.Series(flags, index=s.index)

    z = (x[mask] - mu) / sigma
    flags[mask] = (np.abs(z) > k).astype(int)
    return pd.Series(flags, index=s.index)


def zscore_scores(
    s: pd.Series,
) -> pd.Series:
    """
    Optional: returns absolute z-score as a score (NaN for missing).
    """
    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    scores = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 5:
        return pd.Series(scores, index=s.index)

    mu = np.mean(x[mask])
    sigma = np.std(x[mask])
    if sigma <= 1e-12:
        scores[mask] = 0.0
        return pd.Series(scores, index=s.index)

    scores[mask] = np.abs((x[mask] - mu) / sigma)
    return pd.Series(scores, index=s.index)