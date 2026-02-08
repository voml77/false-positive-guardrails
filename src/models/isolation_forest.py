from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class IFConfig:
    n_estimators: int = 200
    max_samples: str | int = "auto"
    contamination: str | float = "auto"  # keep auto; thresholding handled separately
    random_state: int = 42


def fit_score_isolation_forest(s: pd.Series, cfg: IFConfig = IFConfig()) -> pd.Series:
    """
    Returns anomaly score (higher => more anomalous).
    Model-only: no smoothing, no rolling windows.
    Missing values are ignored for fitting and scoring; scores for NaNs stay NaN.
    """
    x = s.astype(float).to_numpy()
    mask = ~np.isnan(x)

    scores = np.full_like(x, np.nan, dtype=float)
    if mask.sum() < 50:
        return pd.Series(scores, index=s.index)

    X = x[mask].reshape(-1, 1)
    model = IsolationForest(
        n_estimators=cfg.n_estimators,
        max_samples=cfg.max_samples,
        contamination=cfg.contamination,
        random_state=cfg.random_state,
        n_jobs=-1,
    )
    model.fit(X)

    # sklearn: higher decision_function => more normal
    # we invert to "higher => more anomalous"
    normality = model.decision_function(X)
    scores[mask] = -normality
    return pd.Series(scores, index=s.index)