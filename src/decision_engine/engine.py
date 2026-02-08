from __future__ import annotations

import pandas as pd

from src.decision_engine.types import Decision, DecisionResult
from src.decision_engine.rules import DecisionConfig


def apply_decision_engine(
    df: pd.DataFrame,
    model_pred_col: str,
    cfg: DecisionConfig = DecisionConfig(),
) -> pd.Series:
    """
    Operational Logic Layer / Decision Engine

    Consumes raw model predictions (0/1) and outputs operational decisions.
    Deterministic by design.

    MVP-3 rules to implement next:
    - n-in-a-row confirmation
    - cooldown window
    - reason codes

    Returns a Series of strings: NONE / ALERT / SUPPRESSED
    """
    # Placeholder to keep pipeline wiring stable
    # Next weekend: implement actual deterministic rules.
    return pd.Series(["NONE"] * len(df), index=df.index)