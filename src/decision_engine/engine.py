from __future__ import annotations

import pandas as pd

from src.decision_engine.rules import DecisionConfig
from src.decision_engine.types import Decision


def apply_decision_engine(
    df: pd.DataFrame,
    model_pred_col: str,
    cfg: DecisionConfig = DecisionConfig(),
    freq: str = "1min",
) -> pd.DataFrame:
    """
    Deterministic Decision Engine (Operational Logic Layer)

    Rules (MVP-3 v1):
    - n-in-a-row confirmation: raise ALERT only if last N model flags are 1
    - cooldown: after ALERT, suppress any further ALERT for cooldown_minutes

    Returns df with:
    - decision: NONE / ALERT / SUPPRESSED
    - decision_alert: 1 if decision == ALERT else 0
    - reason_code: deterministic explanation
    """
    out = df.copy()
    pred = out[model_pred_col].fillna(0).astype(int)

    # infer steps per minute (we assume minute granularity for MVP)
    if freq == "1min":
        steps_per_min = 1
    else:
        # fallback: assume 1 step = 1 minute
        steps_per_min = 1

    cooldown_steps = int(cfg.cooldown_minutes * steps_per_min)

    decision = []
    reason = []

    consec = 0
    cooldown_left = 0

    for i in range(len(pred)):
        p = int(pred.iloc[i])

        # tick cooldown
        if cooldown_left > 0:
            cooldown_left -= 1

        if p == 1:
            consec += 1
        else:
            consec = 0

        # cooldown dominates
        if cooldown_left > 0:
            decision.append(Decision.SUPPRESSED.value)
            reason.append("SUPPRESSED_COOLDOWN")
            continue

        # confirmed alert?
        if consec >= cfg.n_in_a_row:
            decision.append(Decision.ALERT.value)
            reason.append("ALERT_CONFIRMED_N_IN_ROW")
            cooldown_left = cooldown_steps  # start cooldown AFTER raising
            consec = 0  # reset streak to avoid repeated alerts
            continue

        # otherwise: no alert
        decision.append(Decision.NONE.value)
        reason.append("NO_ALERT")

    out["decision"] = decision
    out["decision_alert"] = (out["decision"] == Decision.ALERT.value).astype(int)
    out["reason_code"] = reason
    return out