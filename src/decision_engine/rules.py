from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionConfig:
    n_in_a_row: int = 3            # confirm alert only if 3 consecutive model flags
    cooldown_minutes: int = 60     # suppress alerts after one alert for this duration