from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Decision(str, Enum):
    NONE = "NONE"
    ALERT = "ALERT"
    SUPPRESSED = "SUPPRESSED"


@dataclass(frozen=True)
class DecisionResult:
    decision: Decision
    reason_code: str
    model_flag: int  # 0/1 (raw model output)
    score: Optional[float] = None  # optional for later