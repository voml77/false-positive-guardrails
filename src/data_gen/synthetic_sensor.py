# src/data_gen/synthetic_sensor.py
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Config
# -------------------------
@dataclass(frozen=True)
class SyntheticConfig:
    seed: int = 42

    # time axis
    start: str = "2026-01-01 00:00:00"
    freq: str = "1min"
    periods: int = 7 * 24 * 60  # 7 days, minute resolution

    # features (keep 1 for MVP-1; designed to extend)
    n_features: int = 1
    feature_names: Tuple[str, ...] = ("value",)

    # base signal
    base_level: float = 10.0
    daily_seasonality_amp: float = 1.2
    weekly_seasonality_amp: float = 0.6
    trend_slope: float = 0.0  # slow global trend (can be 0 initially)

    # industrial dirt
    heteroskedastic_noise: bool = True
    noise_sigma_low: float = 0.15
    noise_sigma_high: float = 0.55

    missing_rate: float = 0.004  # ~0.4% missing points (dropouts)
    dropout_burst_prob: float = 0.02  # chance for dropout bursts
    dropout_burst_min: int = 3
    dropout_burst_max: int = 25

    # anomalies (events)
    anomaly_rate_events_per_day: float = 0.6  # average number of anomaly events per day
    spike_prob: float = 0.45
    level_shift_prob: float = 0.15
    drift_prob: float = 0.10
    burst_prob: float = 0.10  # short period higher variance / oscillations

    # event magnitudes
    spike_magnitude_range: Tuple[float, float] = (2.5, 6.0)  # additive
    level_shift_range: Tuple[float, float] = (-2.0, 3.0)     # additive permanent-ish
    drift_slope_range: Tuple[float, float] = (0.0005, 0.003) # per minute slope
    burst_sigma_multiplier_range: Tuple[float, float] = (2.0, 4.0)

    # event durations (in minutes)
    spike_min: int = 1
    spike_max: int = 3
    level_shift_min: int = 30
    level_shift_max: int = 10 * 12
    drift_min: int = 60
    drift_max: int = 20 * 18
    burst_min: int = 10
    burst_max: int = 120

    # output
    out_dir: str = "reports"
    out_csv_name: str = "mvp1_synth_univariate.csv"
    out_meta_name: str = "mvp1_synth_univariate_meta.json"


# -------------------------
# Event generation helpers
# -------------------------
def _choose_event_type(rng: np.random.Generator, cfg: SyntheticConfig) -> str:
    r = rng.random()
    if r < cfg.spike_prob:
        return "spike"
    if r < cfg.spike_prob + cfg.level_shift_prob:
        return "level_shift"
    if r < cfg.spike_prob + cfg.level_shift_prob + cfg.drift_prob:
        return "drift"
    return "burst"


def _sample_duration(rng: np.random.Generator, cfg: SyntheticConfig, event_type: str) -> int:
    if event_type == "spike":
        return int(rng.integers(cfg.spike_min, cfg.spike_max + 1))
    if event_type == "level_shift":
        return int(rng.integers(cfg.level_shift_min, cfg.level_shift_max + 1))
    if event_type == "drift":
        return int(rng.integers(cfg.drift_min, cfg.drift_max + 1))
    if event_type == "burst":
        return int(rng.integers(cfg.burst_min, cfg.burst_max + 1))
    raise ValueError(f"Unknown event type: {event_type}")


def _apply_event(
    x: np.ndarray,
    y: np.ndarray,
    idx_start: int,
    idx_end: int,
    event_type: str,
    rng: np.random.Generator,
    cfg: SyntheticConfig,
    event_id: int,
) -> Dict:
    """
    x: (T,) values
    y: (T,) labels
    Applies event in [idx_start, idx_end).
    Returns event metadata.
    """
    T = x.shape[0]
    idx_start = max(0, min(T - 1, idx_start))
    idx_end = max(idx_start + 1, min(T, idx_end))

    meta = {
        "event_id": event_id,
        "type": event_type,
        "start_idx": idx_start,
        "end_idx": idx_end,
    }

    y[idx_start:idx_end] = 1  # ground truth anomaly window

    if event_type == "spike":
        mag = float(rng.uniform(*cfg.spike_magnitude_range))
        # make it a short bump rather than one point only
        x[idx_start:idx_end] += mag
        meta["magnitude"] = mag

    elif event_type == "level_shift":
        shift = float(rng.uniform(*cfg.level_shift_range))
        x[idx_start:idx_end] += shift
        meta["shift"] = shift

    elif event_type == "drift":
        slope = float(rng.uniform(*cfg.drift_slope_range))
        # drift accumulates linearly over the window
        t = np.arange(idx_end - idx_start, dtype=float)
        x[idx_start:idx_end] += slope * t
        meta["slope_per_min"] = slope

    elif event_type == "burst":
        mult = float(rng.uniform(*cfg.burst_sigma_multiplier_range))
        # burst = extra noise in a window
        burst_noise = rng.normal(0.0, 1.0, size=(idx_end - idx_start,))
        x[idx_start:idx_end] += mult * cfg.noise_sigma_high * burst_noise
        meta["sigma_multiplier"] = mult

    else:
        raise ValueError(f"Unknown event type: {event_type}")

    return meta


# -------------------------
# Main generator
# -------------------------
def generate(cfg: SyntheticConfig) -> Tuple[pd.DataFrame, Dict]:
    rng = np.random.default_rng(cfg.seed)

    # timeline
    ts = pd.date_range(cfg.start, periods=cfg.periods, freq=cfg.freq)
    T = len(ts)

    # base seasonality
    minutes = np.arange(T, dtype=float)
    daily = cfg.daily_seasonality_amp * np.sin(2 * np.pi * minutes / (24 * 60))
    weekly = cfg.weekly_seasonality_amp * np.sin(2 * np.pi * minutes / (7 * 24 * 60))

    trend = cfg.trend_slope * minutes
    base = cfg.base_level + daily + weekly + trend

    # heteroskedastic noise (variance depends on "activity")
    if cfg.heteroskedastic_noise:
        activity = (np.sin(2 * np.pi * minutes / (24 * 60)) + 1.0) / 2.0  # 0..1
        sigmas = cfg.noise_sigma_low + activity * (cfg.noise_sigma_high - cfg.noise_sigma_low)
    else:
        sigmas = np.full(T, cfg.noise_sigma_low)

    noise = rng.normal(0.0, sigmas, size=T)
    x = base + noise

    # labels
    y = np.zeros(T, dtype=int)
    event_type_arr = np.array(["none"] * T, dtype=object)
    event_id_arr = np.full(T, -1, dtype=int)

    # number of events (Poisson)
    days = T / (24 * 60)
    lam = max(0.0, cfg.anomaly_rate_events_per_day * days)
    n_events = int(rng.poisson(lam=lam))

    events_meta: List[Dict] = []
    for eid in range(n_events):
        et = _choose_event_type(rng, cfg)
        dur = _sample_duration(rng, cfg, et)

        idx_start = int(rng.integers(0, max(1, T - dur)))
        idx_end = idx_start + dur

        meta = _apply_event(x, y, idx_start, idx_end, et, rng, cfg, event_id=eid)
        events_meta.append(meta)

        # tag arrays
        event_type_arr[idx_start:idx_end] = et
        event_id_arr[idx_start:idx_end] = eid

    # missing values (random + bursty dropouts)
    missing_mask = rng.random(T) < cfg.missing_rate

    # burst dropouts
    if rng.random() < cfg.dropout_burst_prob:
        n_bursts = int(rng.integers(1, 4))
        for _ in range(n_bursts):
            burst_len = int(rng.integers(cfg.dropout_burst_min, cfg.dropout_burst_max + 1))
            start = int(rng.integers(0, max(1, T - burst_len)))
            missing_mask[start:start + burst_len] = True

    x_missing = x.astype(float).copy()
    x_missing[missing_mask] = np.nan

    df = pd.DataFrame(
        {
            "ts": ts,
            cfg.feature_names[0]: x_missing,
            "y_true": y,
            "event_type": event_type_arr,
            "event_id": event_id_arr,
            "is_missing": missing_mask.astype(int),
        }
    )

    meta_out = {
        "config": asdict(cfg),
        "n_events": n_events,
        "events": events_meta,
        "class_balance": {
            "positives": int(df["y_true"].sum()),
            "negatives": int((df["y_true"] == 0).sum()),
            "positive_rate": float(df["y_true"].mean()),
        },
    }

    return df, meta_out


def save(df: pd.DataFrame, meta: Dict, cfg: SyntheticConfig) -> Tuple[Path, Path]:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / cfg.out_csv_name
    meta_path = out_dir / cfg.out_meta_name

    df.to_csv(csv_path, index=False)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return csv_path, meta_path


def main() -> None:
    cfg = SyntheticConfig()
    df, meta = generate(cfg)
    csv_path, meta_path = save(df, meta, cfg)

    # quick console summary
    print(f"[OK] Wrote dataset: {csv_path}")
    print(f"[OK] Wrote meta:    {meta_path}")
    print(f"[INFO] Positive rate: {meta['class_balance']['positive_rate']:.4f} "
          f"({meta['class_balance']['positives']} / {cfg.periods})")
    print(f"[INFO] Events: {meta['n_events']}")


if __name__ == "__main__":
    main()