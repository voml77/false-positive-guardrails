from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.metrics import summarize_run, plot_fp_timeline
from src.decision_engine.engine import apply_decision_engine
from src.decision_engine.rules import DecisionConfig

from src.models.vae import fit_score_vae, VAEConfig  # <-- add
# optional: from src.models.ae import fit_score_ae, AEConfig
# optional: from src.models.isolation_forest import fit_score_isolation_forest, IFConfig


def quantile_threshold(scores: pd.Series, q: float = 0.995) -> float:
    s = scores.astype(float)
    s = s[~s.isna()]
    if len(s) < 50:
        return float("inf")
    return float(np.quantile(s.to_numpy(), q))


def flags_from_scores(scores: pd.Series, thr: float) -> pd.Series:
    s = scores.astype(float)
    return ((s.fillna(-np.inf)) > thr).astype(int)


def main():
    reports = Path("reports")
    df = pd.read_csv(reports / "mvp1_synth_univariate.csv")

    value_col = "value"
    y_true_col = "y_true"

    # --- MODEL (choose one): VAE
    score_vae = fit_score_vae(df[value_col], VAEConfig(epochs=25, device="cpu"))
    thr_vae = quantile_threshold(score_vae, q=0.995)
    df["pred_model"] = flags_from_scores(score_vae, thr_vae)

    # BEFORE: model-only
    before = summarize_run(df, value_col, y_true_col, "pred_model")

    # AFTER: decision engine
    cfg = DecisionConfig(n_in_a_row=3, cooldown_minutes=60)
    df2 = apply_decision_engine(df, model_pred_col="pred_model", cfg=cfg, freq="1min")
    after = summarize_run(df2, value_col, y_true_col, "decision_alert")

    out = {
        "model": "VAE",
        "threshold_q": 0.995,
        "decision_config": {"n_in_a_row": cfg.n_in_a_row, "cooldown_minutes": cfg.cooldown_minutes},
        "before_model_only": before,
        "after_decision_engine": after,
    }

    (reports / "mvp3_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    plot_fp_timeline(df2, value_col, y_true_col, "decision_alert", str(reports / "mvp3_fp_timeline_decision_engine.png"))

    print("[OK] Wrote reports/mvp3_results.json and reports/mvp3_fp_timeline_decision_engine.png")


if __name__ == "__main__":
    main()