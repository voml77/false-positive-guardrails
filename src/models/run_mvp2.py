from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from src.eval.metrics import summarize_run, plot_fp_timeline
from src.models.isolation_forest import fit_score_isolation_forest, IFConfig
from src.models.ae import fit_score_ae, AEConfig
from src.models.vae import fit_score_vae, VAEConfig


def quantile_threshold(scores: pd.Series, q: float = 0.995) -> float:
    """
    Threshold chosen so only top (1-q) fraction becomes anomalies.
    Missing scores ignored.
    """
    s = scores.astype(float)
    s = s[~s.isna()]
    if len(s) < 50:
        return float("inf")
    return float(np.quantile(s.to_numpy(), q))


def flags_from_scores(scores: pd.Series, thr: float) -> pd.Series:
    s = scores.astype(float)
    # honest: NaN scores => 0 (no alert)
    return ((s.fillna(-np.inf)) > thr).astype(int)


def main():
    reports = Path("reports")
    df = pd.read_csv(reports / "mvp1_synth_univariate.csv")

    value_col = "value"
    y_true_col = "y_true"

    # ---- Isolation Forest
    score_if = fit_score_isolation_forest(df[value_col], IFConfig(random_state=42))
    thr_if = quantile_threshold(score_if, q=0.995)
    df["pred_if"] = flags_from_scores(score_if, thr_if)

    # ---- Autoencoder
    score_ae = fit_score_ae(df[value_col], AEConfig(epochs=20, device="cpu"))
    thr_ae = quantile_threshold(score_ae, q=0.995)
    df["pred_ae"] = flags_from_scores(score_ae, thr_ae)

    # ---- VAE
    score_vae = fit_score_vae(df[value_col], VAEConfig(epochs=25, device="cpu"))
    thr_vae = quantile_threshold(score_vae, q=0.995)
    df["pred_vae"] = flags_from_scores(score_vae, thr_vae)

    out = {
        "thresholds": {"if_q995": thr_if, "ae_q995": thr_ae, "vae_q995": thr_vae},
        "isolation_forest": summarize_run(df, value_col, y_true_col, "pred_if"),
        "autoencoder": summarize_run(df, value_col, y_true_col, "pred_ae"),
        "vae": summarize_run(df, value_col, y_true_col, "pred_vae"),
        "note": "MVP-2 is model-only. No guardrails applied by design.",
    }

    (reports / "mvp2_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    plot_fp_timeline(df, value_col, y_true_col, "pred_if", str(reports / "mvp2_fp_timeline_if.png"))
    plot_fp_timeline(df, value_col, y_true_col, "pred_ae", str(reports / "mvp2_fp_timeline_ae.png"))
    plot_fp_timeline(df, value_col, y_true_col, "pred_vae", str(reports / "mvp2_fp_timeline_vae.png"))

    print("[OK] Wrote reports/mvp2_results.json and MVP-2 FP timeline plots.")


if __name__ == "__main__":
    main()