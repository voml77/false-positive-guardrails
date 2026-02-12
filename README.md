# False Positive Guardrails

False positives in anomaly detection are not a model problem.
They are a system design problem.

## Motivation
In monitoring and industrial AI systems, high accuracy often hides
a critical issue: excessive false positives that destroy operational trust.

This project explores anomaly detection from a system perspective,
focusing on false positive rate, alert behavior over time,
and decision logic instead of blind model trust.

## MVP-1: Baseline without Guardrails
The first milestone intentionally evaluates baseline methods
(Z-Score, IQR) without any decision logic, smoothing or confirmation.
The goal is to make the operational pain of false positives visible.

Metrics:
- False Positive Rate (FPR)
- Precision / Recall / F1
- Alerts per time unit

No guardrails. No mitigation. By design.

## Observed Behavior (MVP-1)

Even with simple baseline methods (Z-Score, IQR), the system exhibits:

- A noticeable false positive rate (FPR)
- Low precision despite acceptable recall
- A high number of alerts per day

This highlights a core issue of model-only anomaly detection:
operational noise emerges long before operational trust.

At this stage, no mitigation or decision logic is applied by design.

## Observed Behavior (MVP-2)

Using more advanced models (Isolation Forest, Autoencoder, Variational Autoencoder)
on the same data and with identical evaluation criteria shows:

- False positives accumulate over time across all models
- Model complexity does not reliably reduce false positive rate
- Alerts cluster around seasonal patterns and structural transitions
- Latent representations do not prevent operational alert noise

This confirms that false positives persist even with more sophisticated models
when no decision or guardrail logic is applied.

## MVP-3: Decision Engine (Operational Logic Layer)
Next milestone introduces a deterministic decision layer (confirmation, cooldown, overrides)
to reduce alert noise without changing the underlying model.

## MVP-3: Decision Engine (Operational Logic Layer)

MVP-3 introduces a deterministic guardrail layer that consumes raw model predictions
and outputs operational alert decisions.

Implemented guardrails (v1):
- **N-in-a-row confirmation (N=3)**: no single prediction can trigger an alert
- **Cooldown window (60 min)**: after an alert, further alerts are suppressed for a period

Result (on the same dataset and metrics):
- **Model-only** produced frequent alerts (operational noise)
- **Decision Engine** reduced alert noise to near-zero in this run

This is intentional: the goal of MVP-3 is to demonstrate that false positives
are primarily a **system design problem**, not a model problem.