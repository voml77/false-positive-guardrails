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