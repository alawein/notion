# 15 Production Ml Systems

**Total Pages:** 4



--- Page 1 ---

Algorized Interview Prep — Doc 5: Production ML Systems
Page 1
DOC 5 / 8
Production ML Systems & MLOps
Building, deploying, and operating ML systems reliably at scale
Topics Covered
 ML Pipeline Architecture
 Feature Engineering & Data Management
 Model Registry & Versioning
 CI/CD for ML Models
 Monitoring: Drift, Degradation & Alerting
 A/B Testing & Staged Rollouts
 Data Quality & Labeling at Scale
 Interview Q&A; Bank
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 5: Production ML Systems
Page 2
1. ML Pipeline Architecture
1.1 End-to-End Pipeline Components
 Data Ingestion: Raw sensor data → S3/GCS landing zone. Schema validation on arrival (Pydantic/Great
Expectations). Dead-letter queue for malformed records. Partitioned by date + sensor_id for efficient querying.
 Feature Engineering: Deterministic, versioned transformations. Feature Store (Feast, Tecton, or custom)
decouples feature computation from model training. Offline (batch) and online (real-time) feature serving must be
consistent — training/serving skew is the #1 production MLOps failure mode.
 Training Pipeline: Triggered by: new data threshold, scheduled cadence, or manual. Hyperparameter search
(Optuna). Experiment tracking (MLflow, Weights & Biases). Always checkpoint — resumable training for long
runs.
 Evaluation Gate: Automated: must beat incumbent model on held-out evaluation set by threshold (e.g., +0.5%
recall). Regression checks: no degradation on known failure cases. Shadow run: new model runs in parallel with
incumbent for N hours before cutover.
 Deployment: Model registry tag 'production'. Blue-green or canary deployment. OTA for edge devices. Rollback
trigger: if production metrics degrade >X% within 24h, auto-revert.
 Monitoring: Input distribution statistics, prediction distribution, business metrics (false alarm rate, true detection
rate). Alerting on threshold breaches. On-device: lightweight confidence histogram logging.
2. Training/Serving Skew — The #1 Production Failure
Training/serving skew: model sees different data at training time vs. inference time. Causes silent degradation —
model accuracy looks fine in evaluation but fails in production. Sources:
 Feature computation differences: training uses batch statistics, serving uses streaming approximations
 Data preprocessing bugs: different code paths for train and serve (must use same transformation code)
 Label leakage: feature derived from target variable inadvertently included in training features
 Temporal leakage: using future information in features (e.g., rolling mean computed including future frames)
 Schema drift: production data evolves, training data stays fixed
For Algorized: training/serving skew is particularly dangerous because the background clutter removal
(EMA) computed differently during offline training vs. real-time embedded inference could cause the
model to see systematically different input distributions. The fix: use the same stateful EMA
implementation (in C++ or ported to Python) in both pipelines.
3. Monitoring & Drift Detection


--- Page 3 ---

Algorized Interview Prep — Doc 5: Production ML Systems
Page 3
3.1 Types of Drift
<b>Type</b>
<b>Definition</b>
<b>Detection Method</b>
<b>Response</b>
Data drift (covariate shift)
P(X) changes; P(Y|X) same
PSI, KL divergence on input features
Retrain on new data
Concept drift
P(Y|X) changes; task itself changes
Monitor prediction accuracy vs ground truth
Full retraining or domain adaptation
Label drift (prior prob shift)
P(Y) changes; class balance shifts
Monitor prediction class distribution
Recalibrate decision threshold
Feature drift
Individual feature distribution shifts
Per-feature Kolmogorov-Smirnov test
Investigate sensor hardware or environment
Model degradation
Accuracy drops without clear drift
Track business metrics + accuracy on labeled subset
Debug pipeline, retrain
3.2 On-Device Monitoring for Edge Models
 Confidence histograms: 256-bin histogram of softmax max confidence. Cheap (256 bytes). Upload on cloud
sync. Significant shift from training-time histogram signals drift.
 Input feature statistics: Track running mean/variance of range-Doppler map energy per range bin. Deviation
from calibration values signals environment change (new furniture, obstructions).
 Prediction distribution: Rolling count of predicted class frequencies. If 'empty room' prediction spikes
unexpectedly, something changed in the environment or sensor.
 Shadow model disagreement: OTA deploy new model in parallel. Log cases where incumbent and shadow
disagree. High disagreement rate triggers human review before full rollout.
4. CI/CD for ML
 Code CI: Standard: linting (ruff/flake8), type checking (mypy), unit tests (pytest). ML-specific: test data
transformations deterministically, test feature computation edge cases.
 Model CI (evaluation gate): On every model PR: run full evaluation suite on held-out data. Must beat
incumbent by threshold. Check: no regression on top-10 failure cases from previous version. Block merge if tests
fail.
 Data validation CI: On new data batch: schema validation, distribution shift test, label quality check. Alert if
significant drift detected. Do not silently train on corrupted data.
 Deployment CD: Model artifacts in registry → staging → canary (1%) → ramp to 100% over 48h. Automatic
rollback: if error rate >threshold within 24h, revert. OTA for edge: cryptographically signed model packages.
 Infrastructure as code: Model training jobs as reproducible containers (Docker). All dependencies pinned.
Reproducibility: given commit hash + dataset version → identical model. Tools: DVC (data versioning), MLflow
(experiment tracking), GitHub Actions (pipeline orchestration).
5. A/B Testing & Staged Rollouts


--- Page 4 ---

Algorized Interview Prep — Doc 5: Production ML Systems
Page 4
For edge-deployed people-sensing models, A/B testing requires special design — you cannot split a single vehicle
into two groups:
 Device-level A/B: Assign devices (not users) to treatment/control. 50% of fleet runs model A, 50% runs model
B. Stratify by: device type, deployment environment, customer. Requires fleet management infrastructure.
 Shadow mode: New model runs on device but outputs are logged, not acted upon. Zero user impact. Collect
performance data over 1-4 weeks. Best for safety-critical systems (CPD) — never push unvalidated model to
production output.
 Statistical power: How many devices needed? For detecting 0.5% recall improvement at 80% power, 5%
significance: n ≈ (z_α + z_β)² × 2σ² / δ². For binary outcome with δ=0.005 and p≈0.999, need thousands of
device-hours to detect small changes.
 Business metrics vs model metrics: Recall on test set ≠ production recall. Must define business metric (false
alarm rate per day, detection events per week) and measure it in production. Model metrics are proxies; business
metrics are the truth.
6. Interview Q&A; Bank — Production ML
Q: What is the difference between data drift and concept drift? Give a concrete example for
people-sensing.
■ Data drift: input distribution changes but relationship to label unchanged. Example: new office furniture changes
background radar signature. Concept drift: relationship between input and label changes. Example: company policy
changes 'presence' definition to include people sitting very still (new label semantics). Different responses: data drift →
retrain with new data. Concept drift → relabel + retrain.
Q: Describe your approach to building a monitoring system for a model deployed on 10,000 edge
devices.
■ Three-layer: (1) On-device: confidence histograms and input statistics logged locally, uploaded on next sync. (2) Fleet
aggregation: server-side dashboard showing per-firmware-version distribution shifts, anomaly detection on aggregated
metrics. (3) Sentinel devices: 50 instrumented devices in known environments that upload full telemetry for ground-truth
validation. Alert thresholds: >5% confidence distribution shift in >10% of fleet triggers review.
Q: How do you handle a model update that must not degrade safety-critical performance (CPD recall >
99.9%)?
■ Never push directly. Process: (1) Offline evaluation on hardened test set including all known failure modes. (2)
Hardware-in-the-loop (HIL) testing with simulated sensor data. (3) Shadow deployment on 100 devices in non-CPD
mode. (4) 30-day monitoring period. (5) Staged rollout: 1% → 10% → 100% with 7-day hold at each stage. (6) Automatic
rollback if production recall drops below threshold. This is analogous to my Morphism governance pipeline — formal
gates before any state change.
