# Master Cheat Sheet

**Total Pages:** 17



--- Page 1 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 1
Master Interview Cheat Sheet
8 Topics | Cheat-Sheet Density | Algorized Senior Data Scientist
01
AI / ML Fundamentals
Probability, supervised learning, optimization, regularization, evaluation
02
Deep Learning
Backprop, Convolutional Networks, LSTM/GRU, Transformers, sensor
architectures
03
Technical Interview Q and A
Algorized-specific, system design, coding, resume deep-dives
04
Radar Sensing and Signal
Processing
Ultra-wideband physics, Channel Impulse Response pipeline, detection,
tracking, vitals
05
Production ML Systems
Pipelines, training/serving skew, drift, continuous integration, A/B testing
06
Edge AI and Deployment
Quantization, pruning, TFLite Micro, CMSIS-NN, over-the-air updates
07
Resume Bridge
Every claim mapped to job description, gap analysis, STAR story frameworks
08
Career and Cultural Fit
STAR stories, pivot narrative, questions to ask, negotiation reference


--- Page 2 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 2
AI / ML Fundamentals
01
Core theory every senior data scientist must answer cold
Probability and Statistics
<b>Concept</b>
<b>Definition</b>
<b>Interview Trap</b>
Bayes Theorem
P(A given B) = P(B given A) times P(A) divided by P(B)
Do not confuse P(hypothesis given evidence) with P(evidence given hypothesis) -- base-rate neglect
Maximum Likelihood vs Maximum a Posteriori
MLE: maximize P(data given theta). MAP: maximize P(theta given data)
MAP with Gaussian prior = L2 regularization. MAP with Laplace prior = L1 regularization
Bias-Variance Tradeoff
Mean Squared Error = Bias-squared + Variance + Irreducible noise
High bias = underfitting (too simple). High variance = overfitting (too complex). Ensembles reduce variance
p-value
Probability of observing data at least this extreme, assuming null hypothesis is true
NOT the probability the null hypothesis is true. Common misstatement in interviews
Central Limit Theorem
Sample mean of n independent draws approaches Normal(mu, sigma-squared/n) as n grows
Foundation of confidence intervals and hypothesis testing
Supervised Learning Quick Reference
<b>Model</b>
<b>Objective / Key Formula</b>
<b>Strength</b>
<b>Weakness</b>
Linear Regression
Minimize sum of squared residuals. Closed form: beta = (X-transpose X)-inverse X-transpose y
Interpretable, fast
Linear relationships only
Logistic Regression
Cross-entropy loss. Output: sigmoid(w-transpose x). No closed form -- use gradient descent
Calibrated probabilities
Linear decision boundary
Support Vector Machine
Maximum margin hyperplane + kernel trick for nonlinear boundaries
Works in high dimensions, small datasets
Slow on large N (quadratic-cubic)
Random Forest
Bagging (bootstrap aggregation) + random feature subsets at each split
Low variance, robust baseline
Memory-heavy, opaque
Gradient Boosted Trees (XGBoost)
Sequential weak learners. Each tree fits residuals of previous. L1 and L2 regularization
Usually best accuracy on tabular data
Many hyperparameters to tune
Regularization
- L1 / Lasso: lambda times sum of absolute weights. Produces exact zeros -- acts as feature selection. Equivalent to Laplace
prior on weights.
- L2 / Ridge: lambda times sum of squared weights. Shrinks all weights uniformly, never to zero. Equivalent to Gaussian prior.
- Elastic Net: alpha times L1 plus (1-alpha) times L2. Sparsity plus stability when features are correlated.
- Dropout: Randomly zero activations during training (probability p). Disable at inference and scale by (1-p). DO NOT forget to
disable at test time.
- Early Stopping: Halt training when validation loss stops improving. Implicit L2 regularization effect. Use patience of 5-10
epochs.
- Data Augmentation (radar-specific): Add Gaussian noise to Channel Impulse Response frames, random time shifts, amplitude
scaling. Reduces overfitting without new labeled data.
Evaluation Metrics
<b>Metric</b>
<b>Formula</b>
<b>When to Use</b>
Recall (Sensitivity)
True Positives divided by (True Positives + False Negatives)
False negatives are catastrophic -- Child Presence Detection, medical screening. Primary metric at Algorized.
Precision
True Positives divided by (True Positives + False Positives)
False alarms are costly. Secondary in safety-critical systems.
F-beta Score
((1 + beta-squared) times Precision times Recall) divided by (beta-squared times Precision + Recall)
beta greater than 1 weights recall higher. For Child Presence Detection use F2 or F3.
Area Under Precision-Recall Curve
Area under Precision vs. Recall curve
Imbalanced datasets (better than ROC-AUC when positives are rare -- e.g. 5 percent of radar frames have a person)
Log-Loss
Negative sum of y times log(predicted) + (1-y) times log(1-predicted)
Probability calibration quality -- critical for safety-critical confidence scores
Multi-Object Tracking Accuracy (MOTA)
1 minus (False Positives + False Negatives + Identity Switches) divided by Ground Truth count
People-tracking evaluation: penalizes missed detections, false alarms, and lost track identities


--- Page 3 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 3
Cross-Validation and Model Selection
- k-Fold Cross-Validation: Stratified for imbalanced classes. NEVER shuffle time-ordered sensor data -- use walk-forward
(expanding or rolling window) validation instead.
- Nested Cross-Validation: Outer loop estimates generalization performance. Inner loop tunes hyperparameters. Prevents
leakage from tuning into the performance estimate.
- Bayesian Hyperparameter Optimization (Optuna / Hyperopt): Models the objective surface and chooses next trial
intelligently. More efficient than grid or random search.


--- Page 4 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 4
Deep Learning
02
Architectures, training dynamics, and sensor-specific networks
Backpropagation and Training Essentials
- Chain Rule: Gradient of loss with respect to layer-L weights = (gradient of loss w.r.t. activation-L) times (gradient of activation
w.r.t. pre-activation) times (gradient of pre-activation w.r.t. weights). Backprop computes all gradients in one backward pass via
dynamic programming.
- Vanishing Gradient Problem: Repeated multiplication of small Jacobians (less than 1) drives gradients to zero in deep
networks. Three fixes: (1) ReLU activations -- gradient is 1 for positive inputs. (2) Residual (skip) connections -- identity path
provides gradient highway. (3) LSTM/GRU gating -- cell state allows gradients to flow unchanged over long sequences.
- He Initialization: Weights drawn from Normal(0, 2 divided by fan-in). Use for ReLU networks. Xavier initialization uses 2 divided
by (fan-in + fan-out) and is better for tanh or sigmoid. Prevents vanishing or exploding gradients at startup.
- Gradient Clipping: Cap the gradient norm at threshold tau. Essential for Recurrent Neural Networks over long sequences. Less
critical for Transformers which use Layer Normalization.
Activation Functions
<b>Name</b>
<b>Formula</b>
<b>Default Use</b>
ReLU (Rectified Linear Unit)
max(0, x)
Standard for Convolutional Network hidden layers. Dying ReLU risk (always zero for negative inputs) -- Leaky ReLU fixes this with a small slope of 0.01
GELU (Gaussian Error Linear Unit)
x times Phi(x) where Phi is standard normal CDF
Default in Transformers (BERT, GPT). Smooth approximation, empirically outperforms ReLU
Sigmoid
1 divided by (1 + exp(-x))
Binary classification output layer only. Saturates and vanishes at extreme values -- avoid in hidden layers
Softmax
exp(x_i) divided by sum of exp(x_j) over all j
Multi-class output layer. Numerically stable version: subtract max value before taking exp
Swish / SiLU
x times sigmoid(x)
Used in EfficientNet and modern Convolutional Networks. Non-monotonic, empirically strong
Convolutional Networks (CNN)
- Output size formula: (Input width - Filter size + 2 times Padding) divided by Stride, plus 1. Know this for filter size and padding
design.
- Depthwise Separable Convolution: Split standard convolution into: per-channel spatial convolution (depthwise) followed by
1x1 cross-channel convolution (pointwise). Roughly 8 times fewer parameters than standard convolution. Used in MobileNet.
Critical for Algorized edge deployment.
- Global Average Pooling: Average each feature map to a single number. Replaces large fully-connected layers. Reduces
parameters and overfitting while keeping spatial invariance.
- 1D Convolution for radar: Treats range bins as the spatial dimension. Stack 1D convolutions with increasing dilation for
exponentially growing receptive field without increasing depth. Causal variant: no lookahead for real-time streaming.
- Tiny Convolutional Network (relevant to Algorized): Architecture: BatchNorm -> Conv1D x3 -> Global Average Pooling ->
Fully Connected -> Softmax. Under 200 KB after INT8 quantization. Inference under 48 ms on STM32 microcontroller. 99.38
percent accuracy for people counting, retaining 98.22 percent after quantization.
LSTM and GRU (Gated Recurrent Units)
- LSTM gates: Forget gate f_t, Input gate i_t, Output gate o_t (all sigmoid). Cell candidate c-tilde_t (tanh). Cell update: c_t = f_t
element-wise-times c_(t-1) + i_t element-wise-times c-tilde_t. Prevents vanishing gradient via the cell state highway.
- GRU simplification: Merges forget and input gates into a single Update gate. Approximately 25 percent fewer parameters than
LSTM. Similar accuracy on most tasks. Preferred on edge hardware due to lower memory footprint.
- When to use: LSTM or GRU for multi-step temporal radar CIR sequences where long-range dependencies matter (e.g., tracking
a person across 64+ frames). 1D Convolutional Network or Temporal Convolutional Network (TCN) when parallelism matters
more than exact recurrence.


--- Page 5 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 5
Transformers and Self-Attention
- Self-Attention formula: Attention(Q, K, V) = softmax(Q times K-transpose divided by sqrt(d_k)) times V. The scaling factor
sqrt(d_k) prevents softmax saturation in high dimensions.
- Multi-Head Attention: Run h parallel attention heads with different linear projections. Concatenate outputs then project. Each
head captures different patterns (local vs. long-range, semantic vs. positional).
- Computational cost: O(n-squared times d) per layer -- quadratic in sequence length n. Problematic for long radar frame
streams of 1000+ steps. Use local attention windows, TCN, or Mamba state-space models instead.
- Layer Normalization: Normalize across feature dimension per sample (not across batch). Batch-size independent. Default for
Transformers and Recurrent Networks. Pre-normalization (before attention) is more stable than post-normalization.
Architectures for Sensor and Time-Series Data
<b>Architecture</b>
<b>Key Property</b>
<b>Algorized Fit</b>
1D dilated causal Convolutional Network
Parallelizable. Receptive field doubles per layer. Causal (no future lookahead)
Strong edge baseline -- fast inference, deployable on microcontroller
Temporal Convolutional Network (TCN)
Dilated 1D convolutions with residual connections. Receptive field = 2 to the power of number-of-layers
Best pure convolutional option for Channel Impulse Response sequences
Hybrid CNN plus GRU
Convolutional Network extracts per-frame spatial features. GRU models temporal evolution across frames
Good accuracy-to-compute trade-off for multi-person tracking
Transformer (PatchTST)
Patch-based tokenization of time series. High accuracy but quadratic memory cost
Too heavy for bare microcontroller. Suitable for edge SoC or cloud post-processing
Tiny Convolutional Network (TyCNN)
Sub-200 KB CNN. INT8 quantized. Under 48 ms on STM32
The production-relevant architecture for Algorized edge deployment
Q: When NOT to use Batch Normalization?
A: Avoid for: small batches (fewer than 8 samples), Recurrent Networks (use Layer Normalization instead), and streaming inference
where batch statistics are undefined.
Q: Choose an architecture for 256-step UWB Channel Impulse Response sequences on a microcontroller.
A: Tiny Convolutional Network or Temporal Convolutional Network with dilated causal convolutions. Must be causal (no future
lookahead) for real-time streaming. Avoid full Transformer -- too memory-heavy for bare metal.


--- Page 6 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 6
Technical Interview Q and A
03
Algorized-specific questions, system design, coding scenarios, resume deep-dives
Algorized Domain Questions
Q: Walk me through the complete pipeline from a raw UWB ADC sample stream to a people-count integer on an ARM
Cortex-M microcontroller.
A: Analog-to-Digital Converter captures Channel Impulse Response frames at 50-200 Hz. Exponential Moving Average background
subtraction removes static clutter (walls, furniture). Range gating discards near-field noise and far-field returns. 2D Fast Fourier
Transform across range bins (fast-time) and across frames (slow-time) produces a Range-Doppler map. Constant False Alarm Rate
(CFAR) thresholding detects candidate target cells. Kalman filter or Multiple Hypothesis Tracking (MHT) associates detections into
persistent tracks across frames. INT8-quantized 1D Convolutional Network classifies occupancy count from track features. Output
integer sent via GPIO or CAN bus. Total latency target: under 50 ms end-to-end.
Q: What causes false positive detections in a people-sensing radar system and how do you mitigate them?
A: Sources: HVAC units vibrating at 0.2-0.5 Hz (overlaps human breathing signature), fans, pendulum-style moving objects, RF
interference. Mitigations: (1) Collect an explicit false-positive dataset from each known interference source. (2) Train a discriminator
using spatial coherence: genuine human motion produces a localized return in one range bin; HVAC creates correlated noise across all
range bins. (3) Add a temporal consistency gate: a person must be detected for more than N consecutive frames before counting as
present.
Q: A customer complains the model degrades two months after deployment at a new site.
A: Systematic investigation: (1) Check on-device confidence histogram logs -- rising entropy signals distribution shift. (2) Ask customer
about physical changes: new furniture, HVAC added, renovations. (3) Compare input feature statistics (range-Doppler energy per bin)
against training-time calibration. (4) Collect 200-500 labeled frames from the new environment. (5) Fine-tune classifier head only,
freeze feature extractor (analogous to supervised fine-tuning in language model training). (6) Validate on held-out new-environment
data before pushing over-the-air update.
Q: Design a Child Presence Detection system achieving 99.9 percent recall.
A: Data: diverse environments (day/night, temperature extremes, child sizes, sleeping positions), plus an explicit false-positive corpus.
Model: optimize decision threshold for recall above 99.9 percent on held-out safety test set. Use F-beta (beta = 3) as training objective.
Deployment: INT8 C++ model on AUTOSAR-compliant microcontroller, under 50 ms latency. Monitoring: shadow model on 1 percent
of fleet before full rollout. Validation: UN ECE R129 regulatory test protocol plus hardware-in-the-loop simulation suite. Rollback:
automatic revert if production recall drops below threshold within 7 days.
System Design Questions
Q: How would you build a sensor-agnostic foundation people-sensing model?
A: Pre-train on diverse sensor data (UWB, FMCW, Wi-Fi Channel State Information) using sensor-type embedding tokens similar to
language tokens in multilingual models. Use masked signal autoencoder pre-training on raw signal representations for self-supervised
learning. Fine-tune a sensor-specific adapter layer per modality (analogous to Low-Rank Adaptation in language model fine-tuning)
while sharing the core feature extractor. Task-specific classifier heads (presence, count, vitals) trained with small labeled datasets per
application.
Q: Design a data pipeline for a startup simultaneously collecting field data and training models.
A: Three layers: (1) Edge collection: upload anonymized feature vectors (not raw Channel Impulse Response for privacy) with
metadata: sensor ID, firmware version, environment tag. (2) Cloud ingestion: S3 landing zone, schema validation (Pydantic), quality
filter (signal-to-noise threshold), versioned dataset via DVC. (3) Training trigger: new data volume milestone fires retraining job, which
must pass regression gate before any deployment. Design for schema evolution from day one -- the sensor hardware will change.
Resume Deep Dives
Q: How exactly did you achieve 70 percent runtime reduction and 160,000 USD annual savings on DFT workflows?
A: Baseline measurement: SLURM job accounting gave wall-clock time per job type. Interventions: (1) Trained an ML surrogate model
to pre-screen candidate structures, replacing expensive full Density Functional Theory calculations for most of the workflow. (2) Job
batching reduced scheduler overhead. (3) Checkpoint-restart eliminated redundant recomputation after node failures. (4) Memory
layout optimization for SIMD instructions. Validation: automated regression suite confirmed physics accuracy was unchanged across


--- Page 7 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 7
2,300 production jobs. Savings calculated: 24,000 CPU-hours times cost-per-hour times 0.70 reduction.
Q: Your background is language model training -- how does that help a radar sensing company?
A: Three direct bridges: (1) Automated evaluation harness for regression prevention: the benchmark suite and CI gates built to prevent
language model degradation across updates is exactly the infrastructure Algorized needs before every over-the-air model push. Lead
with this -- most immediately actionable. (2) Reward model logic applied to radar frame quality: RLHF trains a model to judge output
quality and weight training samples. Applied here: train a Channel Impulse Response quality discriminator to score frames as clean or
corrupted by multipath, replacing ad-hoc signal-to-noise thresholds. (3) Supervised fine-tuning for environment adaptation: freeze the
feature extractor of the foundation model, fine-tune the classifier head with 200-500 site-specific labeled frames. Same paradigm as
domain-specific supervised fine-tuning.


--- Page 8 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 8
Radar Sensing and Signal Processing
04
UWB physics, Channel Impulse Response pipeline, CFAR detection, tracking, vitals, ARIA platform
Ultra-Wideband Radar Physics
- Definition: RF signal occupying more than 500 MHz bandwidth or more than 20 percent fractional bandwidth. FCC-regulated
band: 3.1 to 10.6 GHz. ARIA HYDROGEN chip: up to 1.8 GHz programmable bandwidth.
- Range resolution: Delta-R = speed-of-light divided by (2 times bandwidth). At 1.8 GHz bandwidth: Delta-R is approximately 8.3
cm. Higher bandwidth means finer range resolution.
- Angular resolution (ARIA 4x4 MIMO): Approximately lambda divided by (N times antenna spacing), approximately 5 degrees
for 16 virtual elements at 6.5 GHz center frequency. Can separate two people roughly 30 cm apart at 3 m range.
- Through-obstacle capability: Penetrates drywall, wood, and plastic. Blocked by metal. Roughly 5-15 dB attenuation per
non-metallic wall. Presence detectable through 2-3 walls.
- Privacy advantage: Captures amplitude vs. delay, not images. No facial recognition possible. Key advantage for
GDPR-sensitive deployments in offices and healthcare.
- UWB vs FMCW radar: UWB: time-domain Channel Impulse Response, low average power, excellent for Internet-of-Things.
FMCW (Frequency-Modulated Continuous Wave): frequency-domain chirp, natural Range-Doppler map per single frame, higher
signal-to-noise ratio, preferred in automotive (Texas Instruments AWR series). Algorized uses UWB with the ARIA HYDROGEN
chip.
Complete Channel Impulse Response Signal Pipeline
<b>Step</b>
<b>Operation</b>
<b>Key Parameter</b>
1. ADC Capture
Transceiver transmits a short pulse; ADC samples the received Channel Impulse Response at gigahertz rate. Output: amplitude vs. delay (equals range) at 50-200 frames per second.
Frame rate: 50-200 Hz
2. Static Clutter Removal
Exponential Moving Average background: B[t] = alpha times B[t-1] + (1-alpha) times S[t]. Dynamic signal: D[t] = S[t] minus B[t]. Removes static reflectors (walls, furniture). Only moving targets remain.
alpha approximately 0.95 (20-frame memory)
3. Range Gating
Keep only range bins corresponding to the room interior (0.3 to 5 m). Discard near-field interference and far-field noise.
Saves 40-60 percent compute
4. Range-Doppler 2D FFT
Stack N consecutive frames. Apply FFT across range bins (fast-time) then FFT across frames (slow-time). Output: power at each range-velocity pair.
N = 32 or 64 frames typically
5. CFAR Detection
Constant False Alarm Rate threshold adapts to local noise. Cell-Averaging CFAR averages neighbor cells; Order-Statistic CFAR uses k-th largest neighbor (more robust near clutter edges).
Guard cells plus training window around Cell Under Test
6. Tracking
Multiple Hypothesis Tracking (MHT): best multi-person accuracy, handles occlusion. Kalman filter: lower compute, suitable for up to 5 targets. Maintains persistent track IDs across frames.
State vector: x, y, velocity-x, velocity-y
7. CNN Classification
INT8-quantized Tiny Convolutional Network on Range-Doppler features classifies occupancy count. Under 48 ms on STM32 microcontroller.
Model under 200 KB after quantization
Vitals Detection (Breathing and Heart Rate)
- Respiration (0.2-0.5 Hz): Chest displacement of 5-20 mm. Well above noise floor. Extract: select range bin where target is
located; compute FFT of amplitude envelope over 10-30 second window; peak in 0.2-0.5 Hz range equals respiration rate.
- Heart rate (1-2 Hz): Chest displacement of only 0.1-0.5 mm. Near the UWB noise floor. Requires: high signal-to-noise ratio
(subject within 2 m), static subject, high frame rate (above 50 Hz), careful bandpass filtering to separate from respiration
harmonics.
- Child Presence Detection application: A sleeping child breathes at 0.3-0.5 Hz, distinguishing from an empty car seat with no
micro-motion. This is the core Algorized product use case -- know this cold.
- Key challenges: HVAC vibration overlaps the breathing frequency band. Respiration harmonics fall in the heart rate band.
Multi-person scenarios require source separation algorithms.
Sensor Fusion: UWB plus Wi-Fi
<b>Fusion Type</b>
<b>How It Works</b>
<b>Pros</b>
<b>Cons</b>
Feature-level (early)
Concatenate UWB and Wi-Fi features before the ML model input
Single model, low latency
Requires synchronized streams; fragile if one sensor fails


--- Page 9 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 9
Decision-level (late)
Independent models per modality; combine confidence scores at output
Robust to sensor dropout
Higher latency; more memory
Kalman filter fusion
UWB position estimates plus Wi-Fi occupancy as separate measurement inputs to one filter
Principled uncertainty handling
Assumes approximately linear dynamics
ARIA HYDROGEN Platform -- Announced MWC Barcelona, February 2026
Algorized and ARIA Sensing launched an AI-powered UWB radar platform at MWC Barcelona 2026. Launch product:
in-cabin automotive Child Presence Detection for OEM and Tier-1 suppliers. HYDROGEN chip: 4x4 MIMO (16 virtual
elements), true 3D detection, digital beamforming at approximately 5 degree angular resolution, up to 1.8 GHz
programmable bandwidth. CEO Natalya Lopareva: 'By combining AI with a purpose-built 3D UWB radar architecture,
we are unlocking capabilities that were simply not possible with legacy silicon.' Extends beyond automotive: smart
buildings, elderly monitoring, consumer electronics, robotics.


--- Page 10 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 10
Production ML Systems and MLOps
05
Pipelines, training-serving skew, drift detection, CI/CD, monitoring
ML Pipeline Components
- Data Ingestion: Raw sensor data lands in S3 or Google Cloud Storage. Schema validation on arrival (Pydantic or Great
Expectations). Partition by date and sensor ID. Dead-letter queue for malformed records.
- Feature Store: Decouples feature computation from model training. Offline batch serving and online real-time serving MUST
use the same transformation code. Training-serving skew is the number one cause of silent production failures.
- Training: Triggered by new data volume threshold or schedule. Track all experiments (MLflow, Weights and Biases). Pin all
dependencies. Reproducibility requirement: given a commit hash plus dataset version, produce an identical model.
- Evaluation Gate: New model must beat incumbent by a defined threshold on the held-out evaluation set. Must also pass
regression suite on known failure cases. Block deployment if either check fails.
- Deployment: Tag model version as production in registry. Blue-green or canary deployment. Over-the-air update for edge
devices. Automatic rollback trigger: if production metrics degrade more than X percent within 24 hours, revert.
- Monitoring: Track input distribution statistics, prediction distribution, and business metrics (false alarm rate, detection events).
Alert on threshold breach. On-device: lightweight confidence histogram uploads on sync.
Training-Serving Skew -- The Number One Production Failure
The model sees different data at training time versus inference time, causing silent degradation. For Algorized
specifically: the Exponential Moving Average background clutter removal computed differently in offline Python
training versus real-time C++ embedded inference will cause the model to receive systematically different input
distributions. Fix: use the same stateful EMA implementation (ported to Python for training, C++ for inference) in both
pipelines. Validate by running the same raw Channel Impulse Response sequence through both pipelines and
comparing features.
Drift Detection Reference
<b>Drift Type</b>
<b>Definition</b>
<b>Detection Method</b>
<b>Response</b>
Data drift (covariate shift)
Input distribution P(X) changes; relationship P(Y given X) unchanged
Population Stability Index (PSI), Kolmogorov-Smirnov test on features
Collect new labeled data; retrain
Concept drift
Relationship P(Y given X) changes; task definition evolves
Monitor accuracy against ground truth labels over time
Full retraining or domain adaptation
Label distribution shift
Prior P(Y) changes; class balance shifts in production
Monitor prediction class distribution over time
Recalibrate decision threshold
On-device (no ground truth)
Cannot measure accuracy directly on constrained hardware
Confidence histogram entropy increase; input feature statistics deviation from calibration baseline
Flag on next cloud sync; trigger human review
Continuous Integration and Deployment for ML
- Code CI: Linting, type checking, unit tests. ML-specific: test data transformations are deterministic; test feature computation on
edge cases.
- Model CI: Every model pull request triggers full evaluation suite. Must beat incumbent by threshold. Regression check on top-10
known failure cases. Block merge if either fails.
- Data CI: Validate new data batch on arrival: schema check, distribution shift test, label quality review. Do not silently train on
corrupted data.
- Deployment CD: Staging -> canary (1 percent of fleet) -> ramp to 100 percent over 48 hours. Automatic rollback if error rate
exceeds threshold.
- Over-the-Air edge deployment: Cryptographically signed model packages. Write to staging partition, not active partition. Run
known-answer self-test before cutover. Maintain rollback partition.
A/B Testing for Edge Devices


--- Page 11 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 11
- Device-level A/B: Assign devices (not users) to treatment and control groups. Stratify by deployment environment type. You
cannot split a single vehicle into two groups.
- Shadow mode (safest): New model runs on-device but its output is logged, not acted upon. Zero user impact. Collect
performance data over 1-4 weeks. Best for safety-critical systems like Child Presence Detection -- never push an unvalidated
model to the production output path.
- Statistical power: Detecting a 0.5 percent recall improvement at 80 percent power with 5 percent significance requires
thousands of device-hours. Plan the rollout timeline accordingly.
Q: How do you monitor 10,000 edge devices with no ground truth labels available?
A: Three-layer approach: (1) On-device: 256-bin confidence histograms and input feature statistics (mean energy per range bin) logged
locally, uploaded on next cloud sync. Cheap: a few hundred bytes per upload. (2) Fleet aggregation dashboard: per-firmware-version
distribution shift metrics, anomaly detection on aggregated confidence histograms. (3) Sentinel devices: 50 fully instrumented devices
in known controlled environments that upload complete telemetry for ground-truth validation. Alert threshold: more than 5 percent
confidence distribution shift in more than 10 percent of fleet triggers human review.


--- Page 12 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 12
Edge AI and Embedded Deployment
06
Quantization, pruning, TFLite Micro, CMSIS-NN, over-the-air updates
Edge Hardware Reference
<b>Class</b>
<b>Example Chips</b>
<b>RAM</b>
<b>Model Budget</b>
<b>Inference Runtime</b>
Bare metal Cortex-M4/M7
STM32F7, NXP i.MX RT1060
512 KB to 2 MB
100-500 KB
TFLite Micro + CMSIS-NN
Application-class Cortex-A
Raspberry Pi CM4, NXP i.MX8
1 to 4 GB
1 to 100 MB
ONNX Runtime, TFLite
Embedded Neural Processing Unit
STM32 Neural-ART, NXP eIQ
varies
1 to 10 MB INT8
Vendor SDK
Automotive SoC
NXP S32G, Renesas R-Car
2 to 16 GB
Unlimited
TensorRT, ONNX Runtime
Model Compression Workflow
- Step 1 -- Structured Pruning: Remove entire filters or channels with low L2 norm. Target 50-70 percent parameter reduction.
Maps cleanly to C arrays (no sparse data structures needed). Train with pruning masks applied.
- Step 2 -- Knowledge Distillation (optional): Train a compact student model to mimic the soft output probabilities of a larger
teacher model. Use when the architecture must change significantly from the original trained model.
- Step 3 -- Post-Training Quantization (try first): Convert FP32 weights and activations to INT8 using a calibration dataset of
500 representative sensor frames. Achieves roughly 4x size reduction. If recall drops more than 2 percent compared to the FP32
baseline, switch to Quantization-Aware Training.
- Step 4 -- Quantization-Aware Training (if Post-Training Quantization insufficient): Simulate quantization during training
using fake quantization nodes. Gradients flow through via the Straight-Through Estimator. Model learns to be robust to rounding
errors. Required for regression tasks (vitals estimation) and models under 1 million parameters.
- Step 5 -- Export: PyTorch -> ONNX (opset 12) -> TFLite -> C byte array via xxd tool. Validate each conversion step: max output
difference vs. previous step must be under 0.001.
Quantization Key Concepts
<b>Concept</b>
<b>Details</b>
INT8 scale factor
Scale S = (max - min) divided by 255. Zero-point = 0 for symmetric (weights), non-zero for asymmetric (activations after ReLU).
Per-channel vs per-tensor
Per-channel quantization: separate scale per output channel of a weight tensor. 1-3 percent better accuracy than per-tensor. Required for depthwise convolutions. Supported by TFLite Micro.
Symmetric quantization
zero-point = 0. Range is -128 to 127. Simpler hardware multiply-accumulate. Better for weight tensors.
Asymmetric quantization
zero-point is non-zero. Range is 0 to 255. Better for activation tensors that have non-zero mean (e.g. after ReLU).
Mixed precision
Keep first layer, last layer, and skip connections in INT16 or FP16. Recover 1-2 percent accuracy at a small memory cost increase. Use when standard INT8 accuracy drop exceeds the safety threshold.
Inference Runtimes for Embedded C++
- TensorFlow Lite Micro (TFLite Micro): Single C++ source file. No operating system dependency. CMSIS-NN backend for ARM
acceleration. Supports INT8 models. Uses a pre-allocated tensor arena (no dynamic memory). Target platforms: STM32, Arduino
Nano 33, Nordic Semiconductor nRF52.
- CMSIS-NN (Cortex Microcontroller Software Interface Standard Neural Network): ARM-optimized SIMD kernels using the
DSP extension on Cortex-M4/M7/M33/M55. Key function: arm_convolve_HWC_q7_fast(). Without CMSIS-NN, inference is 5 to
10 times slower on Cortex-M. Integral backend to TFLite Micro.
- ONNX Runtime (embedded build): Supports ARM Linux targets (NXP i.MX8, Raspberry Pi 4). Dynamic shapes. XNNPACK
execution provider for mobile-optimized ops. About 1-3 MB minimal binary. Good for more powerful edge processors running
Linux.


--- Page 13 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 13
- Custom C++ inference: Hand-write the inference loop as struct arrays and function calls. Zero framework overhead. Maximum
control. Only justified for models under 50 KB or AUTOSAR/SafetyOS certification contexts where third-party libraries are
prohibited.
Over-the-Air Update Architecture
- Model package format: Signed bundle containing: model binary + version metadata + hash of evaluation results. Cryptographic
signature prevents tampered models from being deployed.
- Safe staging: Download to staging flash partition. Do NOT overwrite the active partition until validation passes.
- On-device validation: After staging: run a known-answer test (fixed input -> compare output hash to expected). If mismatch:
abort, delete staging partition, report error upstream.
- Cutover and rollback: Atomic boot pointer swap. First N operating hours: provisional mode with performance monitoring.
Anomaly detected -> auto-rollback to previous version in backup partition.
Q: INT8 Post-Training Quantization drops Child Presence Detection recall by 3 percent. What do you do?
A: 3 percent is too large for a safety-critical application. Steps in order: (1) Switch to Quantization-Aware Training -- model adapts to
rounding errors during training. (2) Use per-channel quantization for depthwise convolution layers if not already applied. (3) Apply
mixed precision: keep the first layer, last layer, and any skip connection layers in INT16. (4) If still insufficient: change architecture to
more quantization-friendly operations, or use knowledge distillation from the FP32 teacher.


--- Page 14 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 14
Resume Bridge
07
Every claim mapped to job requirements, gap analysis with ramp-up plans, STAR story summaries
Resume vs Job Description Alignment
<b>Job Requirement</b>
<b>Your Evidence</b>
<b>Bridge or Risk</b>
PhD in Computer Science or related field
PhD EECS UC Berkeley (December 2025)
Direct match -- no bridging needed
3+ years hands-on data science experience (PhD track)
PhD 5 years + Turing Enterprises + Morphism Systems = 6+ years total
Frame PhD as applied R and D with production deliverables, not pure research
ML on raw sensor / time-series data
Density Functional Theory time-series workflows, High-Performance Computing signal pipelines
Physics-based signal processing is directly analogous at the engineering level. Different physics, same pipeline patterns.
C/C++ edge deployment
C++ High-Performance Computing (VASP, LAMMPS, CUDA), 11+ years
Deep C++ background. Ramp on CMSIS-NN and TFLite Micro is incremental, roughly 1 week.
Sensor fusion and edge AI
Not directly on resume -- acknowledge this
Bridge via: multi-modal High-Performance Computing data fusion, Kalman filter from physics background, studied academic literature on UWB systems
Full ML pipeline (train, deploy, monitor)
DFT HPC pipelines: 2,300+ jobs, monitoring dashboards, regression testing suite
Exact match at the systems engineering level. Domain is different, patterns are identical.
CI/CD and monitoring frameworks
GitHub Actions, MLflow, automated regression testing, monitoring dashboards
Direct match
PyTorch and Python ML libraries
Listed under LLM and Deep Learning skills
Direct match
Three LLM/RLHF to Sensor AI Bridges (Memorize These)
Bridge 1 -- Lead With This (Most Immediately Actionable)
Automated evaluation harness for regression prevention: The benchmark suite and CI gates built to prevent language model
degradation across updates is exactly the infrastructure Algorized needs before every over-the-air model push. Every time a new
model version ships to an automotive edge device, a regression suite of standardized radar scenarios (different rooms, occupancy
counts, interference types) must pass. This is not theoretical -- you have built and operated this infrastructure.
Bridge 2 -- Creative Technical Contribution
RLHF reward model applied to radar frame quality: In RLHF, a reward model judges output quality to filter and weight training
samples. Applied to UWB: train a Channel Impulse Response quality discriminator to score frames as clean versus corrupted by
multipath or interference. This replaces ad-hoc signal-to-noise thresholds with a learned quality oracle that automatically routes
low-quality frames out of the training pipeline.
Bridge 3 -- Domain Adaptation Story
Supervised fine-tuning for environment-specific adaptation: Supervised fine-tuning efficiently adapts a base language model to a new
domain using a small curated dataset. Direct translation: freeze the feature extractor of the foundation people-sensing model,
fine-tune only the classifier head with 200-500 labeled frames from the customer's specific environment (warehouse, office, car
cabin). Same paradigm: preserve general capability acquired during pre-training, adapt cheaply to the specific deployment context.
Gap Analysis -- Honest Assessment with Ramp-Up Plan
<b>Gap</b>
<b>Honest Assessment</b>
<b>Ramp-Up Plan</b>
Direct UWB radar experience
None on resume. Real gap.
Have studied TyCNN and HDL4AR academic literature and understand the full Channel Impulse Response pipeline. Physics background accelerates hands-on ramp.
CMSIS-NN and TFLite Micro
Deep C++ HPC experience, but not embedded MCU inference specifically.
Well-documented API with good tutorials. Estimate 1 week to first working example given existing C++ depth. Quantization theory already understood.
Real-time sensor fusion systems
Not directly on resume.
Multi-modal High-Performance Computing data fusion is conceptually analogous. Kalman filter is standard physics curriculum. Gap is in specific embedded implementation details.
Automotive certification standards
No automotive industry background.
Frame as process engineering: working within constrained formal specifications is familiar from PhD research and from Morphism governance work.


--- Page 15 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 15
How to frame gaps: 'I have not worked with CMSIS-NN directly, but I have deep C++ from High-Performance
Computing and I understand INT8 quantization theory end-to-end. I expect 1-2 weeks to be productive on the
inference side.' More credible than claiming familiarity you do not have. More impressive than just saying you do not
know it.
STAR Story Summaries (8 Prepared -- Know Cold)
Technical Impact Under Constraints
160K USD savings / 70 percent runtime. Baseline via SLURM accounting. ML surrogate replaced expensive DFT steps. Job batching
and checkpoint-restart removed overhead. Regression suite validated accuracy unchanged across 2,300 production jobs.
Production System Failure and Recovery
DFT pipeline silent failure: wrong results, no crash. Root cause: upstream package configuration change. Fix: immutable
configuration management plus automated correctness checks on every job output. Changed engineering practice permanently.
Learning a New Domain Rapidly
Spintronics at KAUST to 2D materials at Berkeley. Different physics, different simulation tools. Approach: 5 key papers, reproduced
each result, weekly domain expert meetings, verification scripts before scaling. First paper submitted at 14 months.
Technical Disagreement with Data
Disagreed with reward model architecture at Turing Enterprises. Ran 3-day ablation on 100 held-out problems. Showed 12 percent
performance gap with quantitative evidence. Team adopted the approach.
Ambiguity and Self-Direction
Morphism Systems: no external specification. Grounded design in category theory and sheaf-theoretic drift detection. Full pipeline
with passing tests and mathematical foundations. Shows: I bring structure to ambiguous problems.
Cross-Functional Communication
PhD defense to mixed committee (EE, materials science, physics). Led with engineering implications first, then physical mechanism,
then math derivation. Passed with minor revisions. Lesson: lead with impact.
Speed vs Correctness Trade-off
Turing delivery deadline conflicted with finding 15 percent dataset labeling errors. Quantified expected accuracy impact (3-5 percent
degradation estimate). Proposed 3-day correction sprint with validation protocol. Team accepted. Final benchmark improved 4.2
percent.
Ownership Beyond Job Description
Built Python/SLURM cost accounting wrapper with anomaly alerts when LBNL accounting tools were inadequate. Adopted by 6 other
lab groups. Saved roughly 80 hours per year across the lab. Shows proactive infrastructure ownership.


--- Page 16 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 16
Career, Personal and Cultural Fit
08
Pivot narrative, behavioral Q and A, questions to ask, negotiation reference
Why Algorized -- Three-Layer Narrative
- Layer 1 -- Problem domain: Five years in physics-based simulation taught you that getting ML models to work on real physical
signals -- not clean benchmark datasets -- is a fundamentally different engineering challenge. UWB radar sensing is exactly that
problem at an interesting intersection of physics and systems engineering.
- Layer 2 -- Stage and ownership: Algorized is scaling, not a 5-person pre-product company. The ARIA Sensing platform launch
at MWC 2026 proves they have hardware partnerships and real product traction. You can have genuine ML stack ownership from
day one.
- Layer 3 -- Safety criticality: Child Presence Detection is a system where your work directly affects whether a child lives. That
level of stakes creates a different engineering culture -- rigorous validation, formal regression testing, careful deployment -- which
is the culture you want to work in.
Proof-of-research signal to drop naturally: 'I saw the ARIA Sensing platform announcement from MWC 2026 -- the 4x4
MIMO 3D beamforming at 5-degree angular resolution changes what is possible for occupant detection in confined
geometries like car cabins.' Signals genuine technical interest, not HR-style enthusiasm.
Pivot Questions -- What to Say
Why leave research and Morphism Systems now?
I want to apply systems and infrastructure work in a product context where impact is direct and measurable. Algorized is rare in
combining first-class physics sensing with first-class systems engineering at a startup scale where I would own the full stack.
Would Morphism Systems conflict with your work here?
Morphism is a governance research project. It does not compete with people-sensing AI. I am comfortable placing it in maintenance
mode. My priority at Algorized would be Algorized. State this directly and move on -- do not over-explain.
Your PhD took 6 years. What took so long?
My dissertation covered a genuinely novel materials physics problem plus four open-source simulation platforms built during that
time. Berkeley computational physics PhDs are typically 5-6 years for ambitious projects. Answer confidently -- do not apologize for
the timeline.
STEM OPT -- any work authorization concerns?
Approximately 2.5 years remaining on STEM Optional Practical Training. Eligible for H-1B in the next lottery cycle. Well-understood
process for venture-capital-backed deep-tech companies. Brief, factual, and calm -- do not over-explain.
Why hire you over a candidate with direct embedded ML experience?
An embedded specialist knows CMSIS-NN but may not have designed a full monitoring and governance pipeline, built training
infrastructure from scratch, or understood the physics of the sensing modality deeply. I bring physics intuition for the signal, ML
systems experience for the pipeline, and governance architecture for reliable deployment. The embedded runtime ramp is the fastest
part.
Questions to Ask -- Prioritized by Stage
Coffee Chat with Product Tech Lead -- Use 2 of these:
- The ARIA HYDROGEN 4x4 MIMO platform at 5-degree angular resolution fundamentally changes what is trackable in a
confined space. How does that change the model architecture you are building toward?
- What is the hardest current failure mode for Child Presence Detection -- the detection itself, or the false positive rate at the
confidence threshold needed for regulatory compliance?


--- Page 17 ---

Master Cheat Sheet -- Algorized Senior Data Scientist Interview Prep
Page 17
- How is ML stack ownership structured today -- is signal processing and ML combined in one team, or are they separate?
- What does a successful first 90 days look like for someone coming into this role?
Technical Panel -- Use 2 of these:
- Is per-site fine-tuning for environment-specific model degradation in the roadmap, or is the foundation model expected to
generalize zero-shot to new environments?
- What does the over-the-air update pipeline look like today for the edge model? Would this role own that infrastructure?
- What is the biggest current bottleneck -- data collection and labeling, model accuracy on edge cases, or the C++ deployment
pipeline?
Cultural Fit
<b>What Algorized Values</b>
<b>How to Demonstrate It</b>
End-to-end ownership ('hands-on role with significant ownership')
Reference DFT pipeline: designed, built, monitored, maintained. No handoff to another team.
Swiss engineering precision (HQ in Etoy, Switzerland; safety-critical products)
Show rigorous validation, regression testing, zero tolerance for silent production failures.
Startup velocity ('dynamic startup environment')
Reference fast delivery under uncertainty: Turing 1-month contract, Morphism self-directed architecture.
Customer proximity ('willingness to travel for on-site support')
Express genuine willingness. Ask proactively about customer site engagement process.
Genuine product passion ('genuine interest in people-sensing')
ARIA platform details plus specific failure mode discussion proves deep interest, not surface enthusiasm.
Negotiation Reference (Senior DS, Campbell CA, PhD)
<b>Component</b>
<b>Market Range</b>
<b>Negotiation Notes</b>
Base Salary
170,000 to 210,000 USD
PhD premium plus systems depth. Anchor at 195,000 USD.
Equity (stock options)
0.1 to 0.5 percent at early stage
Ask: vesting schedule, cliff period, strike price vs. last valuation.
H-1B Sponsorship
Not monetary
Get written commitment to sponsor H-1B in the next lottery cycle as a condition of accepting the offer.
Title
Senior Data Scientist or Senior ML Engineer
Negotiate Senior as a minimum. Ask about the Staff Engineer career path.
3-Sentence Positioning Statement -- Memorize This
I am a computational physicist turned ML systems engineer, with production experience building both
high-throughput scientific computing pipelines and language model training and evaluation infrastructure at scale. I
have deep familiarity with the full model development lifecycle -- from architecture and training through quantized
edge deployment, monitoring, and regression-safe update pipelines -- which maps directly to what Algorized needs to
scale its edge-AI platform. I am drawn to this role because it combines rigorous physics-based sensing with the
systems engineering challenges I find most interesting: reliability, deployment at scale, and real-world safety impact.
