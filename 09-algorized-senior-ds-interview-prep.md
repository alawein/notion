# PDF 9 - Extracted Content

**Total Pages:** 12



--- Page 1 ---

Senior Data Scientist
Interview Preparation Guide
 Algorized — Edge-AI & People-Sensing Systems Focus
 Candidate: Meshal Alawein, PhD (EECS)  |  Role: Senior Data Scientist  |  Campbell, CA
 Interview: Coffee Chat, Orchard Valley Coffee  |  March 2026
 
About Algorized
 VC-funded Silicon Valley deep-tech company with Swiss roots (US office: Campbell, CA; R&D;:
Etoy, Switzerland)
 Builds edge-AI foundation models for real-time human awareness using existing wireless sensors
 Core product: people-sensing models enabling safe human-machine co-presence on
embedded/edge hardware
 Deployment stack: C/C++ runtime on edge devices; primary sensor inputs include FMCW radar +
fusion
 Stage: rapidly scaling startup — explicit expectation of end-to-end ownership, frontend through
embedded
Contents
Section A
Algorized's Edge-AI & Radar Domain
Deep research: FMCW radar ML, edge deployment, sensor fusion, MLOps
Section B
Technical Bridge Questions
How your resume maps directly to the role's requirements
Section C
Advanced Systems & MLOps Concepts
Prepared talking points for senior-level technical depth
Section D
Coffee Chat Strategy & Questions to Ask
Tone calibration, pitch, and high-signal questions


--- Page 2 ---

SECTION A — Algorized's Edge-AI & Radar Domain
Deep research: FMCW radar processing, edge deployment architectures, sensor fusion, post-deployment MLOps
A.1 FMCW Radar for People-Sensing — Core Signal Pipeline
Algorized's models ingest raw radar sensor data. FMCW (Frequency-Modulated Continuous Wave) radar is the
dominant modality for indoor people-sensing: privacy-preserving, operates through darkness and smoke, and
provides simultaneous range-Doppler-angle information at low power budgets.
Signal Processing Chain — Know This Cold
 1. ADC Sampling: Raw IQ (complex baseband) samples from each RX antenna captured per chirp at ~few
MHz. Multiple TX/RX antenna pairs enable MIMO virtual array formation, expanding effective aperture.
 2. Range FFT: DFT applied along fast-time dimension (samples within a single chirp). Beat frequency maps to
distance: range_bin = c * f_beat / (2 * chirp_slope). Produces range profile per chirp.
 3. Doppler FFT: DFT applied along slow-time dimension (chirp index within a frame). Phase shift between
chirps encodes radial velocity. Produces 2D Range-Doppler map.
 4. 2D CFAR Detection: Constant False Alarm Rate thresholding applied to Range-Doppler map. Adaptively
estimates local noise floor; identifies detections above threshold. Common variants: CA-CFAR, OS-CFAR.
 5. Angle Estimation: Phase difference across antenna array elements yields azimuth/elevation. Algorithms:
FFT beamforming (fast, moderate resolution), MUSIC (super-resolution, expensive), CAPON/MVDR
(beamformer with interference cancellation).
 6. Point Cloud Output: Each detection = (range, Doppler velocity, azimuth, elevation, RCS intensity,
timestamp) tuple. This is the primary input space for downstream ML models.
ML Architectures for Radar Point Clouds
 PointNet / PointNet++: Operates on unordered point clouds; permutation-invariant; strong baseline for people
detection/classification from sparse radar returns. PointNet++ adds hierarchical local grouping.
 Range-Doppler CNN: Treats 2D Range-Doppler heatmap as image; 2D conv layers. Fast, deployment-friendly
in C++; loses 3D spatial info but excellent for activity classification.
 Temporal models (LSTM / TCN): Micro-Doppler signatures over time are highly discriminative for gait, activity
recognition, fall detection. TCN (dilated causal convolutions) preferred over LSTM for edge due to parallelism.
 Graph Neural Networks: Model spatial relationships between detections. Effective for multi-person tracking
under occlusion; edges encode proximity and shared-trajectory priors.
 BEV anchor-free detection: Bird's-eye-view grid projection; YOLO-style head for person localization. Common
in automotive radar; increasingly adopted for indoor sensing pipelines.
A.2 Edge AI Deployment — Quantization, Pruning & C/C++ Runtime
The JD explicitly requires C/C++ deployment experience. The PyTorch research model to edge-deployed binary
path is a multi-stage compression and compilation pipeline.


--- Page 3 ---

Quantization
Technique
When to Use
Accuracy Cost
Post-Training Quant (PTQ)
INT8
Fastest path; needs calibration dataset
(100–1000 samples)
~0.5–1% mAP drop typical;
acceptable for most
deployment
Quantization-Aware Training
(QAT)
When PTQ accuracy is unacceptable;
adds ~20% training time
Recovers most PTQ loss; gold
standard for edge
FP16 / BF16
Edge GPUs/NPUs with FP16 support
(Jetson, etc.)
Minimal; near-lossless on
modern hardware
Binary / Ternary Networks
Ultra-constrained MCUs with no FP unit
Significant; task-specific
viability assessment required
Structured Pruning vs. Unstructured Pruning
 Unstructured pruning: Zeroes individual weights. Achieves high sparsity ratios (>90%) but requires sparse
BLAS support at inference time — typically not available on edge C++ runtimes. Net speedup often marginal
without hardware support.
 Structured pruning (channel/filter): Removes entire output channels from conv/linear layers. Directly reduces
FLOPs and memory without any sparse runtime requirement. Preferred for C++ edge targets. Combine with
magnitude or gradient-based importance scores.
 Knowledge distillation: Train compact student to mimic teacher's soft outputs. Orthogonal to pruning;
compounds benefit. Often +2–5% accuracy recovery on top of pruned model.
C/C++ Deployment Runtime Options
 ONNX Runtime (C API): Cross-platform; export from PyTorch via torch.onnx.export(); run inference in C++
with minimal dependencies. Supports INT8 execution providers. Ideal for ARM Linux edge nodes.
 TensorRT (NVIDIA): For edge GPUs (Jetson Orin/Xavier); layer fusion + INT8 calibration; C++ API; 3–10x
latency reduction over PyTorch baseline. Requires NVIDIA hardware.
 TFLite / XNNPACK: ARM MCU and embedded Linux targets; FlatBuffer model format; C delegate interface;
XNNPACK backend gives significant NEON speedup on Cortex-A.
 NCNN / MNN: Tencent/Alibaba frameworks for ARM NEON; zero dynamic allocation; header-only capable.
Common in IoT and smart home radar hardware.
 Compiled C inference: For ultra-tight MCUs (no OS, <256KB SRAM): compute graph compiled to static C
arrays. Tools: Edge Impulse, microTVM, or hand-rolled.


--- Page 4 ---

Interview Bridge — HPC to Edge
 Your CUDA/MPI/C++ HPC experience directly applies: you understand memory layout, cache
locality, SIMD vectorization, and pipeline stalls — the same properties governing edge inference
latency.
 Frame the DFT workflow optimization (70% runtime reduction) as systematic profiling-driven
optimization — exactly the discipline applied to model quantization and runtime profiling.
 Sentence to use: 'My HPC background gives me an intuition for where compute actually lives in a
pipeline — profiling-first, then targeted optimization. That mental model transfers directly to edge ML.'
A.3 Sensor Fusion for People-Sensing
Single-modality radar has known limitations: sparse point clouds, near-range blind zones, angle ambiguity.
Sensor fusion with IMU, camera, or multi-radar arrays improves robustness.
Fusion Architectures
 Early Fusion (raw level): Concatenate raw sensor streams before any model processing. Tightest coupling;
requires strict time synchronization; highest bandwidth demand. Rarely practical for heterogeneous sensors.
 Mid Fusion (feature level): Extract per-modality features independently; fuse at intermediate representation
(learned cross-attention, concatenation + projection). Most common in production; allows modality-specific
encoders; handles asynchronous sampling rates.
 Late Fusion (decision level): Run independent models per modality; combine predictions (voting, learned
combiner). Easiest to make sensor-agnostic and to gracefully degrade when a modality is unavailable.
Sensor-Agnostic Pipeline — Algorized's Explicit JD Requirement
The JD requires 'sensor-agnostic people-sensing model training.' Architectural patterns to satisfy this:
■ Modality dropout during training: randomly mask sensor inputs at training time; forces the model to be robust to
missing modalities at inference.
■ Shared latent space with modality-specific encoders: each sensor type has a learned projection into a common
embedding; the people-sensing head operates on the fused embedding.
■ Foundation model + sensor-specific adapters: pretrain a shared backbone on large multi-sensor corpus; fine-tune
lightweight adapters per deployment configuration.
A.4 MLOps & Monitoring for Edge-Deployed Models
Post-deployment monitoring on C/C++ edge devices is structurally different from cloud ML monitoring: devices
may be offline, have no persistent storage, and cannot run Python-based instrumentation.
On-Device Monitoring (Edge Side)
 Lightweight telemetry: Log prediction confidence histogram, input feature statistics (mean/variance), and
class-count distribution in a fixed circular buffer. Flush to server on heartbeat or network availability.
 On-device anomaly detection: Simple statistical tests (z-score on aggregated input features, prediction
entropy threshold) compiled to C. Zero Python dependency. Triggers server-side review flag when tripped.


--- Page 5 ---

 Shadow mode deployment: Run old and new model in parallel; compare outputs before fleet-wide switch.
Requires no ground truth — just output agreement rate as a proxy signal.
Drift Types & Detection Strategies
 Covariate shift (data drift): Input distribution P(X) changes — new environment, different ceiling height,
seasonal clutter changes. Detect via KS-test or PSI on telemetry feature aggregates.
 Concept drift: P(Y|X) changes — new behavior patterns, sensor degradation. Requires ground-truth feedback;
hardest on edge. Mitigate via periodic human annotation of flagged high-entropy samples.
 Virtual drift: System-level changes — firmware update alters ADC calibration, mounting angle shifted.
Detected by hardware version metadata + A/B deployment tracking rather than statistical tests.
CI/CD for Edge ML
 OTA (over-the-air) model update pipeline with signed firmware images and atomic rollback capability.
 Automated regression test suite against held-out sensor capture datasets before any OTA push. Gate: latency
P95 and accuracy metrics must not degrade beyond threshold.
 Canary deployment: push to 5% of fleet; monitor telemetry for 48h; auto-promote or rollback based on KPIs.
Key Distinction to Communicate
 Cloud MLOps = instrument the model, watch dashboards. Edge MLOps = design for bandwidth
constraints, offline periods, no Python runtime, and atomic firmware rollback.
 Your DFT/HPC regression testing pipelines are structurally analogous: scheduled jobs,
convergence criteria as pass/fail gates, automated requeue on failure. Same mental model.


--- Page 6 ---

SECTION B — Technical Bridge Questions
How your resume maps directly to role requirements — STAR-structured responses
B.1 Governed Software Kernels -> Edge Model Reliability
Your Morphism Systems work is the strongest direct analog to what Algorized needs: reliable, updateable,
auditable edge model deployments.
Q: You built 'governed software kernels with structured repositories and validation gates.' How does that
apply to C/C++ edge AI?
A: The primitives transfer directly. A policy-enforced validation gate becomes a mandatory regression test suite
run against held-out radar capture datasets before any OTA firmware push. Structured output enforcement in
LLM agents maps to typed C structs with bounds-checked deserialization at inference output boundaries on
device. Reproducible upgrade paths become signed delta firmware with atomic commit/rollback semantics.
The governance layer is not framework-specific — it is the discipline of encoding system invariants as
machine-checkable constraints at every API boundary.
Q: Your Morphism work uses sheaf-theoretic drift detection. Can you translate that to a practical radar
monitoring architecture?
A: In a sheaf framing, each edge device is a local stalk providing observations; the global section consistency
check detects when local distributions diverge from fleet baseline — which is covariate shift. Practical
implementation: each device computes a low-dimensional statistical sketch of its range-bin energy distribution
and ships it in the telemetry heartbeat. The server checks cross-device consistency; inconsistency triggers
targeted recalibration — without transmitting raw radar data, satisfying bandwidth and privacy constraints
simultaneously.
B.2 SFT/RLHF & Reward Modeling -> Domain-Specific Model Refinement
Three concrete transfers from your LLM training methodology to Algorized's non-LLM radar domain:
1. Human-in-the-loop annotation as a reward signal
In RLHF, human preference labels train a reward model that shapes generation behavior. Radar analog: route
low-confidence or high-entropy detections from edge devices to human annotators. Train a confidence calibration
model on annotator decisions — analogous to a reward model — that improves the edge model's uncertainty
estimates without requiring full dataset re-annotation. This closes the data quality feedback loop efficiently.
2. Constitutional constraints as physically grounded hard rules
SFT enforces structured output schemas via loss masking on invalid tokens. Edge-AI analog: add a differentiable
constraint layer during training that penalizes physically impossible outputs (person count > room occupancy limit,
radial velocity > human sprint speed, range < physical near-field blind zone). Reduces false positives from
multipath artifacts — a direct product-quality improvement.


--- Page 7 ---

3. Synthetic data generation for rare/dangerous scenarios
RLHF pipelines use synthetic prompts to cover distribution gaps that real data undersamples. For radar: simulate
rare events (falls, crowd surges, near-field occlusion) using radar simulation tools (FEKO, CST Studio, or
raytracing-based simulators) and augment the real capture dataset. Your experience building synthetic data
pipelines for LLM training translates directly to sensor simulation toolchains.
B.3 DFT/HPC Optimization -> ML Pipeline Architecture
Q: You achieved 70% runtime reduction on DFT workflows with 2,300+ jobs. How does that translate to
ML data pipeline design?
A: The methodological transfer is direct: profile the critical path first (I/O vs. compute vs. memory-bound
stages), introduce ML surrogate models to short-circuit expensive computation, and rebuild the scheduler to
maximize hardware utilization. Applied to an ML radar pipeline: profile where latency lives (data loading,
preprocessing, augmentation, batching), replace bottlenecks with vectorized alternatives (CUDA-accelerated
range-Doppler processing instead of CPU), and implement async prefetching so the GPU never stalls on data
I/O. The $160K cost savings demonstrate that I think in terms of business-impact metrics, not just benchmark
scores in isolation.
Q: Your HPC background is scientific computing, not radar. Why should Algorized believe you can ramp
on radar signal processing?
A: Scientific computing at this level is signal processing: DFT is the same mathematical transform underlying
range/Doppler FFT in FMCW radar. I've applied Fourier analysis to large-scale structured datasets (VASP
wavefunctions, SIESTA Green's functions) with the same core operations — FFT pipelines, spectral analysis,
complex array arithmetic. The domain-specific piece — radar phenomenology, clutter statistics, antenna array
calibration — has a ramp time of weeks given this foundation, not months. I can be productive on the ML
architecture and pipeline side immediately.
B.4 Time-Series & Sensor Data Questions
Q: What are the unique challenges of raw radar sensor data versus standard ML datasets?
A: Three categories: (1) Physical artifacts — multipath reflections, mutual interference between co-located
radar units, static clutter. These require both signal-processing-layer preprocessing and learned filtering in the
ML model. (2) Variable cardinality — radar point clouds have different point counts per frame; models must be
cardinality-invariant (PointNet approach) or use fixed-size grid projections (BEV approach). (3) Annotation cost
— obtaining ground-truth person positions and activities from radar alone requires synchronized camera rigs,
homography calibration, and semi-automated labeling pipelines.
Q: Describe your approach to building a scalable, sensor-agnostic data pipeline.
A: Three-layer design: (1) Ingest layer — sensor-specific drivers output a normalized intermediate format
(timestamped detection struct with sensor-type metadata field); new sensor types plug in by implementing a
driver interface, not by modifying pipeline core. (2) Transform layer — stateless preprocessing operators
composable via a DAG; unit-testable in isolation; parameterized per sensor type via config, not code branches.
(3) Storage layer — versioned dataset registry (DVC or internal equivalent) with schema validation at write
time; downstream models pin to a dataset version, enabling fully reproducible training runs and retroactive
reprocessing when algorithms improve.


--- Page 8 ---

SECTION C — Advanced Systems & MLOps Concepts
Prepared talking points for deep technical discussion at senior level
C.1 Full ML Lifecycle — Research to Production (Algorized Context)
Data Collection: Radar captures in annotated environments; calibration rigs with ground truth (MOCAP or
camera); edge-case simulation for rare events (falls, crowds).
Data Validation: Schema checks, outlier detection, class balance audits; automated rejection of corrupted
captures (e.g., ADC saturation artifacts). Hard gate before any training run.
Feature Engineering: Range-Doppler map normalization, point cloud voxelization, Doppler velocity unwrapping,
temporal frame stacking for micro-Doppler features.
Distributed Training: PyTorch DDP; mixed-precision (AMP); gradient checkpointing for memory-constrained
multi-GPU setups; experiment tracking via MLflow/W&B; deterministic seeds.
Multi-Objective Evaluation: Offline: mAP (detection), MOTA/HOTA (tracking), F1 (fall detection). On-device:
inference latency P95, peak SRAM, power draw on target hardware.
Compression Pipeline: Structured pruning → QAT INT8 → ONNX export → C++ runtime benchmark on target
hardware. Each step gated by regression tests against held-out captures.
OTA Deployment: Signed firmware package; staged canary rollout (5% fleet); telemetry monitoring for 48h;
automated rollback on P95 latency or accuracy KPI violation.
C.2 Statistical Monitoring — Technical Depth
Distribution Shift Tests
 Kolmogorov-Smirnov (KS) test: Nonparametric; tests if two 1D distributions are from the same population.
Apply per feature in telemetry. Low computational cost; suitable for fleet-side server monitoring.
 Maximum Mean Discrepancy (MMD): Kernel-based; handles multivariate distributions; more powerful for
high-dimensional feature spaces. Computationally heavier — run server-side on aggregated sketches.
 Population Stability Index (PSI): Binned divergence measure; industry standard for tabular feature
monitoring. Easy to implement in C telemetry for on-device pre-aggregation.
 CUSUM / EWMA: Sequential change-point detection; constant memory; ideal for on-device real-time drift
detection in a C++ inference loop.
C.3 Radar Failure Mode Catalogue
Demonstrating awareness of failure modes signals production depth beyond academic ML. Know at least three
cold:


--- Page 9 ---

Multipath Ghost Detections: Radar signal reflects off walls/floors before reaching a person, creating phantom
detections at incorrect range/angle. Mitigation: training data diversity across environments; geometric consistency
checks (ghost rarely has consistent velocity across frames); floor-bounce suppression filters.
Mutual Interference (Multi-Radar Deployments): Two co-located FMCW radars with overlapping chirp
schedules create beat-frequency artifacts that appear as false targets. Mitigation: time-division multiplexing
(TDM), orthogonal waveform design (different start frequencies, chirp slopes), or interference mitigation via
adaptive CFAR thresholds.
Micro-Doppler Aliasing: Fast limb movements (running, rapid gestures) alias across Doppler bins if pulse
repetition interval is too long. Mitigation: select radar parameters (PRF, chirp bandwidth) at system design time for
expected human velocity range; typically covered by standards like 60GHz ISM band configurations.
Near-Field Blind Zone: FMCW radars cannot resolve targets closer than ~0.3–0.5m (beat frequency below ADC
resolution). People in this zone are missed entirely. Mitigation: sensor placement guidelines enforced at
installation; secondary modality (IR, ultrasound) for near-field coverage in safety-critical zones.
Environmental Non-Stationarity: Room furniture changes, new HVAC equipment, seasonal thermal expansion
of sensor mounting — all cause distribution shift unrelated to human behavior. Mitigation: periodic environment
re-mapping; adaptive background subtraction that updates the static scene model on a slow timescale.
C.4 Data Quality & Preprocessing Techniques
 Background subtraction: Accumulate static scene model via exponential moving average of range-Doppler
maps; subtract to isolate dynamic targets. Tunable time constant for different environments.
 DBSCAN clustering: Group nearby radar detections into person-level clusters. Parameter-free cardinality;
standard in radar post-processing stacks.
 Kalman / Extended Kalman Filter tracking: Propagate person state (position, velocity covariance) across
frames. Associate new detections via Hungarian algorithm. Classic but still production baseline for
latency-constrained C++ edge runtimes.
 Dataset versioning: Raw IQ captures or range-Doppler images stored immutably (object storage). Processing
applied via versioned, reproducible pipelines (DVC, MLflow). Downstream model training pins to dataset
version — full reproducibility.
 Annotation tooling: Synchronized camera rig provides weak supervision; homography calibration maps
camera bounding boxes to radar coordinate frame; enables large-scale semi-automated labeling without
manual per-frame annotation.


--- Page 10 ---

SECTION D — Coffee Chat Strategy & Questions to Ask
Thursday 1:30-2:00 PM | Orchard Valley Coffee, Campbell CA | Product Tech Lead
D.1 Tone & Positioning Calibration
 This is a dialogue, not a panel. Keep answers to 90 seconds max; ask follow-ups; make them feel like a peer
conversation.
 They are evaluating: intellectual curiosity, communication clarity, genuine alignment with the product
problem. Not textbook recall.
 You are evaluating: technical depth of team, product direction, culture, practical scope of the role.
 Lead with your C++ / optimization / deployment experience — your strongest differentiator for this specific
role.
 Ground Morphism Systems in engineering outcomes (validation gates, reproducible rollouts, regression testing)
— not abstract theory.
D.2 30-Second Opening Pitch
Verbatim Draft — Practice Out Loud
 'I'm Meshal — I just finished my PhD at Berkeley in EECS, where I spent years building
production-grade computational pipelines and integrating ML surrogate models into high-throughput
scientific workflows. More recently I've been working on LLM agent infrastructure: governed,
deployment-grade systems with formal validation gates and reproducible rollout pipelines. When I
saw Algorized's work on edge-deployed people-sensing, it hit a very specific intersection: the C++
and optimization muscle from my HPC work, the ML deployment rigor from LLM infrastructure, and a
product domain — real-time human awareness on constrained hardware — that I find genuinely
compelling. I wanted to understand your current technical challenges firsthand.'
D.3 High-Signal Questions to Ask
These signal technical depth and product thinking simultaneously:
On the technical stack
 What does your current edge inference stack look like — ONNX Runtime, a custom C inference engine, or
something vendor-specific to your radar hardware?
 How do you handle model updates on deployed devices today — OTA with rollback capability, or manual
firmware flashing? What is the typical fleet size per customer deployment?
 Is the people-sensing model a single end-to-end network, or a pipeline of specialized modules (detection,
tracking, activity classification)? How do you manage the inter-module latency budget?


--- Page 11 ---

On data and the sensing problem
 What is the biggest data quality challenge right now — annotation cost, environment diversity, or rare-event
coverage for things like fall detection?
 How sensor-agnostic is the current architecture in practice — are you training separate models per radar SKU,
or is there a shared backbone with sensor-specific adapters?
On the role and team
 What does success look like for this role in the first six months? Is it primarily a model accuracy problem, a
deployment infrastructure problem, or a data pipeline problem right now?
 How is responsibility divided between the Switzerland R&D; office and the Campbell team — is
hardware/firmware in Switzerland and ML systems here?
D.4 Things to Avoid
 Do not over-explain your PhD physics research — they are a product company, not an academic audience.
Keep condensed matter references brief and connected to engineering outcomes.
 Do not describe Morphism in abstract theoretical terms (sheaf cohomology, contraction mappings) unless they
explicitly ask — ground everything in system reliability outcomes.
 Do not use the word 'passionate' — it is filler. Use specific language about the actual problem that interests you.
 Do not raise visa or salary topics in the coffee chat — keep those on the recruiter (Megan) track.
D.5 Key Numbers from Your Resume — Have These Ready
Fact
Context / How to Use It
2,300+ DFT jobs; 24,000 CPU-hours
Scale of production pipeline; demonstrates systems-at-scale
mindset
70% runtime reduction
Profiling-driven optimization; connect to edge latency
optimization
$160K annual cost savings
Business impact framing; shows engineering decisions tied to
ROI
SFT/RLHF pipelines + reward modeling
LLM training depth; bridge to sensor model refinement
Governed kernels with validation gates
Deployment reliability / MLOps rigour
16+ peer-reviewed publications
Research credibility baseline; use sparingly
PhD EECS Berkeley, Dec 2025
Satisfies PhD qualification; seniority signal
C++, CUDA, MPI, Python, PyTorch, Docker
Direct overlap with JD tech stack requirements


--- Page 12 ---

Algorized Senior Data Scientist — Interview Prep Guide  |  Meshal Alawein, PhD  |  March 2026
 