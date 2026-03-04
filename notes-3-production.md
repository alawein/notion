# Notes 3: Production ML Systems & Edge AI Deployment

Comprehensive reference for building, deploying, and monitoring ML systems in production, with focus on edge-constrained environments and sensor data pipelines. Covers radar signal processing, model compression, C/C++ inference, MLOps infrastructure, and real-world deployment challenges.

---

## Part 1: UWB & FMCW Radar Fundamentals

### UWB Radar Physics & Hardware

**Ultra-Wideband (UWB) Definition**
- RF signal occupying >500 MHz bandwidth or >20% fractional bandwidth
- Typical center frequency: 3.1-10.6 GHz (FCC regulated)
- Key advantage: Direct time-of-flight ranging with cm-level precision

**Range Resolution Formula**
- ΔR = c / (2B) where c = 3×10⁸ m/s, B = bandwidth
- At B = 1.8 GHz: ΔR ≈ 8.3 cm
- At B = 500 MHz: ΔR ≈ 30 cm
- Higher bandwidth → finer range resolution

**Time-of-Flight Ranging**
- Distance = (c · TOF) / 2
- Measures round-trip time of sub-nanosecond pulses
- Immune to multipath that plagues narrowband RSSI ranging
- Through-wall capability: penetrates non-metallic materials; attenuated ~5-15 dB per wall

**MIMO Antenna Arrays**
- Example (ARIA HYDROGEN): 4×4 MIMO (4 Tx, 4 Rx = 16 virtual elements)
- Enables 3D sensing: azimuth + elevation angles
- Angular resolution ≈ λ / (N·d) where N = virtual elements, d = spacing
- Digital beamforming with ~5° resolution at 6.5 GHz center, 16 elements

**Privacy Advantage**
- Captures amplitude vs. time, not visual images
- No facial recognition possible
- Motion/breathing only — no identifiable biometrics
- Key selling point for GDPR-sensitive deployments

### FMCW Radar Signal Pipeline

**FMCW Fundamentals**
- Frequency-modulated continuous-wave chirp: f(t) = f_c + (B/T_c)·t
- Reflects from target, mixes with transmitted signal → beat frequency proportional to range
- Stacks multiple chirps to extract velocity via Doppler FFT
- Dominant choice for automotive and indoor sensing

**Signal Processing Chain — Know This Cold**

1. **ADC Sampling**: Raw IQ (complex baseband) samples from each RX antenna per chirp at ~MHz rates
2. **Range FFT**: DFT applied along fast-time (samples within single chirp)
   - Beat frequency maps to distance: range_bin = c × f_beat / (2 × chirp_slope)
   - Output: range profile per chirp
3. **Doppler FFT**: DFT along slow-time (chirp index within frame)
   - Phase shift between chirps encodes radial velocity
   - Output: 2D Range-Doppler map
4. **2D CFAR Detection**: Adaptive thresholding on Range-Doppler map
   - CA-CFAR: Threshold T = α · (1/N Σ neighbors); assumes uniform noise
   - OS-CFAR: Threshold = α · k-th largest neighbor; more robust to clutter edges
5. **Angle Estimation**: Phase difference across antenna array
   - FFT beamforming (fast, moderate resolution)
   - MUSIC (super-resolution, expensive compute)
   - CAPON/MVDR (beamformer with interference cancellation)
6. **Point Cloud Output**: Each detection = (range, Doppler, azimuth, elevation, RCS, timestamp) tuple

### CIR Signal Pipeline (Step-by-Step)

**Step 1: CIR Acquisition**
- Transceiver transmits short pulse; ADC samples received signal at GHz rate
- Output: CIR frame S[t] — amplitude vs. delay (range)
- Frame rate: 50-200 Hz
- Stored as real-valued time series per virtual antenna element

**Step 2: Static Clutter Removal**
- Background = EMA of frames: B[t] = α·B[t-1] + (1-α)·S[t]
- Dynamic signal D[t] = S[t] - B[t]
- α ≈ 0.95 → ~20-frame memory
- Removes static reflectors (walls, furniture); only moving targets remain

**Step 3: Range Gating**
- Keep only range bins corresponding to room/vehicle interior (e.g., 0.3m to 5m)
- Discard near-field interference (<0.3m) and far-field noise
- Reduces compute by 40-60%

**Step 4: Range-Doppler Transform**
- Stack N consecutive frames (N=32 or 64)
- Apply 2D FFT: fast-time (range bins), slow-time (frames)
- Output: Range-Doppler map P[r,v] showing power at each range-velocity pair

**Step 5: Feature Extraction**
- From Range-Doppler map: micro-Doppler signature, energy centroid, peak ranges, histograms
- For vitals: FFT of amplitude variations at 0.2-0.5 Hz (respiration) and 1-2 Hz (heart rate)

**Step 6: Model Input**
- Normalized feature tensor fed to CNN or LSTM
- For TyCNN: 2D range-Doppler map (e.g., 64×64) or 1D range profile (64 bins)
- Batch normalize at model input layer

### Vitals Detection (Respiration & Heart Rate)

**Respiration Detection (0.2-0.5 Hz)**
- Chest displacement: 5-20mm (well above UWB noise floor)
- Method: Select range bin where target located; FFT amplitude envelope over 10-30s window
- Peak in 0.2-0.5 Hz band = respiration rate
- Well-established baseline difficulty

**Heart Rate Detection (1-2 Hz)**
- Chest displacement: 0.1-0.5mm (near UWB noise floor)
- Requires: high SNR (close range <2m), static subject, high frame rate (>50 Hz), careful filtering
- Must separate from respiration harmonics: bandpass filter 1-2 Hz then peak detect
- Key challenge: motion artifacts dominate when subject moves

**Key Challenges**
- Motion artifacts overpower signal during movement
- Respiration harmonics fall in heart rate band (aliasing)
- Multi-person scenarios require source separation
- Environmental vibrations (HVAC) create false signatures at 0.3-0.5 Hz

**Automotive CPD Application**
- Detect sleeping child (minimal motion) vs. empty seat
- Child breathing at 0.3-0.5 Hz distinguishes from static object
- Core Algorized product use case — practice explaining clearly

### Common Radar Failure Modes

**Multipath Ghost Detections**
- Signal reflects off walls/floors before reaching person → phantom detections at wrong range/angle
- Mitigation: diverse training data across environments, geometric consistency checks, floor-bounce suppression

**Mutual Interference (Multi-Radar Deployments)**
- Two co-located FMCW radars with overlapping chirp schedules create beat-frequency artifacts
- Appear as false targets
- Mitigation: time-division multiplexing, orthogonal waveforms (different start frequencies), adaptive CFAR

**Micro-Doppler Aliasing**
- Fast limb movements (running, rapid gestures) alias across Doppler bins if PRF too long
- Mitigation: select radar parameters (PRF, bandwidth) at system design for expected human velocities

**Near-Field Blind Zone**
- FMCW cannot resolve targets closer than ~0.3-0.5m (beat frequency below ADC resolution)
- People in this zone are missed entirely
- Mitigation: sensor placement guidelines at installation; secondary modality (IR, ultrasound) for near-field

**Environmental Non-Stationarity**
- Room furniture changes, new HVAC, seasonal thermal expansion
- Causes distribution shift unrelated to human behavior
- Mitigation: periodic environment re-mapping, adaptive background subtraction

---

## Part 2: Edge Hardware Landscape

### Hardware Classes & Constraints

| Class | Examples | RAM | Model Budget | ML Framework |
|-------|----------|-----|--------------|--------------|
| Bare MCU | STM32F7, NXP i.MX RT | 512KB-2MB | 100-500KB | TFLite Micro + CMSIS-NN |
| Application MCU | RPi CM4, NXP i.MX8 | 1-4GB | 1-100MB | ONNX Runtime, TFLite |
| DSP | TI C6748 | 128KB-1MB | 200KB-2MB | Custom C / TI DL |
| Embedded NPU | NXP eIQ, STM32 NUCLEO-NPU | varies | 1-10MB INT8 | Vendor SDK |
| Automotive SoC | NXP S32G, Renesas R-Car | 2-16GB | unlimited | TensorRT, ONNX |

### Memory & Latency Budgets

**Bare MCU Targets**
- Inference latency: <100ms typical; <50ms for safety-critical
- Peak SRAM: <512KB
- Flash footprint: <256KB model binary
- Model size: INT8 quantized <200KB

**Embedded Linux (Cortex-A)**
- Inference latency: <50ms (real-time requirement)
- RAM: Shared with OS; effective model budget 1-100MB
- Can support larger models; benefits from NEON SIMD

---

## Part 3: Model Compression & Quantization

### Post-Training Quantization (PTQ)

**INT8 PTQ Process**
1. Train FP32 model normally
2. Collect calibration dataset (100-1000 representative inputs)
3. Run calibration: measure activation ranges per layer
4. Compute scale factors: S = (max - min) / 255
5. Quantize weights and activations
6. Evaluate accuracy — if drop >2%, switch to QAT

**Calibration Data Selection**
- Representative but NOT full training set
- Cover edge cases
- For radar: diverse environments, occupancy counts, interference scenarios
- Typical size: 500-1000 frames

**Quantization Strategies**

| Strategy | Best For | Trade-offs |
|----------|----------|-----------|
| Symmetric quant | Weights; zero_point=0 | Simpler, but wastes range with non-zero-mean distributions |
| Asymmetric quant | Activations; zero_point≠0 | Better for ReLU outputs, more parameters |
| Per-layer quant | Simple deployment | 1-3% accuracy loss vs. per-channel |
| Per-channel quant | Depthwise convolutions | Higher accuracy, required for depthwise layers |

### Quantization-Aware Training (QAT)

**When to Use QAT**
- PTQ accuracy loss >2%
- Small models with <1M parameters (less redundancy)
- Regression tasks (vitals estimation)
- Models with folded batch normalization
- Any model where FP32→INT8 recall drop exceeds safety threshold

**QAT Process**
- Simulate quantization during training using fake quantization (FQ) nodes
- Gradients flow via Straight-Through Estimator (STE): ∂FQ/∂x = 1 within clamp, 0 outside
- Model learns to be robust to quantization error
- Recovers 80-90% of PTQ loss typically

### Pruning & Knowledge Distillation

| Technique | Mechanism | Size Reduction | Best For |
|-----------|-----------|-----------------|----------|
| Unstructured Pruning | Zero individual weights by magnitude | Up to 90% sparse | Sparse hardware (rare on MCUs) |
| Structured Pruning | Remove entire filters/channels (L2 norm) | 30-70% fewer params | Dense hardware; maps to C arrays |
| Weight Sharing | Cluster weights into K centroids | Depends on K | Combined with quantization |
| Knowledge Distillation | Student learns from teacher soft labels | 10-100× size reduction | Architecture changes required |

**Practical Recipe for Algorized**
- Structured pruning + INT8 QAT is standard 2-step approach
- Start with 50% structured pruning (filter magnitude)
- Then apply QAT
- Benchmark on target MCU
- If latency/accuracy acceptable, done
- If not, add knowledge distillation from larger teacher

---

## Part 4: C/C++ Inference Runtimes

### Runtime Options Comparison

**TensorFlow Lite Micro (TFLM)**
- Single .cc/.h file compilation
- No OS dependency
- CMSIS-NN backend for ARM
- Supports: Conv2D, DepthwiseConv, FC, LSTM, RNN
- Memory allocator from pre-allocated tensor arena
- Use: STM32, Arduino Nano 33, nRF52840
- Limitation: static graph only, no dynamic shapes

**CMSIS-NN**
- ARM-optimized kernels for Cortex-M
- SIMD via DSP extension (M4/M7/M33/M55)
- Key functions: arm_convolve_HWC_q7_fast(), arm_fully_connected_q7()
- Integral to TFLM on ARM
- Without CMSIS-NN: ~5-10× slower on Cortex-M

**ONNX Runtime (ORT)**
- Supports ARM Linux (Cortex-A)
- Dynamic shapes supported
- Execution providers: CPU (default), XNNPACK (mobile-optimized)
- Better for NXP i.MX8, RPi 4, Renesas R-Car than bare MCU

**ONNX Runtime Mobile**
- Minimal build for embedded Linux: ~1-3MB binary
- Supports INT8 quantized models
- Reasonable for i.MX8-class hardware with Linux OS

**Custom C++ Flat Inference**
- Hand-write inference as struct arrays + function calls
- Zero framework overhead
- Maximum control over memory layout
- Used in AUTOSAR/SafetyOS contexts (third-party libs prohibited)
- Only justified for tiny models (<50KB) or certification requirements

### Complete C/C++ Deployment Workflow

1. **Train & Validate (PyTorch)**: FP32 model, full evaluation suite, establish baseline
2. **Quantize (PTQ or QAT)**: INT8 calibration, validate FP32 vs INT8 max diff < tolerance
3. **Export to ONNX**: torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=12)
4. **Validate ONNX**: Confirm ONNX output matches PyTorch
5. **Convert to TFLite**: tf.lite.TFLiteConverter.from_saved_model() or via onnx2tf
   - Apply INT8 representative dataset
   - Export .tflite file
6. **Generate C Source**: xxd -i model.tflite > model_data.cc or use flatc
7. **Integrate TFLM + CMSIS-NN**: Add sources, allocate tensor arena, build interpreter
8. **Benchmark on Target**: Measure inference time, peak RAM, flash footprint vs. requirements
9. **Regression Validation**: Run held-out dataset through C++ path, confirm accuracy within tolerance

### Known-Answer Test (KAT) Methodology

- Compute output of 10 fixed test inputs in Python (to 8 decimal places)
- Store expected outputs as constants
- In C++ inference code, run same inputs, compare within tolerance (max abs diff < 1e-3 for INT8)
- Run as automated test in firmware CI
- Any deviation indicates: quantization error, endianness bug, or preprocessing mismatch

---

## Part 5: ML Pipeline Architecture

### End-to-End Pipeline Components

**Data Ingestion**
- Raw sensor data → S3/GCS landing zone
- Schema validation on arrival (Pydantic/Great Expectations)
- Dead-letter queue for malformed records
- Partitioned by date + sensor_id for efficient querying

**Feature Engineering**
- Deterministic, versioned transformations
- Feature Store (Feast, Tecton, or custom) decouples computation from training
- Offline (batch) and online (real-time) serving must be consistent
- Training/serving skew is the #1 MLOps failure mode

**Training Pipeline**
- Triggered by: new data threshold, scheduled cadence, manual trigger
- Hyperparameter search (Optuna)
- Experiment tracking (MLflow, Weights & Biases)
- Always checkpoint — resumable training for long runs

**Evaluation Gate**
- Automated: must beat incumbent on held-out set by threshold (e.g., +0.5% recall)
- Regression checks: no degradation on known failure cases
- Shadow run: new model runs in parallel for N hours before cutover

**Deployment**
- Model registry tag 'production'
- Blue-green or canary deployment
- OTA for edge devices with signed packages
- Rollback trigger: if production metrics degrade >X% within 24h, auto-revert

**Monitoring**
- Input distribution statistics
- Prediction distribution
- Business metrics (false alarm rate, detection rate)
- Alerting on threshold breaches
- On-device: lightweight confidence histogram logging

### Training/Serving Skew — The #1 Production Failure

**Definition**: Model sees different data at training vs. inference time → silent degradation

**Sources**
- Feature computation differences: training uses batch stats, serving uses streaming approx
- Data preprocessing bugs: different code paths for train/serve
- Label leakage: feature derived from target inadvertently included
- Temporal leakage: using future information in features
- Schema drift: production data evolves, training data stays fixed

**For Algorized Specifically**
- Background clutter removal (EMA) computed differently offline vs. real-time embedded
- Fix: use same stateful EMA implementation in both pipelines

---

## Part 6: Monitoring & Drift Detection

### Types of Drift

| Type | Definition | Detection Method | Response |
|------|-----------|------------------|----------|
| Data Drift | P(X) changes; P(Y\|X) same | PSI, KL divergence on features | Retrain on new data |
| Concept Drift | P(Y\|X) changes; task changes | Monitor accuracy vs ground truth | Full retraining or domain adaptation |
| Label Drift | P(Y) changes; class balance shifts | Monitor prediction distribution | Recalibrate decision threshold |
| Feature Drift | Individual feature distribution shifts | Per-feature KS test | Investigate sensor/environment |
| Model Degradation | Accuracy drops without clear drift | Track business + labeled metrics | Debug pipeline, retrain |

### Distribution Shift Detection Tests

**Kolmogorov-Smirnov (KS) Test**
- Nonparametric; tests if two 1D distributions from same population
- Apply per feature in telemetry
- Low compute cost; suitable for fleet-side monitoring

**Maximum Mean Discrepancy (MMD)**
- Kernel-based; handles multivariate distributions
- More powerful for high-dimensional feature spaces
- Computationally heavier — run server-side on aggregated sketches

**Population Stability Index (PSI)**
- Binned divergence measure
- Industry standard for tabular feature monitoring
- Easy to implement in C telemetry for on-device pre-aggregation
- Formula: PSI = Σ (% baseline - % current) × ln(% baseline / % current)

**CUSUM / EWMA**
- Sequential change-point detection
- Constant memory footprint
- Ideal for on-device real-time drift detection in C++ loop

### On-Device Monitoring (Edge Side)

**Lightweight Telemetry**
- Log prediction confidence histogram (256-bin histogram = 256 bytes)
- Track input feature statistics (mean/variance)
- Log class-count distribution
- Flush to server on heartbeat or network availability

**On-Device Anomaly Detection**
- Simple statistical tests (z-score on aggregated features, entropy threshold)
- Compiled to C; zero Python dependency
- Triggers server-side review flag when tripped

**Shadow Mode Deployment**
- Run old and new model in parallel
- Compare outputs before fleet-wide switch
- Requires no ground truth — just output agreement rate as proxy

**Confidence Histograms**
- 256-bin histogram of softmax max confidence
- Cheap (256 bytes)
- Upload on cloud sync
- Significant shift from training-time histogram signals drift

**Input Feature Statistics**
- Track running mean/variance of range-Doppler energy per range bin
- Deviation from calibration values signals environment change
- Examples: new furniture, obstructions, sensor mounting shift

**Prediction Distribution**
- Rolling count of predicted class frequencies
- If 'empty room' prediction spikes unexpectedly, something changed

**Shadow Model Disagreement**
- OTA deploy new model in parallel
- Log cases where incumbent and shadow disagree
- High disagreement rate triggers human review before rollout

---

## Part 7: CI/CD for ML

### Code CI (Continuous Integration)

- **Standard**: linting (ruff/flake8), type checking (mypy), unit tests (pytest)
- **ML-specific**: test data transformations deterministically, edge cases for feature computation
- **Example**: Test that range normalization is idempotent across batch sizes

### Model CI (Evaluation Gate)

- On every model PR: run full evaluation suite on held-out data
- Must beat incumbent by threshold (e.g., +0.5% recall)
- Regression checks: no degradation on top-10 failure cases
- Block merge if tests fail

### Data Validation CI

- On new data batch: schema validation, distribution shift test, label quality check
- Alert if significant drift detected
- Do NOT silently train on corrupted data
- Example: reject any CIR frame with ADC saturation artifacts

### Deployment CD (Continuous Deployment)

- Model artifacts: registry → staging → canary (1%) → ramp to 100% over 48h
- Automatic rollback: if error rate >threshold within 24h, revert
- OTA for edge: cryptographically signed model packages
- Example: staged rollout gates: 1% for 24h, 10% for 48h, 100%

### Infrastructure as Code

- Model training jobs as reproducible containers (Docker)
- All dependencies pinned
- Reproducibility: given commit hash + dataset version → identical model
- Tools: DVC (data versioning), MLflow (experiment tracking), GitHub Actions (orchestration)

---

## Part 8: A/B Testing & Staged Rollouts

### Device-Level A/B Testing

- For edge-deployed people-sensing: cannot split single vehicle into two groups
- Solution: Assign devices (not users) to treatment/control
- 50% of fleet runs model A, 50% runs model B
- Stratify by: device type, deployment environment, customer

### Shadow Mode Deployment

- New model runs on device but outputs NOT acted upon
- Zero user impact
- Collect performance data over 1-4 weeks
- Best for safety-critical systems (CPD) — never push unvalidated model

### Statistical Power Calculation

- How many devices needed to detect improvement?
- Formula: n ≈ (z_α + z_β)² × 2σ² / δ²
- Example: For 0.5% recall improvement at 80% power, 5% significance: need thousands of device-hours

### Business Metrics vs Model Metrics

- Recall on test set ≠ production recall
- Must define business metric (false alarms/day, detections/week)
- Measure in production — this is the truth
- Model metrics are proxies; business metrics are reality

---

## Part 9: OTA Update Architecture

### Model Package Format

- Signed bundle: model binary + version metadata + evaluation certificate
- Evaluation certificate = hash of eval results
- Cryptographic signature prevents tampered models
- Size: <500KB for MCU targets

### Download & Staging

- Download in background over BLE/Wi-Fi
- Write to staging flash partition
- Do NOT overwrite active partition until validation passes

### On-Device Validation

- After staging: run self-test (known-answer test on fixed input)
- Compare output to expected (stored hash)
- If mismatch: abort, delete staging, report error

### Cutover & Rollback

- On successful validation: atomic swap (update boot pointer)
- First N operating hours in 'provisional' state: monitor performance
- If anomaly detected → auto-rollback to previous version in backup partition

### Fleet Management

- Server tracks: device_id → firmware_version → model_version → last_sync
- Enables targeted rollback for affected devices without updating entire fleet
- Example: if new model causes failures in 50 devices, rollback just those 50

---

## Part 10: Sensor Fusion for People-Sensing

### Fusion Architecture Comparison

| Fusion Type | How It Works | Pros | Cons |
|-------------|-----------|------|------|
| Early (Raw) | Concatenate raw streams before model | Single model, low latency | Requires strict sync; high bandwidth |
| Mid (Feature) | Extract per-modality; fuse at intermediate | Single encoder per modality; handles async | Requires cross-modal registration |
| Late (Decision) | Independent models; combine confidence | Robust to sensor dropout | Higher latency, more memory |
| Kalman | UWB position + Wi-Fi occupancy as measurements | Principled uncertainty | Assumes linear dynamics |
| Cross-Attention | Transformer cross-attention between modalities | Best accuracy | Heavy; not edge-deployable |

### Sensor-Agnostic Pipeline (Algorized Requirement)

**Modality Dropout During Training**
- Randomly mask sensor inputs at training time
- Forces model robust to missing modalities at inference
- Graceful degradation when a sensor fails

**Shared Latent Space**
- Each sensor type has learned projection into common embedding
- People-sensing head operates on fused embedding
- Encoder per modality; shared classifier

**Foundation Model + Adapters**
- Pretrain shared backbone on large multi-sensor corpus
- Fine-tune lightweight adapters per deployment configuration
- Example: MLP adapter layer per sensor modality

---

## Part 11: Production Systems Interview Q&A

**Q: What is the difference between data drift and concept drift? Give concrete example for people-sensing.**

A: Data drift: input distribution changes but relationship to label unchanged. Example: new office furniture changes background radar signature. Concept drift: relationship between input and label changes. Example: company policy changes 'presence' definition to include people sitting very still (new label semantics). Different responses: data drift → retrain with new data. Concept drift → relabel + retrain.

**Q: Describe your approach to building monitoring system for 10,000 edge devices.**

A: Three-layer: (1) On-device: confidence histograms and input statistics logged locally, uploaded on next sync. (2) Fleet aggregation: server-side dashboard showing per-firmware-version distribution shifts, anomaly detection on aggregated metrics. (3) Sentinel devices: 50 instrumented devices in known environments uploading full telemetry for ground-truth validation. Alert thresholds: >5% confidence distribution shift in >10% of fleet triggers review.

**Q: How do you handle model update that must not degrade safety-critical performance (CPD recall >99.9%)?**

A: Never push directly. Process: (1) Offline evaluation on hardened test set including all known failure modes. (2) Hardware-in-the-loop (HIL) testing with simulated sensor data. (3) Shadow deployment on 100 devices in non-CPD mode. (4) 30-day monitoring period. (5) Staged rollout: 1% → 10% → 100% with 7-day hold at each stage. (6) Automatic rollback if production recall drops below threshold.

**Q: Your INT8 quantized model has 3% lower recall than FP32 on test set. What do you do?**

A: 3% is too much for safety-critical. Options in order: (1) Switch PTQ→QAT: train with simulated quantization. (2) Per-channel quantization for depthwise convolutions. (3) Mixed precision: keep sensitive layers (first/last, skip connections) in INT16 or FP16. (4) Architecture change: replace quantization-sensitive operations. (5) Knowledge distillation: retrain compact student from scratch with soft labels.

**Q: What is the tensor arena in TFLite Micro and how do you size it?**

A: TFLM uses pre-allocated byte array for all tensor data (no dynamic allocation). Size by: running model with oversized arena (e.g., 100KB) and calling interpreter.arena_used_bytes(). Add 10-20% margin. If arena too small, interpreter returns kTfLiteError at invoke time — easy to diagnose.

**Q: How do you verify C++ edge model produces same results as Python training model?**

A: Known-answer test (KAT): compute output of 10 fixed test inputs in Python (to 8 decimals). Store expected outputs as constants. In C++ inference code, run same inputs, compare within tolerance (max abs diff < 1e-3 for INT8, 1e-5 for FP32). Run as automated test in firmware CI. Any deviation indicates quantization error, endianness bug, or preprocessing mismatch.

**Q: Walk me through PyTorch people-counting model → ARM Cortex-M7 C++ deployment.**

A: Full chain: (1) PTQ with calibration data from diverse environments. (2) ONNX export: torch.onnx.export(). (3) Quantized conversion: tf.lite.TFLiteConverter with INT8 representative dataset. (4) C source generation: xxd -i model.tflite > model_data.cc. (5) Integrate TFLM + CMSIS-NN: add sources, allocate tensor arena. (6) Benchmark on target: measure latency (<50ms target), peak RAM (<512KB). (7) KAT validation: compare C++ output to PyTorch on fixed inputs.

**Q: How would you handle domain shift when customer deploys in new environment with different room geometry?**

A: Options: (1) Collect labeled data and fine-tune classifier head (SFT-style). (2) Unsupervised domain adaptation using background statistics. (3) Physics-based data augmentation (simulate different room impulse responses). Mention few-shot fine-tuning angle — adapt in 2-3 hours with small labeled dataset.

**Q: What are key differences between FMCW radar and UWB impulse radar for people-sensing?**

A: UWB: time-domain CIR, very high range resolution, low power, better for precise ranging. FMCW: frequency-domain (chirp), produces natural range-velocity map, better SNR for velocity, higher power. UWB preferred for low-power IoT; FMCW for automotive (TI AWR series). FMCW more mature for automotive radar integration.

**Q: How do you validate people-sensing model for safety-critical application (child detection)?**

A: Mention: edge case coverage (small children, unusual positions, sleeping), class imbalance (empty seat is 95% of data), recall vs. precision trade-off (false negative catastrophic), hardware-in-the-loop testing, regulatory testing (UN ECE R129 / FMVSS 213). Staged rollout with extensive monitoring.

---

## Key Implementation Patterns

### Feature Store Pattern

```
Training Path:
  Raw Data → Feature Computation (Offline) → Feature Store → Model Training

Inference Path:
  Live Data → Feature Computation (Online) → Model Inference
  
Critical: Both paths must use identical feature computation code.
```

### Pipeline Architecture Pattern

```
Sensor Input
  ↓
Signal Processing (CFAR, FFT)
  ↓
Feature Extraction (Range-Doppler, Micro-Doppler)
  ↓
ML Model Inference (Quantized CNN/LSTM)
  ↓
Post-processing (Kalman Tracking, Clustering)
  ↓
Output (Count, Activity, Vitals)
```

### Evaluation Gate Pattern

```
Model PR
  ↓
PTQ/QAT Quantization
  ↓
Run Regression Suite (10+ test scenarios)
  ↓
Compare vs. Incumbent (must beat by >0.5%)
  ↓
Run on Known Failure Cases
  ↓
HIL Testing
  ↓
Approved for Shadow Deployment
```

### OTA Rollout Pattern

```
Deploy to 1% Fleet (24h)
  → Monitor Telemetry
  → Decision: Proceed or Rollback
    ↓
Deploy to 10% Fleet (48h)
  → Monitor Telemetry
  → Decision: Proceed or Rollback
    ↓
Deploy to 100% Fleet
  → Continue Monitoring
  → Auto-Rollback Trigger: metrics degrade >threshold
```

---

**Document Summary**: Comprehensive production ML reference covering sensor signal processing, model compression for edge deployment, CI/CD pipelines, monitoring infrastructure, and real-world deployment strategies. Includes specific patterns for safety-critical systems (child detection, vitals), firmware-based inference optimization, and fleet management at scale.
