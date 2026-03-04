# 13 Technical Interview Qa

**Total Pages:** 5



--- Page 1 ---

Algorized Interview Prep — Doc 3: Technical Interview Q&A
Page 1
DOC 3 / 8
Technical Interview Q&A;
Algorized-specific and general senior DS technical questions with full answer frameworks
Topics Covered
 Signal Processing & UWB Radar Questions
 ML System Design Questions
 Python / PyTorch Coding Questions
 Edge Deployment & C/C++ Questions
 Data Pipeline & MLOps Questions
 Hard Problem-Solving Questions
 Resume-Specific Technical Deep Dives
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 3: Technical Interview Q&A
Page 2
1. Signal Processing & UWB Radar
Q: Walk me through the complete pipeline from a raw UWB ADC sample stream to a people-count
integer running on an ARM Cortex-M.
■ 1) ADC captures CIR frames at ~100Hz. 2) Background subtraction (exponential moving average) removes static
clutter. 3) Range-Doppler 2D FFT: FFT across range bins (fast-time) and across frames (slow-time) gives range-velocity
map. 4) CFAR thresholding detects candidate target cells. 5) MHT/Kalman filter associates detections across frames
into tracks. 6) INT8 quantized 1D-CNN classifies occupancy count from track features. 7) Output integer via GPIO/CAN.
Total latency target: <50ms.
Q: What is the difference between UWB impulse radar and FMCW radar? When would you choose
each?
■ UWB: time-domain CIR, very high range resolution (mm), low average power, excellent for IoT/embedded,
privacy-safe. FMCW: frequency-domain chirp, natural range-velocity (Doppler) map from single frame, higher
instantaneous power, better SNR, preferred in automotive (TI AWR series). Algorized uses UWB (ARIA HYDROGEN).
FMCW more common in robotics (lidar alternatives).
Q: How do you remove static clutter from UWB radar data?
■ Three approaches: (1) Exponential Moving Average (EMA) background model — subtract running mean of CIR
frames. Simple, causal, adjustable forgetting factor. (2) SVD/PCA-based background subtraction — decompose frame
matrix, remove low-rank background. Better for structured clutter but higher compute. (3) Adaptive Interference
Cancellation — when reference signal available. For edge MCU: EMA is the practical choice.
Q: What causes false positives in people-sensing radar systems, and how do you mitigate them?
■ Sources: HVAC/fans (micro-Doppler mimics breathing), pendulum-style moving objects, vibrating machinery, other RF
interference. Mitigations: collect 'false positive' dataset explicitly, train discriminator on known interference patterns, add
temporal consistency check (a person must be detected for >N consecutive frames), use multi-modal confirmation (UWB
+ Wi-Fi agreement).
Q: Explain digital beamforming and why ARIA's 5-degree angular resolution matters.
■ Beamforming: combine signals from multiple antenna elements with phase offsets to electronically steer/focus the
beam. 4×4 MIMO array (16 virtual elements) gives angular resolution ≈ λ/D ≈ 5°. This means the system can resolve two
people ~30cm apart at 3m range — critical for distinguishing occupant positions in a car cabin for CPD compliance.
2. ML System Design
Q: Design an end-to-end ML system for child presence detection in a vehicle that must achieve 99.9%
recall.
■ Data: diverse environments (day/night, temperatures, child sizes, positions), balanced false-positive corpus. Model:
quantized CNN on UWB range-Doppler features; optimize threshold for recall>99.9% on hold-out. Deployment: INT8
C++ model on AUTOSAR-compliant MCU, <50ms latency. Monitoring: on-device confidence distribution logging,
shadow model OTA. Validation: UN ECE R129 test protocol, HIL simulation suite. Rollout: staged OTA to 1% fleet first.
Q: How would you build a sensor-agnostic people-sensing foundation model?


--- Page 3 ---

Algorized Interview Prep — Doc 3: Technical Interview Q&A
Page 3
■ Pre-train on diverse sensor data (UWB, FMCW, Wi-Fi CSI) with sensor-type embedding tokens. Use contrastive or
masked autoencoder pre-training on raw signal representations. Fine-tune classifier head per task (presence, count,
vitals) with small labeled datasets. Sensor-agnostic via adapter layers that project sensor-specific inputs into shared
embedding space — analogous to LoRA adapters in LLM fine-tuning.
Q: Your people-counting model degrades 2 months after deployment at a new customer site. Diagnose.
■ Systematic investigation: (1) Check on-device confidence histogram logs — entropy increase signals distribution shift.
(2) Compare input feature statistics to training-time calibration data. (3) Ask customer about physical changes: new
furniture, HVAC added, renovations. (4) Collect a small labeled dataset from the new environment. (5) Fine-tune
classifier head with 50-200 new samples (SFT-equivalent). (6) Validate on held-out new-environment data before OTA
push.
Q: How do you design the data pipeline for a startup that is simultaneously collecting radar data in the
field and training models?
■ Three layers: (1) Edge collection — anonymized feature vectors (not raw CIR for privacy) uploaded on sync.
Metadata: sensor ID, firmware version, environment tag. (2) Cloud ingestion — S3 landing zone, schema validation,
deduplication, quality filter (SNR threshold). (3) Training pipeline — versioned datasets (DVC), automated retraining
trigger on data volume milestones, CI gate (regression suite), staged rollout. Key: design for schema evolution from day
one.
3. Python / PyTorch Coding Scenarios
Expect whiteboard or live-coding questions in later interview rounds. Key patterns to have fluent:
 Custom PyTorch Dataset for CIR frames:
class RadarDataset(Dataset): def __init__(self, cir_frames, labels): ... def __len__(self): return len(self.labels); def
__getitem__(self, idx): return self.cir_frames[idx], self.labels[idx]. Remember: normalize in __getitem__, not __init__.
Lazy loading for large datasets.
 INT8 Post-Training Quantization:
model.eval(); model.qconfig = torch.ao.quantization.get_default_qconfig('x86'); torch.ao.quantization.prepare(model,
inplace=True); [run calibration data]; torch.ao.quantization.convert(model, inplace=True). Then torch.jit.script() for
export.
 Focal Loss for imbalanced detection:
FL(p■) = -α■(1-p■)^γ · log(p■). γ=2 focuses training on hard examples; α■ balances class frequency. Implement as:
loss = F.cross_entropy(logits, targets, reduction='none'); pt = torch.exp(-loss); focal = (1-pt)**gamma * loss.
 Sliding window inference for streaming:
Maintain a circular buffer of N frames. On each new frame: buffer[ptr % N] = frame; ptr += 1. Run inference when
buffer full. Critical: do NOT re-run full preprocessing on every frame — maintain running EMA clutter state.
4. Edge Deployment & C/C++ Questions


--- Page 4 ---

Algorized Interview Prep — Doc 3: Technical Interview Q&A
Page 4
Q: You have a 15MB PyTorch model. Walk me through getting it to run on an ARM Cortex-M7 with
512KB flash.
■ 15MB → 512KB requires ~30× compression. Steps: (1) Architecture reduction — replace large FC layers with Global
Average Pooling. (2) Structured pruning — remove 60-70% of filters (sparsity-aware training). (3) Knowledge distillation
— train compact student from full model. (4) QAT INT8 — 4× size reduction. (5) Export to TFLite Micro → CMSIS-NN
kernels. (6) Profile on target MCU, identify bottleneck layer, apply manual optimization. Result: typically 200-500KB for
CNN-based sensor models.
Q: What is CMSIS-NN and when do you need it?
■ ARM's Cortex Microcontroller Software Interface Standard Neural Network library. Hand-optimized SIMD (DSP
extension) implementations of: depthwise conv, pointwise conv, FC, pooling, activation. Used by TFLite Micro as
backend for Cortex-M. Without CMSIS-NN, inference is 5-10× slower on Cortex-M. Required for meeting <50ms latency
budget on STM32/NXP targets.
Q: How do you validate that a quantized INT8 model is numerically equivalent to the FP32 original?
■ 1) Run the same calibration dataset through both models. 2) Compare per-layer activations: max absolute difference
should be <1% of activation range. 3) Compare final output: classification agreement >99.5% on held-out test set. 4)
Check calibration curve: quantized model confidence should track FP32 confidence. 5) Specifically test edge cases that
are near the decision boundary.
Q: What are the tradeoffs between TFLite Micro, ONNX Runtime Lite, and a hand-written C++ inference
loop?
■ TFLite Micro: best ecosystem, CMSIS-NN support, but inflexible graph. ONNX Runtime (embedded): more portable,
supports dynamic shapes, better for ARM Linux targets (RPi, i.MX8). Custom C++: maximum control, minimum
overhead, but high maintenance burden — justified only for production automotive-grade (AUTOSAR) where third-party
libraries are restricted.
5. Resume-Specific Technical Deep Dives
Prepare a crisp 90-second answer for each of these — they will be asked directly:
Q: You say you achieved 70% runtime reduction on DFT workflows. How specifically did you measure
that, and what were the engineering steps?
■ Be specific: baseline measurement (wall-clock time per job type, logged via SLURM accounting), then optimization
steps: (1) ML surrogate model replaced expensive DFT steps for screening. (2) Job batching reduced SLURM scheduler
overhead. (3) Checkpoint/restart eliminated redundant computation. (4) Memory layout optimization (NumPy array
ordering for SIMD). Validation: regression suite confirmed physics accuracy was not degraded. Measurement: 2,300-job
batch statistics, not single-job cherry-picking.
Q: What does 'governed software kernels' mean at Morphism Systems, and how is it different from
standard software engineering?
■ Standard SE: code correctness. Morphism governance: correctness + reproducibility + audit trail + policy enforcement
+ formal upgrade paths. Concretely: every model deployment produces a signed artifact showing (a) which model
version, (b) evaluation results on standard benchmarks, (c) which data it was trained on, (d) policy constraints it must
satisfy. Upgrades require clearing a validation gate before replacing the incumbent. This is the governance layer that
autonomous AI systems currently lack.


--- Page 5 ---

Algorized Interview Prep — Doc 3: Technical Interview Q&A
Page 5
Q: You built SFT/RLHF pipelines at Turing. What specifically was your contribution vs. the standard
Hugging Face TRL library?
■ Be honest about scope. Likely contributions: domain-specific dataset curation (physics problems), custom reward
function design, evaluation harness for structured output compliance, distributed training stability fixes. The
infrastructure (PPO, DPO, etc.) came from TRL or similar. Your value was in the domain expertise, data quality, and
evaluation design — not reimplementing RL algorithms.
Q: Walk me through a time your production system failed in an unexpected way and how you
diagnosed it.
■ Use the DFT pipeline or LLM system. Key elements: silent failure (no crash, wrong results), detection method
(monitoring caught it vs. user report), root cause analysis (was it data? code? environment?), fix and prevention (added
regression test, monitoring alert). End with what you changed in your engineering practice as a result.
