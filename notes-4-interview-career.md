# Notes 4: Interview, Career & Resume Strategy

Comprehensive interview preparation, behavioral frameworks, resume navigation, and career positioning for Senior Data Scientist roles. Focus: Algorized, with universal principles applicable across AI/ML companies.

**Contents:** Technical interview Q&A | STAR story bank | Resume-to-JD bridging | Career narrative | Cultural fit | Negotiation reference

---

## Part 1: Technical Interview Q&A Foundation

### Signal Processing & UWB Radar (Algorized-Critical)

**Q: Walk me through the complete pipeline from raw UWB ADC samples to a people-count integer on ARM Cortex-M.**

A: (1) **ADC capture** — transceiver transmits pulse; ADC samples received Channel Impulse Response at GHz rate (~100 Hz frame rate). (2) **Clutter removal** — exponential moving average background subtraction (α ≈ 0.95) removes static reflectors. D[t] = S[t] − B[t]. (3) **Range gating** — keep only 0.3–5m interior range, discard near-field and far-field noise (40–60% compute savings). (4) **Range-Doppler 2D FFT** — stack N=32–64 frames, apply FFT across range bins (fast-time), then across frames (slow-time). Outputs power at each range-velocity pair. (5) **CFAR detection** — Constant False Alarm Rate thresholding adapts to local noise; Cell-Averaging CFAR for most cases; Order-Statistic CFAR near clutter edges. (6) **Tracking** — Kalman filter for up to 5 targets (low compute) or Multiple Hypothesis Tracking for multi-person accuracy with occlusion handling. Maintains persistent track IDs. (7) **CNN classification** — INT8-quantized Tiny Convolutional Network on Range-Doppler features classifies occupancy count. Must complete within 50ms end-to-end latency.

**Q: UWB vs FMCW radar — when do you choose each?**

A: 
- **UWB**: time-domain Channel Impulse Response, extremely high range resolution (ΔR = c/(2B) ≈ 8.3cm at 1.8GHz bandwidth), low average power, excellent for IoT/embedded, privacy-safe (amplitude vs. delay, no images), penetrates drywall/wood (5–15dB attenuation per wall). Algorized uses UWB (ARIA HYDROGEN).
- **FMCW**: frequency-domain chirp, natural Range-Doppler map from single frame, higher signal-to-noise ratio, higher instantaneous power, preferred in automotive (TI AWR series) and robotics. Better when you need velocity naturally in the frequency domain.

**Q: How do you remove static clutter?**

A: Three approaches: (1) **Exponential Moving Average** — B[t] = α·B[t−1] + (1−α)·S[t], α ≈ 0.95 (20-frame memory). Simple, causal, adjustable forgetting. (2) **SVD/PCA background subtraction** — decompose frame matrix, remove low-rank background. Better for structured clutter but higher compute. (3) **Adaptive interference cancellation** — when reference signal available. For edge MCU: **EMA is the practical choice**.

**Q: What causes false positives in people-sensing radar?**

A: Sources: HVAC/fans (0.2–0.5Hz micro-Doppler mimics breathing), pendulum-style moving objects, vibrating machinery, other RF interference. Mitigations: (1) Collect explicit false-positive dataset from known interference sources. (2) Train discriminator using spatial coherence: genuine human motion localizes to one range bin; HVAC creates correlated noise across all bins. (3) Temporal consistency gate: person must be detected for >N consecutive frames before counting.

**Q: Explain digital beamforming and ARIA's 5° angular resolution.**

A: Beamforming: combine signals from multiple antenna elements with phase offsets to electronically steer/focus the beam. ARIA 4×4 MIMO array → 16 virtual elements → angular resolution ≈ λ/D ≈ 5°. This resolves two people ~30cm apart at 3m range — critical for car cabin CPD compliance (distinguish child from empty seat). Enables true 3D detection in confined spaces.

**Vitals Detection (Breathing & Heart Rate):**

- **Respiration (0.2–0.5 Hz)**: Chest displacement 5–20mm (well above noise floor). Extract: select range bin where target is; compute FFT of amplitude envelope over 10–30s window; peak in 0.2–0.5Hz band = respiration rate.
- **Heart rate (1–2 Hz)**: Chest displacement only 0.1–0.5mm (near noise floor). Requires: high SNR (subject <2m), static subject, frame rate >50Hz, careful bandpass filtering to separate from respiration harmonics.
- **CPD application**: Sleeping child breathes 0.3–0.5Hz → distinguishes from empty seat with no micro-motion. **This is the core Algorized product.**
- **Challenges**: HVAC vibration overlaps breathing band; respiration harmonics in heart rate band; multi-person source separation.

---

### ML System Design (Production-Grade)

**Q: Design an end-to-end ML system for Child Presence Detection achieving 99.9% recall.**

A: 
- **Data**: Diverse environments (day/night, temperature extremes, child sizes, positions), balanced false-positive corpus.
- **Model**: Quantized CNN on UWB Range-Doppler features; optimize decision threshold for recall >99.9% on held-out safety test set. Use F-beta loss (β≥3) to weight recall.
- **Deployment**: INT8 C++ model on AUTOSAR-compliant MCU, <50ms latency.
- **Monitoring**: On-device confidence distribution logging, shadow model on 1% fleet before full rollout.
- **Validation**: UN ECE R129 regulatory test protocol, hardware-in-the-loop simulation suite.
- **Rollout**: Staged OTA to 1% → 10% → 100% over 48h with automatic rollback if recall drops below threshold within 7 days.

**Q: How would you build a sensor-agnostic foundation people-sensing model?**

A: (1) **Pre-train** on diverse sensor data (UWB, FMCW, Wi-Fi CSI) using sensor-type embedding tokens (like language tokens in multilingual models). (2) **Self-supervised learning** — masked signal autoencoder on raw signal representations. (3) **Adapter layers** — per-sensor adapter that projects sensor-specific inputs into shared embedding space (analogous to LoRA in LLMs). (4) **Fine-tuning** — task-specific classifier heads (presence, count, vitals) trained per task with small labeled datasets. (5) **Cross-sensor transfer** — foundation handles distribution shift; adapters customize per modality.

**Q: Your people-counting model degrades 2 months after deployment at a new site. Diagnose.**

A: Systematic investigation: (1) **Check on-device logs** — confidence histogram entropy increase signals distribution shift. (2) **Compare input statistics** — Range-Doppler energy per bin vs. training calibration. (3) **Ask customer** — new furniture? HVAC added? Renovations? (4) **Collect 200–500 labeled frames** from new environment. (5) **Fine-tune classifier head only**, freeze feature extractor (SFT paradigm). (6) **Validate** on held-out new-environment data before OTA push.

**Q: Design the data pipeline for a startup collecting radar data in the field and training models simultaneously.**

A: 
- **Layer 1 — Edge collection**: Anonymized feature vectors (not raw CIR for privacy), uploaded on sync. Metadata: sensor ID, firmware version, environment tag.
- **Layer 2 — Cloud ingestion**: S3 landing zone, schema validation (Pydantic), deduplication, quality filter (SNR threshold).
- **Layer 3 — Training pipeline**: Versioned datasets (DVC), automated retraining trigger on data volume milestones, CI gate (regression suite), staged rollout.
- **Critical**: Design for schema evolution from day one — hardware will change.

---

### Python/PyTorch Coding Patterns

Whiteboard/live-coding questions — have these patterns fluent:

**Custom PyTorch Dataset for CIR Frames:**
```python
class RadarDataset(Dataset):
    def __init__(self, cir_frames, labels):
        self.cir_frames = cir_frames
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        frame = torch.tensor(self.cir_frames[idx], dtype=torch.float32)
        frame = (frame - frame.mean()) / (frame.std() + 1e-6)  # Normalize in __getitem__
        return frame, self.labels[idx]
```

**INT8 Post-Training Quantization:**
```python
model.eval()
model.qconfig = torch.ao.quantization.get_default_qconfig('x86')
torch.ao.quantization.prepare(model, inplace=True)
# Run calibration data
torch.ao.quantization.convert(model, inplace=True)
quantized = torch.jit.script(model)  # Export
```

**Focal Loss for Imbalanced Detection:**
```python
# FL(p_t) = -α(1-p_t)^γ log(p_t); γ=2 focuses on hard examples
loss = F.cross_entropy(logits, targets, reduction='none')
pt = torch.exp(-loss)
focal_loss = ((1 - pt) ** 2) * loss
```

**Sliding Window Inference for Streaming:**
```python
buffer = np.zeros((N, CIR_BINS))
ptr = 0
ema_clutter = None  # Maintain stateful EMA across frames

while new_frame_arrives():
    frame = read_adc()
    buffer[ptr % N] = frame
    ptr += 1
    
    if ptr >= N:
        # Run inference on full buffer
        # DO NOT re-run EMA — update state incrementally
        ema_clutter = 0.95 * ema_clutter + 0.05 * frame
        dynamic = frame - ema_clutter
        predictions = model(dynamic)
```

---

### Edge Deployment & C/C++ Specifics

**Q: You have a 15MB PyTorch model. Get it to run on ARM Cortex-M7 with 512KB flash.**

A: 15MB → 512KB requires ~30× compression. Steps:
1. **Architecture reduction** — replace large FC layers with Global Average Pooling.
2. **Structured pruning** — remove 60–70% of filters (sparsity-aware training).
3. **Knowledge distillation** — train compact student from full model.
4. **QAT INT8** — 4× size reduction, simulating quantization during training.
5. **Export** — PyTorch → ONNX (opset 12) → TFLite → C byte array.
6. **Profile on target MCU**, identify bottleneck layer, apply manual optimization.
   
**Result**: Typically 200–500KB for CNN-based sensor models. Inference <50ms on STM32.

**Q: What is CMSIS-NN and when do you need it?**

A: ARM Cortex Microcontroller Software Interface Standard Neural Network library. Hand-optimized **SIMD** (DSP extension) implementations of: depthwise conv, pointwise conv, FC, pooling, activation. Used by TFLite Micro as backend for Cortex-M. **Without CMSIS-NN**: inference 5–10× slower. **Required** for meeting <50ms latency budget on STM32/NXP targets.

**Q: How do you validate a quantized INT8 model equals FP32 numerically?**

A: (1) Run same calibration dataset through both. (2) **Per-layer activations**: max absolute difference <1% of activation range. (3) **Final output**: classification agreement >99.5% on held-out test set. (4) **Confidence calibration**: quantized confidence tracks FP32 confidence. (5) **Edge cases**: test near decision boundary especially.

**Q: TFLite Micro vs ONNX Runtime Lite vs custom C++?**

A: 
- **TFLite Micro**: Best ecosystem, CMSIS-NN support, inflexible graph. Bare MCU (STM32).
- **ONNX Runtime**: More portable, supports dynamic shapes, ARM Linux (RPi, i.MX8). ~1–3MB minimal binary.
- **Custom C++**: Maximum control, minimum overhead. Justified only for models <50KB or AUTOSAR certification contexts where third-party libraries prohibited.

---

## Part 2: Behavioral Interview & STAR Stories

### STAR Story Bank (8 Prepared — Know Cold)

**1. Technical Impact Under Constraints: $160K Savings / 70% Runtime Reduction**

*Situation*: 2,300-job DFT backlog consuming $230K/year in compute at LBNL.

*Task*: Reduce cost without degrading physics accuracy.

*Action*: Integrated ML surrogate model to pre-screen candidates, optimized job batching, added checkpoint/restart to eliminate redundant computation after node failures, memory layout optimization for SIMD vectorization.

*Result*: 70% runtime reduction, $160K annual savings. Regression suite validated physics accuracy preserved across all 2,300 production jobs. **Lesson**: Constraint (budget) forced more elegant ML solution than brute-force compute.

---

**2. Production System Failure & Recovery**

*Situation*: DFT workflow failed silently — results computed but incorrect due to basis set misconfiguration in an upstream package.

*Task*: Diagnose before downstream researchers used bad data.

*Action*: Systematic comparison of output checksums across job versions, traced misconfiguration to package update. Added automated regression tests running in <10 minutes.

*Result*: Fixed and changed practice permanently: immutable configuration management + automated correctness validation on every job output.

---

**3. Learning a New Domain Rapidly**

*Situation*: Transition from spintronics (KAUST) to 2D materials/TMDs (Berkeley). Completely different physics, simulation codes, community.

*Task*: Publish first paper within 18 months despite knowledge gap.

*Action*: Identified 5 key papers, reproduced each result, found domain experts for weekly meetings, built verification scripts before scaling.

*Result*: First paper submitted at 14 months. **Lesson**: Systematic skill-building beats trying to absorb everything simultaneously.

---

**4. Disagreement with Technical Decision**

*Situation*: At Turing, team chose generic preference model for physics-reasoning LLM; I disagreed with architecture choice not calibrated to problem structure.

*Task*: Push back without stalling delivery.

*Action*: Ran 3-day ablation with both approaches on 100 held-out problems, showed 12% performance gap quantitatively.

*Result*: Team adopted my approach. **Lesson**: Bring data, not opinion, to technical disagreements.

---

**5. Ambiguity & Self-Direction**

*Situation*: At Morphism Systems, no external specifications. Problem definition, architecture, validation criteria all undefined.

*Task*: Build something rigorously grounded, not "just another config file system."

*Action*: Grounded design in category theory (contraction mapping), built sheaf-theoretic drift detection, wrote formal proof artifacts.

*Result*: Full pipeline with passing tests and mathematical foundations. **Shows**: Thrive with autonomy, bring structure to ambiguous problems.

---

**6. Cross-Functional Communication**

*Situation*: PhD defense — explaining nanoscale quantum mechanical phenomena to committee with EE, materials science, physics members.

*Task*: Make thesis accessible without dumbing down technical content.

*Action*: Structured presentation around engineering implications first, then physical mechanism, then mathematical derivation.

*Result*: Passed with minor revisions; one EE committee member said "clearest TMD defense I've reviewed." **Lesson**: Lead with impact, not derivation.

---

**7. Speed vs Correctness Trade-off**

*Situation*: Turing had tight delivery deadline. Discovered 15% of dataset had labeling errors.

*Task*: Fix properly vs. ship to deadline.

*Action*: Raised issue, quantified expected accuracy impact (3–5% degradation estimate), proposed 3-day dataset correction sprint with clear validation protocol.

*Result*: Team accepted delay; corrected dataset improved final benchmark by 4.2%. **Lesson**: Technical debt in training data is real debt — quantify cost before deciding to take it.

---

**8. Ownership Beyond Job Description**

*Situation*: LBNL HPC cluster's job accounting system inadequate for tracking DFT workflow costs.

*Task*: Build tool myself.

*Action*: Wrote Python/SLURM accounting wrapper with automated cost attribution, per-project tracking, anomaly alerts.

*Result*: Adopted by 6 other lab groups; saved ~80 hours/year of manual tracking across lab. **Shows**: Identify infrastructure gaps, fill them without being asked.

---

### Behavioral Question Framework

**"Tell me about a time you..."** → Use a STAR story from the above bank. Speak at 90 seconds; be ready for follow-up drilling.

**Personal/Pivot Questions:**

- **Why now — why leave research/Morphism for full-time?** Frame positively: "I've been doing systems and infrastructure work independently, want to apply it in a product context where impact is direct and measurable. Algorized is rare in combining physics sensing rigor with systems engineering challenges at a startup scale where I'd own the full ML stack."

- **Morphism Systems — would it conflict?** Direct: "Morphism is governance research; does not compete with people-sensing AI. Comfortable placing in maintenance mode. Priority: Algorized." Do NOT over-explain.

- **Your PhD took 6 years. Why so long?** Confident: "Dissertation covered genuinely novel materials physics (strain-induced flat bands in TMD monolayers) plus four open-source simulation platforms. Berkeley computational physics PhDs typically 5–6 years for ambitious projects. I am proud of the work." Do NOT apologize.

- **Where do you see yourself in 5 years?** Specific: "Technical leader in edge-AI for physical sensing — shipped products people rely on in safety-critical contexts. Whether principal engineer at Algorized scaling, or later technical co-founder, goal is same: own hard problems from physics to production."

- **Why hire you over a candidate with direct embedded ML experience?** Differentiator is depth + breadth + systems thinking: "Embedded specialist knows CMSIS-NN but may lack full monitoring pipeline, training infrastructure design, or deep physics understanding. I bring physics intuition for signal, ML systems for pipeline, governance for deployment. Embedded runtime is fastest ramp."

---

## Part 3: Resume Navigation & Bridging

### Resume ↔ JD Alignment Matrix

| **JD Requirement** | **Your Evidence** | **Bridge Statement** |
|---|---|---|
| PhD CS/related | PhD EECS UC Berkeley (Dec 2025) | Direct match — no bridging needed |
| 5+ yrs / 3+ (PhD) hands-on DS | PhD (5yr) + Turing + Morphism = 6+ yrs | Frame PhD as applied R&D, not pure research |
| ML on raw radar/sensor data | DFT time-series + HPC signal pipelines | Physics-based signal processing directly analogous |
| C/C++ model deployment | C++ HPC (LAMMPS, VASP), CUDA | Deep C++ background; CMSIS-NN incremental (~1 week) |
| Sensor fusion / edge AI | Not directly in resume — acknowledge | Bridge via multi-modal HPC data fusion, Kalman filter background |
| ML pipeline (train/test/deploy/monitor) | DFT pipelines (2,300+ jobs, monitoring, regression tests) | Exact match at systems level; domain different |
| CI/CD + monitoring frameworks | GitHub Actions, MLflow, monitoring dashboards | Direct match |
| PyTorch, scikit-learn | Listed in LLM & DL skills | Direct match |
| Time-series / stream processing | Sequential DFT job chains | Moderate bridge — emphasize temporal nature |

---

### Three LLM-to-Sensor-AI Bridges (Memorize These)

**Bridge 1 — Lead With This (Most Immediately Actionable):**

Automated evaluation harness for regression prevention. The benchmark suite and CI gates built to prevent language model degradation across updates is **exactly** the infrastructure Algorized needs before every over-the-air model push. Every model version shipped to an automotive edge device must pass a regression suite of standardized radar scenarios (different rooms, occupancy counts, interference types). This is not theoretical — you have built and operated this infrastructure.

**Bridge 2 — Creative Technical Contribution:**

RLHF reward model applied to radar frame quality. In RLHF, a reward model judges output quality to filter and weight training samples. Applied to UWB: train a Channel Impulse Response quality discriminator scoring frames as clean vs. corrupted by multipath/interference. Replaces ad-hoc SNR thresholds with learned quality oracle automatically routing low-quality frames out of training pipeline.

**Bridge 3 — Domain Adaptation Story:**

Supervised fine-tuning for environment-specific adaptation. SFT efficiently adapts base model to new domain using small curated dataset. Direct translation: freeze feature extractor of foundation people-sensing model, fine-tune classifier head with 200–500 labeled frames from customer's specific environment (warehouse, office, car cabin). Same paradigm: preserve general capability from pre-training, adapt cheaply to deployment context.

---

### Gap Analysis — Honest Assessment with Ramp-Up

| **Gap** | **Honest Assessment** | **Your Ramp-Up Plan** |
|---|---|---|
| Direct UWB radar experience | None listed on resume | Studied TyCNN and HDL4AR academic literature, understand full CIR pipeline. Physics background accelerates hands-on ramp. |
| CMSIS-NN and TFLite Micro | Deep C++ HPC, but not embedded MCU | Well-documented API with good tutorials. ~1 week to first working example given C++ depth. Quantization theory already understood. |
| Real-time sensor fusion systems | Not directly on resume | Multi-modal HPC data fusion is conceptually analogous. Kalman filter standard physics curriculum. Gap is specific embedded details. |
| Automotive certification (ECE R129) | No automotive background | Frame as process engineering: working within constrained formal specifications familiar from PhD and Morphism governance work. |
| Streaming (Kafka, Flink) | Not listed | SLURM job streaming is adjacent; can ramp in 2–4 weeks. |

**Framing gaps:** "I haven't worked with CMSIS-NN directly, but I have deep C++ from HPC and understand INT8 quantization theory end-to-end. I expect 1–2 weeks to be productive." More credible than claiming familiarity you lack — more impressive than just "I don't know it."

---

### Positioning Statement Variants

**For coffee chat (casual, 30 seconds):**

"My background is production ML systems and physics-based simulation at scale — I've built pipelines processing thousands of jobs and tens of thousands of compute-hours, and I've built the governance and evaluation infrastructure to make model deployments reliable. I want to apply that systems thinking to a product that has direct real-world safety impact, which is exactly what Algorized's edge-AI for presence detection is."

**For technical interviewer (90 seconds):**

"I'm a computational physicist by training who has spent the last few years building production ML infrastructure — SFT/RLHF pipelines for LLM training, HPC workflows at 24,000 CPU-hour scale with automated regression testing, and governance architecture for deploying AI agents reliably. The common thread across all of it is: how do you build a system where a model update cannot silently degrade production behavior? That's the same problem Algorized faces with OTA updates to safety-critical edge devices, and it's the problem I find most technically interesting."

**Full version (3 sentences — memorize this):**

"I am a computational physicist turned ML systems engineer, with production experience building both high-throughput scientific computing pipelines and language model training infrastructure at scale. I have deep familiarity with the full model development lifecycle — from architecture and training through quantized edge deployment, monitoring, and regression-safe update pipelines — which maps directly to what Algorized needs to scale its edge-AI platform. I am drawn to this role because it combines rigorous physics-based sensing with the systems engineering challenges I find most interesting: reliability, deployment at scale, and real-world safety impact."

---

## Part 4: Why Algorized — Authentic Narrative

Three-layer structure for the "Why this company?" question:

**Layer 1 — Problem Domain:**

You spent 5 years in physics-based simulation and learned that getting ML models to work on **real physical signals** (not clean benchmark data) is fundamentally different engineering challenge than standard ML. UWB radar sensing is exactly that problem at an interesting intersection of physics and systems engineering.

**Layer 2 — Stage and Ownership:**

Algorized is scaling, not 5-person pre-product company. ARIA Sensing platform launch at MWC 2026 proves they have hardware partnerships and real product traction. You can have genuine ML stack ownership from day one.

**Layer 3 — Safety Criticality:**

Child Presence Detection is a system where your work directly determines whether a child lives. That level of stakes creates different engineering culture — rigorous validation, formal regression testing, careful deployment — which is the culture you want.

**Proof-of-research signal to drop naturally:**

"I saw the ARIA Sensing platform announcement from MWC 2026 — the 4×4 MIMO 3D beamforming at 5-degree angular resolution changes what's possible for occupant detection in confined geometries like car cabins." Signals genuine technical interest, not HR enthusiasm.

---

## Part 5: Questions to Ask — Prioritized by Stage

### Coffee Chat (with Product Tech Lead) — Use 2–3:

- **Most impressive (lead with this)**: "The ARIA HYDROGEN 4×4 MIMO platform launch at MWC is exciting — the 3D beamforming at 5-degree angular resolution fundamentally changes what's trackable in a confined space. How does that new capability change the model architecture you're building toward?"

- **Technical depth signal**: "What's the hardest current failure mode for CPD — is it the detection itself, or the false positive rate at the confidence threshold needed for regulatory compliance?"

- **Ownership signal**: "How is the ML stack ownership structured today — is there a clear boundary between signal processing and ML teams, or are those responsibilities combined?"

- **Growth signal**: "What does the technical team look like right now, and what would this hire change about what the team can take on?"

- **Personal fit**: "What does a great first 90 days look like for someone coming into this role?"

### Technical Panel — Use 2–3:

- "How do you currently handle environment-specific model degradation — is per-site fine-tuning in the roadmap?"

- "What does the OTA update pipeline look like today for the embedded model? Is that infrastructure something this role would own?"

- "What's your current training data collection strategy — do you have labeled data across diverse customer environments, or is this an active problem?"

- "How do you think about the sensor-agnostic foundation model goal — is the architecture today built with cross-sensor transfer in mind?"

- "What's the biggest current bottleneck — data collection/labeling, model accuracy on edge cases, or the C++ deployment pipeline?"

---

## Part 6: Cultural Fit & Negotiation

### Algorized Values — How to Demonstrate

| **Value** | **Demonstrates** |
|---|---|
| End-to-end ownership | Reference DFT pipeline: designed, built, monitored, maintained. No handoff. |
| Swiss engineering precision (HQ Etoy; safety-critical) | Rigorous validation, regression testing, zero tolerance for silent failures. |
| Startup velocity | Fast delivery under uncertainty; comfortable with ambiguity. Turing 1-month contract, Morphism self-directed. |
| Customer proximity | Express genuine willingness to travel for on-site support. Ask about engagement process. |
| Genuine product passion | Reference ARIA platform specifics + failure mode discussion. Deep interest, not surface enthusiasm. |

---

### Negotiation Reference (Senior DS, Campbell CA, PhD)

| **Component** | **Market Range** | **Notes** |
|---|---|---|
| **Base Salary** | $170K–$210K | PhD premium; anchor at $195K. Startup slight discount vs. FAANG. |
| **Equity (options)** | 0.1–0.5% early-stage | Ask: vesting schedule (4yr cliff 1yr typical?), strike price vs. last valuation. |
| **Signing Bonus** | $10K–$30K | Less common at early-stage startups. |
| **H-1B Sponsorship** | N/A (non-monetary) | Get written commitment to sponsor next lottery cycle. Condition of offer. |
| **Title** | Senior DS or Senior ML Engineer | Push for "Senior" minimum. Ask about Staff Engineer path. |

**Negotiation strategy**: Bundle all four together, not sequentially. Equity value depends on strike price vs. valuation — ask directly.

---

### STEM OPT & Work Authorization

Brief, calm, factual: "~2.5 years remaining on STEM OPT. Eligible for H-1B next lottery cycle. Standard process for VC-backed deep-tech companies. Request written sponsorship commitment as condition of offer."

Do NOT over-explain. This is routine for founders and VCs.

---

## Part 7: Master Reference Tables

### AI/ML Fundamentals — Quick Reference

**Probability & Statistics:**

| Concept | Definition | Interview Trap |
|---|---|---|
| Bayes Theorem | P(A\|B) = P(B\|A)·P(A) / P(B) | Don't confuse P(hyp\|evidence) with P(evidence\|hyp) — base-rate neglect |
| MLE vs MAP | MLE: max P(data\|θ). MAP: max P(θ\|data) | MAP with Gaussian prior = L2 regularization; Laplace prior = L1 |
| Bias-Variance | MSE = Bias² + Variance + noise | Ensembles reduce variance. Regularization reduces variance at cost of bias. |
| p-value | P(data at least this extreme \| null is true) | NOT P(null hypothesis is true). Common misstatement. |

**Supervised Learning Quick Reference:**

| Model | Objective | Strength | Weakness |
|---|---|---|---|
| Linear Regression | min Σ(y − ŷ)² | Interpretable, fast | Linear relationships only |
| Logistic Regression | Cross-entropy, sigmoid output | Calibrated probabilities | Linear boundary |
| SVM | Max-margin + kernel trick | High dimensions, small N | Slow on large N (cubic) |
| Random Forest | Bagging + random features | Low variance, robust | Memory-heavy, opaque |
| XGBoost | Sequential weak learners (boosting) | Usually best tabular accuracy | Many hyperparameters |

**Regularization Techniques:**

- **L1/Lasso**: Σ\|w\| — produces exact zeros, feature selection. Laplace prior.
- **L2/Ridge**: Σw² — shrinks uniformly, never zero. Gaussian prior.
- **Elastic Net**: αL1 + (1−α)L2 — sparsity + stability.
- **Dropout**: Randomly zero activations during training (p). **Disable at test time!**
- **Early Stopping**: Halt when validation loss plateaus. Implicit L2 regularization. Use patience 5–10 epochs.
- **Data Augmentation**: Radar-specific: Gaussian noise on CIR, random time shifts, amplitude scaling.

**Evaluation Metrics (Imbalanced & CPD-Critical):**

| Metric | Formula | When to Use |
|---|---|---|
| **Recall (Sensitivity)** | TP / (TP + FN) | False negatives catastrophic. CPD, medical screening. **Primary metric at Algorized.** |
| **Precision** | TP / (TP + FP) | False alarms costly. Secondary in safety-critical. |
| **F-β Score** | (1+β²)·P·R / (β²·P + R) | β>1 weights recall. Use F2 or F3 for CPD. |
| **Area Under PR Curve** | Area(Precision vs Recall) | Imbalanced datasets (better than ROC-AUC when positives rare, e.g., 5% of CIR frames). |
| **Log-Loss** | −Σ(y·log(ŷ) + (1−y)·log(1−ŷ)) | Probability calibration quality. Critical for safety-critical confidence. |
| **MOTA** | 1 − (FP + FN + ID_switches) / GT | Multi-object tracking: penalizes misses, false alarms, lost IDs. |

---

### Deep Learning Cheat Sheet

**Backpropagation Essentials:**

- **Chain Rule**: ∇L w.r.t. w_L = (∇L w.r.t. a_L) × (∇a_L w.r.t. z_L) × (∇z_L w.r.t. w_L).
- **Vanishing Gradient**: Small Jacobians (<1) drive gradients to zero. Fixes: ReLU (gradient=1 for x>0), residual connections, LSTM gating.
- **He Initialization**: N(0, 2/fan-in) for ReLU. Xavier: N(0, 2/(fan-in+fan-out)) for tanh/sigmoid.
- **Gradient Clipping**: Cap gradient norm at τ. Essential for RNNs; less critical for Transformers (Layer Norm handles it).

**Activation Functions:**

| Name | Formula | Default Use |
|---|---|---|
| ReLU | max(0, x) | Standard CNN hidden layers. Risk: dying ReLU (always zero). Leaky ReLU (slope 0.01) fixes. |
| GELU | x·Φ(x) | Transformers (BERT, GPT). Smooth, empirically strong. |
| Sigmoid | 1/(1+e^−x) | Binary classification output only. Saturates, avoid hidden layers. |
| Softmax | e^x_i / Σe^x_j | Multi-class output. Use log-sum-exp trick for numerical stability. |
| Swish/SiLU | x·sigmoid(x) | Modern CNNs (EfficientNet). Non-monotonic, empirically strong. |

**CNN Operations:**

- **Output size**: (W − F + 2P) / S + 1 (know this).
- **Depthwise Separable Conv**: Per-channel spatial conv + 1×1 cross-channel conv. ~8× fewer params than standard. Critical for edge deployment (MobileNet).
- **Global Average Pooling**: Average each feature map to single number. Replaces FC layers.
- **1D Convolution for Radar**: Range bins as spatial dimension. Stack 1D convs with increasing dilation for exponential receptive field.

**Sensor-Specific Architectures:**

| Architecture | Key Property | Algorized Fit |
|---|---|---|
| 1D Dilated Causal CNN | Parallelizable, exponential receptive field, causal (no future lookahead) | Strong edge baseline — fast, deployable on microcontroller. |
| Temporal Convolutional Network (TCN) | Dilated 1D + residual connections | Best pure convolutional option for CIR sequences. |
| Hybrid CNN+GRU | CNN extracts per-frame features, GRU models temporal evolution | Good accuracy-to-compute trade-off for multi-person tracking. |
| Transformer/PatchTST | Patch-based tokenization, high accuracy, quadratic memory | Too heavy for bare MCU. Edge SoC or cloud post-processing. |
| **Tiny CNN (TyCNN)** | **Sub-200KB INT8, <48ms on STM32** | **Production-relevant for Algorized edge deployment.** |

---

### Production ML Systems Reference

**ML Pipeline Components:**

1. **Data Ingestion**: Raw sensor data → S3, schema validation (Pydantic), deduplication, quality filter (SNR threshold).
2. **Feature Store**: Decouples feature computation from training. Offline batch + online real-time **must use same code**.
3. **Training**: Triggered by data volume or schedule. Track with MLflow. Pin all dependencies. Reproducibility: commit hash + dataset version → identical model.
4. **Evaluation Gate**: New model beats incumbent by threshold on held-out set + passes regression suite on known failure cases.
5. **Deployment**: Tag as production, blue-green or canary deploy. OTA for edge. Automatic rollback if metrics degrade >X% in 24h.
6. **Monitoring**: Input distribution, prediction distribution, business metrics. On-device: lightweight confidence histograms.

**Training-Serving Skew — The #1 Production Failure:**

Model sees different data at training vs. inference. **For Algorized specifically**: Exponential Moving Average background clutter removal computed differently in Python (training) vs. C++ (inference) causes model to receive systematically different input distributions. **Fix**: Use same stateful EMA implementation in both pipelines. Validate by running same raw CIR sequence through both, comparing features.

**Drift Detection Reference:**

| Drift Type | Definition | Detection Method | Response |
|---|---|---|---|
| **Data drift** | P(X) changes; P(Y\|X) unchanged | PSI, Kolmogorov-Smirnov on features | Collect labeled data, retrain |
| **Concept drift** | P(Y\|X) changes, task evolves | Monitor accuracy vs. ground truth | Full retraining or domain adaptation |
| **Label drift** | Prior P(Y) changes, class balance shifts | Monitor class distribution over time | Recalibrate decision threshold |
| **On-device** | No ground truth on constrained hardware | Confidence histogram entropy; input feature statistics deviation | Flag on cloud sync, trigger review |

**A/B Testing for Edge:**

- **Device-level**: Assign entire devices (not users) to treatment/control. Stratify by environment type. Cannot split single vehicle.
- **Shadow mode** (safest): New model runs on-device but output logged, not acted upon. Zero user impact. Collect data 1–4 weeks.
- **Statistical power**: Detecting 0.5% recall improvement at 80% power with 5% significance requires thousands of device-hours.

---

### Edge Deployment Reference

**Hardware Classes:**

| Class | Examples | RAM | Model Budget | Runtime |
|---|---|---|---|---|
| Bare metal Cortex-M | STM32F7, NXP i.MX RT | 512KB–2MB | 100–500KB | TFLite Micro + CMSIS-NN |
| Application Cortex-A | RPi CM4, NXP i.MX8 | 1–4GB | 1–100MB | ONNX Runtime, TFLite |
| Embedded NPU | STM32 Neural-ART | varies | 1–10MB INT8 | Vendor SDK |
| Automotive SoC | NXP S32G, Renesas R-Car | 2–16GB | Unlimited | TensorRT, ONNX Runtime |

**Model Compression Workflow:**

1. **Structured Pruning**: Remove entire filters/channels with low L2 norm. Target 50–70% reduction. No sparse data structures needed.
2. **Knowledge Distillation**: Compact student mimics soft outputs of larger teacher. When architecture must change significantly.
3. **Post-Training Quantization (PTQ)**: FP32 → INT8 using 500 representative CIR frames. ~4× size. If recall drops >2%, switch to QAT.
4. **Quantization-Aware Training (QAT)**: Simulate quantization during training with fake quantization nodes. Gradients flow via Straight-Through Estimator. Model learns robustness to rounding.
5. **Export**: PyTorch → ONNX (opset 12) → TFLite → C byte array. Validate each step: max output difference <0.001.

**Quantization Concepts:**

| Concept | Details |
|---|---|
| INT8 scale factor | S = (max − min) / 255. Symmetric (weights): zero-point = 0. Asymmetric (activations): zero-point non-zero. |
| Per-channel | Separate scale per output channel. 1–3% better accuracy. Required for depthwise conv. |
| Mixed precision | Keep first, last, skip connections in INT16/FP16. Recover 1–2% accuracy at small memory cost. |

**Inference Runtimes:**

- **TFLite Micro**: Single C++ file, no OS dependency, CMSIS-NN backend, INT8, pre-allocated tensor arena. Target: STM32, Arduino Nano 33.
- **CMSIS-NN**: ARM-optimized SIMD (DSP extension). 5–10× speedup on Cortex-M vs. non-optimized. Integral to TFLite Micro.
- **ONNX Runtime**: ARM Linux (RPi, i.MX8), dynamic shapes, XNNPACK provider, ~1–3MB minimal binary.
- **Custom C++**: Maximum control, minimum overhead. Justified only for models <50KB or AUTOSAR certification.

**OTA Update Architecture:**

- **Package format**: Signed bundle (model + version metadata + evaluation results hash). Cryptography prevents tampering.
- **Safe staging**: Download to staging partition, not active partition. Validate before cutover.
- **On-device validation**: Known-answer test (fixed input → compare output hash). Mismatch → abort, delete staging partition.
- **Cutover & rollback**: Atomic boot pointer swap. Provisional mode with monitoring. Anomaly detected → auto-rollback to backup partition.

---

## Part 8: Radar & Edge AI Master Glossary

**Signal Processing Terms:**

- **CIR (Channel Impulse Response)**: Time-domain amplitude vs. delay (range). Raw output of UWB transceiver.
- **Range Resolution**: ΔR = c/(2B). At 1.8GHz: ΔR ≈ 8.3cm.
- **CFAR (Constant False Alarm Rate)**: Adaptive thresholding that accounts for local noise. Cell-Averaging or Order-Statistic variants.
- **Range-Doppler Map**: 2D FFT of stacked CIR frames. Range (x-axis), Doppler/velocity (y-axis).
- **Beamforming**: Electronic steering of antenna beam via phase offsets. ARIA 4×4 MIMO → 16 virtual elements → 5° angular resolution.
- **Multipath**: Signal bouncing off walls/furniture multiple times before arriving. Creates ghost targets.
- **Micro-Doppler**: Oscillations caused by limb motion (breathing, heartbeat), distinguishes human from rigid objects.

**ML Systems Terms:**

- **Training-Serving Skew**: Model sees different data distribution at training vs. inference.
- **Data Drift**: P(X) changes, P(Y|X) unchanged.
- **Concept Drift**: P(Y|X) changes, task definition evolves.
- **Shadow Deployment**: New model runs but output not used. Safest testing strategy for safety-critical systems.
- **Canary Deployment**: Roll out to small fraction (1%) before full deployment. Automatic rollback if metrics degrade.

**Edge Deployment Terms:**

- **Quantization**: Converting FP32 weights/activations to INT8. Reduces model size ~4×.
- **Pruning**: Removing weights/filters. Structured (remove entire filters) vs. unstructured (individual weights).
- **Knowledge Distillation**: Compact student model learns from larger teacher model.
- **Tensor Arena**: Pre-allocated buffer for TFLite Micro inference. Size must be calculated carefully.
- **Known-Answer Test**: Fixed input → compare output hash to expected. Validates model deployment integrity.

---

## Final Preparation Checklist

**Before interviews:**

- [ ] Memorize 3-sentence positioning statement (Part 4).
- [ ] Have 8 STAR stories fluent at 90 seconds each (Part 2).
- [ ] Prepare gap analysis answers with ramp-up plans (Part 3).
- [ ] Know three LLM-to-sensor AI bridges cold (Part 3).
- [ ] Review 5 resume ↔ JD mappings most likely to be drilled (Part 3).
- [ ] Have 2–3 questions to ask per interview stage ready (Part 5).
- [ ] Know Bayesian thinking, bias-variance, regularization (Part 7).
- [ ] Know UWB → CIR → CFAR → tracking → CNN pipeline cold (Part 1).
- [ ] Know CMSIS-NN, TFLite Micro, INT8 quantization (Part 1).
- [ ] Practice "Tell me about a time..." → STAR story flow.

**During interviews:**

- Lead with impact, not implementation details.
- Bring data to technical disagreements, not opinion.
- Acknowledge gaps; pair with clear ramp-up plan.
- Ask about ARIA platform, CPD regulatory requirements, OTA infrastructure.
- Close with genuine interest: "This problem excites me because..."

---

**Total Content:** 8 sections, 7 major parts, 50+ reference tables, 8 STAR stories, 3 positioning variants, comprehensive resume bridges, behavioral Q&A framework.

**Scope:** Technical interview Q&A (signal processing, ML systems, edge deployment), behavioral preparation (STAR story bank), resume navigation (JD alignment, gaps, bridges), career narrative (positioning, cultural fit), negotiation reference.

This is your complete interview toolkit for roles at Algorized and similar edge-AI/embedded-ML companies. Know the STAR stories cold. Lead with the LLM-to-sensor bridges. Demonstrate rigorous systems thinking across the full stack from physics to production.

