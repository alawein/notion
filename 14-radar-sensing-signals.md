# 14 Radar Sensing Signals

**Total Pages:** 5



--- Page 1 ---

Algorized Interview Prep — Doc 4: Radar & Signal Processing
Page 1
DOC 4 / 8
Radar Sensing & Signal Processing
UWB physics, signal pipeline, clutter, tracking, and multi-modal fusion for edge
people-sensing
Topics Covered
 UWB Radar Physics & Hardware
 Channel Impulse Response (CIR) Signal Pipeline
 Range-Doppler Processing
 CFAR Detection & Tracking
 Vitals Detection (Respiration & Heart Rate)
 Multi-Modal Sensor Fusion (UWB + Wi-Fi)
 Algorized's ARIA HYDROGEN Platform
 Interview Q&A; Bank
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 4: Radar & Signal Processing
Page 2
1. UWB Radar Physics & Hardware
1.1 UWB Fundamentals
 Definition: Ultra-wideband: RF signal occupying >500 MHz bandwidth or >20% fractional bandwidth. Typical
center frequency: 3.1-10.6 GHz (FCC regulated). Algorized's ARIA HYDROGEN: up to 1.8 GHz programmable
bandwidth.
 Range resolution: ΔR = c/(2B) where c = 3×10■ m/s, B = bandwidth. At B=1.8 GHz: ΔR ≈ 8.3 cm. At B=500
MHz: ΔR ≈ 30 cm. Higher bandwidth → finer range resolution.
 Time-of-flight ranging: Distance = c·TOF/2. UWB measures the round-trip time of short pulses
(sub-nanosecond) to resolve range with cm precision. Immune to multipath that plagues narrowband RSSI
ranging.
 Through-wall capability: UWB penetrates non-metallic materials (drywall, wood, plastic). Signal attenuated
~5-15 dB per wall. Still detectable for presence sensing up to 2-3 walls. Metal walls block completely.
 Privacy advantage: UWB captures amplitude vs. time, not visual images. No facial recognition possible. No
identifiable biometric beyond motion/breathing. Key selling point for GDPR-sensitive deployments (offices,
healthcare).
 MIMO antenna array: ARIA HYDROGEN: 4×4 MIMO (4 Tx, 4 Rx = 16 virtual elements). Enables 3D sensing.
Angular resolution ≈ λ/(N·d) ≈ 5° at 6.5 GHz center, 16 virtual elements, λ/2 spacing.
2. CIR Signal Pipeline
2.1 From ADC to Features
 Step 1: CIR Acquisition: Transceiver transmits short pulse; ADC samples the received signal at GHz sampling
rate. Output: CIR frame S[t] — amplitude vs. delay (= range). Frame rate: 50-200 Hz. Stored as real-valued time
series per virtual antenna element.
 Step 2: Static Clutter Removal: Background = EMA of frames: B[t] = α·B[t-1] + (1-α)·S[t]; Dynamic signal D[t] =
S[t] - B[t]. α ≈ 0.95 → ~20-frame memory. Removes static reflectors (walls, furniture). Only moving targets
remain.
 Step 3: Range Gating: Keep only range bins corresponding to the room/vehicle interior (e.g., 0.3m to 5m).
Discard near-field interference (<0.3m) and far-field noise. Reduces compute by 40-60%.
 Step 4: Range-Doppler Transform: Stack N consecutive frames (N=32 or 64). Apply 2D FFT: FFT across
range bins (fast-time) and across frames (slow-time). Output: Range-Doppler map P[r, v] — power at each
range-velocity pair.
 Step 5: Feature Extraction: From Range-Doppler map: micro-Doppler signature (velocity profile), energy
centroid, peak ranges, range-velocity histograms. For vitals: FFT of amplitude variations in target range bin at
0.2-2 Hz (respiration) and 1-2 Hz (heart rate).


--- Page 3 ---

Algorized Interview Prep — Doc 4: Radar & Signal Processing
Page 3
 Step 6: Model Input: Normalized feature tensor fed to CNN or LSTM. For TyCNN: 2D range-Doppler map (e.g.,
64×64) or 1D range profile (64 bins) as input. Batch normalize at model input layer.
3. CFAR Detection & Multi-Target Tracking
3.1 CFAR Thresholding
Constant False Alarm Rate detection maintains a fixed probability of false alarm regardless of noise level. The
threshold adapts to local noise statistics:
 CA-CFAR (Cell-Averaging): Threshold T = α · (1/N Σ neighbors). α set to achieve desired P_fa. Assumes
uniform noise; fails near clutter edges.
 OS-CFAR (Ordered Statistics): Threshold = α · k-th largest neighbor. More robust to clutter edges and multiple
targets. Higher compute.
 CFAR in 2D: Apply CFAR on Range-Doppler map: guard cells around CUT (Cell Under Test) + training cells.
Common guard cell size: 2-3 range/Doppler bins. Training window: 8-16 cells.
3.2 Multi-Target Tracking
 Kalman Filter: Optimal linear estimator for Gaussian noise. State: [x, y, vx, vy]. Prediction step: x■■|■■■ =
Fx■■■ + Bu. Update step: incorporate new measurement. Suitable for 2D people tracking with UWB.
 MHT (Multiple Hypothesis Tracking): Maintains a tree of possible track associations. Handles occlusion, track
birth/death, and closely-spaced targets. Computationally expensive; requires pruning (N-scan pruning). Best
accuracy for multi-person scenarios.
 JPDA (Joint Probabilistic Data Association): Probabilistic association of measurements to tracks. More
tractable than MHT. Common in real-time embedded systems. Works well for up to ~5 targets.
 Track management: Track initiation: require M detections in N frames. Track deletion: no association for K
consecutive frames. ID management: persistent track ID across frames for occupancy counting.
4. Vitals Detection
UWB phase shifts caused by sub-mm chest wall movements enable non-contact respiration and heart rate
estimation:
 Respiration (0.2-0.5 Hz): Chest displacement: 5-20mm. Well above UWB noise floor. Extract: select range bin
where target is located; compute FFT of amplitude envelope over 10-30 second window; peak in 0.2-0.5 Hz =
respiration rate.
 Heart rate (1-2 Hz): Chest displacement: 0.1-0.5mm. Near UWB noise floor. Requires: high SNR (close range
<2m), static subject, high frame rate (>50 Hz), careful filtering to separate from respiration harmonics. Bandpass
filter at 1-2 Hz then peak detection.
 Key challenges: Motion artifacts dominate when subject moves. Harmonics of respiration fall in HR band.
Multi-person scenarios require source separation. Environmental vibrations (HVAC) create false signatures at
0.3-0.5 Hz.


--- Page 4 ---

Algorized Interview Prep — Doc 4: Radar & Signal Processing
Page 4
 Automotive CPD application: Detect sleeping child (minimal motion) vs. empty seat. Child breathing at 0.3-0.5
Hz distinguishes from static object. This is the core Algorized product use case — practice explaining this clearly.
5. Multi-Modal Fusion: UWB + Wi-Fi
<b>Fusion Type</b>
<b>How It Works</b>
<b>Pros</b>
<b>Cons</b>
Feature-level (early)
Concatenate UWB + Wi-Fi features before model
Single model, low latency
Requires synchronization
Decision-level (late)
Separate models; combine confidence scores
Robust to sensor dropout
Higher latency, more memory
Kalman fusion
UWB position + Wi-Fi occupancy as measurements
Principled uncertainty handling
Assumes linear dynamics
Cross-attention (deep)
Transformer cross-attention between modalities
Best accuracy
Heavy; not edge-deployable
6. Algorized Platform: ARIA HYDROGEN Details
 HYDROGEN 4×4 SoC specs: 4 Tx, 4 Rx antennas. True 3D detection (azimuth + elevation). Digital
beamforming with ~5° angular resolution. Up to 1.8 GHz programmable bandwidth. Low-power radar framework.
First UWB SoC with integrated 3D detection on a single chip.
 Algorized's edge AI engine: Runs on embedded MCU alongside the HYDROGEN SoC. C/C++ inference.
Supports: people counting and tracking, micro-motion monitoring, behavioral pattern recognition, CPD (launch
product).
 Joint platform capabilities: Micro-motion detection (breathing patterns), respiratory activity monitoring,
occupant positioning, behavioral activity recognition. Applications: automotive CPD (launch), smart buildings,
elderly monitoring, consumer electronics.
 CEO quote (use in coffee chat): Natalya Lopareva (Algorized CEO): 'By combining AI with a purpose-built 3D
UWB radar architecture, we are unlocking capabilities that were simply not possible with legacy silicon.' — MWC
Barcelona, Feb 2026.
7. Interview Q&A; Bank — Radar & Signals
Q: How does UWB achieve mm-range resolution when the speed of light is 3×10■ m/s?
■ ΔR = c/2B. At B=1.8 GHz: ΔR ≈ 8.3 cm. At 1 GHz bandwidth: ~15 cm. The wide bandwidth (short pulse duration ~1/B)
means the radar can resolve reflectors separated by only centimeters in range. This is why UWB is fundamentally
different from narrowband Wi-Fi-based sensing.
Q: Why is 2D FFT used on stacked CIR frames, and what does the output tell you?
■ First FFT (across range bins, fast-time): gives power vs. range — where targets are. Second FFT (across frames,
slow-time): gives Doppler shift vs. range — how fast each range cell is moving. The output range-Doppler map shows all
targets at their range and velocity simultaneously. Stationary clutter concentrates at zero Doppler; moving people spread


--- Page 5 ---

Algorized Interview Prep — Doc 4: Radar & Signal Processing
Page 5
across the non-zero Doppler bins.
Q: A customer complains the system detects presence in an empty room near an industrial HVAC unit.
How do you fix it?
■ HVAC creates periodic micro-vibrations at 0.3-0.5 Hz and harmonic frequencies — overlapping with human breathing
signature. Solution: (1) Collect labeled HVAC interference data at that site. (2) Train a discriminator on motion
frequency, amplitude, and spatial coherence (HVAC affects all range bins uniformly; a person creates a localized return).
(3) Add a coherence check: real person motion shows range-localized return, not room-wide correlated noise.
Key differentiator: Algorized just launched a 4x4 MIMO platform with 3D detection (MWC 2026). Most
academic literature uses 1x1 or 1x4 arrays. The beamforming capability fundamentally changes the
tracking architecture — you can now separate people in angle, not just range and velocity. Mention this
proactively.
