# 16 Edge Ai Deployment

**Total Pages:** 5



--- Page 1 ---

Algorized Interview Prep — Doc 6: Edge AI & Deployment
Page 1
DOC 6 / 8
Edge AI & Embedded Deployment
Quantization, compression, C/C++ inference, and TinyML for constrained hardware
Topics Covered
 Edge Hardware Landscape (MCU, DSP, NPU)
 Model Compression: Quantization (PTQ & QAT)
 Model Compression: Pruning & Distillation
 Inference Runtimes: TFLite Micro, ONNX, CMSIS-NN
 C/C++ Deployment Workflow
 Memory & Latency Optimization
 OTA Update Architecture for Edge Models
 Interview Q&A; Bank
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 6: Edge AI & Deployment
Page 2
1. Edge Hardware Landscape
<b>Class</b>
<b>Examples</b>
<b>RAM</b>
<b>Model Budget</b>
<b>ML Framework</b>
Bare MCU (Cortex-M4/M7)
STM32F7, NXP i.MX RT
512KB-2MB
100-500KB
TFLite Micro + CMSIS-NN
Application MCU (Cortex-A)
RPi CM4, NXP i.MX8
1-4GB
1-100MB
ONNX Runtime, TFLite
DSP (C67xx)
TI C6748
128KB-1MB
200KB-2MB
Custom C/TI DL
Embedded NPU
NXP eIQ, STM32 NUCLEO-NPU
varies
1-10MB INT8
Vendor SDK
Automotive SoC
NXP S32G, Renesas R-Car
2-16GB
unlimited
TensorRT, ONNX
2. Quantization
2.1 Post-Training Quantization (PTQ)
 INT8 PTQ process: 1) Train FP32 model normally. 2) Collect calibration dataset (100-1000 representative
inputs). 3) Run calibration: measure activation ranges per layer. 4) Compute scale factors: S = (max-min) / 255. 5)
Quantize weights and activations. 6) Evaluate accuracy — if drop >2%, switch to QAT.
 Calibration data selection: Representative but NOT the full training set. Cover edge cases. For radar: include
diverse environments, occupancy counts, interference scenarios. 500-1000 frames typical.
 Symmetric vs Asymmetric quant: Symmetric: zero_point=0; range [-128,127]. Simpler, better for weights.
Asymmetric: zero_point≠0; range [0,255]. Better for activations with non-zero mean (after ReLU).
 Per-layer vs Per-channel quant: Per-channel: separate scale per output channel of weight tensor. 1-3% better
accuracy than per-layer. Required for depthwise convolutions. TFLite Micro supports per-channel.
2.2 Quantization-Aware Training (QAT)
Simulate quantization during training using fake quantization (FQ) nodes. Gradients flow through FQ nodes via
Straight-Through Estimator (STE): ∂FQ/∂x = 1 within clamp range, 0 outside. Model learns to be robust to
quantization error. Use when PTQ accuracy loss >2%, which often occurs for:
 Small models with <1M parameters (less redundancy to absorb quantization error)
 Regression tasks (vitals estimation) vs. classification
 Models with batch normalization folded into weights
 Any model where FP32→INT8 recall drop exceeds safety threshold
3. Pruning & Knowledge Distillation


--- Page 3 ---

Algorized Interview Prep — Doc 6: Edge AI & Deployment
Page 3
<b>Technique</b>
<b>How It Works</b>
<b>Size Reduction</b>
<b>Best For</b>
Unstructured Pruning
Set individual weights to zero based on magnitude
Up to 90% sparse
Sparse hardware (not most MCUs)
Structured Pruning
Remove entire filters/channels with low L2 norm
30-70% fewer params
Dense hardware; maps to C arrays
Weight Sharing
Cluster weights into K centroids; store indices
Depends on K
Combined with quantization
Knowledge Distillation
Student learns from teacher's soft labels
10-100× model size reduction
When architecture must change
Neural Architecture Search
Automated search over architecture space
Varies
When custom architecture needed
Practical path for Algorized: Structured pruning + INT8 QAT is the standard 2-step recipe. Start with 50%
structured pruning (filter magnitude), then QAT. Benchmark on target MCU. If latency/accuracy
acceptable, done. If not, add knowledge distillation from a larger teacher model trained on the full
dataset.
4. C/C++ Inference Runtimes
 TensorFlow Lite Micro (TFLM): Single .cc/.h file compilation. No OS dependency. CMSIS-NN backend for
ARM. Supports: Conv2D, DepthwiseConv, FC, LSTM, RNN. Memory allocator works from a pre-allocated tensor
arena. Use: STM32, Arduino Nano 33, nRF52840. Limitation: static graph only, no dynamic shapes.
 CMSIS-NN: ARM-optimized kernels for Cortex-M. SIMD via DSP extension (Cortex-M4/M7/M33/M55). Key
functions: arm_convolve_HWC_q7_fast(), arm_fully_connected_q7(). Integral to TFLM on ARM. Without
CMSIS-NN: ~5-10× slower on Cortex-M.
 ONNX Runtime (embedded): Supports ARM Linux (Cortex-A). Dynamic shapes. Execution providers: CPU
(default), XNNPACK (mobile-optimized). Better for NXP i.MX8, RPi 4, Renesas R-Car than for bare MCU.
 ONNX Runtime Mobile: Minimal build for embedded Linux: ~1-3MB binary. Supports INT8 quantized models.
Reasonable for i.MX8-class hardware with Linux OS.
 Custom C++ flat inference: Hand-write inference as struct arrays + function calls. Zero framework overhead.
Maximum control over memory layout. Used in AUTOSAR/SafetyOS contexts where third-party libs are
prohibited. Only justified for tiny models (<50KB) or certification requirements.
5. Complete C/C++ Deployment Workflow
 1. Train & validate (PyTorch): FP32 model. Full evaluation suite. Establish accuracy baseline.
 2. Quantize (PTQ or QAT): INT8 calibration. Validate: FP32 vs INT8 max output diff < tolerance. If recall drops
>1%: switch to QAT.
 3. Export to ONNX: torch.onnx.export(model, dummy_input, 'model.onnx', opset_version=12). Validate ONNX
output matches PyTorch.
 4. Convert to TFLite: tf.lite.TFLiteConverter.from_saved_model() or via onnx2tf. Apply INT8 representative
dataset. Export .tflite file.


--- Page 4 ---

Algorized Interview Prep — Doc 6: Edge AI & Deployment
Page 4
 5. Generate C source: xxd -i model.tflite > model_data.cc. Or use flatc to embed in firmware. Model becomes a
C byte array.
 6. Integrate TFLM + CMSIS-NN: Add tflite-micro sources and CMSIS-NN to firmware project. Allocate tensor
arena (kTensorArenaSize bytes). Build interpreter and invoke.
 7. Benchmark on target: Measure: inference time (cycles/ms), peak RAM usage, flash footprint. Compare
against requirements: <50ms, <512KB.
 8. Regression validation: Run held-out evaluation dataset through C++ inference path. Confirm accuracy
matches ONNX/PyTorch within tolerance.
6. OTA Update Architecture
 Model package format: Signed bundle: model binary + version metadata + evaluation certificate (hash of eval
results). Cryptographic signature prevents tampered models. Size: <500KB for MCU targets.
 Download and staging: Download in background over BLE/Wi-Fi. Write to staging flash partition. Do NOT
overwrite active partition until validation passes.
 On-device validation: After staging: run self-test (known-answer test on fixed input). Compare output to
expected (stored hash). If mismatch: abort, delete staging, report error.
 Cutover and rollback: On successful validation: atomic swap (update boot pointer). First N operating hours in
'provisional' state: monitor performance. If anomaly detected → auto-rollback to previous version in backup
partition.
 Fleet management: Server tracks: device_id → firmware_version → model_version → last_sync. Enables
targeted rollback for affected devices without updating entire fleet.
7. Interview Q&A; Bank — Edge AI
Q: Your INT8 quantized model has 3% lower recall than FP32 on the CPD test set. What do you do?
■ 3% is too much for a safety-critical application. Options in order: (1) Switch PTQ→QAT: train with simulated
quantization, model adapts. (2) Per-channel quantization instead of per-tensor for depthwise convolutions. (3) Mixed
precision: keep sensitive layers (first/last, skip connections) in INT16 or FP16. (4) Architecture change: replace layers
that are quantization-sensitive (e.g., depthwise conv before BN) with more quantization-friendly operations. (5)
Knowledge distillation: retrain compact student from scratch with soft labels.
Q: What is the tensor arena in TFLite Micro and how do you size it?
■ TFLM uses a pre-allocated byte array for all tensor data (no dynamic memory allocation). Size it by: running the model
with an oversized arena (e.g., 100KB) and calling interpreter.arena_used_bytes() after inference. Add 10-20% margin for
safety. If arena too small, interpreter returns kTfLiteError at invoke time — easy to diagnose.
Q: How do you verify that your C++ edge model produces the same results as your Python training
model?
■ Known-answer test (KAT): compute output of 10 fixed test inputs in Python (to 8 decimal places). Store expected
outputs as constants. In C++ inference code, run same inputs, compare outputs within tolerance (max abs diff < 1e-3 for
INT8, 1e-5 for FP32). Run this as automated test in firmware CI. Any deviation indicates quantization error, endianness


--- Page 5 ---

Algorized Interview Prep — Doc 6: Edge AI & Deployment
Page 5
bug, or preprocessing mismatch.
