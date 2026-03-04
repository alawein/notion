# 12 Deep Learning

**Total Pages:** 5



--- Page 1 ---

Algorized Interview Prep — Doc 2: Deep Learning
Page 1
DOC 2 / 8
Deep Learning
Neural architectures, training dynamics, and modern DL systems for senior DS interviews
Topics Covered
 Neural Network Fundamentals & Backpropagation
 Convolutional Neural Networks (CNNs)
 Recurrent Networks: LSTM & GRU
 Transformer Architecture & Attention
 Training Stability & Normalization Techniques
 Loss Functions & Activation Functions
 Modern Architectures for Time-Series & Sensor Data
 Interview Q&A; Bank
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 2: Deep Learning
Page 2
1. Neural Network Fundamentals
1.1 Forward Pass & Backpropagation
A neural network is a composition of parameterized affine transformations interleaved with nonlinear activations: f(x)
= σ(W■σ(W■■■...σ(W■x + b■)...+ b■■■) + b■). Training minimizes a loss L(■, y) via gradient-based optimization.
 Chain rule: ∂L/∂W■ = ∂L/∂a■ · ∂a■/∂z■ · ∂z■/∂W■ where a■ = σ(z■) and z■ = W■a■■■ + b■. Backprop
efficiently computes all gradients in one backward pass via dynamic programming.
 Vanishing gradients: Repeated multiplication of small Jacobians (<1) in deep networks drives gradients to
zero. Solution: ReLU activations, skip connections (ResNet), batch/layer normalization, proper initialization (He,
Xavier).
 Exploding gradients: Repeated multiplication of large Jacobians. Solution: gradient clipping, weight
regularization. Common in RNNs over long sequences.
 Weight initialization: He initialization: W ~ N(0, 2/n■■) — for ReLU networks. Xavier: W ~ N(0,
2/(n■■+n■■■)) — for tanh/sigmoid. Prevents gradient vanishing/explosion at initialization.
1.2 Activation Functions
<b>Function</b>
<b>Formula</b>
<b>Range</b>
<b>Notes</b>
Sigmoid
1/(1+e■■)
[0,1]
Vanishing gradient for |x|>>0; use for output prob
Tanh
(e■-e■■)/(e■+e■■)
[-1,1]
Zero-centered; still vanishes at extremes
ReLU
max(0,x)
[0,∞)
Simple, fast; dying ReLU problem (x<0 → 0 grad)
Leaky ReLU
max(αx,x), α=0.01
(-∞,∞)
Fixes dying ReLU; α is hyperparameter
GELU
x·Φ(x)
≈ReLU
Smooth; default in BERT/GPT; better than ReLU empirically
Softmax
e■■/Σe■■
[0,1], Σ=1
Output layer for multi-class; numerically stabilize by subtracting max
Swish/SiLU
x·σ(x)
(-∞,∞)
Non-monotonic; strong empirical performance in modern nets
2. Convolutional Neural Networks
2.1 CNN Operations
 Convolution: Output size: (W-F+2P)/S + 1 where W=input width, F=filter size, P=padding, S=stride. Parameter
sharing: same filter applied everywhere — equivariance to translation. Critical for radar range-bin features.
 Pooling: Max pooling: translation invariance, extracts dominant features. Average pooling: smoother, used in
global average pooling (GAP) to replace FC layers. GAP reduces parameters and overfitting.


--- Page 3 ---

Algorized Interview Prep — Doc 2: Deep Learning
Page 3
 Depthwise Separable Convolution: Split standard conv into depthwise (per-channel spatial) + pointwise (1×1
across channels). 8-9× fewer parameters. Used in MobileNet — relevant for Algorized's edge deployment.
 Receptive field: The input region affecting a neuron. Grows with depth and dilation. For 1D-CNN on radar time
series: receptive field determines how many time steps the model can integrate.
 1D CNN for time series: Treats time steps as the spatial dimension. Kernel size controls temporal context
window. Stack 1D convs with increasing dilation (dilated causal conv) for exponentially growing receptive field
without depth — efficient for radar CIR streams.
Algorized relevance: Tiny CNN (TyCNN) for UWB people counting uses 1D convolutions on range bins,
achieving 99.38% accuracy with <200KB model size after INT8 quantization — retaining 98.22%
post-quantization. This is the architecture you should discuss for edge deployment.
3. Recurrent Networks: LSTM & GRU
3.1 LSTM Architecture
LSTM solves vanishing gradient in standard RNNs via gated cell state. Four components per timestep:
 Forget gate: f■ = σ(Wf·[h■■■, x■] + bf) — decides what to remove from cell state
 Input gate: i■ = σ(Wi·[h■■■, x■] + bi); c■■ = tanh(Wc·[h■■■, x■] + bc) — new candidate values
 Cell update: c■ = f■■c■■■ + i■■c■■ — update cell state via elementwise operations
 Output gate: o■ = σ(Wo·[h■■■, x■] + bo); h■ = o■■tanh(c■) — produce hidden state
GRU vs LSTM:
GRU merges forget and input gates into update gate; no separate cell state. Fewer parameters (~25% reduction).
Similar performance on most tasks. Preferred when compute budget is tight — relevant for edge sensor models.
4. Transformer Architecture & Self-Attention
4.1 Self-Attention Mechanism
Attention(Q, K, V) = softmax(QK■/√d■)·V where Q, K, V are linear projections of the input. Scaled dot-product: √d■
prevents softmax saturation at high dimensions.
 Multi-Head Attention: Run h parallel attention heads with different projections; concatenate and project. Each
head can attend to different positional/semantic patterns. h=8 (base) or h=16 (large) standard.
 Positional Encoding: Transformers have no inherent order — add positional encodings. Sinusoidal (fixed) or
learned. For time-series sensor data: learned position embeddings often better.
 Self-attention complexity: O(n²d) per layer — quadratic in sequence length n. Problematic for long radar frame
sequences. Alternatives: local attention, Longformer, linear attention.


--- Page 4 ---

Algorized Interview Prep — Doc 2: Deep Learning
Page 4
 Layer Normalization: Normalize across feature dimension (not batch). Applied before (Pre-LN) or after
(Post-LN) attention. Pre-LN more stable for training; used in GPT-2+.
 FFN sublayer: Two-layer MLP with GELU: FFN(x) = GELU(xW■+b■)W■+b■. Width = 4× model dim typically.
Adds nonlinear capacity to the attention block.
5. Modern Architectures for Sensor / Time-Series Data
 Temporal Convolutional Network (TCN): 1D dilated causal convolutions with residual connections.
Parallelizable (unlike LSTM). Receptive field = 2^L for L layers with doubling dilation. Strong baseline for UWB
CIR time series.
 Hybrid CNN+LSTM (HDL4AR): CNN extracts per-frame spatial features; LSTM models temporal evolution.
Published on UWB HAR (Human Activity Recognition). Combine: 1D-CNN encoder → LSTM sequence model →
classification head.
 Transformer for time series (TST): Patch-based tokenization of time series (PatchTST, TimesNet). Patch size
= sub-sequence length. Position encoding handles temporal order. Higher accuracy than CNN/LSTM on long
sequences; heavier compute.
 State Space Models (S4, Mamba): Recent alternative to attention for long sequences. Linear recurrence with
special parameterization for long-range dependencies. O(n log n) or O(n) — potentially suited for streaming radar
inference.
 Tiny CNN (TyCNN): Sub-200KB CNN for MCU deployment. Architecture: input → BatchNorm → Conv1D (×3)
→ GAP → FC → softmax. Quantized to INT8; inference <48ms on STM32. This is the production-relevant
architecture for Algorized.
6. Training Stability & Normalization
<b>Technique</b>
<b>Normalizes Over</b>
<b>When to Use</b>
Batch Norm
Batch dimension (N)
CNNs, large batch sizes; not for RNNs or small batches
Layer Norm
Feature dimension (C,H,W)
Transformers, RNNs, NLP; batch-size independent
Instance Norm
Spatial dims per sample
Style transfer, image generation
Group Norm
Channel groups per sample
Object detection, small batch sizes
Weight Norm
Reparameterize weights
RL, generative models; avoids batch dependency
7. Interview Q&A; Bank — Deep Learning
Q: Why does Batch Normalization help training, and when would you NOT use it?


--- Page 5 ---

Algorized Interview Prep — Doc 2: Deep Learning
Page 5
■ Reduces internal covariate shift, allows higher LR, has mild regularization effect. Avoid for: small batches (<8), RNNs
(use Layer Norm), online/streaming inference (batch stats undefined).
Q: Explain the vanishing gradient problem and three solutions.
■ (1) ReLU activations — gradient is 1 for x>0. (2) ResNet skip connections — identity path provides gradient highway.
(3) LSTM/GRU gating — cell state allows gradients to flow unchanged over long sequences.
Q: You are designing a model for UWB radar CIR sequences of length 256. What architecture would you
choose?
■ Start with TCN (dilated causal conv): parallelizable, good receptive field, edge-deployable. Compare to 1D-CNN+GRU
hybrid. Avoid full Transformer unless compute budget allows. Key: maintain causal (non-lookahead) structure for
real-time inference.
Q: What is the difference between model parameters, hyperparameters, and architecture choices?
■ Parameters: learned (weights, biases). Hyperparameters: set before training (LR, batch size, regularization λ).
Architecture: structural choices (layer count, activation type, skip connections) — can be searched via NAS.
Q: How would you debug a model that trains fine but produces poor validation accuracy from epoch 1?
■ Classic overfitting: model too large for dataset, or data leakage. Check: dataset size vs. model parameters, train/val
split integrity, augmentation applied only to train, no target leakage in features.
Q: Explain dropout and why you disable it at inference time.
■ Training: randomly zero activations to prevent co-adaptation. Inference: disable and scale outputs by (1-p) — or
equivalently, train with probability p and use all neurons at test time. Forgetting to disable dropout at inference causes
noisy, stochastic predictions.
