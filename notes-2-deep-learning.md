# Notes 2: Deep Learning Architectures & Training

Comprehensive guide to neural network fundamentals, architectures (CNNs, RNNs, LSTMs, Transformers), training dynamics, and modern deep learning systems for production deployment.

---

## Part 1: AI → ML → DL Hierarchy

### The Nested Subset Framework

**Artificial Intelligence (AI)** — Broadest category
- Any computational technique making machines exhibit intelligent behavior
- Includes rule-based systems, search algorithms, expert systems, and learned models
- Example: Chess engines (rule-based and learned)

**Machine Learning (ML)** — Subset of AI
- Systems that learn patterns from data rather than following explicit rules
- Formally: find f* = argmin E[L(f(x), y)] over some function class
- Examples: decision trees, random forests, neural networks, SVMs

**Deep Learning (DL)** — Subset of ML
- Uses neural networks with many layers (>2) for representation learning
- Model discovers its own features from raw data automatically
- Not a separate paradigm, but a way to parameterize f using deep hierarchies
- Dominates for high-dimensional data: images, text, audio, sensor streams

### When to Use Each

**Choose DL when:**
- High-dimensional input data (images, long sequences, sensor streams)
- You cannot hand-engineer good features
- Large amounts of data available (1K+ examples minimum)
- Compute budget allows (GPUs available)

**Choose Classical ML when:**
- Tabular/structured data with <10K rows
- Feature engineering is feasible and interpretability matters
- Limited data (<1K examples)
- Inference latency/compute is critical and models must be tiny

**Real-world example:** In computational materials science, DFT provides quantum mechanical features that are hard to engineer — exactly where ML surrogates (classical RF or DL) provide 60-70% speedup.

---

## Part 2: Neural Networks from First Principles

### The Artificial Neuron — One Unit

A single neuron computes:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = Wx + b
a = σ(z)  where σ is activation function
output = a
```

- **Weights (W):** Learned parameters — control how much each input contributes
- **Bias (b):** Learned parameter — shifts the activation threshold
- **Activation function σ:** Non-linearity that enables learning of complex patterns

**Critical insight:** A single neuron (or single layer) computes a linear function. **Stacking non-linear layers is what gives neural networks their expressive power.**

### Why Depth Matters: Universal Approximation

**Universal Approximation Theorem:** A shallow network with enough neurons can approximate any continuous function.

**But depth is exponentially more efficient:**
- A function needing 2ⁿ neurons in one layer may need only n×d neurons across d layers
- Depth enables compositional representations: each layer builds on the last
- Example: Layer 1 learns edges, Layer 2 learns textures, Layer 3 learns shapes, Layer 4 learns objects

### Activation Functions: The Non-Linearity

| Function | Formula | Range | Key Characteristics |
|----------|---------|-------|---------------------|
| **Sigmoid** | 1/(1+e⁻ˣ) | [0,1] | Vanishes at extremes; use only for output probability |
| **Tanh** | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | [-1,1] | Zero-centered; still suffers vanishing gradient |
| **ReLU** | max(0,x) | [0,∞) | Fast, simple; dying ReLU problem (neurons stuck at 0 for x<0) |
| **Leaky ReLU** | max(αx,x), α≈0.01 | (-∞,∞) | Fixes dying ReLU; allows small negative gradient |
| **GELU** | x·Φ(x) | ≈ReLU | Smooth non-linearity; **default in BERT/GPT** |
| **Swish/SiLU** | x·σ(x) | (-∞,∞) | Non-monotonic; strong empirical performance |
| **Softmax** | eˣⁱ/Σeˣʲ | [0,1], Σ=1 | Output layer for multi-class classification |

**Why not sigmoid everywhere?**
- Sigmoid **saturates** for large |x|: derivative ≈ 0 → vanishing gradients in deep networks
- ReLU fixed this: gradient = 1 for x > 0, preventing gradient disappearance
- **GELU and SiLU dominate transformers** because they're smooth (better optimization) and non-monotonic (richer gradient signal)

---

## Part 3: Backpropagation & Training Fundamentals

### The Training Cycle: One Step

1. **Sample batch** (x_batch, y_batch) from DataLoader with shuffle=True
2. **Forward pass:** ŷ = model(x) with autocast for FP16/BF16 mixed precision
3. **Compute loss:** L = CrossEntropy(ŷ, y) or other loss function
4. **Backward pass:** loss.backward() computes ∂L/∂w via chain rule (automatic differentiation)
5. **Clip gradients:** Prevent exploding gradients by capping norm
6. **Update weights:** optimizer.step() moves weights in direction of negative gradient
7. **Repeat:** For every batch; one pass over full dataset = one epoch

### Backpropagation: Reverse-Mode Automatic Differentiation

Backpropagation applies the chain rule to compute ∂L/∂wᵢ for every parameter efficiently:

```
Chain rule: ∂L/∂w₁ = ∂L/∂aₙ · ∂aₙ/∂aₙ₋₁ · ... · ∂a₂/∂a₁ · ∂a₁/∂w₁
```

- Each term is a Jacobian: ∂f/∂x (how much output changes with input)
- **Vanishing gradients:** If each Jacobian is <1 (e.g., sigmoid in deep net), product → 0 exponentially
  - In 20-layer sigmoid network, gradients at layer 1 ≈ 0 — those weights don't learn
- **Exploding gradients:** If each Jacobian is >1, product → ∞ — updates become chaotic
  - Common in RNNs over long sequences

### Solutions to Gradient Problems

**Vanishing Gradient Solutions:**
1. **ReLU activations:** Gradient is 1 for x>0; doesn't inherently shrink (but still has dying ReLU issue)
2. **Residual connections:** x + F(x) has gradient 1 + ∂F/∂x. The "+1" provides direct gradient highway through all layers
3. **LSTM/GRU gating:** Cell state allows gradients to flow unchanged over long sequences
4. **Layer normalization:** Keeps activation statistics stable; enables training without warmup

**Exploding Gradient Solutions:**
1. **Gradient clipping:** Cap gradient norm at threshold τ: ‖g‖ ← min(‖g‖, τ) · g/‖g‖
2. **Weight regularization:** L2 penalty keeps weights small
3. **Lower learning rate:** Smaller update steps prevent divergence
4. **Proper initialization:** He/Xavier init prevents gradients being born too large

### Weight Initialization

**He Initialization** (for ReLU):
```
W ~ N(0, 2/n_in)  where n_in = number of input neurons
```
- Variance grows with input dimension to keep output variance constant
- Prevents gradient vanishing at initialization

**Xavier Initialization** (for tanh/sigmoid):
```
W ~ N(0, 2/(n_in + n_out))
```
- More conservative; suitable for saturating activations

---

## Part 4: Convolutional Neural Networks (CNNs)

### CNN Operations: Convolution

**Convolution Formula:**
```
Output size = (W - F + 2P) / S + 1
where W=input width, F=filter size, P=padding, S=stride
```

**Key properties:**
- **Parameter sharing:** Same filter applied everywhere → translation equivariance
- **Local connectivity:** Each neuron sees only a local receptive field
- **Critical for radar:** Convolutions on range-bin features capture spatial patterns

### Pooling: Reducing Dimensionality

**Max Pooling:**
- Extracts dominant features; translation invariant
- Takes maximum over a window (e.g., 2×2)
- Reduces spatial dimensions; keeps important signals

**Average Pooling:**
- Smoother; used in Global Average Pooling (GAP)
- Replaces FC layers → fewer parameters, less overfitting
- Used in modern architectures (MobileNet, EfficientNet)

### Receptive Field: How Much Context?

**Receptive field:** The input region affecting a neuron

```
For 1D-CNN on radar:
- Receptive field = how many time steps the model can integrate
- Grows with: depth of network + dilation
- Dilated convolution: skip inputs by dilation rate d → exponential RF growth without depth
```

**Example:** For UWB CIR sequences of length 256:
- 1D dilated conv: RF = 2^L for L layers with doubling dilation
- Parallelizable (unlike LSTM); good for edge deployment

### Depthwise Separable Convolution: Efficient Design

**Standard convolution:** N_in × N_out × K × K parameters (expensive)

**Depthwise separable:**
1. Depthwise: per-channel spatial convolution (1 filter per input channel)
2. Pointwise: 1×1 conv across channels

**Result:** 8-9× fewer parameters with similar accuracy

**Used in:** MobileNet, MobileNetV2 — relevant for **Algorized's edge deployment**

### Case Study: Tiny CNN (TyCNN) for Edge Deployment

**Architecture:**
```
Input (256,)
  ↓
BatchNorm
  ↓
Conv1D (3 layers with ReLU)
  ↓
Global Average Pooling (GAP)
  ↓
FC (output classes)
  ↓
Softmax
```

**Quantized to INT8:**
- Model size: <200KB
- Inference: <48ms on STM32
- Post-quantization accuracy: 98.22% (vs 99.38% baseline)

**Why this matters:** This is the production architecture for Algorized's edge deployment on UWB people counting.

---

## Part 5: Recurrent Networks: LSTM & GRU

### RNNs: The Problem

Standard RNN updates hidden state sequentially:
```
h_t = σ(W_h · [h_{t-1}, x_t] + b_h)
```

**Problems:**
- **Sequential:** Can't parallelize on GPU; T5 must wait for T1→T2→T3→T4→T5
- **Vanishing gradient:** h_t depends on h_{t-1}, which depends on h_{t-2}, etc. → exponential decay
- **Practical limit:** ~7-10 steps before gradients vanish

### LSTM: Long Short-Term Memory

LSTM solves vanishing gradient by introducing a **gated cell state** that can store information indefinitely.

**Four gates per timestep:**

1. **Forget gate:** f_t = σ(W_f·[h_{t-1}, x_t] + b_f)
   - Decides what to remove from cell state [0, 1]

2. **Input gate & candidate:** 
   - i_t = σ(W_i·[h_{t-1}, x_t] + b_i)
   - c̃_t = tanh(W_c·[h_{t-1}, x_t] + b_c)
   - New candidate values to add

3. **Cell update:** c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
   - Element-wise multiplication: forget old + input new
   - **Key:** Cell state can flow unchanged through time

4. **Output gate:** 
   - o_t = σ(W_o·[h_{t-1}, x_t] + b_o)
   - h_t = o_t ⊙ tanh(c_t)
   - Produce hidden state from cell state

**Why it works:** Cell state is updated via element-wise multiplication (⊙), not full matrix multiply. Gradient of element-wise ops is local — doesn't vanish exponentially.

### GRU: Gated Recurrent Unit

**Simplified LSTM:**
- Merges forget and input gates into update gate
- No separate cell state; hidden state only
- **25% fewer parameters** than LSTM
- Similar performance on most tasks

**Preferred when:** Compute budget is tight — relevant for edge sensor models

---

## Part 6: Transformer Architecture & Self-Attention

### The Problem Transformers Solve

**RNN sequential processing:**
```
T1 → T2 → T3 → T4 → T5
```
- O(L) sequential steps — can't parallelize on GPU
- Gradient path length = L → vanishing gradients over long sequences
- Max practical sequence: ~500 tokens

**Transformer parallel processing:**
```
T1 T2 T3 T4 T5 (all in parallel)
```
- O(1) "sequential" steps (all parallel)
- Gradient path from any token to any other = 1 step
- Handles 128K+ token contexts with efficient kernels (Flash Attention)

**The trade-off:**
- RNN: O(L) time, O(1) memory per step
- Transformer: O(1) sequential steps, O(L²) memory for attention matrix

### Scaled Dot-Product Attention

**How attention works:**

1. **Project inputs to Q, K, V:**
   ```
   Q = X · W^Q  ("what am I looking for?")
   K = X · W^K  ("what do I contain?")
   V = X · W^V  ("what I'll contribute")
   ```

2. **Score each position pair:**
   ```
   Scores = Q · K^T / √d_k  (normalize by √d_k to prevent softmax saturation)
   ```

3. **Convert to attention weights via softmax:**
   ```
   Weights = softmax(Scores)  (row-wise; each row sums to 1)
   ```

4. **Weighted sum of values:**
   ```
   Attn(Q,K,V) = softmax(Q·K^T/√d_k) · V
   ```

**Each token attends to ALL other tokens in parallel.**

### Why √d_k Scaling?

For random Q, K vectors, dot product variance grows as d_k. At d_k=64:
- Scores blow up → softmax saturates → gradient ≈ 0
- Dividing by √d_k keeps variance ≈ 1 → stable gradients

### Multi-Head Attention

**Single attention head:** One pattern of what to attend to

**Multi-head attention:** h parallel heads, each with different Q/K/V projections

```
head_i = Attention(Q_i, K_i, V_i)  for i=1..h
Output = Concat(head_1, ..., head_h) · W^O
```

**Each head specializes:**
- Head 1: syntax patterns
- Head 2: coreference resolution  
- Head 3: sentence structure
- Head 4+: other patterns

**Typical:** 8-16 heads; each head_dim = d_model / h

**Advanced: Grouped Query Attention (GQA)**
- Share K, V across groups of Q heads
- Reduces KV cache size by n_groups
- Used in LLaMA-2/3 for efficiency

### Positional Encoding: How Does Transformer Know Position?

Transformers have no inherent order — must add position information.

**Sinusoidal (original):**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```
- No learned parameters
- Works for any sequence length at inference
- Encodes position as high-frequency oscillations

**Learned embeddings:**
- Add learnable position embeddings directly
- Better empirically for some tasks
- Preferred for time-series sensor data (can capture domain-specific position patterns)

**RoPE (Rotary Position Encoding, LLaMA):**
- Rotate Q, K vectors by angle proportional to position
- Relative distances preserved in dot product
- Generalizes beyond training length with scaling

**ALiBi (Attention with Linear Biases, MPT):**
- No position embeddings
- Subtract linear bias from attention scores based on distance
- Strong extrapolation to longer sequences than training

### Transformer Block: The Repeating Unit

**Decoder-Only (GPT/LLaMA style) — Pre-Norm:**

```
x_0 = Embed(tokens) + PositionalEncoding(positions)

For each of N layers:
  ├─ RMSNorm(x)                          [normalize BEFORE sublayer]
  ├─ Multi-Head Self-Attention
  ├─ Residual connection: x ← x + attn_output
  │
  ├─ RMSNorm(x)                          [normalize before FFN]
  ├─ Feed-Forward Network (FFN)
  │  ├─ Linear(d → 4d)
  │  ├─ GELU / SiLU
  │  └─ Linear(4d → d)
  └─ Residual connection: x ← x + ffn_output

x_out ∈ ℝ^(L×d) → LM Head → Logits for next token
```

### Pre-Norm vs Post-Norm

**Post-Norm (original paper):** LayerNorm(x + Sublayer(x))
- Unstable at large scale
- Gradient depends on sublayer output which can vary wildly early in training
- Requires warmup

**Pre-Norm (modern):** x + Sublayer(LayerNorm(x))
- The residual stream x is never normalized out
- Gradients flow cleanly through identity connections
- Enables training without warmup in some regimes
- **Used in GPT-3, LLaMA, all modern LLMs**

### The FFN Layer: What Does It Do?

Mechanistic interpretability research (Geva et al., 2021) shows:

**FFN layers act as key-value memory:**
- First linear layer ("keys"): activates for specific input patterns
- Second linear layer ("values"): retrieves stored information
- **This is how LLMs recall facts without retrieval** — knowledge is stored in FFN weights

**Standard configuration:**
- d_ff = 4×d_model (4× expansion ratio)
- SwiGLU variant: gate(x)·up(x) — used in LLaMA/PaLM with 2/3 × 4 = 8/3 ratio

---

## Part 7: Modern Architectures for Time-Series & Sensor Data

### Temporal Convolutional Network (TCN)

**Design:**
- Stack of 1D dilated causal convolutions with residual connections
- Receptive field = 2^L for L layers with doubling dilation (1→2→4→8→...)
- **Parallelizable** (unlike LSTM)

**Advantages:**
- Strong baseline for UWB CIR sequences
- Much faster training than RNN-based models
- Can process entire sequence in parallel on GPU

**Use case:** For UWB radar CIR sequences of length 256, TCN is the go-to choice.

### Hybrid: CNN + LSTM (HDL4AR)

**Architecture:**
```
Input Sequence
  ↓
1D-CNN Encoder (per-frame feature extraction)
  ↓
LSTM (temporal modeling)
  ↓
Classification Head
```

**Why hybrid:**
- CNN extracts spatial patterns within each frame
- LSTM captures temporal evolution across frames
- Published on UWB HAR (Human Activity Recognition)

### Transformer for Time Series (TST)

**Modern approaches:**
- **PatchTST:** Patch-based tokenization (divide time series into sub-sequences)
- **TimesNet:** Multi-scale temporal patterns

**Advantages:**
- Higher accuracy than CNN/LSTM on long sequences
- Captures complex temporal dependencies

**Disadvantage:** Heavier compute; O(L²) memory still an issue

### State Space Models (S4, Mamba)

**Emerging alternative to attention:**
- Linear recurrence with special parameterization for long-range dependencies
- S4: O(n log n) complexity
- Mamba: O(n) linear complexity
- Potentially suited for streaming radar inference with minimal compute

---

## Part 8: Training Stability & Normalization Techniques

### Normalization Methods: Which Dimension?

| Technique | Normalizes Over | Use When |
|-----------|-----------------|----------|
| **BatchNorm** | Batch dimension (N) | CNNs, large batch sizes (>8) |
| **LayerNorm** | Feature dimension (D) | Transformers, RNNs, any batch size |
| **InstanceNorm** | Spatial dims per sample | Style transfer, image generation |
| **GroupNorm** | Channel groups per sample | Object detection, small batches |
| **RMSNorm** | RMS (skip mean centering) | LLaMA, Mistral — 10-20% faster |

### BatchNorm vs LayerNorm

**BatchNorm:**
```
μ, σ computed across batch dimension
BN(x) = (x - μ_batch) / √(σ²_batch + ε)
```
- Great for CNNs with large batches
- **Avoid for:** RNNs (batch stats undefined at each step), inference with batch_size=1

**LayerNorm:**
```
μ, σ computed across feature dimension (per sample)
LN(x) = (x - μ_sample) / √(σ²_sample + ε)
```
- **Batch-size independent** — works with any batch size
- Variable sequence length support
- **Standard for Transformers and RNNs**

**RMSNorm (LLaMA optimization):**
```
RMSNorm(x) = x / RMS(x) · γ
RMS(x) = √(1/d · Σxᵢ²)
```
- Skip mean centering (mean subtraction not critical)
- 10-20% faster than LayerNorm
- Same quality
- **Used in LLaMA, Mistral, Qwen — modern LLMs**

### Batch Normalization: Benefits & When Not to Use

**Benefits:**
- Reduces internal covariate shift
- Enables higher learning rates (more stable gradients)
- Mild regularization effect (stochastic due to batch statistics)

**When to AVOID:**
- Small batch sizes (<8): batch statistics unreliable
- RNNs: batch stats undefined at each time step
- Online/streaming inference: batch size=1, can't compute batch statistics
- **Use LayerNorm instead in all these cases**

---

## Part 9: Loss Functions & Activation Combinations

### Loss Functions by Task

**Classification (softmax + cross-entropy):**
```
CrossEntropy(ŷ, y) = -Σ y_i · log(ŷ_i)
```
- Output: softmax probabilities
- Numerically stable implementation: subtract max before exp

**Regression (MSE):**
```
MSE = (1/n) Σ (ŷ_i - y_i)²
```
- Penalizes large errors heavily
- Use MAE for outlier-robust regression

**Binary classification (sigmoid + binary cross-entropy):**
```
BCE = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```
- Output: sigmoid probability [0,1]
- Balanced if classes balanced; use class weights if imbalanced

### Proper Output Layer Design

| Task | Output Activation | Loss Function |
|------|------------------|----------------|
| **Multi-class classification** | Softmax | Cross-Entropy |
| **Binary classification** | Sigmoid | Binary Cross-Entropy |
| **Multi-label classification** | Sigmoid (each output independent) | Binary Cross-Entropy per label |
| **Regression** | Linear (none) | MSE / MAE / Huber |
| **Ordinal regression** | Softmax over ordered classes | Cross-Entropy with ordinal structure |

---

## Part 10: Reading Loss Curves & Diagnostics

### Four Common Training Patterns

**✓ Healthy:** Training and validation loss both ↓, small gap
- Model is learning; normal generalization gap
- Keep training until plateau

**✗ Overfitting:** Training loss ↓, validation loss ↑
- Model memorizing training data
- Fix: dropout, L2 regularization, more data, data augmentation, early stopping

**✗ Underfitting:** Both losses plateau high
- Model too simple or learning rate too low
- Fix: bigger model, higher learning rate, remove regularization, more features

**✗ Learning rate too high:** Loss chaotic or NaN
- Updates too large, overshooting minima
- Fix: reduce LR by 10×, add warmup, check gradient clipping

---

## Part 11: Five Training Paradigms

### 1. **Pretraining** — Train from scratch on massive unlabeled data

**When:** You have 1T+ tokens, want foundation model

**Process:**
- Unsupervised objective: Next Token Prediction (NTP) or Masked Language Modeling (MLM)
- Weeks of training, $M+ compute

**Example:** Training LLaMA from scratch on web-scale data

### 2. **Full Fine-Tuning** — Update ALL parameters

**When:** >10K labeled examples, full GPU memory available, task-specific accuracy critical

**Process:**
```
Learning rate: 1e-5 to 5e-5 (much lower than pretraining)
Epochs: 2-5
Risk: catastrophic forgetting (model forgets general knowledge)
```

**Advantage:** Max task-specific performance

**Disadvantage:** Requires full model GPU RAM; expensive

### 3. **LoRA / PEFT** — Low-Rank Adaptation

**When:** 1K-100K labeled examples, limited GPU memory, need to swap adapters

**Process:**
```
Freeze base model
Add low-rank adapters: ΔW = B·A where rank r << d
Update only adapters (0.1-3% of base model params)
```

**Example:** Fine-tune 70B LLaMA on 2×A100 with QLoRA (4-bit base + BF16 adapters)

**Advantages:**
- Tiny compute (1-3% params)
- No catastrophic forgetting (base frozen)
- Swap adapters per use-case

**Disadvantage:** Slightly below full FT quality

### 4. **SFT + DPO** — Instruction Tuning + Alignment

**When:** Build chatbots, assistants; need (prompt, chosen, rejected) preference data

**Process:**
- **SFT:** Supervised Fine-Tuning on (prompt, ideal-response) pairs — teaches following instructions
- **DPO:** Direct Preference Optimization on preference data — aligns model to user preferences
- No explicit reward model needed (unlike PPO)

**Advantage:** Teaches model to follow instructions AND align to preferences

**Disadvantage:** Needs preference-labeled data

**DPO Loss:**
```
-log σ(β·(log π_θ(y_w|x)/π_ref − log π_θ(y_l|x)/π_ref))
β=0.1-0.5 (KL penalty to prevent collapse)
y_w = chosen response
y_l = rejected response
```

### 5. **Continual Pretraining (PT) + SFT** — Domain-specific LLMs

**When:** Want to specialize on domain data (medical, legal, code)

**Process:**
1. Continue NTP on domain-specific corpus
2. Then SFT on task data
3. Better domain adaptation than pure fine-tuning

**Example:** Medical LLMs pre-trained on medical literature, then SFT on clinical QA

---

## Part 12: Optimizers & Learning Rate Schedules

### Optimizer Family Tree

**SGD + Momentum**
```
v_t = β·v_{t-1} - α·∇L
θ ← θ + v_t
```
- Accelerates in consistent directions
- Best final accuracy for CNNs (with proper LR schedule)

**Adam**
```
m̂ = corrected 1st moment (momentum)
v̂ = corrected 2nd moment (adaptive per-param)
θ ← θ - α·m̂/(√v̂ + ε)
```
- **Default for most deep learning**
- Adapts learning rate per parameter
- Robust across different problems

**AdamW** (Modern default)
```
Same as Adam, THEN: θ ← θ - α·λ·θ  (decoupled weight decay)
```
- **Preferred for Transformers/LLMs**
- Separates weight decay from gradient-based update
- Better generalization than vanilla Adam

### Learning Rate Schedules

**Warmup + Cosine (Transformers/LLMs):**
```
LR increases linearly for first 5-10% of training
Then decreases following cosine curve to near zero
```
- Crucial for Transformers
- Avoids unstable early training

**OneCycleLR (CNNs):**
```
LR increases to peak, then decreases
Completes in single epoch or few epochs
```
- Fast convergence for CNNs
- Good for image classification

**ReduceOnPlateau (Adaptive):**
```
Monitor validation loss
Reduce LR by factor 0.1 if no improvement for N epochs
```
- Adaptive; responds to actual training dynamics
- Good for exploratory work

**Cosine Restarts (SGDR):**
```
Cosine annealing with periodic warm restarts
Helps escape local minima
Useful for ensemble snapshots
```

---

## Part 13: Interview Q&A Bank — Deep Learning

**Q: What's the difference between AI, ML, and Deep Learning? How do they relate?**

AI is the broadest category — any technique making machines appear intelligent, including rule-based expert systems and search algorithms. Machine learning is a subset of AI where the system learns patterns from data rather than following hand-crafted rules. Deep learning is a subset of ML that specifically uses multi-layer neural networks to learn hierarchical representations, which works especially well for high-dimensional data like images, text, and audio.

- **Practical distinction:** You choose DL over classical ML when you can't hand-engineer good features
- **When NOT to use DL:** Tabular data with <10K rows — XGBoost/LightGBM almost always wins

**Q: Explain the vanishing gradient problem and three solutions.**

Backpropagation computes gradients via the chain rule — product of Jacobians at every layer. If each Jacobian is <1 (sigmoid saturation), product shrinks exponentially with depth.

Solutions:
1. **ReLU activations:** Gradient is 1 for x>0; doesn't inherently shrink
2. **Residual connections:** x + F(x) has gradient 1 + ∂F/∂x. The "+1" is a direct highway
3. **LSTM/GRU gating:** Cell state allows gradients to flow unchanged over long sequences

**Q: Why does Batch Normalization help training, and when would you NOT use it?**

BN reduces internal covariate shift, allows higher LR, mild regularization effect.

Avoid for:
- Small batches (<8): batch statistics unreliable
- RNNs: batch stats undefined at each step (use LayerNorm)
- Online/streaming inference: batch size=1 (use LayerNorm)

**Q: You're designing a model for UWB radar CIR sequences of length 256. What architecture would you choose?**

Start with **TCN** (temporal convolutional network): dilated causal convolutions are parallelizable, have good receptive field, edge-deployable. Compare to 1D-CNN+GRU hybrid. Avoid full Transformer unless compute budget allows.

Key: maintain **causal** (non-lookahead) structure for real-time inference.

**Q: Explain how the Transformer works. Why did it replace RNNs?**

Transformers process all tokens in parallel using self-attention. Each token produces Q, K, V projections. Attention score = softmax(Q·K^T/√d_k)·V. This is a weighted average of values where weights are learned attention scores.

Why it replaced RNNs:
- RNNs sequential: T1→T2→T3→T4→T5 can't parallelize
- RNNs vanishing gradient: gradient path = L (exponential decay)
- Transformers parallel: all tokens process simultaneously; gradient path = 1 (any two tokens)

Trade-off: RNNs O(L) time O(1) memory; Transformers O(1) steps O(L²) memory

**Q: How would you debug a model that trains fine but produces poor validation accuracy from epoch 1?**

Classic overfitting: model too large, or data leakage.

Check:
- Dataset size vs. model parameters
- Train/val split integrity (no data in both sets)
- Augmentation applied only to train
- No target leakage in features (future info used as features)
- Class imbalance on validation set

**Q: Explain dropout and why you disable it at inference time.**

Training: randomly zero activations with probability p to prevent co-adaptation.

Inference: disable dropout AND scale outputs by 1/(1-p) (or equivalently, train with prob p and use all neurons at test time).

**Why disable:** Forgetting to disable causes stochastic, noisy predictions. With dropout enabled, every forward pass is slightly different.

**Q: What is the difference between model parameters, hyperparameters, and architecture choices?**

- **Parameters:** Learned (weights, biases) — updated during training
- **Hyperparameters:** Set before training (learning rate, batch size, regularization λ)
- **Architecture:** Structural choices (layer count, activation type, skip connections) — can be searched via NAS

---

## Summary: Key Takeaways

1. **Single neurons → Depth:** Non-linearity enables learning; depth enables efficiency
2. **Vanishing gradients:** ReLU, residuals, normalization, proper initialization all help
3. **Backpropagation:** Reverse-mode AD via chain rule; efficient but susceptible to gradient explosion/vanishing
4. **CNNs:** Parameter sharing + local connectivity → good for images, sensor data
5. **RNNs → LSTMs:** Gated cell state solves vanishing gradient for sequences
6. **Transformers:** Self-attention enables parallelism; O(L²) memory trade-off worth it for scaling
7. **Training stability:** Normalization (LayerNorm for modern), proper LR schedules, gradient clipping
8. **Five paradigms:** Pretraining → Full FT vs LoRA vs SFT+DPO vs Continual PT — choose by data/compute
9. **Optimizers:** AdamW default; SGD+Momentum best final accuracy for CNNs
10. **LR schedules:** Warmup+cosine for Transformers; OneCycleLR for CNNs
