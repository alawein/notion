Deep Learning Reference — Interview Prep Deep Learning · Interview Prep Transformer, Training & Pipeline Reference Meshal Alawein · PhD EECS Berkeley Architecture DL Architecture Map Building Blocks Transformer Deep Dive Attention Mechanism Training Workflows 5 Types of Training Full Training Lifecycle Optimizers & Schedules Data & Pipelines Datasets — Types & Sources Data Collection & Quality Full ML Pipeline Tools & Deployment HuggingFace Ecosystem Install & Environment Problems & Fixes Deployment & Serving Interview Q&A Technical Q&A Deep Dive Focused Deep Learning Reference · 2026 
# Transformers, Training 
& *Production Pipelines *
Architecture diagrams, training workflow types, dataset strategies, tooling installation, common failure modes and fixes. Interview-ready depth. Transformer Architecture 5 Training Workflows RLHF / DPO / SFT Failure Modes & Fixes HuggingFace Ecosystem Deployment & Serving 01 Deep Learning Architecture Map Taxonomy ◉ Deep Learning ├── Feedforward / MLP // universal approximation; tabular data baseline │ ├── Input → [Linear → Activation]ᴺ → Output │ └── Activations: ReLU (default), GELU (transformers), SiLU (LLaMA) ├── Convolutional (CNN) // spatial hierarchy; translation equivariance │ ├── Standard: LeNet → VGG → ResNet → EfficientNet │ ├── Depthwise separable: MobileNet (edge-friendly, 8–9× fewer FLOPs) │ └── Key: conv → pool → BN → activation; residual connections for depth ├── Recurrent (RNN/LSTM/GRU) // sequential; variable-length; stateful │ ├── Vanilla RNN: vanishing gradient problem for long sequences │ ├── LSTM: 4 gates (forget/input/output/cell state) — long memory │ └── GRU: 2 gates (reset/update) — 85% LSTM perf, 70% parameters ├── Transformer // attention; parallelism; scales with data + compute │ ├── Encoder-only (BERT): bidirectional; classification, embeddings, NER │ ├── Decoder-only (GPT, LLaMA, Gemma): causal; generation; LLMs │ ├── Encoder-Decoder (T5, BART): seq2seq; translation, summarization │ └── Vision (ViT): patch embeddings; image classification; no convolutions ├── Generative Models // learn data distribution; sample new examples │ ├── VAE: encoder compresses to latent μ,σ; reparameterization trick; ELBO loss │ ├── GAN: generator vs discriminator adversarial game; mode collapse risk │ ├── Diffusion: forward noise process + learned denoising; DDPM, DDIM, stable diffusion │ └── Normalizing Flows: invertible transforms; exact likelihood; slow ├── Graph Neural Networks (GNN) // graph-structured data; relations │ ├── GCN: spectral conv on graphs; assumes fixed graph structure │ ├── GAT: attention-weighted aggregation; dynamic importance │ └── GraphSAGE: inductive; samples neighbor subgraphs; scales to large graphs └── State Space Models (SSM) // linear recurrence; linear-time attention alternative ├── Mamba (2023): selective SSM; O(L) vs O(L²); competes with transformers on long contexts └── Mamba-2: structured state space duality; better hardware utilization When to Use Each Architecture Architecture Input Type Strengths Weaknesses Production Use MLP Tabular, embeddings Simple, fast, interpretable Doesn't exploit spatial/temporal structure Recommendation heads, tabular baselines CNN (ResNet/Eff) Images, spectrograms Parameter efficient; translation-invariant; proven Fixed-size receptive field; not global Vision classification, feature extraction LSTM/GRU Sequences, time-series Variable-length; streaming inference; small models Sequential (not parallelizable); gradient issues Edge devices, streaming prediction Transformer Text, images, audio Global context; parallelizable; scales O(L²) memory; needs lots of data; large LLMs, BERT-style encoders, ViT Diffusion Images, audio, video High-quality generation; stable training Slow inference (many denoising steps); large Image generation, audio synthesis GNN Graphs, molecules Handles relational data natively Over-smoothing at depth; complex pipelines Drug discovery, recommendation, social Mamba Long sequences Linear time; long context; memory efficient Less mature ecosystem; fewer pretrained models Long document, genomics, time-series 02 Building Blocks — Definitions & Diagrams Core Concepts Module 1 
### Linear Layer (Fully Connected / Dense) 

Learns a linear mapping: **y = Wx + b **. W ∈ ℝ d_out × d_in , b ∈ ℝ d_out . Parameters: d_in × d_out + d_out. Every output neuron connects to every input neuron. 
- **Initialization matters: **Xavier for tanh/sigmoid (var = 2/(fan_in+fan_out)). Kaiming/He for ReLU (var = 2/fan_in). Wrong init → vanishing/exploding gradients at layer 1. 
- **Bias term: **Omit when using BatchNorm (BN has its own shift parameter γ, making bias redundant and wasteful). 
Module 2 
### Activation Functions — Why Non-Linearity Matters 

Without activations, stacking linear layers = still a single linear layer. Non-linearities let the network approximate arbitrary functions (Universal Approximation Theorem). 
ReLU 
#### ReLU(x) = max(0, x) 

Default for hidden layers. Gradient = 1 for x>0, 0 otherwise. Dead neurons if most inputs are negative. Fix: Leaky ReLU, PReLU. 
GELU 
#### GELU(x) = x·Φ(x) 

Used in BERT, GPT, ViT. Smooth, probabilistic gating. Outperforms ReLU on NLP tasks. Slightly slower to compute. 
SiLU / Swish 
#### SiLU(x) = x·σ(x) 

Used in LLaMA, PaLM. Self-gated. Non-monotonic (can output negative values). Better gradient flow than ReLU. 
Module 3 
### Normalization Layers 

Stabilize activations during training. Prevent internal covariate shift. Enable higher learning rates. Critical for training deep models. 
- **Batch Normalization: **Normalizes over the batch dimension. Mean/var computed per feature across samples. Has running statistics for inference. Problem: small batch sizes → noisy estimates. Doesn't work on RNNs (variable-length sequences). Formula: ŷ = γ(x − μ B )/σ B + β 
- **Layer Normalization: **Normalizes over the feature dimension per sample. Works for any batch size (even batch=1). Standard in transformers. Consistent behavior between train and inference. 
- **RMSNorm: **Simplified LayerNorm — removes mean subtraction. Used in LLaMA, Mistral. 10–20% faster, equivalent performance. 
- **Group Norm: **Normalizes over groups of channels. Used in vision (detection, segmentation) where batch size is small per GPU. 
Norm Type Normalizes Over Use In Batch Size Sensitivity BatchNorm Batch × spatial dims CNNs for image tasks High — needs large batches LayerNorm Feature dim (per sample) Transformers, NLP None — per-sample RMSNorm Feature dim (no centering) LLaMA, Mistral, Gemma None GroupNorm Channels within groups Detection, small-batch CV Low Module 4 
### Residual / Skip Connections 

The single most important architectural innovation for training deep networks. Instead of learning H(x), the layer learns F(x) = H(x) − x (the residual). The output is x + F(x). 
- **Why it works: **Gradient flows directly through skip connection: ∂L/∂x = ∂L/∂(x+F) = ∂L/∂(x+F) · (1 + ∂F/∂x). The "+1" term prevents gradient vanishing regardless of depth. 
- **Identity initialization: **At init, F(x) ≈ 0, so the block starts as an identity. Adding capacity is incremental. Easier optimization landscape. 
- **Enabling depth: **Without residuals: practical limit ~20 layers (VGG). With residuals: ResNet-1000 trains stably. Pre-norm transformers use it in every block. 
Module 5 
### Dropout 

During training, randomly zero neurons with probability p (typically 0.1–0.5). Prevents co-adaptation. Forces redundant representations. 
- **During inference: **Dropout is OFF. Weights are scaled by (1−p) to compensate for the missing activations — this is handled automatically by PyTorch. 
- **Inverted dropout: **Modern implementation scales activations by 1/(1−p) during training so no scaling needed at test time. 
- **Where to apply: **After linear layers, before the activation, in attention weights. NOT on batch normalization layers. 
- **MC Dropout: **Keep dropout on at inference for epistemic uncertainty estimation — run N forward passes, measure prediction variance as uncertainty proxy. 
Module 6 
### Embedding Layer 

A lookup table mapping discrete tokens (integer IDs) to dense vectors. E ∈ ℝ vocab_size × d_model . The ith row is the embedding for token i. Learned end-to-end with the task. 
- **Positional encoding: **Transformers have no inherent notion of order. Add positional information to embeddings. Sinusoidal (original), learned (BERT), RoPE (LLaMA, GPT-NeoX), ALiBi (extrapolates to longer sequences). 
- **Shared embeddings: **In LLMs, input embedding matrix and the output projection (lm_head) are often tied — same weights. Reduces parameters by vocab_size × d_model. 
- **Byte-Pair Encoding (BPE): **Most modern tokenizers (GPT, LLaMA) use BPE — subword tokenization that balances vocabulary size vs. sequence length. 
03 Transformer Architecture — Deep Dive Critical The Transformer (Vaswani et al., 2017) is the dominant architecture for LLMs, vision, audio, and multimodal AI. Every Senior DS must understand it at implementation depth — not just "attention is all you need." ◉ Complete Transformer Block (Decoder-only, Pre-Norm variant — LLaMA / GPT-4 style) Input Token IDs [1, 2, 7, 42 …] integer sequence ↓ Embedding Layer vocab_size × d_model ↓ Positional Encoding RoPE / ALiBi / Sinusoidal ↓ x₀ ∈ ℝ L × d_model sequence of embeddings × N Transformer Blocks RMSNorm / LayerNorm Pre-norm: normalize BEFORE attention ↓ Multi-Head Self-Attention Wᴼ·Q Query proj Wᴷ·K Key proj Wᵛ·V Value proj softmax(QKᵀ/√dₖ)·V Output Projection Wᴼ → concat heads ↓ + residual (x = x + attn_out) RMSNorm / LayerNorm second normalization ↓ Feed-Forward Network (FFN) Linear(d_model→d_ff) → SiLU/GELU → Linear(d_ff→d_model) d_ff = 4 × d_model (standard) or SwiGLU variant ↓ + residual (x = x + ffn_out) Output xₙ ∈ ℝ L × d_model after N blocks ↓ Final LayerNorm stabilize output ↓ LM Head (unembedding) Linear(d_model→vocab) ↓ Logits ∈ ℝ L × vocab one per token position ↓ softmax → probabilities or cross-entropy loss Pre-Norm vs. Post-Norm Original (Post-Norm) 
### x = LayerNorm(x + Sublayer(x)) 

Normalize AFTER adding the residual. Used in original "Attention is All You Need" paper. Training instability at large scale — requires careful warmup. GPT-2 and earlier models. 
Modern (Pre-Norm) — Preferred 
### x = x + Sublayer(LayerNorm(x)) 

Normalize BEFORE the sublayer. More stable training at large scale. Gradient flows unimpeded through the residual path. Used in GPT-3, LLaMA, Mistral, Gemma. Enables training without warmup in some cases. 
KV Cache — Why It Matters for Inference Inference Optimization 
### Key-Value Cache 

During autoregressive generation, for each new token, we recompute K and V for all previous tokens — wasteful. KV cache stores past K and V tensors and reuses them. Only compute K, V for the new token each step. 
- **Memory cost: **2 × n_layers × n_heads × seq_len × d_head × 2 bytes (FP16). For LLaMA-7B at seq=4096: ~2GB. This is why long contexts have high memory requirements. 
- **Multi-Query Attention (MQA): **Share one K, V head across all Q heads. Reduces KV cache by n_heads×. Used in Falcon, PaLM. 
- **Grouped Query Attention (GQA): **Share K, V across groups of Q heads. Used in LLaMA-2, Mistral. Balance between quality and cache size. 
Causal Masking (Decoder) ◉ Causal Attention Mask — prevents attending to future tokens Token: T1 T2 T3 T4 T5
 T1 [ 1 0 0 0 0 ] ← T1 can only see T1
 T2 [ 1 1 0 0 0 ] ← T2 can see T1, T2
 T3 [ 1 1 1 0 0 ] ← T3 can see T1–T3
 T4 [ 1 1 1 1 0 ]
 T5 [ 1 1 1 1 1 ] ← T5 sees all past tokens

Masked positions → -∞ before softmax → 0 attention weight
This is what makes language modeling self-supervised:
predict next token from past context only. 04 Attention Mechanism Core Algorithm ◉ Scaled Dot-Product Attention — Step by Step Input X 
L × d_model → Wᴼ 
→ Q Wᴷ 
→ K Wᵛ 
→ V ↓ Step 1: Project X into Q, K, V with learned weight matrices STEP 2: COMPUTE ATTENTION SCORES scores = Q · Kᵀ ∈ ℝᴸˣᴸ (all pairwise dot products) scaled = scores / √dₖ (prevent saturation) If causal decoder: add −∞ mask to upper triangle STEP 3: SOFTMAX → WEIGHTED SUM weights = softmax(scaled) ∈ ℝᴸˣᴸ, each row sums to 1 output = weights · V ∈ ℝᴸˣᵈₖ Each output token = weighted average of all value vectors FULL FORMULA Attention(Q,K,V) = softmax( QKᵀ / √dₖ ) · V Why √dₖ Scaling? Key Insight 
### Preventing Softmax Saturation 

Dot products Q·Kᵀ grow in magnitude as dₖ increases. For random vectors, E[qᵢ·kⱼ] ≈ 0 but Var[qᵢ·kⱼ] = dₖ. At large dₖ (e.g., 64), scores become large → softmax saturates → gradients ≈ 0. Dividing by √dₖ keeps variance ≈ 1 regardless of dₖ. 
Multi-Head Attention Multi-Head 
### Why Multiple Heads? 

Single attention head collapses all relationships into one attention pattern. Multi-head runs h parallel heads, each with its own projections (Wᵢᴼ, Wᵢᴷ, Wᵢᵛ), concatenates outputs, applies final projection Wᴼ. 
- **Each head d_k = d_model/h: **Same total compute, but different "views" of the input. Head 1 might track syntactic dependencies, head 2 coreference, etc. 
- **Typical values: **GPT-2 (12 heads, 64 d_k), LLaMA-7B (32 heads, 128 d_k), GPT-4 (96 heads estimated). 
- **MHA → MQA → GQA: **Full multi-head (1 KV per Q head) → Multi-Query (1 KV shared) → Grouped Query (1 KV per group). GQA is now standard for production LLMs. 
Flash Attention — IO-Aware Implementation Optimization 
### Flash Attention (Dao et al., 2022) 

Standard attention materializes the full L×L attention matrix — reading/writing it to GPU HBM memory multiple times. Flash Attention tiles the computation into blocks that fit in SRAM (fast), computing attention without ever materializing the full matrix. 
- **Result: **Same mathematical output. 2–4× faster. Uses O(√L) memory instead of O(L²). No approximation. 
- **Flash Attention 2: **Better parallelism, avoids redundant computation. 2× faster than v1. 
- **Usage: **`torch.nn.functional.scaled_dot_product_attention `in PyTorch 2.0+ uses Flash Attention automatically when available. 
05 5 Types of Training Workflows Workflows These are the five fundamentally different ways to train or adapt a deep learning model. Each has different data requirements, compute costs, and use cases. Knowing which to apply in which situation is a Senior DS core skill. ① PRETRAINING — Learn from Scratch on Massive Data Raw corpus 
1T–15T tokens → Tokenize 
BPE/SentencePiece → Random init 
Kaiming / scaled → Train: NTP loss 
AdamW + cosine LR → Base model 
capable but unaligned **Loss: **Cross-entropy next-token prediction. **Scale: **300B–15T tokens, weeks on thousands of GPUs. **Data: **CommonCrawl, GitHub, Wikipedia, books, arXiv. **Cost: **$2M–$100M+. **Who does this: **OpenAI, Google, Meta, Anthropic, Mistral. ② FULL FINE-TUNING — Adapt All Parameters on Task Data Pretrained 
Base or instruct model → Task dataset 
1K–1M labeled examples → Lower LR 
1e-5 to 5e-5 → Train all params 
full gradient flow → Task-specific 
all weights updated **When to use: **Enough labeled data (>10K), enough GPU memory (full model), max performance needed. **Risk: **Catastrophic forgetting of pretrained knowledge if LR too high. **Mitigate: **Lower learning rate, warm restarts, data mixing. ③ PEFT / LoRA — Efficient Fine-Tuning (freeze most; train adapters) Pretrained 
frozen weights → Add LoRA 
low-rank matrices A,B → Train only A,B 
0.1–3% params → Merge ΔW=AB 
into base model → Adapted model 
no inference overhead LoRA: W' = W + ΔW = W + B·A, where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ, r ≪ d **r (rank): **4–64. Higher r = more capacity but more params. r=8 is a good default. **Apply to: **Wᴼ, Wᴷ, Wᵛ, Wᴼᵘᵗ (attention projections). Optionally FFN layers. **Memory: **70B model fine-tunable on 2×A100 (vs. 8×A100 for full FT). ④ INSTRUCTION TUNING + RLHF — Align Model to Human Preferences Base Pretrained Model ↓ SFT (Stage 1) instruction-following pairs Data: (instruction, ideal response) pairs. 1K–100K examples. Human-written or GPT-4 distillation. Loss: cross-entropy on response tokens only (not instruction). SFT Model ↓ Reward Model (Stage 2) learn human preference Data: (prompt, chosen, rejected) triples. Human annotators rank model outputs. RM learns: score(preferred) > score(rejected). Bradley-Terry model. SFT + RM ↓ PPO / DPO (Stage 3) policy optimization PPO: maximize RM reward with KL penalty vs. SFT reference. DPO: skip RM entirely — train directly on (chosen, rejected) pairs. DPO is simpler and usually preferred now. DPO Loss: −𝔼 [ log σ( β · (log π_θ(y_w|x)/π_ref(y_w|x) − log π_θ(y_l|x)/π_ref(y_l|x)) ) ] ⑤ CONTINUAL PRETRAINING + DOMAIN ADAPTATION General LLM 
e.g., LLaMA-3 → Domain corpus 
10B–100B domain tokens → Continue NTP 
lower LR: 1e-4 to 1e-5 → Domain-adapted 
still has general knowledge → Then SFT 
task-specific behavior **Use case: **Medical LLMs (train on PubMed), code models (train on GitHub), scientific ML (train on arXiv + code). **Key risk: **Catastrophic forgetting. Mix domain data with 5–10% general data to preserve general capabilities. Use replay buffers. Choosing a Training Strategy Strategy When Data Needed Compute Forgetting Risk Pretraining No suitable base model exists 100B–15T tokens Extreme ($M+) N/A (from scratch) Full Fine-Tuning Large labeled dataset, max perf >10K task examples High (full model) High — need low LR LoRA / PEFT Limited compute, fast iteration 1K–100K examples Low (1–3% params) Low (frozen base) SFT only Good base model, instruction data 1K–50K pairs Low–Medium Low SFT + RLHF/DPO Alignment, safety, quality SFT + 10K–1M prefs High (reward model) Low (KL penalty) Continual PT Specialized domain, large corpus 1B–100B domain tokens High Medium — replay helps Few-shot prompting No training at all, prototype 0–32 in-context examples Minimal None 06 Full Training Lifecycle — End to End Workflow ◉ Complete ML Training Lifecycle 1. Problem 
Definition metric, baseline, ROI → 2. Data 
Collection sources, quality → 3. EDA & 
Preprocessing distributions, cleaning → 4. Model 
Selection architecture, baseline → 5. Training 
& Debug loss, overfit, LR → 6. Evaluation metrics, slicing → 7. Deploy 
& Monitor CI/CD, drift Step 5: Training — The Debugger's Checklist Most training failures are not model architecture problems. They are data, initialization, or optimization problems. Debug in this order. 
#### Symptom: Loss Not Decreasing 
- **Learning rate too low **→ increase 10× 
- **Learning rate too high **→ loss explodes or oscillates 
- **Vanishing gradients **→ check gradient norms 
- **Data labels are wrong **→ inspect 50 random samples 
- **Wrong loss function **→ CE for classification, MSE for regression 

#### Diagnostic Protocol 
- **Overfit one batch first **— 1 batch, 100 epochs → loss should → 0 
- **Log gradient norms **→ norm < 1e-5: vanishing; > 100: exploding 
- **LR range test **→ sweep LR 1e-7 to 1e-1, find max before loss spikes 
- **Visualize batch samples **→ confirm labels match data 
- **Check activation stats **→ dead neurons, saturation 
Reading Loss Curves ✓ Healthy Training train val epochs → Train and val loss both decrease. Val slightly higher — normal generalization gap. Converging together = healthy. ✗ Overfitting train ↓ val ↑ epochs → Train loss → 0, val loss rises. Diverging gap = overfitting. Fix: dropout, regularization, more data, early stopping. ✗ Underfitting (High Bias) both plateau high epochs → Both losses plateau at high values. Model capacity too small, or training too short. Fix: bigger model, more epochs, better features. ✗ LR Too High (Exploding) oscillating / NaN loss epochs → Chaotic, oscillating, or NaN loss. Fix: reduce LR by 10×. Add gradient clipping. Use warmup schedule. train_template.py — Battle-tested training loop pattern Python import torch, logging from torch import nn from torch.cuda.amp import GradScaler, autocast from torch.optim.lr_scheduler import CosineAnnealingLR class Trainer : def __init__ (self, model, train_dl, val_dl, cfg):
 self.device = 'cuda' if torch.cuda.is_available() else 'cpu' self.model = model.to(self.device)
 self.opt = torch.optim.AdamW(model.parameters(),
 lr=cfg.lr, weight_decay=cfg.wd)
 self.sched = CosineAnnealingLR(self.opt, T_max=cfg.epochs, eta_min=cfg.lr/ 100 )
 self.scaler = GradScaler() # automatic mixed precision self.train_dl, self.val_dl = train_dl, val_dl
 self.cfg = cfg
 self.best_val, self.patience_cnt = float ( 'inf' ), 0 def _step (self, batch, train= True ):
 x, y = [t.to(self.device) for t in batch] with autocast(): # FP16 forward pass logits = self.model(x)
 loss = nn.functional.cross_entropy(logits, y) if train:
 self.opt.zero_grad(set_to_none= True )
 self.scaler.scale(loss).backward()
 self.scaler.unscale_(self.opt)
 nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
 self.scaler.step(self.opt); self.scaler.update()
 acc = (logits.argmax( 1 ) == y).float().mean() return loss.item(), acc.item() def train (self): for epoch in range(self.cfg.epochs):
 self.model.train()
 tr = [self._step(b) for b in self.train_dl]
 self.model.eval() with torch.no_grad():
 va = [self._step(b, train= False ) for b in self.val_dl]
 tr_l, tr_a = [sum(x)/ len (x) for x in zip(*tr)]
 va_l, va_a = [sum(x)/ len (x) for x in zip(*va)]
 self.sched.step()
 logging.info( f"Ep {epoch:3d} | tr {tr_l:.4f}/{tr_a:.3f} | va {va_l:.4f}/{va_a:.3f}" ) # Checkpoint best model if va_l < self.best_val:
 self.best_val, self.patience_cnt = va_l, 0 torch.save(self.model.state_dict(), "best.pt" ) else :
 self.patience_cnt += 1 if self.patience_cnt >= self.cfg.patience:
 logging.info( "Early stopping triggered" ); break 07 Optimizers & Learning Rate Schedules Critical ◉ Optimizer Family Tree ◉ Gradient Descent ├── SGD // w ← w − η∇L (baseline, no memory) ├── SGD + Momentum // v ← βv − η∇L; w ← w + v (accelerates, smooths) ├── Nesterov Accelerated Gradient (NAG) // look-ahead gradient ├── Adaptive Methods // scale LR per-parameter │ ├── AdaGrad: sum squared grads → divide. Sparse features. Decays to 0. │ ├── RMSProp: exponential moving average of squared grads. Fixes AdaGrad decay. │ ├── Adam: RMSProp + momentum. m̂/(√v̂+ε). Default for most DL tasks. │ ├── AdamW: Adam + decoupled weight decay. Preferred for transformers. │ ├── LAMB: AdamW + layer-wise LR scaling. Large-batch training (batch >8K). │ └── Lion (2023): sign(momentum). Memory efficient, matches Adam on transformers. └── Second-Order // use Hessian or approximation └── L-BFGS: Newton-style. Classical ML fine-tuning. Not practical for large NNs. Key Distinction 
### Adam vs. AdamW — Why It Matters 

In Adam, L2 regularization adds λw to the gradient, which then gets divided by the adaptive scale √v̂. So weight decay is effectively λw/√v̂ — it scales with gradient history and doesn't apply equally across parameters. 

AdamW separates: w ← w − η·m̂/(√v̂+ε) − η·λ·w. The weight decay η·λ·w is applied directly to weights, independent of gradient history. This is mathematically correct decoupling. Always use AdamW when you want weight decay. 
LR Schedule Guide ◉ Learning Rate Schedule Comparison Cosine Annealing smooth decay → min_lr η = η_min + 0.5(η_max−η_min)(1+cos(πt/T)). Smooth, avoids sharp transitions. Use with SGDR warm restarts for ensemble snapshots. Warmup + Cosine Decay warmup | cosine decay Linear warmup (100–1000 steps) then cosine decay. Standard for transformers. Prevents early instability at high LR. OneCycleLR ↑ warmup | ↓↓ cool Up then down, fast and aggressive. Best for CNNs/tabular. Use `pct_start=0.3 `. Converges in fewer total steps. Rule of Thumb Training a transformer (LLM, BERT, ViT) → **AdamW + warmup + cosine decay **. Training a CNN from scratch → **SGD + momentum + OneCycleLR **. Fine-tuning a pretrained model → **AdamW, lower LR (1e-5 to 5e-5), cosine or constant **. 08 Datasets — Types, Sources & Issues Data ◉ Dataset Types ├── Pretraining Corpora // large, unlabeled, web-scale │ ├── CommonCrawl (80T tokens): web pages, noisy, filtered versions: C4, RefinedWeb, RedPajama │ ├── The Pile (825GB): GitHub, ArXiv, Wikipedia, Books, PubMed, StackExchange │ ├── Dolma (3T tokens): OLMo's training corpus, fully documented │ └── FineWeb (15T tokens, HuggingFace): filtered CommonCrawl, high quality ├── Instruction Tuning // (prompt, response) pairs │ ├── OpenHermes-2.5: GPT-4 synthetic, 1M examples, high quality │ ├── ShareGPT: real user-ChatGPT conversations │ ├── FLAN: Google's instruction tuning dataset, 1800+ tasks │ └── Alpaca: Stanford, 52K GPT-3.5 generated, cheap baseline ├── Preference / RLHF // (prompt, chosen, rejected) triples │ ├── Anthropic/hh-rlhf: 170K human preference pairs │ ├── OpenAssistant (OASST2): multilingual, 66K conversations │ └── UltraFeedback: GPT-4 scored, 64K responses from 4 models ├── Vision // image classification, detection, segmentation │ ├── ImageNet-1K: 1.2M images, 1000 classes. ILSVRC standard. │ ├── LAION-5B: 5B image-text pairs. Training data for CLIP, SD. │ └── COCO: 330K images, detection/caption/segmentation annotations ├── Code // source code for code LLMs │ ├── The Stack (BigCode): 6T tokens, 300+ languages │ └── CodeContests, APPS: competitive programming with solutions └── Scientific / Specialized ├── PubMed (biomedical): 30M abstracts ├── arXiv: 2M+ papers (physics, math, CS, biology) └── ChEMBL, QM9: molecular property prediction Dataset Quality Problems & How to Fix Them Problem Symptom Detection Fix Label noise Model learns wrong patterns; test accuracy ceiling Manual review of low-confidence predictions; inter-annotator agreement metrics (Cohen's κ) Confident Learning (cleanlab), re-annotation, noise-robust loss (symmetric CE) Class imbalance Model ignores minority class; high accuracy but bad recall Class frequency histogram Class weights, oversampling, focal loss, stratified splits Data leakage Validation accuracy unrealistically high; fails in production Check if test examples overlap with train; verify features are available at inference time Temporal split for time-series; GroupKFold; audit feature creation pipeline Distribution shift Model degrades over time in production PSI on features; monitor output distributions More diverse collection, temporal mixing, continuous retraining Near-duplicates Memorization; inflated metrics; val set contamination MinHash LSH deduplication; SimHash Fuzzy deduplication (datasketch, dedup tools in HF datasets) Toxicity / PII Model reproduces harmful content or private info Regex filters, classifier-based detection Block/redact PII; filter with toxicity classifiers; document removal Low diversity Good on seen topics, fails on edge cases Embedding clustering — check topic coverage Active learning; data augmentation; targeted collection Data Splits — Getting Them Right Critical 
### Split Strategy Depends on Data Structure 
- **Random split (IID data): **Shuffle → 70/15/15 train/val/test. Only valid when all examples are truly independent. 
- **Temporal split (time-series): **ALWAYS split by time. Train on past, validate on future. Never random split — that's leakage. Early examples train, later ones validate. 
- **Group split (correlated examples): **Same user/patient/document can't appear in both train and val. Use GroupKFold or group-stratified split. Otherwise model memorizes identity, not patterns. 
- **Holdout test set rule: **Test set is touched exactly ONCE — after all development is done. If you tune hyperparameters looking at test performance, your test set is now a second val set and you need a new test set. 
09 Data Collection & Quality Pipeline Practical ◉ Data Pipeline: Raw → Training-Ready Collect 
APIs, scraping, annotation, synthetic → Validate 
schema, range, nulls → Deduplicate 
exact + fuzzy → Filter Quality 
heuristics + classifiers → Preprocess 
tokenize, normalize → Store + Version 
parquet, DVC, HF datasets data_quality.py — Essential data quality checks for ML training Python import pandas as pd import numpy as np from datasketch import MinHash, MinHashLSH # pip install datasketch def quality_report (df: pd.DataFrame) -> dict: """Run all quality checks. Returns dict of issues.""" issues = {} # 1. Missing values null_rate = df.isnull().mean()
 issues[ 'high_null_cols' ] = null_rate[null_rate > 0.05 ].to_dict() # 2. Duplicate rows issues[ 'duplicate_rows' ] = df.duplicated().sum() # 3. Outlier detection per numeric column outliers = {} for col in df.select_dtypes(include=np.number):
 z = (df[col] - df[col].mean()) / df[col].std()
 outliers[col] = (np.abs(z) > 5 ).sum()
 issues[ 'outliers_per_col' ] = {k: v for k, v in outliers.items() if v > 0 } # 4. Class balance (if 'label' column exists) if 'label' in df.columns:
 counts = df[ 'label' ].value_counts(normalize= True )
 issues[ 'class_imbalance_ratio' ] = (counts.max() / counts.min()).round( 2 ) return issues def fuzzy_dedup_texts (texts: list[str], threshold= 0.8 , num_perm= 128 ) -> list[bool]: """Returns boolean mask: True = keep, False = near-duplicate.""" lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
 keep = [ True ] * len(texts) for i, text in enumerate(texts):
 m = MinHash(num_perm=num_perm) for word in text.lower().split():
 m.update(word.encode( 'utf-8' ))
 result = lsh.query(m) if result: # near-duplicate found → mark as duplicate keep[i] = False else :
 lsh.insert( str (i), m) return keep def psi (expected: np.ndarray, actual: np.ndarray, bins= 10 ) -> float: """Population Stability Index — detect distribution drift. PSI>0.2 = retrain.""" exp_pct = np.histogram(expected, bins=bins)[ 0 ] / len(expected) + 1e-8 act_pct = np.histogram(actual, bins=bins)[ 0 ] / len(actual) + 1e-8 return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))) 10 Full ML Pipeline Architecture Systems ◉ Production ML System — Complete Architecture DATA INGESTION LAYER Raw Sources databases, APIs, files, streams → Kafka / Pub-Sub stream ingestion → Validation Gate schema + quality checks → Feature Store versioned, low-latency TRAINING & EXPERIMENTATION LAYER Data Loader PyTorch Dataset/DataLoader → Training Job GPU cluster, DDP → Experiment Tracking MLflow / W&B → Model Registry versioned artifacts SERVING & MONITORING LAYER Model Registry production model → Inference Server TorchServe / vLLM / Triton → API Gateway rate limiting, auth → Monitoring drift, latency, errors ↺ Retrain Trigger PSI > 0.2, acc drop PyTorch DataLoader — Best Practices dataloader.py — Efficient DataLoader patterns Python from torch.utils.data import Dataset, DataLoader import torch import numpy as np class TabularDataset (Dataset): def __init__ (self, X: np.ndarray, y: np.ndarray):
 self.X = torch.from_numpy(X).float()
 self.y = torch.from_numpy(y).long() def __len__ (self): return len(self.X) def __getitem__ (self, idx): return self.X[idx], self.y[idx] # ── Production DataLoader config ────────────────────────────────────────── train_loader = DataLoader(
 train_dataset,
 batch_size= 256 ,
 shuffle= True , # shuffle every epoch num_workers= 8 , # parallel CPU data loading (match CPU cores) pin_memory= True , # pin to page-locked RAM → faster GPU transfer persistent_workers= True , # keep worker processes alive between epochs drop_last= True , # drop incomplete last batch (stabilizes BatchNorm) prefetch_factor= 2 , # prefetch 2 batches per worker ) # ── For large datasets that don't fit in RAM: IterableDataset ───────────── class StreamingDataset (torch.utils.data.IterableDataset): def __init__ (self, file_paths):
 self.paths = file_paths def __iter__ (self):
 worker_info = torch.utils.data.get_worker_info()
 paths = self.paths if worker_info: # split files across workers paths = paths[worker_info.id::worker_info.num_workers] for path in paths: for item in self. _read_file (path): yield item 11 HuggingFace Ecosystem Tooling ◉ HuggingFace Library Map huggingface/ // the ML ecosystem hub ├── transformers // 300K+ pretrained models, unified API │ ├── AutoModel, AutoTokenizer: load any model by name string │ ├── Trainer: full training loop, logging, checkpointing, evaluation │ └── Pipeline API: zero-code inference for common tasks ├── datasets // Arrow-backed, streaming, memory-mapped │ ├── load_dataset('name'): 50K+ public datasets, or load from files │ ├── Streaming: no full download needed — iterate on-the-fly │ └── map(): apply preprocessing in parallel (multiprocessing, Arrow backend) ├── peft // parameter-efficient fine-tuning │ ├── LoraConfig: specify rank r, target modules, alpha │ ├── get_peft_model(): wrap any model with PEFT adapters │ └── Supports: LoRA, QLoRA, Prefix Tuning, IA³, Adapters ├── trl // transformer reinforcement learning │ ├── SFTTrainer: supervised fine-tuning with instruction masking │ ├── DPOTrainer: direct preference optimization │ └── PPOTrainer: full RLHF with reward model ├── accelerate // distributed + mixed-precision, zero code change │ ├── accelerate config → auto-detects hardware (multi-GPU, TPU, CPU) │ └── DeepSpeed integration: ZeRO-1/2/3 for very large models ├── optimum // export + optimization │ ├── ONNX export with dynamic axes │ ├── Intel OpenVINO, NVIDIA TensorRT, ONNX Runtime backends │ └── BetterTransformer: kernel fusion for faster CPU inference └── hub // model + dataset hosting, versioning └── push_to_hub() / pull: git-based model versioning with LFS Core Usage Patterns huggingface_patterns.py — Most important HuggingFace patterns Python from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments from datasets import load_dataset from peft import LoraConfig, get_peft_model, TaskType from trl import SFTTrainer, DPOConfig, DPOTrainer import torch # ── 1. Load any model + tokenizer ───────────────────────────────────────── model_id = "meta-llama/Meta-Llama-3-8B" # any HF model ID tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
 model_id,
 torch_dtype=torch.bfloat16, # bfloat16 for Ampere+ GPUs device_map= "auto" , # auto-shard across available GPUs attn_implementation= "flash_attention_2" , # Flash Attention 2 if available ) # ── 2. LoRA for efficient fine-tuning ───────────────────────────────────── lora_cfg = LoraConfig(
 task_type=TaskType.CAUSAL_LM,
 r= 16 , # rank — higher = more capacity, more params lora_alpha= 32 , # scaling: effective lr = lr * lora_alpha / r target_modules=[ "q_proj" , "k_proj" , "v_proj" , "o_proj" ], # attention projections lora_dropout= 0.05 ,
 bias= "none" ,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters() # → trainable params: 13,631,488 || all params: 8,044,261,376 || 0.17% # ── 3. SFT Fine-tuning with TRL ─────────────────────────────────────────── dataset = load_dataset( "HuggingFaceH4/ultrachat_200k" , split= "train_sft" )

sft_trainer = SFTTrainer(
 model=model,
 tokenizer=tokenizer,
 train_dataset=dataset,
 dataset_text_field= "messages" , # column with instruction-response pairs max_seq_length= 2048 ,
 args=TrainingArguments(
 output_dir= "./outputs/sft" ,
 per_device_train_batch_size= 4 ,
 gradient_accumulation_steps= 4 , # effective batch = 4*4 = 16 learning_rate= 2e-4 ,
 lr_scheduler_type= "cosine" ,
 warmup_ratio= 0.05 ,
 num_train_epochs= 3 ,
 fp16= True ,
 logging_steps= 10 ,
 save_strategy= "steps" ,
 save_steps= 500 ,
 report_to= "wandb" ,
 ),
)
sft_trainer.train() # ── 4. DPO for preference alignment ─────────────────────────────────────── pref_dataset = load_dataset( "HuggingFaceH4/ultrafeedback_binarized" , split= "train_prefs" ) # Format: {'prompt': str, 'chosen': list[dict], 'rejected': list[dict]} dpo_trainer = DPOTrainer(
 model=model,
 ref_model= None , # auto-creates frozen copy args=DPOConfig(
 beta= 0.1 , # KL penalty — higher = closer to SFT ref output_dir= "./outputs/dpo" ,
 per_device_train_batch_size= 4 ,
 learning_rate= 5e-7 , # very low LR for DPO num_train_epochs= 1 ,
 fp16= True ,
 ),
 train_dataset=pref_dataset,
 tokenizer=tokenizer,
)
dpo_trainer.train() 12 Install & Environment Setup Practical setup.sh — Complete ML environment from scratch bash # ── STEP 1: conda environment ───────────────────────────────────────────── conda create -n ml python=3.11 -y
conda activate ml # ── STEP 2: PyTorch with CUDA (match CUDA version to driver) ───────────── # Check CUDA version: nvidia-smi | head -1 pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Verify: python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)" # ── STEP 3: HuggingFace ecosystem ───────────────────────────────────────── pip install transformers datasets accelerate peft trl
pip install bitsandbytes # 4-bit / 8-bit quantization (QLoRA) pip install flash-attn --no-build-isolation # Flash Attention 2 (needs CUDA 11.6+) # ── STEP 4: Training utilities ──────────────────────────────────────────── pip install wandb mlflow # experiment tracking pip install hydra-core omegaconf # config management pip install einops # readable tensor operations # ── STEP 5: Data processing ─────────────────────────────────────────────── pip install pandas numpy scipy scikit-learn
pip install xgboost lightgbm # gradient boosting pip install datasketch # MinHash deduplication # ── STEP 6: Serving + deployment ────────────────────────────────────────── pip install onnxruntime-gpu # ONNX runtime inference pip install vllm # high-throughput LLM serving (PagedAttention) pip install fastapi uvicorn # REST API serving # ── STEP 7: Verify GPU access ───────────────────────────────────────────── python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
 print(f'GPU name: {torch.cuda.get_device_name(0)}')
 print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB')
" GPU Memory Requirements Guide Model Params FP32 BF16 4-bit (QLoRA) Min GPU for FT GPT-2 117M 0.5GB 0.25GB ~0.12GB Any (4GB+) LLaMA-3-1B 1B 4GB 2GB ~0.5GB 4GB VRAM LLaMA-3-8B 8B 32GB 16GB ~5GB 1× A100 (40GB) BF16; 1× RTX 3090 QLoRA LLaMA-3-70B 70B 280GB 140GB ~40GB 2× A100 80GB QLoRA; 8× A100 BF16 BERT-base 110M 0.44GB 0.22GB — Any (2GB+) ViT-B/16 86M 0.34GB 0.17GB — Any (2GB+) QLoRA Rule of Thumb QLoRA (4-bit base + LoRA adapters) trains a 7B model on a single 24GB GPU (RTX 3090). A 70B model on 2× A100 40GB. The base model weights are quantized to 4-bit (NF4 format) using bitsandbytes. Adapter weights stay in BF16. 13 Common Problems & Fixes Debugging Training Problems Problem Symptom Cause Fix NaN loss Loss becomes NaN after N steps LR too high, gradient explosion, data has NaN/Inf Clip gradients (max_norm=1.0), lower LR, check input data for NaN, use fp32 for sensitive ops Dead ReLUs Large portions of network output zero always LR too high early training, negative bias initialization Use Leaky ReLU, PReLU, or GELU; proper weight init; lower LR during warmup Vanishing gradient Early layers don't learn; loss stalls Sigmoid/tanh deep networks, no skip connections Use ReLU/GELU, add residual connections, gradient highway (LSTM, Transformer) Slow training GPU utilization <50% DataLoader bottleneck (CPU-bound), small batches Increase num_workers, pin_memory=True, persistent_workers=True, larger batch OOM (out of memory) CUDA out of memory error Batch too large, model too large, activation memory Reduce batch, gradient checkpointing, mixed precision (BF16), gradient accumulation Overfitting Train loss ↓, val loss ↑ (diverging) Too much model capacity relative to data Dropout, weight decay, data augmentation, early stopping, smaller model Underfitting Both train and val loss plateau high Model too small, LR too small, too few epochs Larger model, higher LR, more epochs, better features Catastrophic forgetting Fine-tuned model loses general capability LR too high, too many FT steps Lower LR (1e-5), LoRA (freeze base), mix task data with replay data Memory Optimization Toolkit memory_tricks.py — Reduce GPU memory without losing performance Python import torch from torch.utils.checkpoint import checkpoint_sequential # ── 1. Mixed Precision (BF16/FP16) — 2× memory, ~2× speed ──────────────── from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler() with autocast(dtype=torch.bfloat16): # prefer bfloat16 on Ampere+ output = model(x)
 loss = criterion(output, y) # BF16: same dynamic range as FP32 (8 exponent bits), less precision # FP16: better precision but needs GradScaler to prevent underflow # ── 2. Gradient Checkpointing — trade compute for memory ────────────────── # Don't store all activations — recompute during backward pass # ~2× slower backward, but O(√L) memory instead of O(L) model.gradient_checkpointing_enable() # HuggingFace models # OR for custom models: output = checkpoint_sequential(model_layers, segments= 4 , input=x) # ── 3. Gradient Accumulation — large effective batch on small GPU ────────── accumulation_steps = 8 for i, (x, y) in enumerate(dataloader): with autocast():
 loss = criterion(model(x), y) / accumulation_steps # normalize scaler.scale(loss).backward() if (i + 1 ) % accumulation_steps == 0 :
 scaler.unscale_(optimizer)
 torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0 )
 scaler.step(optimizer); scaler.update(); optimizer.zero_grad() # effective batch = batch_size * accumulation_steps (same as large batch) # ── 4. 4-bit Quantization (QLoRA) — 4× memory reduction ────────────────── from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
 load_in_4bit= True ,
 bnb_4bit_quant_type= "nf4" , # NormalFloat4 — optimal for normal distributions bnb_4bit_compute_dtype=torch.bfloat16, # upcast for computations bnb_4bit_use_double_quant= True , # quantize quantization constants (extra 0.5 bpw) )
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config) LLM-Specific Issues Issue Description Fix Hallucination Model generates plausible but false information confidently RAG (ground in retrieved docs), RLHF reward for abstaining, constrained generation, uncertainty calibration Reward hacking Model maximizes reward model score without being actually good (finds loopholes) KL penalty vs. SFT reference, diverse reward signals, human spot-checks, conservative β in DPO/PPO Repetition Model repeats phrases or gets stuck in loops Repetition penalty in sampling, temperature, top-p/top-k, n-gram blocking Mode collapse (fine-tuning) Model always outputs same response regardless of input LR too high; reduce LR, add more training data diversity, reduce epochs Context window overflow Input longer than max_seq_len Chunk with overlap, sliding window, use models with larger context (128K), ROPE scaling Slow generation (LLM) High latency per token KV cache, speculative decoding, vLLM PagedAttention, smaller model, 4-bit quantization 14 Deployment & Serving Production ◉ Model Deployment Path: PyTorch → Production Trained Model 
.pt / .safetensors → Quantize 
INT8 / INT4 / FP16 → Export 
ONNX / TorchScript → Compile 
TRT / TVM / ORT → Serve 
vLLM / Triton / FastAPI → Monitor 
latency / drift / errors Inference Serving Options Tool Best For Key Feature Throughput vLLM LLM serving (GPT, LLaMA, Mistral) PagedAttention: dynamic KV cache allocation, no fragmentation. OpenAI-compatible API. Highest (~10× vs naive) TGI (HuggingFace) HuggingFace model serving Continuous batching, tensor parallelism, flash attention, GPTQ support High Triton Inference Server Any model (multi-framework) Model ensemble, dynamic batching, concurrent model execution, gRPC/HTTP High (production-grade) TorchServe PyTorch models (non-LLM) Native PyTorch, multi-model, A/B testing built-in, model versioning Medium ONNX Runtime Cross-platform (including edge) Hardware-agnostic, INT8/FP16, CPU optimization (OpenBLAS, AVX512) Medium (best on CPU) FastAPI + raw PyTorch Custom logic, prototypes Full control, simple deployment, no abstraction overhead Low-Medium (no batching) serve.py — FastAPI inference server + ONNX export Python # ── Export to ONNX ──────────────────────────────────────────────────────── import torch
model.eval()
dummy_input = torch.randn( 1 , 128 ) # batch=1, feature_dim=128 torch.onnx.export(
 model, dummy_input, "model.onnx" ,
 export_params= True ,
 opset_version= 17 ,
 input_names=[ 'features' ], output_names=[ 'logits' ],
 dynamic_axes={ 'features' : { 0 : 'batch_size' }}, # dynamic batch dimension ) # ── FastAPI inference server ────────────────────────────────────────────── from fastapi import FastAPI from pydantic import BaseModel import onnxruntime as ort import numpy as np

app = FastAPI() # Load ONNX model with GPU provider (falls back to CPU) sess = ort.InferenceSession( "model.onnx" ,
 providers=[ 'CUDAExecutionProvider' , 'CPUExecutionProvider' ]) class Request (BaseModel):
 features: list[list[float]] @app.post ( "/predict" ) async def predict (req: Request):
 x = np.array(req.features, dtype=np.float32)
 logits = sess.run([ 'logits' ], { 'features' : x})[ 0 ]
 probs = np.exp(logits) / np.exp(logits).sum(axis=- 1 , keepdims= True ) return { "class" : int(probs.argmax(axis=- 1 )[ 0 ]), "confidence" : float(probs.max(axis=- 1 )[ 0 ])} # Run: uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 4 15 Technical Interview Q&A — Deep Dive Interview ARCH How would you decide between a CNN, LSTM, and Transformer for a sequence classification task? MID ▶ Framework - **Start with the data size: **<1K examples → LSTM (fewer params, less prone to overfit). 1K–100K → CNN or LSTM. >100K → consider Transformer. 
- **Sequence length: **Long sequences (>512) with global dependencies → Transformer. Short sequences → CNN or LSTM. Very long (>8K) → Mamba or Longformer (linear attention). 
- **Deployment target: **Edge device with <1MB budget → LSTM or TCN (both quantize well). Server → Transformer. CPU-only → LSTM or CNN (Transformers are memory-bound on CPU). 
- **CNN strengths: **Detects local patterns (n-grams for text, local motifs for time-series). Fast, parallelizable, quantization-friendly. TCN with dilated convolutions extends receptive field exponentially. 
- **LSTM strengths: **Streaming inference (process one token at a time, maintain state). Variable-length sequences. Doesn't need positional encoding. 
- **Transformer strengths: **Global context, pre-trainable on large corpora, transfer learning. Worth the complexity only if you can leverage pre-trained weights or have >100K examples. 
- **Practical answer: **Always start with the simplest baseline (logistic regression on mean-pooled features, then LSTM, then fine-tune BERT if needed). Justify complexity with empirical evidence, not preference. 
TRAIN What is gradient clipping and why is it critical for transformer training? MID ▶ Core Answer Gradient clipping rescales the gradient vector so its L2 norm doesn't exceed a threshold: if ‖g‖ > max_norm, then g ← g × max_norm/‖g‖. This prevents "gradient explosion" — where large gradients cause catastrophically large parameter updates that destroy learned weights. - **Why transformers specifically: **Attention scores can produce very sharp distributions (near one-hot), causing large gradients when those attend to wrong tokens. Also, depth (many layers) means gradients multiply together — can amplify. 
- **Typical value: **max_norm=1.0 is standard. LLMs often use 1.0. CNNs are more stable, less often needed. 
- **Gradient norm logging: **Log the gradient norm every N steps. If it's consistently hitting the clip threshold → LR may be too high. If it's never near the threshold → clipping is no-op (fine). 
- **Code: **`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) `— call after `loss.backward() `and before `optimizer.step() `. 
TRAIN Explain weight initialization — why does it matter and what strategies exist? HARD ▶ Core Answer Poor initialization causes vanishing or exploding activations and gradients from layer 1, before any learning occurs. The goal is to start with activations that have unit variance throughout the network so gradients flow stably from the start. - **Xavier (Glorot) initialization: **W ~ Uniform(−√(6/(n_in+n_out)), +√(6/(n_in+n_out))). Maintains unit variance for tanh/sigmoid activations. Default for linear layers without ReLU. 
- **Kaiming (He) initialization: **W ~ N(0, 2/n_in). Maintains variance for ReLU (which kills half the neurons). Default for any layer followed by ReLU. Most common in practice. 
- **Orthogonal initialization: **W is an orthogonal matrix. Preserves gradient norms exactly at initialization. Used in RNNs to prevent gradient issues at depth. 
- **Zero initialization problem: **Never initialize all weights to zero — symmetry breaking is impossible. All neurons receive identical gradients, learn identical features, network has no capacity benefit from width. 
- **LLM-specific (GPT-2 trick): **Scale residual projections by 1/√(2N) where N is the number of layers. Prevents residual stream from growing as O(√N) at initialization. Used in GPT-2, GPT-3, LLaMA. 
TRANSFORMER What is the difference between BERT and GPT? Why have decoder-only models won? MID ▶ Core Answer - **BERT (encoder-only, bidirectional): **Pre-trained with Masked Language Modeling. Every token attends to all other tokens. Excellent for understanding tasks (classification, NER, extractive QA) but cannot generate text. Max input context at pre-training was 512 tokens. 
- **GPT (decoder-only, causal): **Pre-trained with next-token prediction. Causal mask — each token only sees the past. Can generate arbitrarily long text autoregressively. Scales to much larger contexts (8K, 128K+). 
- **Why decoder-only won: **(1) Next-token prediction scales predictably — more data + more compute = better model. (2) The same architecture handles generation, classification, and reasoning via prompting. (3) Emergent few-shot/chain-of-thought abilities appear only in large decoder models. (4) RLHF naturally applies to autoregressive models. BERT's ceiling was ~340M params; GPT family scales to 1T+. 
- **When BERT still wins: **Embedding/retrieval tasks. Small-model classification where generation isn't needed. Knowledge distillation targets. Fine-tuning with limited data. 
TRANSFORMER What are the limits of current LLMs? What problems aren't solved? HARD ▶ Senior Answer — Shows Depth of Understanding - **Hallucination: **LLMs confabulate plausible-sounding but false information. Fundamental tension: they are trained to be fluent, not accurate. RAG helps for knowledge retrieval but doesn't solve reasoning errors. 
- **Long context degradation: **Despite 128K+ context windows, models "forget" information from early in the context ("lost in the middle" problem). Attention is diluted across many positions. 
- **Compositional generalization: **LLMs struggle to correctly combine concepts in novel ways not seen in training. Systematic reasoning (e.g., multi-step arithmetic, formal logic) is brittle. 
- **Sample efficiency: **Humans learn from dozens of examples; LLMs need millions of tokens. Still no true few-shot learning in the mechanistic sense — they rely on pattern matching from pre-training. 
- **Evaluation difficulty: **We lack reliable metrics for open-ended generation quality. Human evaluation is gold standard but expensive. Benchmark contamination (test data in training) inflates reported scores. 
- **World model: **LLMs have no persistent memory or world state — each context is independent. They model text, not reality. 
Interview bridge "My work at Turing on reward modeling touched this directly — the fundamental challenge is that reward models learn to score text quality, but text quality and factual correctness are different. A hallucinated but well-written answer scores higher than an accurate but awkward one." PIPELINE Walk me through how you'd set up a complete fine-tuning pipeline for an LLM from scratch. HARD ▶ Step-by-Step Answer - **1. Define the task precisely: **What is the input format? What is the desired output? What metric measures success? Don't start coding until you can answer all three. 
- **2. Choose base model: **Consider: context length needed, license (LLaMA requires research license, Mistral is Apache 2.0), size (7B fits on 1 GPU with QLoRA, 70B needs multiple), domain (CodeLLaMA for code, BioMedLM for medical). 
- **3. Curate training data: **Minimum ~1K instruction-response pairs for SFT. Clean, diverse, high-quality. Format as ChatML or instruction-response format. Deduplicate. Human-review a random sample. 
- **4. Set up environment: **Install transformers, peft, trl, accelerate, bitsandbytes. Configure accelerate for single/multi-GPU. Set up W&B for tracking. 
- **5. Configure LoRA: **r=16, lora_alpha=32, target_modules=["q_proj","k_proj","v_proj","o_proj"]. Load model in 4-bit (QLoRA) if memory-constrained. 
- **6. Train with SFTTrainer: **Use TRL's SFTTrainer. Key params: per_device_train_batch_size=4, gradient_accumulation_steps=4, learning_rate=2e-4, warmup_ratio=0.05, cosine schedule. Monitor training loss + validation loss. 
- **7. Evaluate: **Human evaluation on 100 examples (not just benchmark scores). Task-specific metrics. Compare against base model and GPT-4 on a held-out set. 
- **8. Merge and serve: **Merge LoRA weights into base model (no inference overhead). Export to ONNX or serve with vLLM/TGI. Set up monitoring. 
PIPELINE What is catastrophic forgetting and how do you mitigate it during fine-tuning? MID ▶ Core Answer Catastrophic forgetting: when a model fine-tuned on a specific task "forgets" its general capabilities — the gradient updates for the new task overwrite the weights encoding general knowledge from pretraining. - **Root cause: **High learning rate forces large weight updates in the direction of task loss, pushing weights far from pretraining configuration. 
- **LoRA (best fix): **Freeze pretrained weights entirely. Train only low-rank adapters. Base weights are mathematically unchanged → zero forgetting of pretrained knowledge. 
- **Lower LR: **Fine-tuning LR should be 1-2 orders of magnitude lower than pretraining LR. Typical: 1e-5 to 5e-5 for full FT (vs. 3e-4 for pretraining). 
- **Data replay: **Mix fine-tuning data with a small fraction (~5–10%) of general pretraining data. The model sees general-purpose examples during every update step. 
- **EWC (Elastic Weight Consolidation): **Add regularization term that penalizes changes to parameters that were important for previous tasks (measured by Fisher information). Adds compute but is principled. 
- **KL penalty (RLHF context): **During PPO/DPO, add KL(π_θ || π_ref) penalty to prevent policy from diverging too far from SFT reference. β controls the tradeoff. 
DATA How do you handle a dataset where 99% of samples are class 0 and 1% are class 1? MID ▶ Structured Answer - **First: change the evaluation metric. **AUC-PR (area under precision-recall curve) is more informative than AUC-ROC for extreme imbalance. A classifier that predicts all zeros achieves 0.99 accuracy but 0 recall — that's useless. 
- **Second: weighted loss. **`pos_weight = torch.tensor([99.0]) `in BCEWithLogitsLoss. Gives 99× more weight to positive class errors. Free, effective, no data modification needed. 
- **Third: threshold tuning. **Don't use 0.5 as the classification threshold. Use the precision-recall curve to choose the operating point that matches your business requirement (e.g., maximize recall at 80% precision). 
- **Fourth (if still needed): Focal Loss. **γ=2 focuses training on hard examples. Down-weights easy negatives so the model concentrates on the 1% positives. Better than class weights for very extreme imbalance. 
- **Only if above is insufficient: SMOTE. **Oversample minority class by interpolating in feature space. Risks: can generate unrealistic examples, may hurt calibration. Use only on training split, never on val/test. 
Common Mistake **Never apply oversampling before splitting into train/val. **If you oversample first, synthetic points leak into the val set — your validation metrics are optimistic and your model will fail in production. Quick Reference Attn: softmax(QKᵀ/√d)·V LoRA: W' = W + B·A, r≪d DPO: −log σ(β·(log π_w − log π_l)) BN: ŷ = γ(x−μ)/σ + β Kaiming: W~N(0, 2/n_in) for ReLU Xavier: W~N(0, 2/(n_in+n_out)) for tanh Residual: output = x + F(x) GELU(x) = x·Φ(x) AdamW: w ← w − η·m̂/(√v̂+ε) − η·λ·w GradClip: g ← g·max_norm/‖g‖ QLoRA: 4-bit NF4 + BF16 adapters PSI > 0.2 → retrain signal KV cache: 2·L·H·d_head·n_layers bytes Flash Attn: O(√L) mem, same result Deep Learning Reference · Transformers · Training · Pipelines · Deployment · Meshal Alawein · February 2026 