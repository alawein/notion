# Notes 1: AI/ML Foundations & Fundamentals

Comprehensive guide to core AI/ML concepts, taxonomy, classical machine learning, statistics, probability, and optimization theory. Foundation for understanding modern deep learning and production systems.

---

## Part 1: AI Ecosystem Hierarchy & Taxonomy

### Complete AI Taxonomy Tree

**Root: Artificial Intelligence (AI)**
All intelligent machine systems — ranges from simple IF-THEN rules to modern neural networks.

```
AI (ROOT)
├── Rule-based Systems
│   ├── Expert Systems
│   └── Decision Trees
└── Machine Learning (CORE)
    ├── Supervised Learning (Labeled data)
    │   ├── Classification (predict categories)
    │   │   ├── Binary Classification
    │   │   └── Multi-class Classification
    │   └── Regression (predict numbers)
    │       ├── Linear Regression
    │       └── Non-linear Regression
    ├── Self-Supervised Learning (Generate own labels)
    │   ├── BERT: masked word prediction
    │   ├── GPT: next-token prediction
    │   └── Contrastive Learning
    ├── Reinforcement Learning (Agent + Reward signal)
    │   ├── AlphaGo
    │   ├── Robotic Control
    │   └── RLHF for LLMs
    └── Deep Learning (Multi-layer neural networks)
        ├── ANN: Artificial Neural Networks (tabular/structured)
        ├── CNN: Convolutional (images & spatial)
        ├── RNN: Recurrent (sequences & temporal)
        ├── LSTM: Long Short-Term Memory (long dependencies)
        └── Transformer: Self-attention (parallel, foundation of LLMs)
```

### Neural Architecture Breakdown

**ANN (Artificial Neural Network)**
- Foundational architecture: Input → Hidden layers → Output
- Neurons compute weighted sum of inputs + bias, pass through activation function
- Training: forward pass → loss → backpropagation → gradient descent
- Use cases: loan approval, sales forecasting, churn prediction

**CNN (Convolutional Neural Network)**
- Filters scan image extracting edges, textures, shapes
- Pipeline: Convolution → Max Pooling → Flatten → Dense ANN
- Max pooling compresses feature maps (32×32 → 16×16)
- Use cases: face recognition, tumor detection, self-driving cars, iPhone FaceID

**RNN (Recurrent Neural Network)**
- Hidden state loops back: H_t-1 becomes input at next step
- Short-term memory (~7-10 steps max) due to vanishing gradient
- Suited for sequences and temporal data
- Use cases: stock prediction, sentiment analysis, autocomplete, speech-to-text

**LSTM (Long Short-Term Memory)**
- RNN upgrade with gated memory cells (Forget, Input, Output gates)
- Extends memory to 100+ steps by maintaining memory cell flow
- Solves vanishing gradient problem
- Use cases: language translation, chatbots, long time-series, subtitle generation

**Transformer (Powers LLMs)**
- Self-attention mechanism: every token attends to every other token simultaneously
- Massively parallelizable; scales to billions of parameters
- No recurrence needed; purely attention-based
- Foundation of all modern LLMs: BERT, GPT family, T5, Vision Transformer

### Generative AI & Foundation Models

**Generative AI:** Models that create new content (images, text, audio, video) resembling training data

**GANs (Generative Adversarial Networks)**
- Generator creates fakes from noise; Discriminator judges real vs fake
- Competitive training loop drives photorealistic generation
- Use cases: synthetic faces, art generation, deepfakes, image super-resolution

**Diffusion Models**
- Forward process: gradually add noise to images
- Reverse process: learn to remove noise step by step
- Text-to-image conditioning via text prompts
- Examples: DALL-E 3, Stable Diffusion, Midjourney, Imagen

**Foundation Models**
- Large pre-trained on massive data via self-supervised learning (SSL)
- Capture broad world knowledge; versatile base
- Fine-tuned (SFT) for specific downstream tasks with small labeled datasets
- Examples: BERT, GPT-3, T5, PaLM, LLaMA

**Large Language Models (LLMs)**
- Transformer-based, billions of parameters, human-like text
- Training pipeline: Pre-training (SSL) → SFT (Supervised) → RLHF (Reinforcement Learning from Human Feedback)
- Capable of reasoning, coding, translation, creative writing
- Examples: ChatGPT, Claude, Gemini, GPT-4, LLaMA 3

---

## Part 2: Probability & Statistics Foundations

### Core Probability Concepts

**Bayes' Theorem**
```
P(A|B) = P(B|A) · P(A) / P(B)
```
- Foundation of Bayesian inference, Naive Bayes classifiers, probabilistic graphical models
- In ML context: **posterior ∝ likelihood × prior**
- Critical for understanding MAP vs MLE

**Conditional Independence**
- X ⊥ Y | Z: knowing Z makes X and Y independent
- Critical assumption in Naive Bayes (features independent given class)
- Foundation for HMMs and many graphical models

**Law of Total Expectation**
```
E[X] = E[E[X|Y]]
```
- Used to decompose bias-variance tradeoff
- Helps analyze nested models and hierarchical structures

**Key Distributions to Know**
- **Gaussian (Normal):** CLT, closed-form integrals, universal approximator
- **Bernoulli/Binomial:** Binary and multiclass classification
- **Categorical/Multinomial:** Softmax outputs, discrete distributions
- **Poisson:** Count data, event frequencies
- **Beta:** Conjugate prior for Bernoulli
- **Dirichlet:** Conjugate prior for Categorical
- **Exponential:** Time-to-event, failure rates
- **Uniform:** Random initialization, baseline distributions

**MLE vs MAP**
- **MLE:** maximize P(data|θ) — frequentist approach
- **MAP:** maximize P(θ|data) = P(data|θ) · P(θ) — Bayesian approach
- **MAP connections to regularization:**
  - MAP with Gaussian prior ≡ L2 regularization (Ridge)
  - MAP with Laplace prior ≡ L1 regularization (Lasso)

### Statistical Foundations

**Bias-Variance Decomposition**
```
MSE = Bias² + Variance + Irreducible Noise
```
- High bias → underfitting (model too simple)
- High variance → overfitting (model too complex)
- Bias-variance tradeoff: can't minimize both simultaneously
- Bagging reduces variance; boosting reduces bias; ensemble methods balance both

**Central Limit Theorem**
- Sample mean of n i.i.d. draws approaches N(µ, σ²/n) as n→∞
- Foundation of hypothesis testing, confidence intervals, A/B testing
- Enables normal approximation for large samples

**p-values and Hypothesis Testing**
- **p-value:** Probability of observing data at least as extreme, assuming H₀ true
- **Common misinterpretation:** p-value is NOT probability H₀ is true
- Type I error (α): rejecting true H₀; Type II error (β): failing to reject false H₀
- Power = 1 - β: ability to detect true effect

**A/B Testing Pitfalls**
- Multiple comparisons problem: use Bonferroni correction or FDR
- Novelty effects: initial users favor new treatment
- Sample ratio mismatch: traffic doesn't split as expected
- Simpson's paradox: trend reverses when data segmented

**Correlation vs Causation**
- **Pearson correlation:** measures linear association only
- **Causal inference requires:** randomized experiments OR causal models (DAGs, do-calculus)
- Spurious correlations invalidate many naive ML interpretations
- Confounding variables create false associations

---

## Part 3: Supervised Learning Algorithms

### Linear Models Foundation

| Algorithm | Key Concept | Strength | Weakness |
|-----------|-------------|----------|----------|
| **Linear Regression** | β* = (X'X)⁻¹X'y | Interpretable, fast, closed-form | Linear relationships only |
| **Logistic Regression** | σ(w'x), cross-entropy loss | Calibrated probabilities, linear | Linear decision boundary |
| **SVM (RBF)** | Max margin + kernel trick | High-dimensional, small data | Slow on large N |
| **Decision Tree** | Gini/info gain splits | Interpretable, non-linear | High variance, prone to overfitting |
| **kNN** | Majority vote of k neighbors | Non-parametric, flexible | O(N) inference, sensitive to k |

### Ensemble Methods (Critical for Senior Interviews)

**Bagging (Bootstrap Aggregating)**
- Train M models on bootstrapped subsets; average predictions
- Reduces variance, not bias
- **Random Forest:** Bagging + random feature subsets at each split
- **Out-of-bag (OOB) error:** Free validation estimate without extra data
- Good for noisy, high-dimensional data

**Boosting**
- Train models sequentially; each focuses on errors of previous
- **AdaBoost:** Reweight samples by error; minimize exponential loss
- **Gradient Boosting:** Fit residuals iteratively; minimize arbitrary loss
- **XGBoost:** Adds L1/L2 regularization, second-order gradients, column subsampling
- Reduces bias more than variance; can overfit if tuned aggressively

**Stacking (Meta-Learning)**
- Train meta-model on out-of-fold predictions of base models
- More flexible than simple averaging
- Risk of overfitting the meta-learner; requires careful cross-validation

**Random Forest vs XGBoost**
- **RF:** Parallel training, good baseline, less hyperparameter-sensitive, faster
- **XGBoost:** Usually higher accuracy, requires careful tuning (learning rate, max_depth, subsampling)
- Use RF for noisy high-dim data; XGBoost when accuracy is paramount

---

## Part 4: Model Evaluation & Selection

### Evaluation Metrics by Use Case

| Metric | Formula | When to Use | Interpretation |
|--------|---------|-------------|-----------------|
| **Accuracy** | TP+TN / Total | Balanced classes only | % correct predictions |
| **Precision** | TP / (TP+FP) | High cost of false positives | % predicted positives that are correct |
| **Recall (Sensitivity)** | TP / (TP+FN) | High cost of false negatives (medical, CPD) | % actual positives found |
| **F1 Score** | 2·P·R / (P+R) | Imbalanced classes, balanced FP/FN cost | Harmonic mean of precision & recall |
| **F-β** | ((1+β²)·P·R) / (β²·P+R) | β>1 weights recall; β<1 weights precision | Trade-off metric |
| **AUC-ROC** | Area under TPR vs FPR | Threshold-independent ranking quality | Probability model ranks random positive above random negative |
| **PR-AUC** | Area under Precision-Recall | **Imbalanced datasets** (better than ROC) | Focus on positive class performance |
| **Log-Loss** | -Σ y·log(ŷ)+(1-y)·log(1-ŷ) | Probability calibration quality | Penalizes confident wrong predictions |
| **RMSE** | √(Σe²/n) | Regression; penalizes outliers more | Root mean squared error |
| **MAE** | Σ\|e\|/n | Regression; robust to outliers | Mean absolute error |
| **MAPE** | Σ\|e/y\|/n × 100% | Regression; interpretable % error | Mean absolute percentage error |

### Cross-Validation & Hyperparameter Tuning

**k-Fold Cross-Validation**
- Partition data into k folds; train on k-1, validate on 1; rotate
- Reduces evaluation variance; k=5 or 10 standard
- **Stratified k-fold:** Maintains class distribution for classification

**Leave-One-Out (LOO)**
- k=n: every sample as validation set once
- Low-bias but high-variance estimate; expensive
- Use for very small datasets (< 100 samples)

**Time-Series Cross-Validation (Walk-Forward) — CRITICAL**
- **Never shuffle time-ordered data:** causes temporal leakage
- **Expanding window:** Train on all past data up to time t; validate on t+1 to t+p
- **Rolling window:** Train on fixed window ending at t; validate on t+1 to t+p
- Essential for radar sensor streams and sequential CIR frames

**Hyperparameter Tuning Methods**
- **Grid Search:** Exhaustive but expensive; good for small parameter spaces
- **Random Search:** Often better per compute budget; samples parameter space uniformly
- **Bayesian Optimization (Optuna, Hyperopt):** Models objective surface; chooses next trial intelligently

**Nested Cross-Validation**
- Outer loop for performance estimation
- Inner loop for hyperparameter tuning
- Prevents information leakage from tuning into final performance estimate

---

## Part 5: Optimization Theory

### Gradient Descent Variants

| Variant | Update Rule | When to Use |
|---------|-------------|------------|
| **Batch GD** | θ ← θ - α·∇L(θ; all data) | Small datasets; stable, smooth convergence |
| **SGD** | θ ← θ - α·∇L(θ; x_i) | Large datasets; noisy but fast, online learning |
| **Mini-batch SGD** | θ ← θ - α·∇L(θ; batch) | **Standard practice;** GPU-efficient |
| **Momentum** | v ← βv - α∇L; θ ← θ+v | Accelerates in consistent directions; escapes plateaus |
| **RMSProp** | θ ← θ - α/(√E[g²]+ε)·g | Non-stationary objectives; adapts per parameter |
| **Adam** | Combines momentum + RMSProp | **Default for deep learning;** robust, adaptive |
| **AdamW** | Adam + decoupled weight decay | **Preferred for Transformers;** better generalization |

### Convexity & Loss Landscapes

**Convex Functions**
- Single global minimum; gradient descent guaranteed to converge (with appropriate α)
- Linear regression, logistic regression losses are convex
- **Neural network losses are NOT convex** → multiple local minima possible

**Local Minima vs Saddle Points**
- High-dimensional spaces: saddle points more common than local minima
- Saddle points have zero gradient but mixed curvature (some directions up, some down)
- Modern optimizers (Adam, momentum) effectively escape saddle points

**Learning Rate Schedules**
- **Constant:** Simple, often suboptimal
- **Cosine annealing:** Reduces LR following cosine curve; excellent for deep learning
- **Warmup + decay:** Crucial for Transformers (avoid unstable early training)
- **Cyclical LR:** Can escape local minima by periodically increasing LR

**Gradient Clipping**
- Cap gradient norm at threshold τ: norm(∇L) → min(norm(∇L), τ) · ∇L/norm(∇L)
- Essential for RNNs/LSTMs to prevent exploding gradients
- Less critical for Transformers (stabilized by layer normalization)

---

## Part 6: Regularization & Generalization

### Regularization Techniques

**L2 (Ridge) Regularization**
```
Loss = MSE + λ Σ w²
```
- Penalizes large weights uniformly; shrinks toward zero
- Equivalent to Gaussian prior on weights (MAP interpretation)
- Never zeroes weights completely; smooth shrinkage

**L1 (Lasso) Regularization**
```
Loss = MSE + λ Σ |w|
```
- Produces sparse solutions (many exact zeros)
- Equivalent to Laplace prior on weights
- Useful for feature selection; identifies irrelevant features

**Elastic Net**
```
Loss = MSE + α·L1 + (1-α)·L2
```
- Best of both: sparsity from L1 + stability from L2
- Essential when features are highly correlated

**Dropout**
- Randomly zero activations with probability p during training
- Ensemble interpretation: each forward pass is a different thinned network
- **Common settings:** p=0.1-0.3 for CNNs; p=0.5 for fully connected
- **Critical:** Disable during inference; scale activations by 1/(1-p) during training (or use inverse dropout)

**Batch Normalization**
- Normalize activations to zero mean/unit variance per mini-batch
- Reduces internal covariate shift; enables higher learning rates
- Has regularization effect due to stochastic batch statistics
- Significantly stabilizes training of deep networks

**Early Stopping**
- Monitor validation loss; stop when stops improving
- Implicit L2 regularization effect
- **Patience parameter:** e.g., stop if no improvement for 10 epochs
- Prevents overfitting without explicit penalty term

**Data Augmentation**
- Artificially expand training set with realistic transformations
- For tabular/structured: noise injection, feature scaling
- For images: rotations, flips, crops, color jitter
- For radar data: add Gaussian noise to CIR, random time shifts, amplitude scaling, synthetic environment simulation
- Reduces overfitting without collecting new data

### Generalization Theory

**Bias-Variance-Complexity Trade-off**
- Model complexity ↑ → bias ↓, variance ↑
- Simple models: high bias, low variance (underfitting)
- Complex models: low bias, high variance (overfitting)
- Sweet spot: balanced complexity

**Regularization's Role**
- Constrains model complexity
- Shifts towards simpler models
- Reduces generalization gap (train loss - test loss)

---

## Part 7: Unsupervised Learning & Dimensionality Reduction

### Clustering Methods

**k-Means**
- Iteratively assign points to nearest centroid; update centroids
- Assumes spherical clusters of similar size
- Sensitive to initialization; use k-means++ for better starting point
- Choose k via elbow method (plot within-cluster variance vs k) or silhouette score
- Complexity: O(nkd) per iteration

**DBSCAN (Density-Based Spatial Clustering)**
- Finds arbitrary-shape clusters based on point density
- No need to specify k; automatically identifies noise/outliers
- Key parameters: ε (radius), minPts (minimum points in ε-neighborhood)
- Fails in varying-density clusters; all-or-nothing for ε choice

**GMM (Gaussian Mixture Model)**
- Soft cluster assignments via EM algorithm
- E-step: compute responsibility (probability point belongs to each cluster)
- M-step: update parameters (means, covariances) using responsibilities
- Use AIC/BIC for model selection (automatic k)
- More principled probabilistic approach than k-means

**Hierarchical Clustering**
- Agglomerative (bottom-up): merge closest clusters iteratively
- Divisive (top-down): recursively split clusters
- Dendrogram shows full cluster tree; no need to prespecify k
- Ward linkage minimizes within-cluster variance

### Dimensionality Reduction

**PCA (Principal Component Analysis)**
- Linear projection onto directions of maximum variance
- Eigenvectors of covariance matrix: principal components
- Explained variance ratio guides component selection (cumulative sum)
- Fast, interpretable, but captures linear structure only
- Common approach: keep components explaining 90-95% of variance

**t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Nonlinear embedding for visualization (2D/3D)
- Preserves local structure; distances not meaningful globally
- **NOT suitable for general dimensionality reduction** (only visualization)
- Perplexity controls neighborhood size; typically 5-50
- Computationally expensive; slow on large datasets

**UMAP (Uniform Manifold Approximation and Projection)**
- Faster than t-SNE; better preserves global structure
- Increasingly preferred for high-dimensional sensor/embedding visualization
- Useful for inspecting UWB feature spaces and sensor fusion representations
- More scalable than t-SNE

**Autoencoders**
- Neural network encoder-decoder architecture
- Latent bottleneck forces compressed representation
- **Variational Autoencoder (VAE):** Learns smooth latent distribution
- Useful for anomaly detection on sensor data
- Unsupervised pretraining for deep networks

---

## Part 8: Interview Q&A Bank — ML Fundamentals

**Q: Explain the bias-variance trade-off and give a concrete example from a past project.**
- Don't just define it — connect to a real choice you made
- E.g., choosing Random Forest over a deep network for a small radar dataset because variance reduction mattered more than model capacity
- Show understanding of when each matters

**Q: When would you choose L1 over L2 regularization?**
- L1 when you want feature selection / sparsity (many irrelevant features)
- L2 when all features potentially matter and you want smooth shrinkage
- Elastic net when both conditions apply
- Connect to your domain: e.g., L1 for high-dim sensor data

**Q: Your model has 97% accuracy on a balanced test set but fails in production. What do you investigate?**
- Train/test distribution shift (most common)
- Label noise in training data
- Time leakage (future information leaks into training)
- Feature unavailability at inference time
- Class imbalance in production (test set was balanced, production isn't)
- Data pipeline bugs (preprocessing differences)
- Systematically eliminate each

**Q: How do you choose a threshold for a binary classifier?**
- Depends on cost asymmetry (cost of FP vs FN)
- Plot precision-recall and ROC curves
- Pick threshold based on F-β: β>1 if recall matters (FN costly), β<1 if precision matters (FP costly)
- Use calibration curves to verify probability outputs are trustworthy
- Frame with domain: e.g., in medical diagnosis, high recall (catch all positives) is critical

**Q: Explain why shuffling time-series data before train/test split is wrong.**
- Temporal leakage: model sees future information during training
- Violates independence assumption; inflates performance estimates
- For sensor streams (like radar CIR, WiFi signals), **always use walk-forward validation**
- Time gap between train and validation windows prevents leakage
- Train: [t=0 to t=100], Validation: [t=101 to t=150], Test: [t=151 to t=200]

**Q: What is the curse of dimensionality and how do you address it?**
- As dimensionality grows: data becomes sparse, distance metrics lose meaning, volume grows exponentially
- In high dimensions, most data lies near boundaries; little in interior
- Mitigations: feature selection, domain-informed engineering, PCA/UMAP, regularization
- Tradeoff: dimensionality reduction loses information but improves generalization

**Q: How do you handle class imbalance in a people-sensing dataset where 95% of frames are 'empty'?**
- Oversample minority class (SMOTE), undersample majority, use class-weighted loss
- Evaluate with PR-AUC instead of AUC-ROC (ROC insensitive to class imbalance)
- Frame classifier as anomaly detection (one-class learning): learn "people present" as outlier
- Cost-sensitive learning: penalty matrix α(i,j) for misclassifying class i as j
- Collect more positive examples if budget allows

---

## Summary: Key Takeaways

1. **AI Taxonomy:** Understand the hierarchy from rule-based → ML → DL → Generative AI → LLMs
2. **Probability & Stats:** Bayes' theorem, distributions, bias-variance decomposition, hypothesis testing
3. **Supervised Learning:** Know strengths/weaknesses of linear models, trees, ensembles (RF, XGBoost)
4. **Evaluation:** Choose metrics based on domain (PR-AUC for imbalance, walk-forward for time-series)
5. **Optimization:** Adam default, but know learning rate schedules, gradient clipping, convexity
6. **Regularization:** L1/L2, dropout, batch norm, early stopping — all serve to reduce overfitting
7. **Unsupervised:** PCA for linear, t-SNE/UMAP for visualization, clustering for data exploration
8. **Interview Strategy:** Connect theory to concrete past experiences; show domain awareness
