# 11 Ai Ml Fundamentals

**Total Pages:** 6



--- Page 1 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 1
DOC 1 / 8
AI / ML Fundamentals
Classical machine learning, statistics, optimization & probability for senior interviews
Topics Covered
 Probability & Statistics Foundations
 Supervised Learning Algorithms
 Model Evaluation & Selection
 Optimization Theory
 Regularization & Generalization
 Unsupervised & Dimensionality Reduction
 Ensemble Methods
 Interview Q&A; Bank
Candidate
Meshal Alawein, PhD EECS UC Berkeley
Role
Senior Data Scientist — Algorized, Campbell CA
Series
Interview Preparation Document Series (8 volumes)


--- Page 2 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 2
1. Probability & Statistics Foundations
1.1 Core Probability
 Bayes' Theorem: P(A|B) = P(B|A)·P(A) / P(B). Foundation of Bayesian inference, Naive Bayes classifiers, and
probabilistic graphical models. In ML: posterior ∝ likelihood × prior.
 Conditional independence: X ⊥ Y | Z means knowing Z makes X and Y independent. Critical for Naive Bayes
(features independent given class) and HMMs.
 Law of Total Expectation: E[X] = E[E[X|Y]]. Used to decompose bias-variance tradeoff and analyze nested
models.
 Distributions to know cold: Gaussian (CLT, closed-form integrals); Bernoulli/Binomial (classification);
Categorical/Multinomial (softmax); Poisson (count data); Beta (conjugate prior for Bernoulli); Dirichlet (conjugate
prior for Categorical); Exponential (time-to-event); Uniform (random initialization).
 MLE vs MAP: MLE: maximize P(data|θ). MAP: maximize P(θ|data) = P(data|θ)·P(θ). MAP with Gaussian prior =
L2 regularization. MAP with Laplace prior = L1 regularization.
1.2 Key Statistical Concepts for Interviews
 Bias-Variance Decomposition: MSE = Bias² + Variance + Irreducible noise. High bias → underfitting (simple
model). High variance → overfitting (complex model). Ensemble methods (bagging) reduce variance; boosting
reduces bias.
 Central Limit Theorem: Sample mean of n i.i.d. draws approaches N(µ, σ²/n) as n→∞. Foundation of
hypothesis testing and confidence intervals.
 p-value: Probability of observing data at least as extreme as seen, assuming H■ true. NOT the probability H■ is
true. Common misinterpretation in ML evaluation.
 A/B Testing: Two-sample hypothesis test. Watch for: multiple comparisons (Bonferroni correction), novelty
effects, sample ratio mismatch, Simpson's paradox in segmentation.
 Correlation vs Causation: Pearson correlation measures linear association. Causal inference requires
randomized experiments or causal models (DAGs, do-calculus). Spurious correlations invalidate many naive ML
interpretations.
2. Supervised Learning Algorithms
2.1 Linear Models
<b>Algorithm</b>
<b>Key Formula / Concept</b>
<b>Strength</b>
<b>Weakness</b>
Linear Regression
β* = (X■X)■¹X■y
Interpretable, fast
Linear only
Logistic Regression
σ(w■x), cross-entropy loss
Calibrated probs
Linear boundary


--- Page 3 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 3
SVM (RBF)
max margin + kernel trick
High-dim, small data
Slow on large N
Decision Tree
Gini / info gain splits
Interpretable
High variance
kNN
Majority vote of k neighbors
Non-parametric
O(N) inference
2.2 Ensemble Methods (Critical for Senior Interviews)
 Bagging (Bootstrap Aggregating): Train M models on bootstrapped subsets. Average predictions. Reduces
variance, not bias. Random Forest = bagging + random feature subsets. Out-of-bag (OOB) error as free
validation estimate.
 Boosting: Train models sequentially; each focuses on errors of previous. AdaBoost: reweight samples.
Gradient Boosting: fit residuals. XGBoost adds L1/L2 regularization, second-order gradients, and column
subsampling for speed and robustness.
 Stacking: Meta-model trained on out-of-fold predictions of base models. More flexible than simple averaging.
Risk of overfitting the meta-learner; use cross-validation carefully.
 Random Forest vs XGBoost: RF: parallel training, good baseline, less hyperparameter-sensitive. XGBoost:
usually higher accuracy, but requires careful tuning (learning rate, max_depth, subsampling). RF better for noisy
high-dim data; XGBoost better when accuracy is paramount.
Algorized context: ensemble methods apply directly to sensor fusion — you could run independent
models per sensor modality (UWB, Wi-Fi) and combine via stacking or confidence-weighted averaging.
Frame this when asked about sensor fusion architecture.
3. Model Evaluation & Selection
3.1 Evaluation Metrics
<b>Metric</b>
<b>Formula</b>
<b>Use When</b>
Accuracy
TP+TN / Total
Balanced classes only
Precision
TP / (TP+FP)
Cost of false positives is high
Recall (Sensitivity)
TP / (TP+FN)
Cost of false negatives is high (CPD, medical)
F1 Score
2·P·R / (P+R)
Imbalanced classes, balanced FP/FN cost
F-beta
((1+β²)·P·R) / (β²·P+R)
β>1 weights recall; β<1 weights precision
AUC-ROC
Area under TPR vs FPR curve
Threshold-independent ranking quality
PR-AUC
Area under Precision-Recall curve
Imbalanced datasets (better than ROC)
Log-Loss
-Σ y·log(■)+(1-y)·log(1-■)
Probability calibration quality
RMSE / MAE
√(Σe²/n) / Σ|e|/n
Regression; RMSE penalizes outliers more


--- Page 4 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 4
MAPE
Σ|e/y|/n × 100%
Regression with interpretable % error
3.2 Cross-Validation & Model Selection
 k-Fold CV: Partition data into k folds; train on k-1, validate on 1; rotate. Reduces evaluation variance. k=5 or 10
standard. Stratified k-fold for class imbalance.
 Leave-One-Out (LOO): k=n. Expensive but low-bias estimate. Useful for very small datasets (< 100 samples).
Rare in practice due to compute cost.
 Time-Series CV (Walk-Forward): CRITICAL for radar sensor data: never shuffle time-ordered data. Use
expanding or rolling window. Algorized's sequential CIR frames must use walk-forward validation.
 Hyperparameter Tuning: Grid Search: exhaustive but expensive. Random Search: often better per compute
budget. Bayesian Optimization (Optuna, Hyperopt): models objective surface, chooses next trial intelligently.
 Nested CV: Outer loop for performance estimation, inner loop for hyperparameter tuning. Avoids information
leakage from tuning into performance estimate.
4. Optimization Theory
4.1 Gradient Descent Variants
<b>Variant</b>
<b>Update Rule</b>
<b>When to Use</b>
Batch GD
θ ← θ - α·∇L(θ; all data)
Small datasets; stable convergence
SGD
θ ← θ - α·∇L(θ; x■)
Large datasets; noisy but fast
Mini-batch SGD
θ ← θ - α·∇L(θ; batch)
Standard practice; GPU-efficient
Momentum
v ← βv - α∇L; θ ← θ+v
Accelerates in consistent directions
RMSProp
θ ← θ - α/(√E[g²]+ε)·g
Non-stationary objectives
Adam
Combines momentum + RMSProp
Default for deep learning
AdamW
Adam + decoupled weight decay
Preferred for Transformers
4.2 Convexity & Loss Landscapes
 Convex functions: Single global minimum. GD guaranteed to converge (with appropriate α). Linear/logistic
regression losses are convex. Neural network losses are NOT convex.
 Local minima vs saddle points: In high-dimensional spaces, saddle points (zero gradient, mixed curvature)
are more common than local minima. Modern optimizers (Adam, momentum) escape saddle points effectively.
 Learning rate schedules: Constant: simple, often suboptimal. Cosine annealing: reduces LR following cosine
curve; excellent for deep learning. Warmup + decay: crucial for Transformers (avoid unstable early training).
Cyclical LR: can escape local minima.


--- Page 5 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 5
 Gradient clipping: Cap gradient norm at threshold τ. Essential for RNNs/LSTMs to prevent exploding gradients.
Less critical for Transformers (already stabilized by layer norm).
5. Regularization & Generalization
 L2 (Ridge): Adds λΣw■² to loss. Penalizes large weights, shrinks uniformly. Equivalent to Gaussian prior on
weights. Never zeroes weights out.
 L1 (Lasso): Adds λΣ|w■| to loss. Produces sparse solutions (many exact zeros). Equivalent to Laplace prior.
Useful for feature selection.
 Elastic Net: α·L1 + (1-α)·L2. Best of both: sparsity + stability when features are correlated.
 Dropout: Randomly zero activations with probability p during training. Ensemble interpretation: each forward
pass is a different thinned network. Use p=0.1-0.3 for CNNs, 0.5 for fully connected. Do NOT use during
inference.
 Batch Normalization: Normalize activations to zero mean/unit variance per mini-batch. Reduces internal
covariate shift. Enables higher LR. Has regularization effect (stochastic due to batch statistics).
 Early Stopping: Monitor validation loss; stop when it stops improving. Implicit L2 regularization effect. Use
patience parameter (e.g., 10 epochs).
 Data Augmentation: Artificially expand training set. For radar: add Gaussian noise to CIR, random time shifts,
amplitude scaling, synthetic environment simulation. Reduces overfitting without collecting new data.
6. Unsupervised Learning & Dimensionality Reduction
6.1 Clustering
 k-Means: Iterative centroid assignment. Assumes spherical clusters. Sensitive to initialization (use k-means++).
Choose k via elbow method or silhouette score. O(nkd) per iteration.
 DBSCAN: Density-based. Finds arbitrary-shape clusters. No need to specify k. Identifies noise/outliers. Key
params: ε (radius), minPts. Fails in varying-density clusters.
 GMM (EM): Soft cluster assignments via Gaussian mixture. EM algorithm alternates E-step (responsibilities)
and M-step (parameter update). AIC/BIC for model selection.
 Hierarchical: Agglomerative (bottom-up) or divisive (top-down). Dendrogram shows full cluster tree. Ward
linkage minimizes within-cluster variance. No need to prespecify k.
6.2 Dimensionality Reduction
 PCA: Linear projection onto directions of maximum variance (eigenvectors of covariance matrix). Explained
variance ratio guides component selection. Fast, interpretable, but only captures linear structure.
 t-SNE: Nonlinear embedding for visualization (2D/3D). Preserves local structure. NOT suitable for general
dimensionality reduction (distances not meaningful globally). Perplexity controls neighborhood size.
 UMAP: Faster than t-SNE, better preserves global structure. Increasingly preferred for high-dim
sensor/embedding visualization. Useful for inspecting UWB feature spaces.


--- Page 6 ---

Algorized Interview Prep — Doc 1: AI/ML Fundamentals
Page 6
 Autoencoders: Neural network encoder-decoder. Latent bottleneck forces compressed representation.
Variational Autoencoder (VAE) learns a smooth latent distribution. Useful for anomaly detection on sensor data.
7. Interview Q&A; Bank — ML Fundamentals
Q: Explain the bias-variance trade-off and give a concrete example from a past project.
■ Don't just define it — connect to a real choice you made. E.g., choosing Random Forest over a deep network for a
small radar dataset because variance reduction mattered more than model capacity.
Q: When would you choose L1 over L2 regularization?
■ L1 when you want feature selection / sparsity (many irrelevant features). L2 when all features potentially matter and
you want smooth shrinkage. Elastic net when both apply.
Q: Your model has 97% accuracy on a balanced test set but fails in production. What do you
investigate?
■ Train/test distribution shift, label noise, time leakage, feature unavailability at inference time, class imbalance in
production, data pipeline bugs. Systematically eliminate each.
Q: How do you choose a threshold for a binary classifier?
■ Depends on cost asymmetry. Plot PR curve and ROC; pick threshold based on F-beta (β>1 if recall matters more, as
in CPD). Use calibration curve to ensure probability outputs are trustworthy.
Q: Explain why shuffling time-series data before train/test split is wrong.
■ Temporal leakage: the model sees future information during training. For sensor streams (like radar CIR), always use
walk-forward validation with a time gap between train and validation windows.
Q: What is the curse of dimensionality and how do you address it?
■ As dimensionality grows, data becomes sparse, distance metrics lose meaning, and volume grows exponentially.
Mitigations: feature selection, PCA/UMAP, regularization, domain-informed feature engineering.
Q: How do you handle class imbalance in a people-sensing dataset where 95% of frames are 'empty'?
■ Oversample minority class (SMOTE), undersample majority, use class-weighted loss, PR-AUC instead of AUC-ROC
for evaluation, and design the classifier as anomaly detection (one-class learning).
