---
layout: default
title: Logistic Regression
parent: Supervised Learning
nav_order: 2
math: true
---

# Logistic Regression

## Introduction
Logistic Regression is a fundamental machine learning algorithm for **binary classification** tasks, although it can be generalized to multi-class problems. It is used to predict the probability of an object belonging to one of two classes (e.g., spam/not spam, sick/healthy, etc.).

- **Learning Type**: Supervised Learning.
- **Output**: A probability from 0 to 1, which can be converted into a class using a threshold, typically 0.5.
- **Core Idea**: It models the relationship between features and probability using the logistic function, also known as the **sigmoid function**, which "squashes" a linear combination of features into the range [0, 1].

Unlike linear regression, which predicts continuous values, logistic regression works with categorical outcomes.

### Connection to Neural Networks
**Logistic Regression = a single-layer neural network** with a sigmoid activation function:
- Input layer → linear combination (weighted sum) → sigmoid → output (probability)
- It is trained via backpropagation, although for a single layer, this is simply gradient descent.
- It is a cornerstone for understanding deep learning: a multi-layer network is a stack of logistic regressions with non-linearities.

### Probabilistic Interpretation: Maximum Likelihood Estimation (MLE)
Training logistic regression is equivalent to **Maximum Likelihood Estimation (MLE)**:
\[ L(\mathbf{w}) = \prod_{i=1}^m p(y_i | \mathbf{x}_i; \mathbf{w}) = \prod_{i=1}^m \hat{y}_i^{y_i} (1-\hat{y}_i)^{1-y_i} \]
The log-likelihood is:
\[ \log L(\mathbf{w}) = \sum_{i=1}^m [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)] \]
Minimizing cross-entropy is equivalent to maximizing the log-likelihood (with an opposite sign).

**Bayesian interpretation**: L2 regularization corresponds to a Gaussian prior on the weights, \(\mathbf{w} \sim \mathcal{N}(0, \sigma^2 I)\), where \(\lambda = 1/(2\sigma^2)\). This is **Maximum A Posteriori (MAP)** estimation instead of MLE.

---

## 📝 Quick Cheat Sheet (review 15 min before an interview)

### Key Formulas
1. **Model**: \( p(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}} \)
2. **Loss**: \( J = -\frac{1}{m}\sum_i [y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)] + \frac{\lambda}{2}\lVert\mathbf{w}\rVert_2^2 \)
3. **Gradient**: \( \nabla_\mathbf{w}J = \frac{1}{m}X^T(\hat{\mathbf{y}} - \mathbf{y}) + \lambda\mathbf{w} \)
4. **Logits**: \( \log\frac{p}{1-p} = \mathbf{w}^T\mathbf{x} + b \)

### 3 Key Insights
1. **MLE → Cross-entropy**: Minimizing the loss function = maximizing the likelihood.
2. **Convexity**: Guarantees finding the global minimum (a unique solution).
3. **Linearity**: The decision boundary is a hyperplane \(\mathbf{w}^T\mathbf{x}+b=0\).

### Trade-offs for Discussion
| Question | Answer |
|---|---|
| **L1 vs L2?** | L1→sparsity, feature selection; L2→stability with multicollinearity. |
| **SGD vs Batch GD?** | SGD→faster, for online learning; Batch→more precise, stable. |
| **LogReg vs SVM?** | LogReg→probabilities; SVM→margin maximization, no native probabilities. |
| **LogReg vs Boosting?** | LogReg→faster, interpretable; Boosting→higher accuracy. |
| **ROC-AUC vs PR-AUC?** | ROC-AUC for balanced classes; PR-AUC for imbalanced (rare positive class). |

### What to Draw on a Whiteboard
1. ✅ Sigmoid (S-curve): 0→0.5→1
2. ✅ Decision boundary in 2D (a straight line)
3. ✅ Derivation of MLE → Cross-entropy
4. ✅ Gradient Descent (Optimization landscape)

### Frequent Questions + Answers
**Q: Why not MSE?**  
A: MSE→vanishing gradients with large errors; cross-entropy is convex and related to MLE.

**Q: Perfect separation?**  
A: Weights→∞ without regularization; solution: L2 or Firth's penalized likelihood.

**Q: Class imbalance?**  
A: Class weighting, SMOTE, threshold adjustment; metric: PR-AUC > ROC-AUC.

**Q: Online learning?**  
A: SGD with a decaying learning rate; FTRL-Proximal for sparse data; important to monitor concept drift.

### Complexity
- Let:
    - **n** — number of samples,
    - **d** — number of features,
    - **iterations** — number of optimization iterations.
- **Training**: O(iterations × n × d); Newton's method: O(n·d² + d³).
- **Inference**: O(d) — very fast!
- **Memory**: O(d) — only weights are stored.

---

## Mathematical Foundation
### Linear Combination
The model starts with a linear combination of features:
\[ z = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n = \mathbf{w}^T \mathbf{x} + b \]
where:
- \(\mathbf{x}\) — feature vector,
- \(\mathbf{w}\) — weights vector,
- \(b\) — bias / intercept.

### Sigmoid Function
To obtain a probability, the sigmoid function is applied:
\[ p(y=1 | \mathbf{x}) = \sigma(z) = \frac{1}{1 + e^{-z}} \]
- For \(z \to \infty\), \(\sigma(z) \to 1\),
- For \(z \to -\infty\), \(\sigma(z) \to 0\),
- At \(z = 0\), \(\sigma(z) = 0.5\).

The sigmoid graph is an S-shaped curve, ideal for modeling probabilities.

### Prediction
- If \(p > 0.5\), predict class 1,
- Otherwise — class 0.

For multi-class classification, **softmax** is used (a generalization of sigmoid):
\[ p(y=k | \mathbf{x}) = \frac{e^{z_k}}{\sum_{i=1}^K e^{z_i}} \]

## Interpretation via Logits and Odds
Probability can be interpreted through the **odds ratio**:
\[ \text{odds} = \frac{p}{1-p} \]
The **logit** is the logarithm of the odds and depends linearly on the features:
\[ \mathrm{logit}(p) = \log\frac{p}{1-p} = \mathbf{w}^T \mathbf{x} + b \]
- An increase of feature \(x_j\) by 1 increases the log-odds by \(w_j\), and the odds themselves are multiplied by \(e^{w_j}\).
- This provides a convenient interpretation of the coefficients and a basis for probability calibration.

### Decision Boundary
The decision rule at a 0.5 threshold is equivalent to a logit threshold of 0:
\[ p > 0.5 \iff \mathbf{w}^T \mathbf{x} + b > 0 \]
Thus, the decision surface is a hyperplane perpendicular to the weight vector \(\mathbf{w}\).

## Model Training
### Loss Function
**Cross-entropy** is used for training, specifically **binary cross-entropy** for binary classification:
\[ L(y, \hat{y}) = - [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})] \]
where:
- \(y\) — ground truth label,
- \(\hat{y} = \sigma(z)\) — predicted probability.

For the entire dataset, the **cost function** is:
\[ J(\mathbf{w}) = \frac{1}{m} \sum_{i=1}^m L(y_i, \hat{y}_i) \]

This is better than **Mean Squared Error (MSE)** because cross-entropy heavily penalizes incorrect predictions (e.g., if y=1 and \(\hat{y}\) is close to 0, loss → ∞).

### With Regularization and Class Weights
The full cost function with L2 (no penalty for bias \(b\)) and class weights \(w^{(cls)}_i\):
\[ J(\mathbf{w}, b) = \frac{1}{m} \sum_{i=1}^m w^{(cls)}_i \cdot \Big(-y_i\log\hat{y}_i - (1-y_i)\log(1-\hat{y}_i)\Big) + \frac{\lambda}{2} \lVert \mathbf{w} \rVert_2^2 \]
Gradients:
\[ \nabla_{\mathbf{w}} J = \frac{1}{m} X^\top \big( w^{(cls)} \odot (\hat{\mathbf{p}} - \mathbf{y}) \big) + \lambda\,\mathbf{w},\quad \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m w^{(cls)}_i(\hat{p}_i - y_i) \]
- The intercept \(b\) is usually not penalized (or has a separate weak penalty).
- For class imbalance: weights are chosen inversely proportional to class frequencies, or `class_weight='balanced'` is used.

### Optimization: Gradient Descent
The model is trained by minimizing J(w) using gradient descent:
1. Initialize weights (usually with zeros or small random numbers).
2. Compute gradients:
   \[ \frac{\partial J}{\partial w_j} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) x_{i,j} \]
   \[ \frac{\partial J}{\partial b} = \frac{1}{m} \sum_{i=1}^m (\hat{y}_i - y_i) \]
3. Update weights:
   \[ w_j \leftarrow w_j - \alpha \frac{\partial J}{\partial w_j} \]
   where \(\alpha\) is the **learning rate**.

Variants:
- **Batch Gradient Descent**: Gradients over the entire dataset (slow for large data).
- **Stochastic Gradient Descent (SGD)**: Per one sample (faster, but "noisy").
- **Mini-batch Gradient Descent**: A compromise (most common).

Additionally: momentum, Adam, RMSprop to accelerate convergence.

### Vector Form, Hessian, and Newton's Method / IRLS
Vectorized gradient (with L2):
\[ \nabla_{\mathbf{w}} J = \frac{1}{m} X^\top(\hat{\mathbf{p}} - \mathbf{y}) + \lambda\,\mathbf{w},\quad \frac{\partial J}{\partial b} = \frac{1}{m}\sum_i (\hat{p}_i - y_i) \]
**Hessian matrix**:
\[ H = \frac{1}{m} X^\top R X + \lambda I,\quad R = \mathrm{diag}(\hat{p}_i(1-\hat{p}_i)) \]
A **Newton's method** step: \( \Delta\theta = H^{-1} g \) with \(g=[\nabla_\mathbf{w}J,\ \partial J/\partial b]\). This is equivalent to **Iteratively Reweighted Least Squares (IRLS)**.
- Pros: quadratic convergence near the minimum, good for small \(d\).
- Cons: computationally expensive (O(nd²)+solving O(d³)), scales poorly with large \(d\).

### Choice of Solvers
- **lbfgs**: default for dense data, a quasi-Newton method.
- **liblinear**: coordinate descent; good for small/sparse data, only for binary classification.
- **sag/saga**: stochastic gradients for large samples; `saga` supports L1/elastic net.
- **newton-cg**: a variant of Newton's method without an explicit Hessian; suitable for medium-sized problems.

Practice: for L1 or huge n — `saga`; for multinomial — `lbfgs`/`saga`; for very sparse and binary — `liblinear`.

## Regularization
To avoid **overfitting**, a penalty for large weights is added:
- **L2 (Ridge)**: \( J(\mathbf{w}) + \lambda \sum w_j^2 \) — shrinks weights towards zero, but does not make them exactly zero.
- **L1 (Lasso)**: \( J(\mathbf{w}) + \lambda \sum |w_j| \) — can zero out weights, performing **feature selection**.
- **Elastic Net**: A combination of L1 and L2.

\(\lambda\) is a **hyperparameter**, tuned via **cross-validation**.

## Model Evaluation
### Metrics
- **Accuracy**: proportion of correct predictions (weak for imbalanced classes).
- **Precision/Recall/F1-score**: quality for the positive class; macro/weighted averaging for multi-class.
- **ROC-AUC**: area under the ROC curve (TPR vs FPR), robust to imbalance.
- **PR-AUC**: area under the Precision-Recall curve; more informative than ROC for a rare positive class.
- **LogLoss**: average cross-entropy — sensitive to probability calibration.
- **Brier score**: mean squared error of probabilities; reflects calibration well.

### Confusion Matrix
Table:
| | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP (True Positive) | FN (False Negative) |
| **Actual Negative** | FP (False Positive) | TN (True Negative) |

### Cross-Validation
Split data into train/test, use k-fold for reliability.

### Probability Calibration and Threshold Selection
- **Calibration**: **Platt scaling** or **Isotonic Regression**; important for correct probabilities.
- **Threshold selection**: based on a business metric (max F1, fixed TPR/FPR, cost-sensitive). The threshold should be tuned on a validation set, not just set to 0.5.

## Preprocessing and Feature Engineering
- **Scaling**: `StandardScaler`/`RobustScaler` improves GD convergence and makes regularization comparable across axes.
- **Categorical features**: **one-hot encoding** for nominal; **target encoding** or Weight of Evidence (WOE) is possible but requires careful cross-validation to avoid data leakage.
- **Sparsity**: effective with `liblinear`/`saga`; L1 drops irrelevant features.
- **Missing values**: **imputation** — median/most frequent, missing indicators.
- **Non-linearities**: **polynomial/interaction features**, splines, binning — preserve interpretability.
- **Imbalance**: `class weights`, **oversampling** (SMOTE), **undersampling**.

## Edge Cases and Numerical Stability

### Problems and Solutions

1. **Perfect separation**
   - **Problem**: If classes are linearly separable, weights go to ±∞, probabilities → 0 or 1.
   - **Symptoms**: Very large coefficients, convergence warnings.
   - **Solutions**:
     - L2 regularization (mandatory!).
     - **Firth's penalized likelihood**, which adds a Jeffrey's prior.
     - More data or remove the strongly separating feature.

2. **Multicollinearity**
   - **Problem**: High correlation between features → unstable coefficients.
   - **Detection**: **Variance Inflation Factor (VIF)** > 10.
   - **Solutions**:
     - L2 regularization (Ridge) — stabilizes.
     - **Principal Component Analysis (PCA)** for dimensionality reduction.
     - Remove correlated features (correlation matrix).

3. **Outliers**
   - **Problem**: Affect the position of the decision boundary.
   - **Solutions**:
     - `RobustScaler` instead of `StandardScaler`.
     - **Winsorization** — clipping by percentiles.
     - L1 regularization (more robust).
     - **Huber loss** instead of cross-entropy (for strong outliers).

4. **Numerical stability**
   
   a) **Calculating log(sigmoid(z))**:
   ```python
   # BAD: can result in -inf for large negative z
   log_prob = log(1 / (1 + exp(-z)))
   
   # GOOD: numerically stable
   def log_sigmoid(z):
       if z >= 0:
           return -log(1 + exp(-z))
       else:
           return z - log(1 + exp(z))
   ```
   
   b) **Calculating log(1 - sigmoid(z))**:
   ```python
   # GOOD: log(1 - σ(z)) = log(σ(-z))
   log_one_minus_prob = log_sigmoid(-z)
   ```
   
   c) **LogSumExp trick** for softmax:
   ```python
   # BAD: overflow for large z
   exp(z_k) / sum(exp(z))
   
   # GOOD
   m = max(z)
   exp(z_k - m) / sum(exp(z - m))
   ```

5. **Data issues**
   - **Imbalance**: class_weight='balanced' or weighting in loss
   - **Missing values**: Imputation (mean/median/mode) + indicator for missing
   - **Different scales**: StandardScaler/MinMaxScaler is a must
   - **Categories**: OHE + handle_unknown='ignore' for unseen in production

## Multi-class and Multi-label
- **Multinomial logistic regression** with softmax optimizes a general cross-entropy; better than **One-vs-Rest** for overlapping classes.
- **One-vs-Rest (OvR)**: trains K binary models; simpler for sparse data and with L1.
- **Multi-label**: independent binary logistic regression for each label (sigmoid), separate thresholds.
- **Metrics**: micro/macro/weighted F1, mAP, macro ROC-AUC/PR-AUC.

## Production Practices
- Complexity: one epoch of GD — O(n·d); Newton's — O(n·d^2)+O(d^3). For large d, use `saga`/`lbfgs`.
- **Online learning**: SGD/mini-batch with a decaying learning rate; FTRL-Proximal for sparse features and L1.
- Preprocessing: fix scalers/OHE dictionaries; for unseen categories — `handle_unknown` or **feature hashing**.
- **Monitoring**: logloss/AUC/PR-AUC over time, calibration (reliability), **PSI (Population Stability Index)** / **concept drift**, stability of score distributions.
- Calibration/threshold: periodically retrain the calibrator and revisit the threshold based on business goals.
- Interpretability: coefficients as log-odds; comparability only with correct feature scaling.
- Reliability: numerical stability in **serving** — clipping logits, protect the pipeline from NaN/Inf.

## Advantages and Disadvantages
### Advantages
- **Simplicity**: Easy to understand and implement; minimal hyperparameters.
- **Interpretability**: Weights show feature importance; coefficients have a clear meaning (log-odds).
- **Efficiency**: Trains quickly, works on large data; **inference** O(d) — very fast.
- **Probabilistic output**: Provides calibratable probabilities, not just classes (important for ranking, cost-sensitive decisions).
- **Scalability**: Works well with online learning (SGD, FTRL); easily parallelizable.
- **Stability**: Convex loss function → guaranteed convergence to a global minimum.
- **Low variance**: Less prone to overfitting compared to more complex models (with proper regularization).

### Disadvantages
- **Linearity**: Performs poorly with non-linear relationships without **feature engineering** — polynomials, interactions, or kernels.
- **Feature engineering**: Requires manual work to identify non-linearities and interactions.
- **Sensitivity**: To outliers, multicollinearity, and class imbalance.
- **Not for complex patterns**: For hierarchical features, deep patterns, images/text, neural networks/trees are better.
- **Assumptions**: Requires relatively independent features; issues with perfect separation.

### When NOT to use logistic regression
- **Complex non-linear relationships**: Use gradient boosting (CatBoost, XGBoost), Random Forest.
- **Many interacting features**: Trees automatically find splits; logistic regression requires explicit creation of cross-features.
- **Tabular data with categories**: Gradient boosting is usually better out-of-the-box.
- **Images/audio/video**: Deep learning — CNN, ResNet, Vision Transformers.
- **Sequences/text**: RNN, LSTM, Transformers (BERT, GPT).
- **Very high dimensionality without regularization**: May require **dimensionality reduction** or L1.
- **When probabilities are not needed**: SVM might be faster for pure classification.

## Comparison with other algorithms

| Aspect | LogReg | SVM | Decision Trees | Gradient Boosting | Neural Networks |
|---|---|---|---|---|---|
| **Interpretability** | ✅ High | ⚠️ Medium (kernel is complex) | ✅ High | ⚠️ Low | ❌ Very low |
| **Probabilities** | ✅ Native | ❌ Requires calibration | ⚠️ Leaf frequencies | ✅ Good | ✅ Good |
| **Non-linearity** | ❌ Only with FE | ✅ Kernel trick | ✅ Automatic | ✅ Excellent | ✅ Best |
| **Training speed** | ✅ Fast | ⚠️ Slow | ✅ Fast | ⚠️ Medium | ❌ Slow |
| **Inference** | ✅ O(d) | ✅ O(#SV×d) | ✅ O(depth) | ⚠️ O(trees×depth) | ⚠️ O(layers×width) |
| **Overfitting** | ✅ Robust | ✅ Robust | ❌ Prone | ⚠️ Requires tuning | ❌ Prone |
| **Categories** | ❌ Needs OHE | ❌ Needs OHE | ✅ Native | ✅ Native (CatBoost) | ⚠️ Embeddings |
| **Small data** | ✅ Good | ✅ Good | ⚠️ Medium | ❌ Poor | ❌ Poor |
| **Big data** | ✅ Excellent (SGD) | ❌ Poor | ⚠️ Medium | ✅ Good | ⚠️ Requires GPU |
| **Online learning** | ✅ Easy (SGD/FTRL) | ❌ Hard | ❌ Hard | ❌ Hard | ⚠️ Possible |

### When to choose what
- **LogReg**: Baseline for tabular data; when interpretability and probabilities are needed; online learning; large sparse data (CTR prediction).
- **SVM**: Small data with non-linearities; when probabilities are not important; text with TF-IDF.
- **Random Forest**: Quick baseline; feature importance; small data.
- **Gradient Boosting**: Best quality on tabular data; competitions; when time can be spent on tuning.
- **Deep Learning**: Images, text, audio; when there is a lot of data; complex patterns.

## Applications
### Common use cases
- **Medicine**: Predicting diseases, risk of complications (interpretability is critical for doctors).
- **Marketing**: Click-through rate (CTR), conversion prediction, customer churn.
- **Finance**: Credit scoring (default/no default), fraud detection (interpretability needed for regulators).
- **NLP**: Sentiment analysis, spam detection, document classification (with TF-IDF/ngrams).
- **E-commerce**: Purchase/no purchase, product recommendations (first level of filtering).

## Implementation Example

### Pseudocode (for a whiteboard)
```python
def sigmoid(z):
    return 1.0 / (1.0 + exp(-z))

def train_minibatch_l2(X, y, epochs, lr, batch_size, lambda_):
    w = zeros(X.shape[1])
    b = 0.0
    best_val = inf
    patience, wait = 5, 0
    for _ in range(epochs):
        for Xb, yb in iterate_minibatches(X, y, batch_size):
            z = Xb @ w + b
            p = sigmoid(z)
            err = p - yb
            grad_w = (Xb.T @ err) / len(yb) + lambda_ * w
            grad_b = err.mean()
            w -= lr * grad_w
            b -= lr * grad_b
        # validation and early stopping
        val_loss = logloss_val(X_val, y_val, w, b) + 0.5 * lambda_ * (w @ w)
        if val_loss + 1e-6 < best_val:
            best_val, wait = val_loss, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return w, b

def predict_label(X, w, b, threshold=0.5):
    p = sigmoid(X @ w + b)
    return (p > threshold).astype(int)
```

### Real implementation from scratch (numpy)
```python
import numpy as np

class LogisticRegression:
    """Logistic regression with L2 regularization"""
    
    def __init__(self, learning_rate=0.01, lambda_=0.01, 
                 max_iter=1000, tol=1e-4):
        self.lr = learning_rate
        self.lambda_ = lambda_
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.loss_history = []
    
    def _sigmoid(self, z):
        """Numerically stable sigmoid"""
        return np.where(z >= 0, 
                       1 / (1 + np.exp(-z)),
                       np.exp(z) / (1 + np.exp(z)))
    
    def _compute_loss(self, X, y):
        """Binary cross-entropy + L2"""
        m = X.shape[0]
        z = X @ self.w + self.b
        y_pred = self._sigmoid(z)
        
        # Numerically stable loss
        epsilon = 1e-15  # to avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cross_entropy = -np.mean(
            y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)
        )
        l2_penalty = 0.5 * self.lambda_ * np.sum(self.w ** 2)
        return cross_entropy + l2_penalty
    
    def fit(self, X, y, X_val=None, y_val=None, verbose=False):
        """Training with mini-batch GD"""
        m, n = X.shape
        self.w = np.zeros(n)
        self.b = 0.0
        
        for iteration in range(self.max_iter):
            # Forward pass
            z = X @ self.w + self.b
            y_pred = self._sigmoid(z)
            
            # Compute gradients
            error = y_pred - y
            grad_w = (X.T @ error) / m + self.lambda_ * self.w
            grad_b = np.mean(error)
            
            # Update parameters
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
            
            # Compute loss
            loss = self._compute_loss(X, y)
            self.loss_history.append(loss)
            
            # Early stopping
            if len(self.loss_history) > 1:
                if abs(self.loss_history[-1] - self.loss_history[-2]) < self.tol:
                    if verbose:
                        print(f"Converged at iteration {iteration}")
                    break
            
            # Validation
            if verbose and iteration % 100 == 0:
                val_str = ""
                if X_val is not None:
                    val_loss = self._compute_loss(X_val, y_val)
                    val_str = f", Val Loss: {val_loss:.4f}"
                print(f"Iter {iteration}, Train Loss: {loss:.4f}{val_str}")
        
        return self
    
    def predict_proba(self, X):
        """Predict probabilities"""
        z = X @ self.w + self.b
        return self._sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """Predict classes"""
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def get_coefficients(self):
        """Get coefficients"""
        return {'weights': self.w, 'bias': self.b}


# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, 
                               n_informative=15, random_state=42)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Training
    model = LogisticRegression(learning_rate=0.1, lambda_=0.01, max_iter=1000)
    model.fit(X_train, y_train, X_test, y_test, verbose=True)
    
    # Prediction
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)
    
    # Evaluation
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
```

## Whiteboard Interview Presentation Strategy

### Recommended Structure (20-30 minutes)

#### 1. Problem Statement (2 min)
**What to say:**
- "Logistic regression is a supervised algorithm for binary classification"
- "Difference from linear regression: output is a probability in [0,1], not a continuous value"
- "Example: predict the probability of a click on an ad"

**What to draw:**
```
Input: x = [x₁, x₂, ..., xₙ]
Output: P(y=1|x) ∈ [0, 1]
→ Decision: ŷ = 1 if P > 0.5, else 0
```

#### 2. Mathematical Model (5 min)
**What to draw (in order):**

a) Linear combination:
```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b = wᵀx + b
```

b) Sigmoid graph (MUST draw!):
```
    σ(z)
    1 ┤      ────────
      │    ╱
  0.5 ┤  ╱
      │╱
    0 ┤────────
      └────┴────┴──── z
         -5  0  5
```
Formula: σ(z) = 1/(1+e^{-z})

c) Decision boundary (2D example):
```
    x₂
     ┤  ○ ○ ○ │ • • •
     │  ○ ○   │   • •
     │  ○     │     •
     └────────┴────── x₁
           w₁x₁+w₂x₂+b=0
```
"The separating hyperplane is perpendicular to the weight vector w"

#### 3. Interpretation via odds (2 min)
**What to write:**
- odds = p/(1-p)
- logit(p) = log(odds) = wᵀx + b
- "The coefficient wⱼ shows how the log-odds change when xⱼ changes by 1"
- exp(wⱼ) = multiplicative effect on odds

#### 4. Loss Function (5 min)
**Must show the derivation via MLE!**

a) Probabilistic model:
```
P(y|x) = ŷʸ(1-ŷ)¹⁻ʸ
```

b) Likelihood:
```
L(w) = ∏ᵢ P(yᵢ|xᵢ)
```

c) Log-likelihood:
```
log L = Σᵢ [yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

d) Loss = -log L (binary cross-entropy):
```
J(w) = -1/m Σᵢ [yᵢlog(ŷᵢ) + (1-yᵢ)log(1-ŷᵢ)]
```

e) With regularization:
```
J(w) = -1/m Σᵢ [...] + λ/2·||w||_2^2
```

**Why not MSE?** "Cross-entropy penalizes confident errors more strongly; it's a convex function; gradients do not vanish"

#### 5. Optimization (5 min)
**What to write:**

a) Gradient:
```
∂J/∂wⱼ = 1/m Σᵢ (ŷᵢ - yᵢ)xᵢⱼ + λwⱼ
∂J/∂b = 1/m Σᵢ (ŷᵢ - yᵢ)
```
"A beautiful form! Similar to linear regression"

b) Vector form:
```
∇_w J = 1/m·Xᵀ(ŷ - y) + λw
```

c) Gradient Descent update:
```
w ← w - α·∇_w J
```

**Variants:**
- Batch GD (entire dataset)
- SGD (one sample, noisy, suitable for online)
- Mini-batch (compromise, standard)
- Advanced: Adam, momentum

d) Alternative — Newton's method:
```
Δw = H⁻¹·∇J
H = 1/m·XᵀRX + λI (Hessian)
```
"Quadratic convergence, but O(d³) — bad for large d"

#### 6. Practical Aspects (5-7 min)
**Must mention:**

a) **Metrics:**
- Accuracy (weak for imbalance)
- Precision/Recall/F1
- ROC-AUC vs PR-AUC (draw the axes!)
- LogLoss for calibration

b) **Class imbalance:**
- Class weights: wᵢ⁽ᶜˡˢ⁾ in loss
- Oversampling (SMOTE) vs undersampling
- Adjust threshold

c) **Regularization:**
- L2 (shrinks weights)
- L1 (feature selection)
- How to choose λ: cross-validation

d) **Feature engineering:**
- Scaling is critical! (StandardScaler)
- Polynomial features for non-linearity
- Categories → One-Hot Encoding

e) **Production challenges:**
- Online learning (SGD, FTRL)
- Fast inference: O(d)
- Calibration (Platt scaling)
- Monitoring: PSI, drift

#### 7. When to use / not to use (2 min)
**Quickly list:**

✅ **When YES:**
- Need interpretability
- Real-time inference
- Baseline for tabular
- Online learning
- CTR prediction, fraud detection

❌ **When NO:**
- Complex non-linearities (→ boosting)
- Images/text (→ deep learning)
- Many categorical features (→ CatBoost)

#### 8. Connection to other algorithms (2 min)
- "LogReg = single-layer neural network"
- vs SVM: no probabilities, hinge loss
- vs Naive Bayes: discriminative vs generative
- Generalization: softmax for multi-class

### What you MUST draw
1. ✅ **Sigmoid graph** (S-curve)
2. ✅ **Decision boundary** (straight line in 2D)
3. ✅ **ROC curve** (TPR vs FPR)
4. ✅ **Loss function formula** with derivation via MLE
5. ✅ **Gradient descent** (arrows down a surface)

### Key points for a Senior level
**Show depth of understanding:**

1. **Theory:**
   - Derivation via MLE/MAP
   - Connection to information theory (cross-entropy)
   - Convexity of loss → guarantee of a global minimum
   
2. **Math:**
   - Derivation of gradients (don't just say "apply chain rule", show it)
   - Hessian and Newton's method
   - Numerical stability (log-sum-exp trick)

3. **Production experience:**
   - Online learning (FTRL for CTR)
   - Calibration is important for business
   - Monitoring (PSI, KL divergence for drift)
   - Feature versioning

4. **Trade-offs:**
   - L1 vs L2: sparsity vs stability
   - Batch vs SGD: accuracy vs speed
   - Threshold tuning for a business metric

### Typical interviewer questions
**Be prepared to answer questions from the following categories:**

---

#### 📚 Theoretical and mathematical questions

**1. "Why use cross-entropy instead of MSE for classification?"**
<details>
<summary>Answer</summary>

- **MSE problems**: 
  - Gradients vanish with large errors (sigmoid saturation)
  - Non-convex loss function for classification → local minima
  - No probabilistic interpretation
  
- **Cross-entropy advantages**:
  - Convex function → guarantee of a global minimum
  - Directly related to MLE (maximum likelihood estimation)
  - Penalizes confident errors more strongly
  - Gradient has a simple form: ∇ = (ŷ - y)·x
  
**Gradient formula for MSE**: large errors → σ'(z) → 0 (saturation)  
**Gradient formula for CE**: gradient is proportional to the error, independent of σ'(z)
</details>

**2. "Prove that the logistic regression loss function is convex"**
<details>
<summary>Answer</summary>

Need to show that the **Hessian is positive semi-definite**.

Hessian: H = (1/m)·X^T·R·X, where R = diag(p̂ᵢ(1-p̂ᵢ))

- All diagonal elements of R > 0, since 0 < p̂ < 1
- For any vector v: v^T·H·v = (1/m)·(Xv)^T·R·(Xv) ≥ 0
- Therefore, H is positive semi-definite → J(w) is convex

**Conclusion**: A single global minimum, any gradient descent method will converge.
</details>

**3. "What is the difference between MLE and MAP? How is this related to regularization?"**
<details>
<summary>Answer</summary>

- **MLE (Maximum Likelihood)**: 
  - Maximizes P(D|w) — the probability of the data given the weights
  - No regularization
  - Can overfit
  
- **MAP (Maximum A Posteriori)**:
  - Maximizes P(w|D) ∝ P(D|w)·P(w) via Bayes' theorem
  - P(w) is the prior distribution
  - Takes into account our assumptions about the weights
  
**Connection to regularization**:
- L2 (Ridge) ↔ Gaussian prior: P(w) = N(0, σ²I), λ = 1/(2σ²)
- L1 (Lasso) ↔ Laplace prior: P(w) ∝ exp(-λ|w|)
- Regularization = MAP with the corresponding prior
</details>

**4. "Why is the decision boundary linear if a non-linear sigmoid function is used?"**
<details>
<summary>Answer</summary>

**Key insight**: The sigmoid is applied AFTER the linear combination.

- Decision boundary: {x : P(y=1|x) = 0.5}
- P(y=1|x) = σ(w^T·x + b) = 0.5
- σ(z) = 0.5 ⟺ z = 0 (property of sigmoid)
- Therefore: w^T·x + b = 0 — **the equation of a hyperplane**

**The sigmoid is needed for**:
- Transforming (-∞, +∞) to [0, 1] (probabilities)
- Smooth transition between classes
- Probabilistic interpretation

**For a non-linear boundary**: need non-linear features (polynomials, interactions) or the kernel trick.
</details>

**5. "Derive the gradient of the cross-entropy loss with respect to the weights"**
<details>
<summary>Answer</summary>

**Step-by-step derivation**:

Loss for one example: L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

where ŷ = σ(z) = 1/(1+e^(-z)), z = w^T·x + b

1. ∂L/∂ŷ = -y/ŷ + (1-y)/(1-ŷ)

2. ∂ŷ/∂z = σ(z)·(1-σ(z)) = ŷ·(1-ŷ)  [derivative of sigmoid]

3. Chain rule: ∂L/∂z = (∂L/∂ŷ)·(∂ŷ/∂z)
   = [-y/ŷ + (1-y)/(1-ŷ)]·ŷ·(1-ŷ)
   = -y(1-ŷ) + (1-y)ŷ
   = ŷ - y  ✨ **A beautiful form!**

4. ∂z/∂w = x

5. Final: **∂L/∂w = (ŷ - y)·x**

For a batch: **∇_w J = (1/m)·X^T·(ŷ - y)**
</details>

**6. "What is a logit? How is it related to log-odds?"**
<details>
<summary>Answer</summary>

**Definitions**:
- **Odds**: odds = p/(1-p) — ratio of probabilities
- **Log-odds**: log(p/(1-p))
- **Logit**: logit(p) = log(p/(1-p)) = z = w^T·x + b

**Interpretation**:
- The logit is the **inverse function** of the sigmoid
- logit(σ(z)) = z
- σ(logit(p)) = p

**Why is this useful**:
1. Transforms a probability [0,1] to (-∞, +∞)
2. Makes the dependency on features **linear**
3. The coefficient wⱼ is the change in log-odds for a 1-unit change in xⱼ
4. exp(wⱼ) is the **multiplicative effect** on odds

**Example**: w₁ = 0.5 → if x₁ increases by 1, the odds are multiplied by e^0.5 ≈ 1.65
</details>

**7. "Why not regularize the bias (intercept)?"**
<details>
<summary>Answer</summary>

**Reasons**:

1. **Shift invariance**: the bias just shifts the decision boundary, it doesn't affect its orientation
2. **Physical meaning**: the bias shows the base frequency of the class, it shouldn't be penalized
3. **Scale**: After standardizing features (mean=0), bias ≈ log(n₁/n₀), its magnitude is justified
4. **Math**: Regularizing the bias can shift the boundary to a non-optimal position

**In code**: 
```python
penalty = lambda * sum(w²)  # without b
```

**Exception**: In some cases (very small data), the bias can be weakly regularized.
</details>

---

#### 🔧 Practical questions

**8. "How to handle class imbalance?"**
<details>
<summary>Answer</summary>

**Approaches** (better to combine):

1. **Class weights** (recommended):
   ```python
   class_weight = {0: 1, 1: n_neg/n_pos}
   # or sklearn: class_weight='balanced'
   ```
   Effect: penalizes errors on the minority class more

2. **Resampling**:
   - Oversampling: SMOTE, ADASYN (generates synthetic examples)
   - Undersampling: RandomUnderSampler (loses data)
   - Hybrid: SMOTEENN, SMOTETomek
   
3. **Threshold adjustment**:
   - Don't use 0.5!
   - Tune on a validation set for a metric (max F1, fixed recall)
   
4. **Metrics**:
   - ❌ Accuracy (misleading with imbalance)
   - ✅ PR-AUC (more important than ROC-AUC for a rare positive class)
   - ✅ F1-score, Precision@k, Recall@k

</details>

**9. "Do you need to scale features for logistic regression? Why?"**
<details>
<summary>Answer</summary>

**Yes, critically important!**

**Reasons**:

1. **Gradient Descent convergence**:
   - Different scales → elongated loss surface
   - GD takes small steps for large features, large steps for small ones
   - Slow convergence, zigzag trajectory
   
2. **Regularization**:
   - L2 penalizes ||w||² — the sum of squared weights
   - Without scaling: large features get small weights → less penalized
   - Regularization becomes **unfair**
   
3. **Interpretation**:
   - Coefficients can only be compared with the same scale
   
**Methods**:
- `StandardScaler`: (x - mean)/std → mean=0, std=1 (standard)
- `MinMaxScaler`: [0, 1] (if positives are needed)
- `RobustScaler`: uses the median, robust to outliers

**Important**: Apply the same scaler on test/production!

**Exception**: Tree-based methods (no scaling needed).
</details>

**10. "What to do with categorical features?"**
<details>
<summary>Answer</summary>

**Options**:

1. **One-Hot Encoding** (standard):
   ```python
   pd.get_dummies(df, drop_first=True)
   # drop_first=True to avoid multicollinearity
   ```
   - ✅ Simple, interpretable
   - ❌ High dimensionality with many categories
   
2. **Target Encoding** (mean encoding):
   - Replace the category with the mean of the target
   - ⚠️ **DANGER**: data leakage! Need leave-one-out or CV
   ```python
   # Correctly:
   for train_idx, val_idx in kfold.split(X):
       target_mean = y[train_idx].groupby(X.category).mean()
       X_val.category = X_val.category.map(target_mean)
   ```
   
3. **Frequency Encoding**:
   - Frequency of the category in the data
   - Loses information about the target
   
4. **WOE (Weight of Evidence)**:
   - WOE = ln(P(X|Y=1) / P(X|Y=0))
   - Popular in credit scoring
   
5. **Hash Trick**:
   - For a very large number of categories (millions)
   - Used in CTR prediction

**For unseen categories**: `handle_unknown='ignore'` in OHE or a default value.
</details>

**11. "How to choose the hyperparameter λ (regularization)?"**
<details>
<summary>Answer</summary>

**Methods**:

1. **Cross-Validation** (recommended):
   ```python
   from sklearn.model_selection import GridSearchCV
   
   param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}  # C = 1/λ
   grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
   grid.fit(X_train, y_train)
   best_lambda = 1 / grid.best_params_['C']
   ```
   
2. **Validation curve**:
   - Plot train/val loss vs λ
   - Choose λ where val loss is minimal
   
3. **Regularization path**:
   - Plot coefficients vs λ
   - See when features start to be zeroed out (L1)

**Trade-off**:
- λ → 0: underfitting (high bias)
- λ → ∞: overfitting (high variance)
- Optimal λ: minimizes generalization error

**Heuristics**:
- Small data → more λ
- Many features → more λ
- Multicollinearity → L2 with λ ~ 0.01-1

**Metrics for selection**: ROC-AUC, F1, LogLoss on validation.
</details>

**12. "What to do with missing values?"**
<details>
<summary>Answer</summary>

**Strategies**:

1. **Deletion**:
   - Delete rows: if few missing values (<5%)
   - Delete features: if many missing values (>50%)
   - ⚠️ Loss of information
   
2. **Imputation**:
   - Mean/Median (numerical): simple, but ignores patterns
   - Mode (categorical): most frequent value
   - ✅ **Median is better**: robust to outliers
   
3. **Missing indicator**:
   ```python
   df['feature_missing'] = df['feature'].isna().astype(int)
   df['feature'].fillna(median, inplace=True)
   ```
   Retains the information "was it missing"
   
4. **Advanced imputation**:
   - KNN Imputer: fills based on k nearest neighbors
   - Iterative Imputer: models each feature from others
   - ⚠️ Computationally expensive
   
5. **Separate category**:
   - For categorical: 'missing' as a separate category
   - Can carry useful information

**Important**: 
- Fit imputer on train, transform on test
- Do not allow data leakage

**In production**: Fix the imputation strategy (save median/mode from train).
</details>

---

#### 🏭 Production questions

**13. "What to do with perfect separation?"**
<details>
<summary>Answer</summary>

**Problem**: There exists a hyperplane that completely separates the classes → weights → ±∞

**Symptoms**:
- Very large coefficients (|w| > 100)
- Warnings: "convergence", "ill-conditioned"
- Perfect predictions on train (100% accuracy)

**Causes**:
- Small data + many features
- A feature almost completely determines the class (leak)
- Sparse data

**Solutions**:

1. **L2 regularization** (main):
   - Penalizes large weights
   - Always use λ > 0!
   
2. **Firth's penalized likelihood**:
   - Adds a Jeffrey's prior
   - Bias correction for small samples
   
3. **More data**:
   - If possible, collect more examples
   
4. **Feature selection**:
   - Remove the feature causing separation
   - Check for data leakage

**Detection**: np.max(np.abs(model.coef_)) > threshold
</details>

**14. "How to do online learning for logistic regression?"**
<details>
<summary>Answer</summary>

**Task**: Update the model as new data arrives (streaming).

**Methods**:

1. **SGD (Stochastic Gradient Descent)**:
   ```python
   from sklearn.linear_model import SGDClassifier
   
   model = SGDClassifier(loss='log', learning_rate='adaptive')
   
   # Incremental learning
   for X_batch, y_batch in data_stream:
       model.partial_fit(X_batch, y_batch, classes=[0,1])
   ```
   
2. **Learning rate decay**:
   - Decrease α over time: α_t = α₀ / (1 + t)
   - Or adaptive: α_t depends on the gradient (AdaGrad, Adam)
   
3. **FTRL-Proximal** (Follow-The-Regularized-Leader):
   - Specifically for online with L1
   - Used at Google for CTR
   - Supports sparse updates
   
4. **Mini-batch**:
   - Update not per 1 example, but per batch (100-1000)
   - More stable than pure SGD

**Challenges**:

- **Concept drift**: The distribution changes over time
  - Monitoring: PSI, KL divergence
  - Solution: Decay old data, retrain
  
- **Feature drift**: New categories, change in scale
  - Solution: Online scaling, hash trick for categories
  
- **Calibration**: Probabilities can drift
  - Solution: Periodic recalibration

</details>

**15. "How to monitor a model in production?"**
<details>
<summary>Answer</summary>

**What to monitor**:

1. **Quality metrics**:
   - LogLoss, AUC, F1 over time
   - Precision, Recall by class
   - Calibration (Brier score, reliability curve)
   - ⚠️ Degradation → retrain
   
2. **Data drift**:
   - **PSI (Population Stability Index)**:
     ```
     PSI = Σ (p_actual - p_expected) * ln(p_actual / p_expected)
     ```
     - PSI < 0.1: no drift
     - 0.1 < PSI < 0.25: small drift
     - PSI > 0.25: significant drift
     
   - **KL divergence**: D_KL(P||Q) between train and production
   
3. **Prediction drift**:
   - Distribution of scores (probabilities)
   - Mean prediction, quantiles
   - Shift → loss of calibration
   
4. **Feature drift**:
   - Feature statistics: mean, std, min, max
   - New categories (unseen values)
   - Frequency of missing values
   
5. **Operational metrics**:
   - Latency (inference time)
   - Throughput (requests/sec)
   - Memory usage
   - Error rate (NaN, Inf)

**Alerts**:
- PSI > 0.2 → warning
- AUC drop > 5% → critical
- Latency > SLA → alert

**Dashboard**: Grafana, Tableau with real-time metrics.
</details>

**16. "How to ensure numerical stability in production?"**
<details>
<summary>Answer</summary>

**Problems**:

1. **Overflow/underflow** in exp():
   - exp(-1000) → 0 (underflow)
   - exp(1000) → inf (overflow)
   
2. **log(0)** → -inf

3. **Division by zero**

**Solutions**:

1. **Clip logits**:
   ```python
   z = np.clip(z, -500, 500)  # before sigmoid
   ```
   
2. **Stable log-sigmoid**:
   ```python
   def log_sigmoid_stable(z):
       return -np.log1p(np.exp(-np.abs(z))) - np.maximum(z, 0)
   ```
   
3. **Clip probabilities**:
   ```python
   probs = np.clip(probs, 1e-15, 1 - 1e-15)  # before log()
   ```
   
4. **LogSumExp trick** (for softmax):
   ```python
   def softmax_stable(z):
       z_max = np.max(z)
       return np.exp(z - z_max) / np.sum(np.exp(z - z_max))
   ```
   
5. **Check for NaN/Inf**:
   ```python
   assert not np.any(np.isnan(predictions))
   assert not np.any(np.isinf(predictions))
   ```

**Testing**: Unit tests with edge cases (very large/small values).

**In serving**: Defensive programming, graceful degradation on NaN.
</details>

---

#### ⚖️ Comparative questions

**17. "LogReg vs SVM: when to use which?"**
<details>
<summary>Answer</summary>

| Aspect | Logistic Regression | SVM |
|---|---|---|
| **Loss function** | Cross-entropy (log loss) | Hinge loss |
| **Probabilities** | ✅ Native, calibratable | ❌ Needs calibration (Platt scaling) |
| **Decision boundary** | Soft margin (probabilistic) | Hard margin (margin maximization) |
| **Outliers** | Sensitive | More robust (margin) |
| **Sparse data** | ✅ Good with L1 | ⚠️ Medium |
| **Speed** | ✅ Fast (O(nd)) | ⚠️ Slow (O(n²) to O(n³)) |
| **Kernel trick** | Hard | ✅ Easy (RBF, poly) |
| **Interpretation** | ✅ Log-odds | ⚠️ Harder with kernel |
| **Online learning** | ✅ SGD | ❌ Hard |

**When LogReg**:
- Need probabilities (CTR, scoring)
- Online learning
- Big data
- Interpretability

**When SVM**:
- Small data
- Non-linearities (with kernel)
- Classification without probabilities
- Outliers (robust margin)
</details>

**18. "LogReg vs Naive Bayes?"**
<details>
<summary>Answer</summary>

**Key difference**:
- **LogReg**: Discriminative — models P(y|x) directly
- **Naive Bayes**: Generative — models P(x|y) and P(y), then applies Bayes' rule

**Trade-offs**:

| Aspect | LogReg | Naive Bayes |
|---|---|---|
| **Assumptions** | Fewer (only the form of P(y\|x)) | Strong (feature independence) |
| **Data needed** | More | Less (converges faster) |
| **Asymptotic performance** | Better | Worse |
| **Training speed** | Slower | ✅ Faster |
| **Small sample** | Can overfit | ✅ Better |
| **Correlation features** | ✅ Not a problem | ❌ Violates assumptions |

**Connection**: Under Gaussian assumptions, NB gives a linear boundary (similar to LogReg).

**When NB**:
- Very small data
- Baseline (quick to train)
- Text classification (bag-of-words)
- Features are truly independent

**When LogReg**:
- Enough data
- Correlated features
- Need probability calibration

**Combo**: NB as a warm start for LogReg weights.
</details>

**19. "When is logistic regression better than gradient boosting, when is it worse?"**
<details>
<summary>Answer</summary>

**LogReg is BETTER when**:

1. **Inference speed is needed**: O(d) vs O(trees × depth)
2. **Online learning**: SGD vs batch retraining
3. **Interpretability**: coefficients vs feature importance
4. **Small data**: less overfitting
5. **Simple patterns**: linear dependencies
6. **Production constraints**: easier to deploy, less memory

**Boosting is BETTER when**:

1. **Non-linearities**: finds them automatically
2. **Interactions**: implicit feature interactions
3. **Categories**: native support (CatBoost)
4. **Tabular data**: usually better out-of-the-box
5. **Competitions**: SOTA on Kaggle
6. **There is time**: for hyperparameter tuning

**Hybrid approach**:

1. **Cascade**:
   - LogReg filters (fast): top-1000 candidates
   - Boosting ranks (accurate): top-10 final
   
2. **Feature extraction**:
   - Boosting creates features (leaf indices)
   - LogReg uses them as input
   
3. **Ensemble**:
   - Blend predictions: 0.3·LogReg + 0.7·CatBoost
</details>

---

#### 🔍 Troubleshooting and edge cases

**20. "The model shows 100% accuracy on train. What to do?"**
<details>
<summary>Answer</summary>

**Possible causes**:

1. **Perfect separation**:
   - Check: max(|coef_|) is very large
   - Solution: L2 regularization
   
2. **Data leakage**:
   - Target in features
   - Future information in train
   - Check: feature importance, correlation with target
   
3. **Overfitting**:
   - Too many features
   - No regularization
   - Solution: increase λ, feature selection
   
4. **Duplicates**:
   - Train == Test (by accident)
   - Check: df.duplicated()
   
5. **Trivial task**:
   - The task is genuinely simple
   - Check on new data

**Diagnostics**:
```python
print(f"Max |coef|: {np.max(np.abs(model.coef_))}")
print(f"Val accuracy: {model.score(X_val, y_val)}")

# Feature importance
importance = pd.DataFrame({
    'feature': features,
    'coef': model.coef_[0]
}).sort_values('coef', key=abs)
```

**Red flags**:
- Train acc = 100%, Val acc < 70% → overfitting
- One coefficient >> others → leakage or separation
</details>

**21. "The coefficients are very large and unstable. Why?"**
<details>
<summary>Answer</summary>

**Causes**:

1. **Multicollinearity**:
   - Features are highly correlated
   - VIF > 10
   - Solution: remove correlated features, PCA, L2
   
2. **No scaling**:
   - Different feature scales
   - Solution: StandardScaler
   
3. **Perfect separation**:
   - Weights → ∞
   - Solution: regularization
   
4. **Low regularization**:
   - λ is too small
   - Solution: increase λ
   
5. **Numerical instability**:
   - Ill-conditioned matrix X^T·X
   - Solution: L2 or another solver

**Diagnostics**:
```python
# Correlation matrix
corr = X.corr()
high_corr = np.where(np.abs(corr) > 0.9)

# VIF
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
```

**Solution**:
1. StandardScaler
2. Remove features with VIF > 10
3. Increase C (decrease λ) in sklearn
</details>

**22. "The model predicts only one class. How to fix it?"**
<details>
<summary>Answer</summary>

**Causes**:

1. **Strong class imbalance**:
   - 99% class 0, 1% class 1
   - The model optimizes accuracy → always predicts 0
   
2. **Incorrect threshold**:
   - 0.5 might be unsuitable
   - All probabilities < 0.5
   
3. **Poor calibration**:
   - Probabilities are shifted towards 0 or 1
   
4. **Data problem**:
   - Train/test from different distributions

**Solutions**:

1. **Class weights**:
   ```python
   model = LogisticRegression(class_weight='balanced')
   ```
   
2. **Adjust threshold**:
   ```python
   # Find the optimal threshold on validation
   from sklearn.metrics import f1_score
   
   best_f1, best_thresh = 0, 0.5
   for thresh in np.arange(0.1, 0.9, 0.05):
       y_pred = (probs > thresh).astype(int)
       f1 = f1_score(y_val, y_pred)
       if f1 > best_f1:
           best_f1, best_thresh = f1, thresh
   ```
   
3. **Resampling**: SMOTE for the minority class

4. **Check the probability distribution**:
   ```python
   plt.hist(model.predict_proba(X)[:, 1], bins=50)
   ```

**Diagnostics**:
- If all probs < 0.5 → adjust threshold
- If probs are polarized (0 or 1) → calibration
- If imbalance is 99:1 → class weights
</details>

---

**23. "How to detect and prevent click fraud in an advertising system?"**
<details>
<summary>Answer</summary>

**Features for fraud detection**:

1. **User behavior**:
   - Click rate (too high → bot)
   - Time between clicks (too regular)
   - Mouse movement patterns
   - Session duration
   - Bounce rate
   
2. **Device/Network**:
   - IP reputation, geolocation
   - User-Agent (device fingerprint)
   - VPN/proxy detection
   - Multiple clicks from the same IP
   
3. **Ad engagement**:
   - Conversion rate post-click
   - Time on landing page
   - Scroll depth
   
4. **Temporal patterns**:
   - Clicks at unusual times (3 AM)
   - Burst activity
   - Periodicity

**Model**:
```python
# LogReg for binary: fraud / legitimate
features = [
    'click_rate_last_hour',
    'clicks_from_ip',
    'time_since_last_click',
    'device_reputation',
    'conversion_rate',
    'session_duration',
    # ... embeddings, aggregations
]

model = LogisticRegression(
    class_weight={0: 1, 1: 10},  # fraud is more important
    C=0.1  # regularization
)
```

**Threshold tuning**:
- High precision: few false positives (don't block legitimate users)
- Threshold ~ 0.9 (strict)

**Real-time scoring**:
- Each click → score fraud probability
- > 0.9 → block immediately
- 0.5-0.9 → manual review queue
- < 0.5 → allow

**Feedback loop**:
- Manual labels → retrain weekly
- Adversarial learning (fraud adapts)

**Metrics**:
- Precision@k (top-k suspected frauds)
- $ saved (blocked fraudulent clicks)
- False positive rate (legitimate blocks)

**In production**:
- A/B test: fraud detection ON vs OFF
- Measure: ROI, advertiser satisfaction
</details>

---

### Final Checklist

Before an interview, make sure you can:
- [ ] Derive the loss via MLE from scratch
- [ ] Derive gradients by hand
- [ ] Draw the sigmoid and decision boundary
- [ ] Explain the difference with 3+ other algorithms
- [ ] Discuss numerical stability and edge cases
- [ ] Explain the choice of hyperparameters (λ, solver)

---

## 🚀 Advanced Topics (for in-depth discussion)

### 1. Generative vs Discriminative models
**If the interviewer asks about the connection to Naive Bayes:**

- **Discriminative (LogReg)**: Models P(y|x) directly
- **Generative (Naive Bayes)**: Models P(x|y) and P(y), then applies Bayes' rule

**Connection**: Under Gaussian distribution assumptions, Naive Bayes gives a linear boundary, similar to logistic regression.

**Trade-off**:
- LogReg: needs more data, but fewer assumptions, asymptotically better
- Naive Bayes: converges faster on small data, but has strong assumptions (feature independence)

### 2. Connection to the Exponential Family
Logistic regression is a special case of a **Generalized Linear Model (GLM)**:
- Linear predictor: \(\eta = \mathbf{w}^T\mathbf{x}\)
- Link function: logit (for a Bernoulli distribution)
- Other GLMs: Poisson regression (log link), Gaussian (identity link)

**Why this is important**: It shows that logistic regression is part of a general framework that can be extended to other distributions.

### 3. Kernel Logistic Regression
Like SVM, the kernel trick can be applied:
\[ z = \mathbf{w}^T\phi(\mathbf{x}) + b \]
where \(\phi\) is a non-linear mapping to a high-dimensional space.

**In practice**: Rarely used (heavy), easier to do feature engineering or use neural networks.

### 4. Calibration in production
**Platt Scaling**: Train another logistic regression on top of the scores:
\[ P_{\text{calibrated}} = \sigma(A \cdot z + B) \]

**Isotonic Regression**: Non-parametric monotonic calibration.

**Metric**: Brier score, calibration curve (reliability diagram)

### 5. FTRL (Follow-The-Regularized-Leader) for online learning
Used at Google for CTR prediction:
\[ \mathbf{w}_{t+1} = \arg\min_\mathbf{w} \left( \mathbf{w}^T \sum_{s=1}^t \mathbf{g}_s + \frac{\lambda_1}{2}\lVert\mathbf{w}\rVert_1 + \frac{\lambda_2}{2}\lVert\mathbf{w}\rVert_2^2 \right) \]

**Advantages**:
- Supports L1 (sparse) online
- Works well with hashed features
- Can add new features dynamically

### 6. Feature Hashing (Hash Trick)
For very high-dimensional sparse data:
\[ x_{\text{hash}} = \text{hash}(\text{feature\_name}) \mod D \]

**Pros**: Fixed dimensionality, no dictionary needed
**Cons**: Collisions, loss of interpretability

**Application**: Advertising systems with millions of categorical features

### 7. Weighted Logistic Regression for imbalanced data
Add weights to the loss:
\[ J = -\sum_i w_i [y_i\log\hat{y}_i + (1-y_i)\log(1-\hat{y}_i)] \]

**Strategies**:
- `class_weight='balanced'`: \(w_{\text{class}} = \frac{n}{k \cdot n_{\text{class}}}\)
- Custom weights based on the business cost of errors

### 8. Ordinal Logistic Regression
For ordinal classes (1 star < 2 stars < ... < 5 stars):
- Proportional odds model
- Preserves order during training

### 9. Distributed Training
**For big data:**
- **Parameter Server**: A central server stores weights, workers compute gradients
- **AllReduce**: Decentralized aggregation of gradients (used in Horovod)
- **Data parallelism**: Different machines process different batches

---

## ⚠️ Common mistakes in interviews (what to avoid)

### Conceptual mistakes
1. ❌ "Logistic regression predicts classes" → ✅ "It predicts probabilities, classes are determined by a threshold"
2. ❌ "The sigmoid is for non-linearity" → ✅ "It's for transforming to [0,1], the boundary is still linear"
3. ❌ "MSE is suitable for classification" → ✅ "Cross-entropy is better (convex, MLE, no saturation effect)"
4. ❌ "Accuracy is a good metric" → ✅ "Weak for imbalance, F1/AUC are better"
5. ❌ "Gradient descent always finds the minimum" → ✅ "Only for convex functions (this is true for logistic regression)"

### Mathematical mistakes
1. ❌ Forgetting regularization in formulas
2. ❌ Incorrect sign in the gradient
3. ❌ Confusing log-odds and odds
4. ❌ Regularizing the bias (usually not done)

### Practical mistakes
1. ❌ "No need to scale features" → ✅ "Critical for GD and regularization"
2. ❌ "L1 is always better than L2" → ✅ "Depends on the task: L1→sparse, L2→stable"
3. ❌ Ignoring class imbalance
4. ❌ "Logistic regression doesn't work for non-linear data" → ✅ "You can add polynomials"

### Production mistakes
1. ❌ "The model is trained → done" → ✅ "Need to monitor for drift, retrain regularly"
2. ❌ Not discussing calibration
3. ❌ Ignoring numerical stability
4. ❌ A fixed 0.5 threshold without justification
