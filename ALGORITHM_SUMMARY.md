## LASSO, kNN, CV

### Supervised learing + Tuning + Evaluation
1. split for final evaluation: Randomly split data into train and test
2. Preprocess
    Handle missing value/outliers
    Encode categorical variables
    Standardize features (LASSO is scale-sensitive)
3. k-fold CV on train to choose $\lambda$
    Ramdomly partition training set into k-fold
    For each $\lambda$:
        For each k = 1, …, k:
        1) let train_k = train excluding fold k; let valid_k = fold k
        2) fit preprocessing on train_k
        3) fit model    $\widehat{\operatorname{f}_{\lambda,\ -k}}$ using train_k
        4) predict $\widehat{y_i}=\widehat{f_{\lambda,-k}}\left(x_i\right) for i\ \in valid\ i $
        5) compute validation error: $er\operatorname{r}_{\lambda,k}=\frac{1}{\left|vali\operatorname{d}_k\right|}\sum_{i\invali\operatorname{d}_k}L\left(\operatorname{y}_i,\widehat{\operatorname{y}_i}\right)$
    Compute CV Score $\mathbit{CV}\left(\mathbf{\lambda}\right)=\frac{\mathbf{1}}{\mathbit{k}}\sum_{\mathbit{k}=\mathbf{1}}^{\mathbit{k}}{\mathbit{er}\mathbit{r}_{\mathbf{\lambda},\mathbit{k}}}$
4. Select $\lambda$
    Min rule : arg min cv($\lambda$)
    1se rule: choose the simplest model whose cv($\lambda$) is within 1se of the min
5. Refit final model
    Refit preprocessing on the full training set: arg min $\frac{1}{n}\sum_{i=1}^{n}\left(y_i-f\left(x_i\right)\right)^2+\lambda^\ast R\left(f\right)$
    Fit $\widehat{f_{\lambda^\ast}}$ on the full training set
6. Final test evaluation

### R Code
```r
Fit <- gamlr(X,y, verb = TRUE)
Plot(Fit) interpretation!!
Coef(Fit, select = which.min(AIC(Fit))) ; Coef(Fit, select = which.min(BIC(Fit)))
Cvfit <- cv.gamlr(X,y, verb = TRUE)
coef(Cvfit)
coef(Cvfit, select = “min”)
```

## TREE

### Regression Tree
Initialize root node containing all training data
While there exists a leaf node m that  is allowed to split (n_m > mincut):
    For each predictor j = 1, … , p:
        Consider condidate split point s in node m 
        Compute SSE(j,s) = SSE_left + SSE_right, where each side preicts by its sample mean y
        Choose (j*, s*) that minimizes SSE(j,s)
        Split node m into left/right children using \operatorname{x}_{\operatorname{j}^\ast}\le\ \operatorname{s}^\ast
    Return the grown tree T_0

### Cost-complexity pruning + CV
Given a large tree T_0
For each candidate subtree size m:
    Obtain T_m by weakest-link pruning:
    Estimate out-of-sample loss of T_m using k-fold CV:
        Split data into k folds
        For fold r = 1, …, k:
            Fit large tree on training fold
            Prune to size m
            Compute validation MSE on fold r
        CVLoss(m) = average validation MSE over folds
    Choose m^\ast = arg min CVLoss(m)
    Return the pruned tree T_{m^\ast}

### R Code
```r
T0 <- tree (y~., data, mincut =1)
Cv <- cv.tree(T0, k=10)
Plot(Cv) !! interpretation 
Best_size <- cv$size[which.min(cv$dev)]
T0_pruned <- prune.tree(T0, best = Best_size)
```
 
## Random Forest

Initialize root node containing all training data.
While there exists a leaf node m that is allowed to split (e.g., n_m > mincut):
    For each predictor j = 1,...,p:
        Consider candidate split points s in node m
            (for continuous x_j: midpoints of sorted unique values in node m)
        Compute SSE(j,s) = SSE_left + SSE_right
            where each side predicts by its sample mean y
    Choose (j*, s*) that minimizes SSE(j,s).
    Split node m into left/right children using x_{j*} ≤ s*.
Return the grown tree T0.

Given a large tree T0.
For each candidate subtree size k (or each λ on the pruning path):
    Obtain Tk by weakest-link pruning (nested sequence of subtrees).
    Estimate out-of-sample loss of Tk using K-fold CV:
        split data into K folds
        for fold r = 1,...,K:
            fit large tree on training folds
            prune to size k
            compute validation MSE on fold r
        CVLoss(k) = average validation MSE over folds
Choose k* = argmin_k CVLoss(k).
Return the pruned tree Tk*

### R Code
```r
library(tree)
T0 <- tree(lcavol ~ ., data = prostate, mincut = 1) # grow a large tree
cv <- cv.tree(T0, K = 10)                           # K-fold CV along pruning path (tree sizes)
best_size <- cv$size[which.min(cv$dev)]             # choose best size (leaf nodes)
T_hat <- prune.tree(T0, best = best_size)           # prune
plot(T_hat); text(T_hat, digits = 2)
``` 

## Neural Network Pseudo Code

### I. Core Concepts of Neural Networks
1. Model Structure
Neural Network Essence: Nonlinear regression with adaptive basis functions

f(X) = Σ βm · φm(X; θm)  
       ↑      ↑
    weights  basis functions learned from data
2. Single Hidden Layer Neural Network Formula
Input Layer → Hidden Layer:
    Zm = w₀m⁽¹⁾ + w₁m⁽¹⁾ᵀ · X,  m = 1,...,M

Hidden Layer → Output Layer:
    f(X) = w₀⁽²⁾ + w₁⁽²⁾ᵀ · σ(Z)

where $\sigma(·)$ is the activation function
3. Common Activation Functions
Name	Formula	Characteristics
Sigmoid	$\sigma\left(z\right)=\frac{1}{1+e^{-z}}$	Output in (0,1), classic choice
ReLU	$\sigma\left(z\right)=ma\operatorname{x}{\left(0,z\right)}$	Modern default, computationally efficient
tanh	$\sigma\left(z\right)=\frac{e^z-e^{-z}}{e^z+e^{-z}}$	Output in (-1,1)
 
### II. Parameter Count Calculation ⭐ Exam Focus
For p inputs → M hidden neurons → K outputs:

Total Parameters = (p + 1) × M + (M + 1) × K
                    ↑              ↑
             input→hidden     hidden→output
             (including bias)  (including bias)

Example: MNIST (784 inputs, 30 hidden, 10 outputs)
= (784 + 1) × 30 + (30 + 1) × 10
= 785 × 30 + 31 × 10
= 23,550 + 310
= 23,860 parameters
 
 
### III. Training Neural Networks - Pseudo Code
Step 1: Data Preparation
ALGORITHM: Data Preprocessing
─────────────────────────────────
INPUT: Raw dataset D
OUTPUT: Training set, Test set

1. Load data
2. Convert response variable to factor type (for classification)
3. Check class imbalance
   - Calculate proportion of each class
   - e.g., purchase: 2.58% response rate → highly imbalanced
4. Split into train/test sets
5. Feature scaling (Scaling matters!)
   - Neural networks are sensitive to input scale
Step 2: Model Fitting
ALGORITHM: Neural Network Training
─────────────────────────────────
INPUT: Training data (X_train, y_train)
HYPERPARAMETERS: 
  - size: number of hidden neurons (typically 5-100)
  - decay: L2 regularization parameter (weight decay)
  - maxit: maximum iterations

1. Initialize weights (close to 0 → initial model is approximately linear)

2. FOR iter = 1 TO maxit:
   
   Forward Pass
   2.1 Compute hidden layer: Z = σ(W⁽¹⁾X + b⁽¹⁾)
   2.2 Compute output layer: ŷ = σ(W⁽²⁾Z + b⁽²⁾)
   
   Compute Loss
   2.3 L = (1/n) Σ(yᵢ - ŷᵢ)² + λ·||W||²
                               ↑
                        regularization term (prevents overfitting)
   
   Backpropagation
   2.4 Compute gradients ∂L/∂W using chain rule:
       ∂L/∂W⁽²⁾ = ∂L/∂ŷ · ∂ŷ/∂Z · ∂Z/∂W⁽²⁾
       
   Parameter Update (Gradient Descent)
   2.5 W ← W - t · ∇L(W)
       where t is the learning rate
   
   Check Convergence
   2.6 IF ||∇L||₂ < ε THEN BREAK

3. RETURN trained model

IMPORTANT NOTES:
- Non-convex optimization → can only find local optima
- Recommend multiple random initializations, select best
 
Step 3: Stochastic Gradient Descent (SGD)
ALGORITHM: Mini-batch SGD
─────────────────────────────────
WHY NEEDED: When n is large, computing full gradient is too slow

1. Randomly partition data into J mini-batches

2. FOR epoch = 1 TO num_epochs:
     FOR j = 1 TO J:
       2.1 Take the j-th mini-batch
       2.2 Compute gradient for this batch
       2.3 Update parameters: W ← W - tˢ · ∇L_batch
   
3. Learning rate tˢ should:
   - Decrease as iterations increase
   - Not too fast (otherwise stops before learning all data)
   - Not too slow (otherwise loses computational advantage)
 
### IV. Prediction and Evaluation - Pseudo Code
Step 4: Generate Predictions
ALGORITHM: Neural Network Prediction
─────────────────────────────────
INPUT: Trained model, test data X_test
OUTPUT: Predicted probabilities p̂

1. Compute predicted probabilities via forward pass
   p̂ = predict(model, X_test, type="raw")

2. OPTIONAL: Convert to class predictions
   ŷ = I{p̂ > threshold}
   
   For imbalanced data: 
   - Threshold 0.5 may not be appropriate
   - Consider lowering threshold (e.g., 0.1) to capture more positives
Step 5: Confusion Matrix Evaluation
ALGORITHM: Classifier Evaluation
─────────────────────────────────
                    Predicted
                  FALSE    TRUE
              ┌──────────┬──────────┐
Actual FALSE  │    TN    │    FP    │
              ├──────────┼──────────┤
       TRUE   │    FN    │    TP    │
              └──────────┴──────────┘

METRICS:
- Accuracy = (TP + TN) / Total
- Sensitivity (Recall) = TP / (TP + FN)  ← How many positives captured
- Specificity = TN / (TN + FP)
- Precision = TP / (TP + FP)  ← Of predicted positives, how many are true

FOR IMBALANCED DATA:
- Accuracy can be misleading (predicting all 0s achieves 97%)
- Focus more on Sensitivity and Precision
 
### V. Application: Customer Targeting
Step 6: Targeting Decision
ALGORITHM: Prediction-based Targeting Strategy
─────────────────────────────────
SCENARIO: Retailer can only send N marketing emails

METHOD 1 (Naive): Send to first N customers in database
  → Poor performance, essentially random selection

METHOD 2 (ML-driven): 
  1. Use trained model to predict purchase probability p̂ᵢ for each customer
  2. Sort customers by p̂ᵢ in descending order
  3. Send to top N ranked customers
  
PERFORMANCE COMPARISON (sending 40 emails):
  - Method 1: 1 purchase
  - Method 2: 16 purchases → 16x improvement!

DECISION RULES:
  - Unconstrained: Send when p̂ᵢ > c/r
    (c = mailing cost, r = purchase revenue)
  - Constrained: Select top N% customers by p̂
Lift Curve Interpretation
LIFT CURVE: Evaluates model targeting capability
─────────────────────────────────
X-axis: Proportion of customers selected (sorted by predicted probability)
Y-axis: Proportion of actual purchasers captured

THREE REFERENCE LINES:
1. Perfect Model (upper bound): 
   - Reaches 100% purchasers first, then horizontal
   
2. Your Model (middle):
   - Closer to upper bound = better
   
3. Random Guessing (diagonal):
   - Selecting 10% customers = capturing 10% purchasers

INTERPRETATION:
- Curve bowing toward upper-left corner = better model
- On the graph, selecting 10% customers may capture 30% purchasers
 
### VI. Regularization Methods
THREE METHODS TO PREVENT OVERFITTING:
─────────────────────────────────

1. Early Stopping
   - Don't train to convergence, stop early
   - Effect: Final model stays closer to linear
   - Implementation: Monitor validation loss, stop when it starts increasing

2. Weight Decay (L2 Regularization)
   - Objective function: L(c) + λ·||c||²
   - λ selected via cross-validation
   - Effect: Constrains weight magnitudes

3. Dropout
   - Randomly "turn off" some neurons during training
   - Effect: Prevents neuron co-adaptation
 
### VII. R Code Template

```r
# 1. Data Preparation
trainDf$y <- as.factor(trainDf$purchase)
testDf$y <- as.factor(testDf$purchase)
# Check class balance
table(trainDf$y)  # Check response rate

# 2. Fit Neural Network
library(nnet)
nnetfit <- nnet(y ~ ., 
                data = trainDf,
                size = 10,      # Number of hidden neurons
                decay = 0.5,    # L2 regularization
                maxit = 10000)  # Maximum iterations

# 3. Prediction
phat_nnet <- predict(nnetfit, testDf, type = "raw")

# 4. Confusion Matrix
table(Actual = testDf$y, 
      Predicted = phat_nnet > 0.5)

# Using caret package for detailed metrics
library(caret)
confusionMatrix(
  data = factor(phat_nnet > 0.5, labels = c("0","1")),
  reference = testDf$y
)

# 5. Customer Targeting
# Sort by predicted probability
sorted_idx <- order(-phat_nnet)

# Select top 40 customers
top40 <- sorted_idx[1:40]
sum(testDf$y[top40] == "1")  # Count actual purchases
```  
 
