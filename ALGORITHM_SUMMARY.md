## LASSO, kNN, CV

### Supervised learing + Tuning + Evaluation
1. Split for final evaluation: Randomly split data into train and test
2. Preprocess
    - Handle missing value/outliers
    - Encode categorical variables
    - Standardize features (LASSO is scale-sensitive)
3. k-fold CV on train to choose $\lambda$
    - Ramdomly partition training set into k-fold
    - For each $\lambda$:

        For each k = 1, …, k:

        1) let train_k = train excluding fold k; let valid_k = fold k

        2) fit preprocessing on train_k

        3) fit model    $\widehat{\operatorname{f}_{\lambda,\ -k}}$ using train_k

        4) predict $\widehat{y_i}=\widehat{f_{\lambda,-k}}\left(x_i\right) for i\ \in valid\ i$

        5) compute validation error: $err_{\lambda,k}=\frac{1}{\left|valid_k\right|}\sum_{i\in valid_k}L\left(\operatorname{y}_i,\widehat{\operatorname{y}_i}\right)$

    - Compute CV Score $CV(\lambda)=\frac{1}{k}\sum_{k=1}^{k}err_{\lambda,k}$
4. Select $\lambda$
    - Min rule : arg min cv($\lambda$)
    - 1se rule: choose the simplest model whose cv($\lambda$) is within 1se of the min
5. Refit final model
    - Refit preprocessing on the full training set: arg min $\frac{1}{n}\sum_{i=1}^{n}\left(y_i-f\left(x_i\right)\right)^2+\lambda^\ast R\left(f\right)$
    - Fit $\widehat{f_{\lambda^\ast}}$ on the full training set
6. Final test evaluation

### R Code
```r
Fit <- gamlr(X,y, verb = TRUE)
Plot(Fit) #interpretation!!
Coef(Fit, select = which.min(AIC(Fit))) ; Coef(Fit, select = which.min(BIC(Fit)))
Cvfit <- cv.gamlr(X,y, verb = TRUE)
coef(Cvfit)
coef(Cvfit, select = “min”)
```

## TREE

### Regression Tree

**Algorithm: Growing a Regression Tree**

**Initialize:** Root node containing all training data

**While** there exists a leaf node m that is allowed to split (n_m > mincut):

    **For** each predictor j = 1, …, p:
    
    - Consider candidate split points s in node m
      - *For continuous x_j:* midpoints of sorted unique values in node m
    - Compute SSE(j,s) = SSE_left + SSE_right
      - Each side predicts by its sample mean ȳ
    
    **Choose** (j*, s*) that minimizes SSE(j,s)
    
    **Split** node m into left/right children using x_{j*} ≤ s*

**Return** the grown tree T_0

### Cost-Complexity Pruning + CV

Given a large tree T₀:

For each candidate subtree size m:
    
    **Obtain** Tₘ by weakest-link pruning
    
    **Estimate** out-of-sample loss of Tₘ using k-fold CV:
        1. Split data into k folds
        2. For fold r = 1, …, k:
           - Fit large tree on training folds
           - Prune to size m
           - Compute validation MSE on fold r
        3. CVLoss(m) = average validation MSE over folds

**Choose** m* = argmin_m CVLoss(m)

**Return** the pruned tree T_{m*}

### R Code
```r
T0 <- tree (y~., data, mincut =1)
Cv <- cv.tree(T0, k=10)
Plot(Cv) !! interpretation 
Best_size <- cv$size[which.min(cv$dev)]
T0_pruned <- prune.tree(T0, best = Best_size)
```
 
## Random Forest

### Regression Tree

**Initialize:** Root node containing all training data

**While** there exists a leaf node m that is allowed to split (n_m > mincut):

    **For** each predictor j = 1, …, p:
    
    - Consider candidate split points s in node m
      - *For continuous x_j:* midpoints of sorted unique values in node m
    - Compute SSE(j,s) = SSE_left + SSE_right
      - Each side predicts by its sample mean ȳ
    
    **Choose** (j*, s*) that minimizes SSE(j,s)
    
    **Split** node m into left/right children using x_{j*} ≤ s*

**Return** the grown tree T_0
### Cost-Complexity Pruning + CV

Given a large tree T₀:

For each candidate subtree size m:
    
    **Obtain** Tₘ by weakest-link pruning
    
    **Estimate** out-of-sample loss of Tₘ using k-fold CV:
        1. Split data into k folds
        2. For fold r = 1, …, k:
           - Fit large tree on training folds
           - Prune to size m
           - Compute validation MSE on fold r
        3. CVLoss(m) = average validation MSE over folds

**Choose** m* = argmin_m CVLoss(m)

**Return** the pruned tree T_{m*}

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

#### 1. Model Structure

**Neural Network Essence:** Nonlinear regression with adaptive basis functions

$$f(X) = \sum_{m=1}^{M} \beta_m \cdot \phi_m(X; \theta_m)$$

where $\beta_m$ are weights and $\phi_m(X; \theta_m)$ are basis functions learned from data.

#### 2. Single Hidden Layer Neural Network

**Input Layer → Hidden Layer:**
$$Z_m = w_{0m}^{(1)} + (w_{1m}^{(1)})^T X, \quad m = 1,\ldots,M$$

**Hidden Layer → Output Layer:**
$$f(X) = w_0^{(2)} + (w_1^{(2)})^T \sigma(Z)$$

where $\sigma(\cdot)$ is the activation function.

#### 3. Common Activation Functions

| Name | Formula | Characteristics |
|------|---------|-----------------|
| **Sigmoid** | $\sigma(z) = \frac{1}{1+e^{-z}}$ | Output in $(0,1)$, classic choice |
| **ReLU** | $\sigma(z) = \max(0, z)$ | Modern default, computationally efficient |
| **tanh** | $\sigma(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | Output in $(-1,1)$ |

 
### II. Parameter Count Calculation ⭐ Exam Focus

**Formula:** For p inputs → M hidden neurons → K outputs:

$$\text{Total Parameters} = (p + 1) \times M + (M + 1) \times K$$

where:
- $(p + 1) \times M$ = input→hidden layer (including bias)
- $(M + 1) \times K$ = hidden→output layer (including bias)

**Example:** MNIST (784 inputs, 30 hidden, 10 outputs)

$$\begin{align}
&= (784 + 1) \times 30 + (30 + 1) \times 10 \\
&= 785 \times 30 + 31 \times 10 \\
&= 23,550 + 310 \\
&= 23,860 \text{ parameters}
\end{align}$$

 
### III. Training Neural Networks - Pseudo Code

#### Step 1: Data Preparation

**ALGORITHM: Data Preprocessing**

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

#### Step 2: Model Fitting

**ALGORITHM: Neural Network Training**

INPUT: Training data (X_train, y_train)

HYPERPARAMETERS:

- `size`: number of hidden neurons (typically 5-100)

- `decay`: L2 regularization parameter (weight decay)

- `maxit`: maximum iterations

**Procedure:**

1. Initialize weights (close to 0 → initial model is approximately linear)

2. **FOR** iter = 1 TO maxit:
    
    **Forward Pass**
    - 2.1: Compute hidden layer: $Z = \sigma(W^{(1)}X + b^{(1)})$
    - 2.2: Compute output layer: $\hat{y} = \sigma(W^{(2)}Z + b^{(2)})$
    
    **Compute Loss**
    - 2.3: $L = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 + \lambda \|W\|^2$
    
    **Backpropagation**
    - 2.4: Compute gradients using chain rule:
      $$\frac{\partial L}{\partial W^{(2)}} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial Z} \cdot \frac{\partial Z}{\partial W^{(2)}}$$
    
    **Parameter Update (Gradient Descent)**
    - 2.5: $W \leftarrow W - t \cdot \nabla L(W)$ where $t$ is the learning rate
    
    **Check Convergence**
    - 2.6: **IF** $\|\nabla L\|_2 < \epsilon$ **THEN BREAK**

3. **RETURN** trained model

**Important Notes:**

- Non-convex optimization → can only find local optima

- Recommend multiple random initializations, select best

#### Step 3: Stochastic Gradient Descent (SGD)

**ALGORITHM: Mini-batch SGD**

**Why Needed:** When n is large, computing full gradient is too slow

1. Randomly partition data into J mini-batches

2. **FOR** epoch = 1 TO num_epochs:
    - **FOR** j = 1 TO J:
      - 2.1: Take the j-th mini-batch
      - 2.2: Compute gradient for this batch
      - 2.3: Update parameters: $W \leftarrow W - t^s \cdot \nabla L_{\text{batch}}$

3. Learning rate $t^s$ should:
    - Decrease as iterations increase
    - Not too fast (otherwise stops before learning all data)
    - Not too slow (otherwise loses computational advantage)
### IV. Prediction and Evaluation

#### Step 4: Generate Predictions

**ALGORITHM: Neural Network Prediction**

INPUT: Trained model, test data X_test

OUTPUT: Predicted probabilities p̂

1. Compute predicted probabilities via forward pass
    ```
    p̂ = predict(model, X_test, type="raw")
    ```

2. Convert to class predictions (optional)
    ```
    ŷ = I{p̂ > threshold}
    ```
    
    **For imbalanced data:** 
    - Threshold 0.5 may not be appropriate
    - Consider lowering threshold (e.g., 0.1) to capture more positives

#### Step 5: Confusion Matrix Evaluation

**ALGORITHM: Classifier Evaluation**

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Accuracy** | (TP + TN) / Total | Overall correctness |
| **Sensitivity (Recall)** | TP / (TP + FN) | How many positives captured |
| **Specificity** | TN / (TN + FP) | How many negatives captured |
| **Precision** | TP / (TP + FP) | Of predicted positives, how many are true |

**For Imbalanced Data:**

- Accuracy can be misleading (predicting all 0s achieves 97%)

- Focus more on Sensitivity and Precision

### V. Application: Customer Targeting

#### Step 6: Targeting Decision

**ALGORITHM: Prediction-based Targeting Strategy**

SCENARIO: Retailer can only send N marketing emails

**Method 1 (Naive):** Send to first N customers in database
- Result: Poor performance, essentially random selection

**Method 2 (ML-driven):**
1. Use trained model to predict purchase probability $\hat{p}_i$ for each customer
2. Sort customers by $\hat{p}_i$ in descending order
3. Send to top N ranked customers

**Performance Comparison (sending 40 emails):**

| Method | Purchases | Improvement |
|--------|-----------|------------|
| Method 1 (Random) | 1 | baseline |
| Method 2 (ML-driven) | 16 | **16x improvement** |

**Decision Rules:**
- **Unconstrained:** Send when $\hat{p}_i > \frac{c}{r}$ where c = mailing cost, r = purchase revenue
- **Constrained:** Select top N% customers by $\hat{p}_i$

#### Step 7: Lift Curve Interpretation

**LIFT CURVE:** Evaluates model targeting capability

**Axes:**

- X-axis: Proportion of customers selected (sorted by predicted probability)

- Y-axis: Proportion of actual purchasers captured

**Three Reference Lines:**

1. **Perfect Model** (upper bound)
    - Reaches 100% purchasers first, then horizontal at 100%
    
2. **Your Model** (middle curve)
    - Closer to upper bound = better performance
    
3. **Random Guessing** (diagonal baseline)
    - Selecting 10% of customers ≈ capturing 10% of purchasers

**Interpretation:**

- Curve bowing upward toward upper-left corner = stronger model

- Example: Selecting top 10% of customers may capture 30% of purchasers

- Gap between your model and diagonal = added value from ML targeting

### VI. Regularization Methods

**Three Methods to Prevent Overfitting:**

#### 1. Early Stopping
- **Concept:** Stop training before convergence
- **Effect:** Final model stays closer to linear
- **Implementation:** Monitor validation loss, stop when it starts increasing

#### 2. Weight Decay (L2 Regularization)
- **Objective:** $L(W) + \lambda \|W\|^2$
- **Tuning:** Select $\lambda$ via cross-validation
- **Effect:** Constrains weight magnitudes

#### 3. Dropout
- **Concept:** Randomly deactivate neurons during training
- **Effect:** Prevents neuron co-adaptation
- **Usage:** Common in deep networks, reduces overfitting

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
 
