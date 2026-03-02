# Check and install required packages
if (!require(keras)) install.packages("keras")
if (!require(randomForest)) install.packages("randomForest")
if (!keras::is_keras_available()) {
  keras::install_keras()
}


rm(list = ls())

library(randomForest)
library(caret)

# 1. 设置 Python 环境
library(reticulate)
use_python("/usr/bin/python3", required = TRUE)
py_require("tensorflow")
library(keras)

train <- read.csv("./week7-code/mnist/mnist_train.csv")
test <- read.csv("./week7-code/mnist/mnist_test.csv")
head(test)

# Create a 28*28 matrix with pixel color values
par(mfrow=c(3,3), pty='s')
all_img <- array(dim=c(21,28*28))

for(dim in 1:9) {
  all_img[dim,] <- apply(train[dim,-1], 2, sum)
  number <- train[dim,1]
  z <- array(all_img[dim,], dim=c(28,28))
  z <- z[,28:1] ##right side up
  image(1:28, 1:28, z, main=number, col=gray.colors(255))
}
###===========================
### Initialize keras
# Prepare data for keras
###===========================
X_train <- as.matrix(train[, -1]) / 255  # normalize pixel values, remove label

# alternative standardization
# X_train <- scale(as.matrix(train[, -1]))  # standardize to mean=0, sd=1

y_train <- to_categorical(train$label, num_classes = 10)
X_test <- as.matrix(test) / 255

###====================================
# Build a simple neural network model
###====================================
model_Rectifier <- keras_model_sequential() %>% # a linear stack of layers
  layer_dense(units = 30, input_shape = 784) %>% # dense: fully-connected layer, one hidden layer with 30 units
  layer_dropout(0.2) %>%
  layer_activation("relu") %>%
  layer_dense(units = 10, activation = "softmax") # 10 output units, probability of each task useful in assignment2

###========================
# Compile the model
###========================
model_Rectifier %>% compile(
  loss = "categorical_crossentropy", # for multi-class classification
  optimizer = optimizer_adam(), # default, gradient-based optimizer, adam is a type of optimizer
  metrics = c("accuracy")
)

# Train the model
# the auto-generated plot shows the training progress
# and helps diagnose overfitting
history <- model_Rectifier %>% fit(
  X_train, y_train,
  epochs = 20, # important! 1 epochs = 1 complete set 
  batch_size = 128, # how many observations used in each iteration
  validation_split = 0.1 # reserve 10% of data in X_train, y_train for evaluation during the traininig process, after each epoch
)

# view the summary of this learned model
summary(model_Rectifier)

# save the model weights -- this is crucial for large models since 
# each training session can take a long time to run
model_Rectifier %>% save_model_hdf5("mnist_model.h5")

# load the model weights
model_Rectifier <- load_model_hdf5("mnist_model.h5")

## print confusion matrix
# Get predictions on training data
y_pred <- predict(model_Rectifier, X_train)
y_pred_classes <- k_argmax(y_pred) %>% as.array()
y_true_classes <- k_argmax(y_train) %>% as.array()

# Use caret confusion matrix function
conf_matrix <- confusionMatrix(factor(y_pred_classes), factor(y_true_classes))
print(conf_matrix)

# Calculate metrics using keras evaluation
metrics <- model_Rectifier %>% evaluate(X_train, y_train)
cat("\nTraining Accuracy:", round(metrics['accuracy'] * 100, 2), "%\n")



## Try predicting on the test set
## no real label provided, but we can read it

# Function to display digits
displayDigit <- function(X){
  m <- matrix(unlist(X), nrow = 28, byrow = T)
  m <- t(apply(m, 2, rev))
  image(m, col=grey.colors(255))
}

# Test predictions
par(mfrow=c(3,3), pty='s')
for (i in 1:9){
  Xtest <- test[sample(1:nrow(test), 1), ]
  displayDigit(Xtest)
  x <- as.matrix(Xtest) / 255
  yhat <- predict(model_Rectifier, x)
  pred_digit <- which.max(yhat) - 1
  title(main = paste("Predicted:", pred_digit))
}

#----------------------------------#
par(mfrow=c(3,3), pty='s')
for (i in 1:9){
  idx <- sample(1:nrow(test), 1)
  Xtest <- test[idx, ]
  
  # 如果 test 有 label 列，去掉它
  if ("label" %in% names(test)) {
    x <- as.matrix(Xtest[, -1]) / 255  # 去掉第一列 label
  } else {
    x <- as.matrix(Xtest) / 255
  }
  
  # 确保是 784 列
  dim(x) <- c(1, 784)
  
  displayDigit(Xtest)
  yhat <- predict(model_Rectifier, x)
  pred_digit <- which.max(yhat) - 1
  title(main = paste("Predicted:", pred_digit))
}

#----------------------------------#区别于上一个code chunk是修复了displaydigit function
# 修复的 displayDigit 函数
displayDigit <- function(X){
  # 如果 X 包含 label 列（785列），去掉第一列
  if (length(X) == 785) {
    X <- X[-1]
  }
  m <- matrix(as.numeric(X), nrow = 28, byrow = TRUE)
  m <- t(apply(m, 2, rev))
  image(1:28, 1:28, m, col = grey.colors(255), xaxt = 'n', yaxt = 'n', xlab = '', ylab = '')
}

# 测试预测
par(mfrow = c(3,3), mar = c(1,1,2,1))
for (i in 1:9){
  idx <- sample(1:nrow(test), 1)
  Xtest <- test[idx, ]
  
  # 准备预测数据（去掉 label 列如果有）
  if ("label" %in% names(test)) {
    x <- as.matrix(Xtest[, -1]) / 255
  } else {
    x <- as.matrix(Xtest) / 255
  }
  dim(x) <- c(1, 784)
  
  # 显示图片
  displayDigit(as.numeric(Xtest))
  
  # 预测
  yhat <- predict(model_Rectifier, x)
  pred_digit <- which.max(yhat) - 1
  title(main = paste("Predicted:", pred_digit))
}

#=============================
### Random Forest Model
#=============================
# Ensure the label column is a factor for classification
train$label <- as.factor(train$label)

inClass <- FALSE
if (inClass == FALSE) {
  rf_fit <- randomForest(label ~ ., data = train, ntree = 20)  # Adjust ntrees for better performance
  saveRDS(rf_fit, file = "mnist_rf_model.rds")  # Save the model
} else {
  rf_fit <- readRDS("mnist_rf_model.rds")  # Load the saved model
}

# Print model summary
print(rf_fit)

# Evaluate performance on the training set
train_predictions <- predict(rf_fit, train)
conf_matrix <- table(Predicted = train_predictions, Actual = train$label)
print("Confusion Matrix:")
print(conf_matrix)

### 
###
# TBC: training many models to see which may do well
# need to set aside a validation set from train data