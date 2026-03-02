##### *** MNIST Digit Recognition *** #####

rm(list = ls())

train <- read.csv("./mnist_train.csv")
test <- read.csv("./mnist_test.csv")
head(test)

# Create a 28*28 matrix with pixel color values
par(mfrow=c(3,3),pty='s')
all_img<-array(dim=c(21,28*28))

for(dim in 1:9)
{
  
all_img[dim,]<-apply(train[dim,-1],2,sum)
number<-train[dim,1]
z<-array(all_img[dim,],dim=c(28,28))
z<-z[,28:1] ##right side up
image(1:28,1:28,z,main=number,col=gray.colors(255))
}

### Initialize h2o: to run this, you need to follow the error/warning messages
# to install Java
library(h2o)
## start a local h2o cluster
localH2O = h2o.init(nthreads = -1) # use all CPUs (8 on my personal computer :3)

## MNIST data as H2O
train$label = as.factor(train$label) # convert digit labels to factor for classification

summary(train$label) # balanced labels
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)

## train a simple neural network model
model_Rectifierwd =
  h2o.deeplearning(x = 2:785,  # column numbers for predictors
                   y = 1,   # column number for label
                   training_frame = train_h2o, # data in H2O format
                   activation = "RectifierWithDropout", # algorithm
                   input_dropout_ratio = 0.2, # % of inputs dropout
                   balance_classes = FALSE,
                   hidden = c(30), # 1 layer of 30 nodes
                   epochs = 25) # no. of epochs


## print confusion matrix
h2o.confusionMatrix(model_Rectifierwd) # not bad for a first try

## Try predicting on the test set
## no real label provided, but we can read it
displayDigit <- function(X){
  m <- matrix(unlist(X),nrow = 28,byrow = T)
  m <- t(apply(m, 2, rev))
  image(m,col=grey.colors(255))
}

par(mfrow=c(3,3),pty='s')
for (i in 1:9){
  Xtest <- test[sample(1:nrow(test), 1), -1]
  displayDigit(Xtest)
  x <- as.h2o(Xtest)
  yhat <- h2o.predict(model_Rectifierwd, x)
  print(yhat)
}

 
## Train a Random forest model
inClass = FALSE
if (inClass == FALSE) {
  rf_fit = h2o.randomForest(x = 2:785, 
                            y = 1,
                            training_frame = train_h2o,
                            ntrees = 20, # adjust to 3000 if want better performance
                            # stopping_rounds = 2,
                            model_id = "rf_MNIST"
  )
  h2o.saveModel(rf_fit, path = "mnist" )  
} else  {
  rf_fit = h2o.loadModel(file.path("mnist", "rf_MNIST"))
} 

summary(rf_fit)  

# performance on train
h2o.confusionMatrix(rf_fit)


####################################################################
# training many models to see which may do well
# need to set aside a validation set from train data

if (inClass == FALSE) {
  # it will take some time to train all
  
  EPOCHS = 2
  args <- list(
    list(epochs=EPOCHS),
    list(epochs=EPOCHS, activation="Tanh"),
    list(epochs=EPOCHS, hidden=c(512,512)),
    list(epochs=5*EPOCHS, hidden=c(64,128,128)),
    list(epochs=5*EPOCHS, hidden=c(512,512), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(50,50), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5),
    list(epochs=5*EPOCHS, hidden=c(100,100,100), 
         activation="RectifierWithDropout", input_dropout_ratio=0.2, l1=1e-5)
  )
  
  run <- function(extra_params) {
    str(extra_params)
    print("Training.")
    model <- do.call(h2o.deeplearning, modifyList(list(x=2:785, y=1,
                                                       training_frame=train_h2o), extra_params))
    sampleshist <- model@model$scoring_history$samples
    samples <- sampleshist[length(sampleshist)]
    time <- model@model$run_time/1000
    print(paste0("training samples: ", samples))
    print(paste0("training time   : ", time, " seconds"))
    print(paste0("training speed  : ", samples/time, " samples/second"))
    
    print("Scoring on test set.")
    p <- h2o.performance(model, test_h2o)
    cm <- h2o.confusionMatrix(p)
    test_error <- cm$Error[length(cm$Error)]
    print(paste0("test set error  : ", test_error))
    
    c(paste(names(extra_params), extra_params, sep = "=", collapse=" "), 
      samples, sprintf("%.3f", time), 
      sprintf("%.3f", samples/time), sprintf("%.3f", test_error))
  }
  
  writecsv <- function(results) {
    table <- matrix(unlist(results), ncol = 5, byrow = TRUE)
    colnames(table) <- c("parameters", "training samples",
                         "training time", "training speed", "test set error")
    table
  }
  
  table = writecsv(lapply(args, run))
  save(table, file="mnist.h2o.table_results.RData")
  
} else {
  load("mnist.h2o.table_results.RData")
  table
}

print(table)

h2o.shutdown(prompt=FALSE)