##### *** Tabloid Mailing Data *** #####
rm(list = ls())
setwd("./week7-code/tabloid")
library(nnet)
source("plot.nnet.R")

## purchase among 10000 households in the mailing experiment
## response value is binary 0/1
trainDf = read.csv("Tabloid_train.csv")
testDf = read.csv("Tabloid_test.csv")

trainDf$purchase = as.factor(trainDf$purchase)
testDf$purchase = as.factor(testDf$purchase)
names(trainDf)[1]="y"
names(testDf)[1]="y"

## Let's look at the raw data: How many actually responded?
## note that the response is highly unbalanced
## 2.58% responds
summary(trainDf$y)

## setup storage for results
phatL = list() #store the test phat for the different methods here

### Logistic regression

## First, let's fit a logistic regression with glm
lgfit = glm(y~.,trainDf,family=binomial)
print(summary(lgfit))
phat_logit = predict(lgfit, testDf, type="response")
phatL$logit = matrix(phat_logit, ncol=1) #logit phat

## look at in-sample fit
phat_IS <- predict(lgfit, trainDf, type = "response")
plot(phat_IS ~ trainDf$y, col = c("red", "blue"),
     xlab = "purchase", ylab = "phat", ylim = c(0,1.05), cex.text = 0.7)

### Neural network model

## with 1 hidden layer, no tuning
## other packages: keras, h2o, caret for deep NN and efficient computation
nnetfit = nnet(y~., data=trainDf, size=10, decay=0.5, maxit=10000)
print(summary(nnetfit))

## visualize the estimated network
## size of the line represents the magnitude of parameter estimates
plot.nnet(nnetfit)

### Prediction and plotting fit
phat_nnet = predict(nnetfit,testDf,type="raw")
phatL$nnet = matrix(phat_nnet, ncol=1) #nnet phat
phat_IS_nnet <- predict(nnetfit, trainDf, type = "raw")

## Confusion matrix
library(caret)

## TP, TN, FP, FN
table(Actual_purchase = testDf$y, Predicted_purchase = phat_logit > 0.5)
## main source of error: false negative

## same table with caret built-in function
## report other measures of classier performance
confusionMatrix(data = factor(phat_logit > 0.5, labels = c("0", "1")), reference = testDf$y)

confusionMatrix(data = factor(phat_nnet > 0.5, labels = c("0", "1")), reference = testDf$y)


## Compare the prediction (on the test set) from logistic regression vs NN
plot(phatL$logit, phatL$nnet)
# Can further explore for whom do these two methods disagree

### Who should we mail to?
train_aug <- cbind(trainDf, phat_IS, phat_IS_nnet)
train_aug <- train_aug[,c(1,6,7)]

## target the first 40
train_aug[1:40,]

## rerank based on nnet-predicted probabilities
sorted_phat <- order(-phat_IS_nnet)
train_aug[sorted_phat[1:40], ]
sum(train_aug[sorted_phat[1:40],'y']==1)
# we get 15/258 purchases out of the first 40 customers targeted!

## rarank
sorted_phat <- order(-phat_IS)
train_aug[sorted_phat[1:40], ]
# we get 16
sum(train_aug[sorted_phat[1:40],'y']==1)
