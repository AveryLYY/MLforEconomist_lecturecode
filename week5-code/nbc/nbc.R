### *** data on tv shows from NBC *** ###

library(gamlr)

## new packages

library(tree)


## read in the NBC show characteristics

nbc <- read.csv("./week5-code/nbc/nbc_showdetails.csv")

# PE (Persons Estimates) - number of people watching a program
# GRP (Gross Rating Points) – a common advertising and media metric that represents the total exposure of a show

## create a separate binary column for each Genre,
## make it a data frame, and re-name them for convenience 
## as always, you should do this explicitly to avoid bugs

nbc$Genre <- as.factor(nbc$Genre)
x <- as.data.frame(model.matrix(PE ~ Genre + GRP, data=nbc)[,-1])


names(x) <- c("reality","comedy","GRP")

PE <- nbc$PE

## here's the regression tree.

nbctree <- tree(PE ~ ., data=x, mincut=1)

## Let's compare trees that code factors as (1) dummy variables, (2) factors

#结果中的reality < 0.5 = reality = 0
par(mfrow=c(1,2))

nbctree1 <- tree(PE ~ ., data=x[,-3], mincut=1)
plot(nbctree1)
text(nbctree1, cex=.75, font=2)

# using factors generates tree which is not very easy to read
nbctree2 <- tree(PE ~ nbc$Genre, mincut=1)
plot(nbctree2)
text(nbctree2, cex=.75, font=2)


lm(PE~nbc$Genre)



## Back to our tree using both genre and GRP


nbctree <- tree(PE ~ ., data=x, mincut=1)


## now plot it


par(mfrow=c(1,2))

plot(nbctree, col=8)

text(nbctree, cex=.75, font=2)


## add a look at fit using the predict function

par(mai=c(.8,.8,.2,.2))

plot(PE ~ GRP, data=nbc, col=c(4,2,3)[nbc$Genre], pch=20, ylim=c(45,90))

newgrp <- seq(1,3000,length=1000)


newdata=data.frame(GRP=newgrp, comedy=0, reality=0) # Prediction for category drama

predicted<-predict(nbctree,newdata)

# Predicted values for genre drama

lines(newgrp, predicted, col=4)




newdata=data.frame(GRP=newgrp, comedy=1, reality=0) # Prediction for category comedy

predicted<-predict(nbctree,newdata)

# Predicted PE for genre comedy

lines(newgrp, predicted, col=3)



newdata=data.frame(GRP=newgrp, comedy=0, reality=1) # Prediction for category reality

predicted<-predict(nbctree,newdata)

# Predicted PE for genre reality

lines(newgrp, predicted, col=2)

legend("bottomright",fill=c(4,2,3),c("drama","reality","comedy"))




