rm(list = ls())
#### comscore web-browsing data
library(gamlr)
library(glmnet)

## Browsing History. 
## The table has three colums: [machine] id, site [id], [# of] visits
setwd("./ECON6820-ml/lecture-codes/comscore")
web <- read.csv("CS2006domains.csv")

subset(web,web$id==5625)

## Tell R that 'id' is a factor; we know there are 10000 household machines

n <- 1e4

web$id <- factor(web$id, levels=1:n)

## Read in the actual website names, and use these to create a site factor
## We know that there are 1000, and these are their names (in correct order)

d <- 1e3

sitenames <- scan("CS2006sites.txt", what="character")

web$site <- factor(web$site, levels=1:d, labels=sitenames)

subset(web,web$id==5625)

o<-order(web$id)

web<-web[o,]

## get total visits per-machine and % of time on each site
## tapply(a,b,c) does c(a) for every level of factor b.

machinetotals <- as.vector(tapply(web$visits,web$id,sum)) 

## it returns matrix; we'll make it a vector

web$visitpercent <- 100*web$visits/machinetotals[web$id]

## use this info in a sparse matrix
## this is something you'll be doing a lot; familiarize yourself.

xweb <- sparseMatrix(
	i=as.numeric(web$id), 
	j=as.numeric(web$site), 
	x=web$visitpercent,
	dims=c(nlevels(web$id),nlevels(web$site)),
	dimnames=list(id=levels(web$id), site=levels(web$site)))

## now read in the spending data 

yspend <- read.csv("CS2006totalspend.csv", row.names=1)  # us 1st column as row names

yspend <- as.matrix(yspend) ## good practice to move from dataframe to matrix

## run a regression

spender <- gamlr(xweb, log(yspend), verb=TRUE)
# other common arguments: nlambda, lambda.min.ratio

spender2<-glmnet(xweb,log(yspend),family="gaussian")


plot(spender) ## path plot


## a few examples


B <- coef(spender) ## the coefficients selected under AICc

B <- B[-1,] # drop intercept 

B[which.min(B)] ## low spenders spend a lot of time here

B[which.max(B)] ## big spenders hang out here

