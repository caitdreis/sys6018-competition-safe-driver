#Kaggle Linear Model

#--------------------- Working Directory and Read in Data
#set working directory & read in data
setwd("~/Dropbox (Personal)/Academic/University of Virginia/Data Science Institute/Fall 2017/SYS 6018/sys6018-competition-safe-driver-master")
train <- read.csv("train.csv")

#--------------------- Packages
library(tidyr)
library(psych)
library(tidyverse)
library(e1071)
library(car)
library(caret) #for
install.packages("MLmetrics")
library(MLmetrics)

#--------------------- Data Cleaning & Imputation
train %>% 
  sapply(function(x) sum(x==-1))
# ps_car_03_cat has 411,231 missings and ps_car_05_cat has 266,551 missings, so we'll drop these columns.

train <- train %>%
  select(-ps_car_03_cat, -ps_car_05_cat)
# We will ignore the -1s present in ordinal and categorical variables, on the assumption that keeping
# "-1" as a factor will help our model if missingness is predictive, and it (theoretically) shouldn't
# make much difference if it is not predictive.

# However, we need to deal with significant number of -1s in the continuous variables ps_reg_03 and ps_car_14.
ps_reg_03_df <- train %>% 
  select(ps_reg_03) %>%
  filter(ps_reg_03!=-1)

ps_reg_03_mean <- mean(ps_reg_03_df$ps_reg_03)
# 0.8940473

ps_car_14_df <- train %>% 
  select(ps_car_14) %>%
  filter(ps_car_14!=-1)

ps_car_14_mean <- mean(ps_car_14_df$ps_car_14)
# 0.3746906

# Replace -1s with imputed means
train$ps_reg_03[train$ps_reg_03==-1] <- ps_reg_03_mean
train$ps_car_14[train$ps_car_14==-1] <- ps_car_14_mean

#--------------------- Signficance Testing of Variables & Simple Linear Regression in Training Set
sig.lm <- aov(target ~., data=train) #testing significance of variables in the training set
summary(sig.lm) #output for only significant variables shown below
#                 Df Sum Sq Mean Sq F value   Pr(>F)    
#ps_ind_01           1      7   7.209 206.797  < 2e-16 ***
#ps_ind_05_cat       1     17  17.113 490.917  < 2e-16 ***
#ps_ind_06_bin       1     18  17.769 509.749  < 2e-16 ***
#ps_ind_07_bin       1     11  11.375 326.329  < 2e-16 ***
#ps_ind_08_bin       1      4   4.140 118.764  < 2e-16 ***
#ps_ind_15           1      8   8.407 241.168  < 2e-16 ***
#ps_ind_16_bin       1      8   8.209 235.504  < 2e-16 ***
#ps_ind_17_bin       1     15  15.165 435.051  < 2e-16 ***
#ps_reg_01           1     11  10.685 306.516  < 2e-16 ***
#ps_reg_02           1      9   9.099 261.010  < 2e-16 ***
#ps_car_02_cat       1      5   4.920 141.141  < 2e-16 ***
#ps_car_04_cat       1      5   4.679 134.221  < 2e-16 ***
#ps_car_07_cat       1     16  16.221 465.321  < 2e-16 ***
#ps_car_08_cat       1      4   4.120 118.183  < 2e-16 ***
#ps_car_12           1      3   2.666  76.493  < 2e-16 ***
#ps_car_13           1      6   6.015 172.565  < 2e-16 ***
#Residuals      595157  20747   0.035         

anova_sig.lm <- aov(sig.lm) #creates an ANOVA table to look at significance of the testing
anova_sig.lm #provides the sum of squares across all the columns

#--------------------- Tradition Cross Validation (see K-fold below)
#Set parameters for cross validation
set.seed(100)
train_CV <- 0.7

#create randomly generated sample for cross validation
cv_train <- sample(1:nrow(train),round(train_CV*nrow(train)))

#create test and train datasets
train_sample <- train[cv_train,] #subsets data into training
test_sample <- train[-cv_train,] #subsets data into testing

#build a linear model with all the variables of significance
line.lm <- lm(target ~ ps_car_01_cat +
                ps_car_02_cat+
                ps_car_04_cat+
                ps_car_07_cat+
                ps_car_09_cat+
                ps_car_13+
                ps_ind_04_cat+
                ps_ind_05_cat+
                ps_ind_15+
                ps_ind_16_bin+
                ps_ind_17_bin+ 
                ps_reg_01, train_sample) 
#Residual standard error: 0.1865 on 416635 degrees of freedom
#Multiple R-squared:  0.006774,	Adjusted R-squared:  0.006745 
#F-statistic: 236.8 on 12 and 416635 DF,  p-value: < 2.2e-16

summary(line.lm)
#Estimate Std. Error t value Pr(>|t|)    
#(Intercept)    1.318e-02  2.433e-03   5.415 6.13e-08 ***
#ps_car_01_cat  2.472e-04  1.246e-04   1.984  0.04723 *  
#ps_car_02_cat -2.562e-03  8.847e-04  -2.895  0.00379 ** 
#ps_car_04_cat -3.410e-05  1.675e-04  -0.204  0.83865    
#ps_car_07_cat -1.364e-02  8.623e-04 -15.812  < 2e-16 ***
#ps_car_09_cat  6.012e-04  3.111e-04   1.932  0.05331 .  
#ps_car_13      3.819e-02  1.798e-03  21.238  < 2e-16 ***
#ps_ind_04_cat  5.511e-03  5.904e-04   9.335  < 2e-16 ***
#ps_ind_05_cat  4.507e-03  2.146e-04  21.001  < 2e-16 ***
#ps_ind_15     -9.013e-04  8.764e-05 -10.284  < 2e-16 ***
#ps_ind_16_bin -2.220e-03  7.637e-04  -2.907  0.00365 ** 
#ps_ind_17_bin  1.647e-02  1.060e-03  15.541  < 2e-16 ***
#ps_reg_01      9.405e-03  1.030e-03   9.131  < 2e-16 ***

#Predict on subset of testing data in linear regression
pred_test <- predict(line.lm, test_sample) #predict based on the linear model in the testing data
mse1 <- sum((test_sample$target-pred_test)^2); mse1 #6275.981

#--------------------- Linear models with K-fold cross validation set at 10
#boosted linear model
bstlm.model <- train(target ~ ps_car_01_cat +
                       ps_car_02_cat+
                       ps_car_04_cat+
                       ps_car_07_cat+
                       ps_car_09_cat+
                       ps_car_11_cat+ 
                       ps_car_13+
                       ps_ind_04_cat+
                       ps_ind_05_cat+
                       ps_ind_15+
                       ps_ind_16_bin+
                       ps_ind_17_bin+ 
                       ps_reg_01, train, method = "BstLm", #tuning mstop (# Boosting Iterations)
                     # or nu (Shrinkage)
                     trControl = trainControl(method = "cv", number = 10, #can iterate over best kfold number
                                              verboseIter = TRUE))
summary(bstlm.model)

pred_test1 <- predict(bstlm.model, train) #predict based on the linear model in the testing data
mse2 <- sum((train$target-pred_test1)^2); mse2 #20806.14

#Generalized boosted linear model
train$target <- factor(train$target) #need to make target a factor with 2 levels
glmboost.model <- train(target ~ ps_car_01_cat +
                          ps_car_02_cat+
                          ps_car_04_cat+
                          ps_car_07_cat+
                          ps_car_09_cat+
                          ps_car_11_cat+ 
                          ps_car_13+
                          ps_ind_04_cat+
                          ps_ind_05_cat+
                          ps_ind_15+
                          ps_ind_16_bin+
                          ps_ind_17_bin+ 
                          ps_reg_01, train, method='glmboost', #could tune with mtry or #Randomly Selected Predictors
                        trControl = trainControl(method = "cv", number = 10,
                                                 verboseIter = TRUE))
summary(glmboost.model)

pred_test3 <- predict(glmboost.model, train) #predict based on the linear model in the testing data

#Regularized Logistic Regression, takes significant time
reg.model <- train(target ~ ps_car_01_cat +
                     ps_car_02_cat+
                     ps_car_04_cat+
                     ps_car_07_cat+
                     ps_car_09_cat+
                     ps_car_11_cat+ 
                     ps_car_13+
                     ps_ind_04_cat+
                     ps_ind_05_cat+
                     ps_ind_15+
                     ps_ind_16_bin+
                     ps_ind_17_bin+ 
                     ps_reg_01, train, method='regLogistic', 
                   trControl = trainControl(method = "cv", number = 10,
                                            verboseIter = TRUE))
summary(reg.model)

pred_test4 <- predict(reg.model, train) #predict based on the linear model in the testing data
mse4 <- sum((train$target-pred_test1)^2); mse4

#--------------------- Alternative Coding Structure for Linear models with K-fold cross validation set at 10
#Reference Dr. Gerber: Calculates unnormalized Gini index from ground truth and predicted probabilities.
unnormalized.gini.index = function(ground.truth, predicted.probabilities) {
  if (length(ground.truth) !=  length(predicted.probabilities))
  {
    stop("Actual and Predicted need to be equal lengths!")}
  
  # arrange data into table with columns of index, predicted values, and actual values
  gini.table = data.frame(index = c(1:length(ground.truth)), predicted.probabilities, ground.truth)
  
  # sort rows in decreasing order of the predicted values, breaking ties according to the index
  gini.table = gini.table[order(-gini.table$predicted.probabilities, gini.table$index), ]
  
  # get the per-row increment for positives accumulated by the model 
  num.ground.truth.positivies = sum(gini.table$ground.truth)
  model.percentage.positives.accumulated = gini.table$ground.truth / num.ground.truth.positivies
  
  # get the per-row increment for positives accumulated by a random guess
  random.guess.percentage.positives.accumulated = 1 / nrow(gini.table)
  
  # calculate gini index
  gini.sum = cumsum(model.percentage.positives.accumulated - random.guess.percentage.positives.accumulated)
  gini.index = sum(gini.sum) / nrow(gini.table) 
  return(gini.index)
}

#Calculates normalized Gini index from ground truth and predicted probabilities.
normalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  model.gini.index = unnormalized.gini.index(ground.truth, predicted.probabilities)
  optimal.gini.index = unnormalized.gini.index(ground.truth, ground.truth)
  return(model.gini.index / optimal.gini.index)
}

train$target <- factor(train$target)

K <- 10 #can iterate over number of K folds
rand_nums <- sample(NROW(train), NROW(train))
splits <- cut(1:NROW(train), K)
output <- lapply(1:K, function(X){
  d <- rand_nums[which(levels(splits)[X] == splits)]
  glm.fit <- glm(target ~ ps_car_13 + ps_reg_03 + ps_car_06_cat + ps_car_14,
                 family = 'binomial',
                 data = train[d,])
  yhat <- predict.glm(glm.fit, newdata = train[-d,], type = 'response')
  gini_coeff = normalized.gini.index(as.numeric(train[-d,]$target), 
                                     yhat)
})
output #Gini indexes for each of the 10 folds
#[1] 0.1693486
#[2] 0.1685148
#[3] 0.1692306
#[4] 0.1656972
#[5] 0.1640496
#[6] 0.1648028
#[7] 0.168505
#[8] 0.1670531
#[9] 0.1694688
#[10] 0.1712042

#--------------------- Preparing Testing Data
test <- read_csv("test.csv")

# Impute mean for missing values in ps_reg_03 and ps_car_14. 
psreg03_all <- data.frame(c(train$ps_reg_03, test$ps_reg_03))
names(psreg03_all) <- "var"
psreg03_all <- psreg03_all %>% 
  filter(var!=-1)

psreg03_mean <- mean(psreg03_all$var)

ps_car_14_all <- data.frame(c(train$ps_car_14, test$ps_car_14))
names(ps_car_14_all) <- "var"
ps_car_14_all <- ps_car_14_all %>% 
  filter(var!=-1)

pscar14_mean <- mean(ps_car_14_all$var)

# Replace -1s with imputed means
test$ps_reg_03[test$ps_reg_03==-1] <- psreg03_mean
test$ps_car_14[test$ps_car_14==-1] <- pscar14_mean

table(test$ps_car_02_cat) # mode is 1
table(test$ps_car_11) # mode is 3

test$ps_car_02_cat[test$ps_car_02_cat==-1] <- 1
test$ps_car_11[test$ps_car_11==-1] <-3

#--------------------- Optimal Model Testing in Test Set
#predictions using the best model
preds2 <- predict.glm(glm.fit, test, type = "response"); preds2

#Prepare to write predictions for preds2 to CSV
write.table(preds2, file="Kaggle.csv", row.names = FALSE, sep=";")
preds2 <- read.csv("Kaggle.csv", header=TRUE, sep=";")
preds2$target <- preds2$x
preds2$id <- subset(test, select=c("id")) #only take id column from testing data
preds2$id <- preds2$x
preds2$x <- NULL
preds2 <- preds2[ , order(names(preds2))] #sort columns to fit identified order

#write to csv for submission
write.table(preds2, file = "Linear_preds_cd2.csv", 
            row.names=F, col.names=T, sep=",")
