#Kaggle Linear Model

#--------------------- Working Directory and Read in Data
#set working directory & read in data
setwd("~/Dropbox (Personal)/Academic/University of Virginia/Data Science Institute/Fall 2017/SYS 6018/sys6018-competition-safe-driver-master")
train <- read.csv("train.csv", header=T)

#--------------------- Packages
library(tidyr)
library(psych)

#--------------------- Data Cleaning & Imputation
#Investigate if there is any data cleaning to 
#Count the NAs across all the columns in the training data, sums the length of NA values
na_count <-sapply(train, function(y) sum(length(which(is.na(y)))))
na_count #shows no NAs across the columns

#however, upon reading through the dataset, values that are listed as -1 are actually unknown
train[train==-1]<-NA #replace all -1 in dataset with NA
na_count <-sapply(train, function(y) sum(length(which(is.na(y))))) #re=search for NAs
na_count #shows multiple columns with NAs

#imput none for categorical columns only
train <- train[ , order(names(train))] #sort columns to identify categorical ones
train[, 22:31][is.na(train[, 22:31])] <- "None" #columns 22-31
train[, 33][is.na(train[, 33])] <- "None" #column 33
train[, 39][is.na(train[, 39])] <- "None" #column 39
train[, 41:42][is.na(train[, 41:42])] <- "None" #column 41 through 42

#recheck NAs in columns
na_count <-sapply(train, function(y) sum(length(which(is.na(y)))))
na_count #ps_car_11, ps_car_12, ps_car_14, ps_reg_03 still have missing values

#Build function to impute median value for missing numerical variables
impute_median <- function(df, colnm){ #assigns imputation to new variable impute_median
  for(cols in colnm){ #for every column in all the columns
    df[is.na(df[cols]),cols] <- median(df[,cols], na.rm = T) #if the columns include missing values take the median
  }
  return(df) #return the new dataset
}

#impute missings in numerical columns with median values
train$ps_car_11 <- as.numeric(train$ps_car_11)
train_impute <- impute_median(train,c("ps_car_11","ps_car_12", "ps_car_14", "ps_reg_03"))
na_count <-sapply(train_impute, function(y) sum(length(which(is.na(y)))))
na_count #no missing values

#descriptive statistics
describe(train)

#--------------------- Tradition Cross Validation (see K-fold below)
#Set parameters for cross validation
set.seed(100)
train_CV <- 0.7

#create randomly generated sample for cross validation
cv_train <- sample(1:nrow(train_impute),round(train_CV*nrow(train_impute)))

#create test and train datasets
train_sample <- train_impute[cv_train,] #subsets data into training
test_sample <- train_impute[-cv_train,] #subsets data into testing

#--------------------- Signficance Testing of Variables & Simple Linear Regression in Training Set
model.lm <- aov(target ~., data=train_sample) #testing significance of variables in the training set
summary(model.lm) #output for only significant variables shown below
#                 Df Sum Sq Mean Sq F value   Pr(>F)    
#ps_car_01_cat      12     25   2.048  59.062  < 2e-16 ***
#ps_car_02_cat       2     12   5.781 166.699  < 2e-16 ***
#ps_car_03_cat       2      8   4.212 121.443  < 2e-16 ***
#ps_car_04_cat       1      7   7.135 205.739  < 2e-16 ***
#ps_car_07_cat       2      8   3.897 112.358  < 2e-16 ***
#ps_car_09_cat       5      7   1.334  38.476  < 2e-16 ***
#ps_car_11_cat     103     13   0.129   3.710  < 2e-16 ***
#ps_car_13           1      4   4.177 120.431  < 2e-16 ***
#ps_ind_04_cat       2      4   1.786  51.490  < 2e-16 ***
#ps_ind_05_cat       7     24   3.438  99.129  < 2e-16 ***
#ps_ind_06_bin       1      2   2.389  68.877  < 2e-16 ***
#ps_ind_15           1      6   6.433 185.492  < 2e-16 ***
#ps_ind_16_bin       1      5   5.052 145.654  < 2e-16 ***
#ps_ind_17_bin       1      6   6.479 186.801  < 2e-16 ***
#ps_reg_01           1      3   3.031  87.406  < 2e-16 ***
#Residuals      416460  14444   0.035

anova_model.lm <- aov(model.lm) #creates an ANOVA table to look at significance of the testing
anova_model.lm #provides the sum of squares across all the columns

#build a linear model with all the variables with a p value <2e-16
model2.lm <- glm(target ~ ps_car_01_cat +
                  ps_car_02_cat+
                  ps_car_03_cat+
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
                  ps_reg_01, family=quasipoisson(link = "log"), train_sample) 
summary(model2.lm)

#Predict on subset of testing data in linear regression
pred_test <- predict(model2.lm, test_sample, type="prob") #predict based on the linear model in the testing data
mse1 <- sum((test_sample$target-pred_test)^2); mse1 #6263.703

#boosted linear model
new_train$target <- as.factor(new_train$target)
levels(new_train$target) #"0" "1"
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
                       ps_reg_01, new_train, method = "BstLm", #tuning mstop (# Boosting Iterations)
                     # or nu (Shrinkage)
                     trControl = trainControl(method = "cv", number = 10, #can iterate over best kfold number
                                              verboseIter = TRUE))
summary(bstlm.model)

#--------------------- Linear models with K-fold cross validation set at 10
#Generalized boosted linear model
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
                       ps_reg_01, new_train, method='glmboost', #could tune with mtry or #Randomly Selected Predictors
                     trControl = trainControl(method = "cv", number = 10,
                                              verboseIter = TRUE))
summary(glmboost.model)

#Regularized Logistic Regression
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
                          ps_reg_01, new_train, method='regLogistic', 
                        trControl = trainControl(method = "cv", number = 10,
                                                 verboseIter = TRUE))
summary(reg.model)

#--------------------- Cleaning and Simple Linear Regression in Testing Set
test <- read.csv("test.csv")

#Investigate if there is any data cleaning to 
#Count the NAs across all the columns in the training data, sums the length of NA values
na_count <-sapply(test, function(y) sum(length(which(is.na(y)))))
na_count #shows no NAs across the columns

#however, upon reading through the dataset, values that are listed as -1 are actually unknown
test[test==-1]<-NA #replace all -1 in dataset with NA
na_count <-sapply(test, function(y) sum(length(which(is.na(y))))) #re=search for NAs
na_count #shows multiple columns with NAs

#imput none for categorical columns only
#test <- test[ , order(names(test))] #sort columns to identify categorical ones
test[, 3][is.na(test[, 3])] <- "None" #columns 3 ps_ind_02_cat
test[, 5:6][is.na(test[, 5:6])] <- "None" #column 5 and 6 ps_ind_04_cat  ps_ind_05_cat
test[, 23:25][is.na(test[, 23:25])] <- "None" #column 23 through 25 ps_car_01_cat, ps_car_02_cat, ps_car_03_cat
test[, 27][is.na(test[, 27])] <- "None" #column 27 ps_car_05_cat
test[, 29][is.na(test[, 29])] <- "None" #column 28 ps_car_07_cat
test[, 31][is.na(test[, 31])] <- "None" #column 31 ps_car_09_cat

#recheck NAs in columns
#na_count <-sapply(test, function(y) sum(length(which(is.na(y)))))
#na_count 
#ps_reg_03 #58, ps_car_14 #36 still have missing values

#Build function to impute median value for missing numerical variables
impute_median <- function(df, colnm){ #assigns imputation to new variable impute_median
  for(cols in colnm){ #for every column in all the columns
    df[is.na(df[cols]),cols] <- median(df[,cols], na.rm = T) #if the columns include missing values take the median
  }
  return(df) #return the new dataset
}

#impute missings in numerical columns with median values
test_impute <- impute_median(test,c("ps_reg_03", "ps_car_14"))

#recheck NAs in columns
#na_count2 <-sapply(test_impute, function(y) sum(length(which(is.na(y)))))
#na_count2 

#reformat columns
test_impute$ps_car_11_cat <- as.character(test_impute$ps_car_11_cat)
#test_impute$target <- NULL

#Predict on subset of testing data in linear regression
pred_test3 <- predict(model2.lm, newdata=test_impute, type="prob") #predict based on the linear model in the testing data
#pred_test3 <- ifelse(pred_test3 > 0.5,1,0); pred_test3 #if we wanted a binary classification

#Transform prediction to exponential
#preds.test3 <- sapply(pred_test3, exp) #applies exponential transformation to the prediction testing if desired

#Prepare to write predictions for pred_test3 to CSV
write.table(pred_test3, file="Kaggle.cdlm9.csv", row.names = FALSE, sep=";")
pred_test3 <- read.csv("Kaggle.cdlm9.csv", header=TRUE, sep=";")
View(pred_test3)
pred_test3$target <- pred_test3$x
pred_test3$id <- subset(test_impute, select=c("id")) #only take id column from testing data
pred_test3$id <- pred_test3$x
pred_test3$x.id <- NULL
pred_test3 <- pred_test3[ , order(names(pred_test3))] #sort columns to fit identified order

#write to file
write.table(pred_test3, file = "Kaggle.lmsubmission.csv", 
            row.names=F, col.names=T, sep=",")
