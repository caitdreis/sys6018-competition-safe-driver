#Kaggle Linear Model

#--------------------- Working Directory and Read in Data
#set working directory & read in data
setwd("~/Dropbox (Personal)/Academic/University of Virginia/Data Science Institute/Fall 2017/SYS 6018/sys6018-competition-safe-driver-master")
train <- read.csv("train.csv", header=T)

#--------------------- Packages
library(tidyr)

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

#--------------------- Cross Validation
#Set parameters for cross validation
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
#ps_calc_01          1      0   0.175   5.058   0.0245 *  
#ps_car_01_cat      12     22   1.872  54.071  < 2e-16 ***
#ps_car_02_cat       2     10   5.010 144.727  < 2e-16 ***
#ps_car_03_cat       2      9   4.434 128.082  < 2e-16 ***
#ps_car_04_cat       1      7   6.993 202.005  < 2e-16 ***
#ps_car_05_cat       2      1   0.360  10.398 3.05e-05 ***
#ps_car_07_cat       2      8   4.077 117.770  < 2e-16 ***
#ps_car_08_cat       1      1   1.403  40.517 1.95e-10 ***
#ps_car_09_cat       5      7   1.395  40.288  < 2e-16 ***
#ps_car_11           1      0   0.277   7.992   0.0047 ** 
#ps_car_11_cat     103     13   0.128   3.699  < 2e-16 ***
#ps_car_12           1      1   1.048  30.277 3.75e-08 ***
#ps_car_13           1      5   4.572 132.070  < 2e-16 ***
#ps_car_15           1      0   0.107   3.101   0.0782 .  
#ps_ind_01           1      2   1.793  51.794 6.18e-13 ***
#ps_ind_02_cat       4      2   0.419  12.096 7.85e-10 ***
#ps_ind_03           1      1   0.873  25.207 5.15e-07 ***
#ps_ind_04_cat       2      2   1.182  34.157 1.47e-15 ***
#ps_ind_05_cat       7     22   3.112  89.906  < 2e-16 ***
#ps_ind_06_bin       1      2   2.357  68.078  < 2e-16 ***
#ps_ind_07_bin       1      2   1.618  46.731 8.15e-12 ***
#ps_ind_08_bin       1      1   0.818  23.642 1.16e-06 ***
#ps_ind_12_bin       1      0   0.199   5.757   0.0164 *  
#ps_ind_15           1      7   6.501 187.802  < 2e-16 ***
#ps_ind_16_bin       1      4   4.048 116.932  < 2e-16 ***
#ps_ind_17_bin       1      7   6.505 187.917  < 2e-16 ***
#ps_ind_18_bin       1      0   0.116   3.348   0.0673 .  
#ps_reg_01           1      4   4.040 116.702  < 2e-16 ***
#ps_reg_02           1      1   1.431  41.340 1.28e-10 ***
#ps_reg_03           1      1   0.537  15.512 8.20e-05 *** 
#Residuals      416460  14417   0.035

anova_model.lm <- aov(model.lm) #creates an ANOVA table to look at significance of the testing
anova_model.lm #provides the sum of squares across all the columns

#build a linear model with all the variables with a p value <2e-16
model2.lm <- lm(target ~ ps_car_01_cat +
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
                  ps_reg_01, train_sample) 
summary(model2.lm)
#Residual standard error: 0.1861 on 416506 degrees of freedom
#Multiple R-squared:  0.008901,	Adjusted R-squared:  0.008565 
#F-statistic: 26.53 on 141 and 416506 DF,  p-value: < 2.2e-16

#build a binomial logistic model with all the variables with a p value <2e-16
model3.lm <- glm(target ~ ps_car_01_cat +
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
                   ps_reg_01, family=binomial(link='logit'),data=train_sample) 
summary(model3.lm)

#Predict on subset of testing data in linear regression
pred_test <- predict(model2.lm, test_sample, type="response") #predict based on the linear model in the testing data
mse1 <- sum((test_sample$target-pred_test)^2); mse1 #6290.186

#Predict on subset of testing data in logistic regression
pred_test2 <- predict(model3.lm, test_sample, type="response") #predict based on the linear model in the testing data
mse2 <- sum((test_sample$target-pred_test2)^2); mse2 #6288.693

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
na_count <-sapply(test, function(y) sum(length(which(is.na(y)))))
na_count 
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
na_count2 <-sapply(test_impute, function(y) sum(length(which(is.na(y)))))
na_count2 

#reformat columns
test_impute$ps_car_11_cat <- as.character(test_impute$ps_car_11_cat)
#test_impute$target <- NULL

#Predict on subset of testing data in linear regression
pred_test3 <- predict(model2.lm, test_impute) #predict based on the linear model in the testing data
pred_test3 <- ifelse(pred_test3 > 0.5,1,0); pred_test3

#Predict on subset of testing data in logistic regression
pred_test4 <- predict(model3.lm, test_impute) #predict based on the linear model in the testing data
pred_test4 <- ifelse(pred_test4 > 0.5,1,0); pred_test4

#Write out predictions for pred_test3 to CSV
write.table(pred_test3, file="Kaggle.LM1.csv", row.names = FALSE, sep=";")
pred_test3 <- read.csv("Kaggle.LM1.csv", header=TRUE, sep=";")
View(pred_test3)
pred_test3$target <- pred_test3$x
pred_test3$x <- NULL
pred_test3$id <- NA
pred_test3$id <- seq.int(nrow(pred_test3))
write.table(pred_test3, file="Kaggle.LM12.csv", row.names = FALSE, col.names=TRUE, sep=",")

#Write out predictions for pred_test4 to CSV
write.table(pred_test4, file="Kaggle.LM3.csv", row.names = FALSE, sep=";")
pred_test4 <- read.csv("Kaggle.LM3.csv", header=TRUE, sep=";")
pred_test4$id <- NA
pred_test4$id <- seq.int(nrow(pred_test4))
View(pred_test4)
pred_test4$target <- pred_test4$x
pred_test4$x <- NULL
write.table(pred_test4, file="Kaggle.Lm.csv", row.names = FALSE, col.names=TRUE, sep=",")
