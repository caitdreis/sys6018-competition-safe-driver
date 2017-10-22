#Kaggle Linear Model

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
train[, 41:42][is.na(train[, 41:42])] <- "None" #column 33

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

#--------------------- Cross Validation
#Set parameters for cross validation
train_CV <- 0.7

#create randomly generated sample for cross validation
cv_train <- sample(1:nrow(train_impute),round(train_CV*nrow(train_impute)))

#create test and train datasets
train_sample <- train_impute[cv_train,] #subsets data into training
test_sample <- train_impute[-cv_train,] #subsets data into testing

#--------------------- Signficance Testing of Variables & Simple Linear Regression
model.lm <- aov(target ~., data=train_impute) #testing significance of variables in the training set
summary(model.lm) #output for only significant variables shown below
#                 Df Sum Sq Mean Sq F value   Pr(>F)    
#ps_car_01_cat      12     35   2.958  85.025  < 2e-16 ***
#ps_car_02_cat       2     15   7.398 212.656  < 2e-16 ***
#ps_car_03_cat       2     13   6.549 188.254  < 2e-16 ***
#ps_car_04_cat       9     14   1.550  44.567  < 2e-16 ***
#ps_car_05_cat       2      1   0.379  10.903 1.84e-05 ***
#ps_car_06_cat      17      6   0.355  10.201  < 2e-16 ***
#ps_car_07_cat       2      9   4.719 135.654  < 2e-16 ***
#ps_car_08_cat       1      2   1.951  56.072 7.00e-14 ***
#ps_car_09_cat       5      8   1.564  44.970  < 2e-16 ***
#ps_car_11           1      0   0.243   6.990 0.008198 ** 
#ps_car_11_cat      94     13   0.134   3.857  < 2e-16 ***
#ps_car_12           1      1   0.978  28.117 1.14e-07 ***
#ps_car_13           1      5   5.397 155.160  < 2e-16 ***
#ps_ind_01           1      2   2.382  68.479  < 2e-16 ***
#ps_ind_02_cat       4      2   0.553  15.907 5.01e-13 ***
#ps_ind_03           1      1   1.290  37.078 1.14e-09 ***
#ps_ind_04_cat       2      4   2.165  62.225  < 2e-16 ***
#ps_ind_05_cat       7     32   4.546 130.684  < 2e-16 ***
#ps_ind_06_bin       1      3   3.102  89.159  < 2e-16 ***
#ps_ind_07_bin       1      2   2.142  61.569 4.28e-15 ***
#ps_ind_08_bin       1      1   0.984  28.284 1.05e-07 ***
#ps_ind_12_bin       1      0   0.202   5.801 0.016022 *  
#ps_ind_15           1      9   9.233 265.412  < 2e-16 ***
#ps_ind_16_bin       1      6   6.300 181.105  < 2e-16 ***
#ps_ind_17_bin       1      9   9.257 266.103  < 2e-16 ***
#ps_ind_18_bin       1      0   0.132   3.809 0.050988 .  
#ps_reg_01           1      6   5.947 170.960  < 2e-16 ***
#ps_reg_02           1      2   2.204  63.360 1.72e-15 ***
#ps_reg_03           1      1   0.526  15.117 0.000101 ***
#Residuals      595008  20698   0.035           

anova_model.lm <- aov(model.lm) #creates an ANOVA table to look at significance of the testing
anova_model.lm #provides the sum of squares across all the columns
anova_model.lm <- anova(model.lm) 
anova_significance <- data.frame(anova_model.lm$`Pr(>F)`) #creates a dataframe of the p-values from the significance testing
anova_significance$varname <- rownames(anova_model.lm) #creates new variable from the rownames of the linear model
colnames(anova_significance) <- c("p_value", "varname") #combines columns p-value and varname
anova_significance <- anova_significance[order(anova_significance$p_value),] #order the p-values in the signifance object

#Iteratively test variables based on increasing p value
for(i in 3:nrow(anova_significance)){ #for each row in the signifiance object based on the ANOVA testing
  train_df <- train_sample[,c(head(anova_significance$varname,i),"target")] 
  test_df <- test_sample[,c(head(anova_significance$varname,i),"target")] 
  temp_lin_model <- lm(target~., train_df) #creates a temporary linear model object with the signifiance dataframe
  
  temp_pred <- predict(temp_lin_model,test_sample) #uses the temporary linear model to predict based on the testing data
  mse <- sum((test_df$target-temp_pred)^2) #computes the mean square error for all variables
  print(i) #prints list of variable numbers
  print(mse) #prints list of mean square errors
}

#build model with optimum number of variables
train_data <- train_impute[,c(head(anova_significance$varname,28),"target")] #uses the top variables and lists in the training data
model2.lm <- lm(target~., train_data) #create a new linear model with target and the top signifiance variables
