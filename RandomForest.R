# Kaggle Random Forest
# Adrian Mead
library(tidyverse)
library(randomForest)
library(reshape2)
library(ggplot2)
library(GGally)
library(foreach)
source('gini.R')

# Random Forests are an extension of decision trees that attempt to deal with the high-variance issues associated
# with decion trees in a number of ways. First, by buildng many trees from bootstrapped samples of the training 
# data. And second by using some subset of randomy selected predictors with which to build the tree. These two 
# methods manage to decorrelate trees and greatly improve predictive accuracy. Random forests are nice because
# they don't need columns to be transformed/normalized, you can include as many unimportant features as possible 
# with little effect on the predictive accuracy, and manage to achieve solid accuracy with modeling non-linear 
# relationships. As such, we think that they make a solid choice for the safe driver data set and the problem
# of classifying people that will file claims.

# Read in the data; replace -1 with NA
train <- read_csv('train.csv')#, na = c(-1, '-1', -1.0))

# Want to make sure that everything which ends in 'cat' or 'bin' is a factor/logical
sub_cols <- names(train) %>%
  substr(nchar(names(train))-2,
         nchar(names(train)))
# Get just the columns with cat or bin at the end and make them factors and logicals
corrected_cat_columns <- lapply(train[, names(train)[which(sub_cols %in% c('cat','bin'))]],
      factor) %>%
  as.tibble()
# corrected_bin_columns <- lapply(train[, names(train)[which(sub_cols %in% c('bin'))]],
#                                 as.logical) %>%
#   as.tibble()

# Put in NAs for the continous variables since the import process doesn't catch the -1's
a <- bind_cols(lapply(train[, names(train)[-which(sub_cols %in% c('cat', 'bin'))]], function(X){
  X[which(X == -1)] <- NA
  return(X)
}))

# Now combine them back in with the old columns
# We chose not to replace the -1's in the categorical/logical variables with NA. This is because we think there may be 
# information in the the very fact that these categorical variables are NA. So keep it and let it be its own factor.
# However we will want to take care of continuous variable NAs and impute.
drop_nas_train <- bind_cols(a,
  # train[, names(train)[-which(sub_cols %in% c('cat', 'bin'))]],
                         corrected_cat_columns)#,
                         # corrected_bin_columns)

# Going to remove the columns for which more than 50% of data is missing
too_many_NAs <- sapply(drop_nas_train, function(X){
  if(class(X) %in% c('factor', 'logical')){
    sum(X == -1) > NROW(drop_nas_train) * .5
  } else{
    sum(is.na(X)) > NROW(drop_nas_train) * .5
  }
})
names(which(too_many_NAs))
# # It looks like "ps_car_03_cat" has too many NAs, so remove it
drop_nas_train$ps_car_03_cat <- NULL

# Now go in and do imputation on the remaining continuous values
# Mode for the categorical variables
getmode <- function(v, na.rm = TRUE) {
  unique <- unique(na.omit(v)) #identify the unique values within the data
  unique[which.max(tabulate(match(v, unique)))] #tabulate the unique values
}


# And mean for the continous variables
# Remember that we didn't put any NA values back into the categorical variables
imputation_values <- lapply(drop_nas_train, function(X){
  if(class(X) %in% c('factor', 'logical')){ # So categorical variables
    return(getmode(X))
  } else if(class(X) == 'numeric'){ # So continuous variables
    return(mean(X, na.rm=T))
  } else{ # should catch integer
    return(as.integer(mean(X, na.rm=T)))
  }
})

# Now replace all of these NA's with the imputed values
impute_train <- drop_nas_train %>% 
  replace_na(imputation_values)

# Make the integer columns into ordered (in the dataset we are told the integer columns are ordinal)
impute_train <- bind_cols(lapply(impute_train, function(X){
  if(class(X) == 'integer'){
    return(ordered(X))
  } else{
    return(X)
  }
}))

# Check for multicollinearity
alias(lm(target~.-id, impute_train))
# It looks like we have some perfect multicollinearity (some variables are linear combinations of others)
# We can solve this by removing them: 'ps_ind_09_binTRUE' and 'ps_ind_13_binTRUE'
impute_train$ps_ind_09_bin <- NULL
impute_train$ps_ind_13_bin <- NULL

# Looking at multicollinearity
vif(lm(target~.-id, impute_train))
# Not really any columns with a large amount of multicollinearity judging by GVIF^(1/(2*Df))

# Make the target variable a logical
impute_train$target <- factor(train$target)
# Make the id an integer
impute_train$id <- train$id

# randomForest function is having trouble with factors with levels over 53, so remove those columns that violate
which(sapply(impute_train, function(X) length(levels(X))) > 53)
# It looks like the column, 'ps_car_11_cat' has this problem
impute_train$ps_car_11_cat <- NULL

# It looks like we don't have to do too much messing with distribution. Things are pretty normally distributed in general
# This is an in-built function for doing Cross-Validation in randomForest. But instead I choose to use Out-Of-Bag (OOB),
# as that significantly drops runtime
# a <- rfcv(sample[,-c(1,2,10:50)], sample$target, cv.fold=5, scale="log", step=0.5)

set.seed(100)
# We're going to have to cut down the amount of data that we're looking at. Unfortunately, if you try and run a random 
# forest on all ~600K observations, your code will run for hours. You can achieve similar predictive accuracy by sampling
# intelligently. We ended up using all of the observations that filed claims and an equal number that did not
sample <- bind_rows(impute_train[impute_train$target==0,][sample(NROW(impute_train[impute_train$target==0,]), 21694),], # Half are non-claims
                    impute_train[impute_train$target==1,][sample(NROW(impute_train[impute_train$target==1,]), 21694),]) # And the other half are claims

# We produce a pairwise plot of the predictors to try and decide if we need to do any transformations of the variables
ggpairs(sample[which(lapply(sample, class) %in% c('integer', 'numeric'))])
# For the most part it looks pretty good. There aren't any particularly skewed distributions; most variables are either normal 
# or uniform

# I decided to use Out-Of-Bag error estimation to decide what value of m to use. m > 15 doesn't
# seem to produce super-strong results. It also takes longer and longer to run the model
# So iterate from 1 to 20 and pick the m that does the best
oob_estimates <- bind_rows(lapply(X = seq(1, 20), function(m){
  print(m)
  # Use ntree=100 to get a sense of what m-values are better
  # Introduce some parallelization as well
  rf.portseguro <- foreach(ntree=rep(25, 4), .combine=combine, .packages='randomForest', .multicombine=TRUE) %dopar% {randomForest(sample[-c(1,2)],
                                                                                                                                    sample$target,
                                                                                                                                    ntree=ntree,
                                                                                                                                    mtry = m)}
  # Make the prediction
  yhat <- randomForest:::predict.randomForest(rf.portseguro, type = 'prob')
  # Return a nice object with all of the information we need
  return(tibble(m = m, 
                num_trees = 800, 
                gini_coeff = normalized.gini.index(as.numeric(sample$target), 
                                                   yhat[,2]),
                rf = rf.portseguro))
}))
oob_estimates
# We found that m=10 is one of the best-sized trees for finding the lowest OOB

# Now we want to go through with m=10 and find the size tree that works well with the data. Again,
# using OOB error estimates
oob_estimates_ntree <- bind_rows(lapply(X = seq(100, 1000, 50), function(n){
  print(n)
  # Use m = 10 as that value worked above
  rf.portseguro <- randomForest(target ~ .-id, data = sample, mtry = 10, ntree = n, type = 'prob')
  yhat <- randomForest:::predict.randomForest(rf.portseguro, type = 'prob')
  return(tibble(m = 10, 
                num_trees = n, 
                gini_coeff = normalized.gini.index(as.numeric(sample$target), 
                                                   yhat[,2]),
                rf = rf.portseguro))
}))
oob_estimates_ntree
# Looking at the results we see that there is diminishing returns for ntree > 800, so we can pick ntree=800 
# and use that

# Looks like with a sample size of 43K and ntrees=800, we can get the best gini_coeff with m=10

# Looking at what we can do with parallelization. Use ntree = 800, m = 10, and the 43K sample-size
rf.model <- foreach(ntree=rep(200, 4), .combine=combine, .packages='randomForest', .multicombine=TRUE) %dopar% {randomForest(sample[-c(1,2)],
                                                                                                 sample$target,
                                                                                                 ntree=ntree,
                                                                                                 mtry = 10,
                                                                                                 importance = TRUE)}
# Interested in what variables are the most important
varImpPlot(rf.model)
# So just staring at it, it looks like the most important variables are (in decreasing order):
# ps_car_13
# ps_reg_03
# ps_car_06_cat
# ps_car_14
# etc.

# remember_me <- rf.model
# predictions
yhat <- randomForest:::predict.randomForest(rf.model, type = 'prob')
# OOB error estimation
gini_coeff <- normalized.gini.index(as.numeric(sample$target), 
                                   yhat[,2])
gini_coeff

# Now try it on a validation set -- not CV but gives us a gut check on how our model looks 
set.seed(200)
# test_these <- sample(NROW(impute_train), 10000)
# validation <- impute_train[test_these,]
validation <- impute_train[-which(impute_train$id %in% sample$id), ] # pick all observations not used in the training sample
# predictions
yhat <- predict(rf.model, newdata = validation, type = 'prob')
# OOB error estimation
gini_coeff = normalized.gini.index(as.numeric(validation$target), 
                                   yhat[,2])
gini_coeff




######################
##    SUBMISSION    ##
######################
# Read in the data; replace -1 with NA
test <- read_csv('test.csv')#, na = c(-1, '-1', -1.0))

# Want to make sure that everything which ends in 'cat' or 'bin' is a factor/logical
sub_cols <- names(test) %>%
  substr(nchar(names(test))-2,
         nchar(names(test)))
# Get just the columns with cat or bin at the end and make them factors and logicals
corrected_cat_columns <- lapply(test[, names(test)[which(sub_cols %in% c('cat','bin'))]],
                                factor) %>%
  as.tibble()

# Put in NAs for the continous variables
b <- bind_cols(lapply(test[, names(test)[-which(sub_cols %in% c('cat', 'bin'))]], function(X){
  X[which(X == -1)] <- NA
  return(X)
}))

# Now combine them back in with the old columns
drop_nas_test <- bind_cols(b,
                            # train[, names(train)[-which(sub_cols %in% c('cat', 'bin'))]],
                            corrected_cat_columns)#,
# corrected_bin_columns)

# Remove the same columns
drop_nas_test$ps_car_03_cat <- NULL

imputation_values <- lapply(drop_nas_test, function(X){
  if(class(X) %in% c('factor', 'logical')){ # So categorical variables
    return(getmode(X))
  } else if(class(X) == 'numeric'){ # So continuous variables
    return(mean(X, na.rm=T))
  } else{ # should catch integer
    return(as.integer(mean(X, na.rm=T)))
  }
})

# Now replace all of these NA's with the imputed values
impute_test <- drop_nas_test %>% 
  replace_na(imputation_values)

# Make the integer columns into ordered
impute_test <- bind_cols(lapply(impute_test, function(X){
  if(class(X) == 'integer'){
    return(ordered(X))
  } else{
    return(X)
  }
}))

# Remove more columns as decided above
impute_test$ps_ind_09_bin <- NULL
impute_test$ps_ind_13_bin <- NULL

# Go and make sure the id column is an integer
impute_test$id <- test$id

# And remove another column
impute_test$ps_car_11_cat <- NULL

# And make our predictions using the model
yhat <- predict(rf.model, newdata = impute_test, type = 'prob')

# And produce the correctly formatted csv for submission
to_submit <- tibble(id = impute_test$id,
       target = yhat[,2])
write_csv(to_submit, 'RandomForestSubmission4-upsample-alltargets.csv')
