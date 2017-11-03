# Kaggle Random Forest
library(tidyverse)
library(randomForest)
library(reshape2)
library(ggplot2)
library(GGally)
library(foreach)
source('gini.R')

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

###########
# Put in NAs for the continous variables
a <- bind_cols(lapply(train[, names(train)[-which(sub_cols %in% c('cat', 'bin'))]], function(X){
  X[which(X == -1)] <- NA
  return(X)
}))

###########

# Now combine them back in with the old columns
drop_nas_train <- bind_cols(a,
  # train[, names(train)[-which(sub_cols %in% c('cat', 'bin'))]],
                         corrected_cat_columns)#,
                         # corrected_bin_columns)

# Going to remove the columns for which more than 50% of data is missing
too_many_NAs <- sapply(drop_nas_train, function(X){
  sum(is.na(X)) > NROW(drop_nas_train) * .5
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

# Make the integer columns into ordered
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
impute_train$id <- train$id

# randomForest function is having trouble with factors with levels over 53, so remove those columns that violate
which(sapply(impute_train, function(X) length(levels(X))) > 53)
# It looks like the column, 'ps_car_11_cat' has this problem
# So we're going to engineer a replacement to this with fewer than 53 columns
# impute_train %>%
#   group_by(ps_car_11_cat) %>%
#   summarize(percent_1s = sum(as.integer(target)) / count(ps_car_11_cat))
impute_train$ps_car_11_cat <- NULL


# It looks like we don't have to do too much messing with distribution. Things are pretty normally distributed in general
# This is an in-built function for doing Cross-Validation in randomForest
# a <- rfcv(sample[,-c(1,2,10:50)], sample$target, cv.fold=5, scale="log", step=0.5)


set.seed(100)
sample_these <- sample(NROW(impute_train), 10000)
# sample <- bind_rows(impute_train[impute_train$target==0,][sample(NROW(impute_train[impute_train$target==0,]), 5000),], # Half are non-claims
#                     impute_train[impute_train$target==1,][sample(NROW(impute_train[impute_train$target==1,]), 5000),]) # And the other half are claims
sample <- impute_train[sample_these,]
# We produce a pairwise plot of the predictors to try and decide if we need to do any transformations of the variables
# ggpairs(sample[which(lapply(sample, class) %in% c('integer', 'numeric'))])
# For the most part it looks pretty good. There aren't any particularly skewed distributions; most variables are either normal 
# or uniform

# I decided to use Out-Of-Bag error estimation to decide what value of m to use. Above m=15 doesn't
# seem to produce super-strong results. It also takes longer and longer to run the model
oob_estimates <- bind_rows(lapply(X = seq(1, 20), function(m){
  print(m)
  # Use ntree=100 just to get a quick sense of what m-values are better
  rf.portseguro <- randomForest(target ~ .-id, data = sample, mtry = m, ntree = 100, type = 'prob')
  yhat <- randomForest:::predict.randomForest(rf.portseguro, type = 'prob')
  return(tibble(m = m, 
                num_trees = 100, 
                gini_coeff = normalized.gini.index(as.numeric(sample$target), 
                                                   yhat[,2])))
}))
oob_estimates
# OUTPUT:
# A tibble: 20 x 3
#        m num_trees gini_coeff
#    <int>     <dbl>      <dbl>
#  1     1       100  0.0727741
#  2     2       100  0.1085188
#  3     3       100  0.1671652
#  4     4       100  0.1622811
#  5     5       100  0.1602194
#  6     6       100  0.1393153
#  7     7       100  0.1500228
#  8     8       100  0.1704952
#  9     9       100  0.1631520
# 10    10       100  0.1494739
# 11    11       100  0.1778428
# 12    12       100  0.1644787
# 13    13       100  0.1516196
# 14    14       100  0.1931824
# 15    15       100  0.1767996
# 16    16       100  0.1256602
# 17    17       100  0.1789918
# 18    18       100  0.1494306
# 19    19       100  0.1725981
# 20    20       100  0.1520909
# Looks like with a sample size of 10K and ntrees=100, we can get the best gini_coeff with m=14
# With further investigation and adjusting ntrees we find that n=400 gets solid behavior

# Looking at what we can do with parallelization
rf.model <- foreach(ntree=rep(250, 4), .combine=combine, .packages='randomForest', .multicombine=TRUE) %dopar% {randomForest(sample[-c(1,2)],
                                                                                                 sample$target,
                                                                                                 ntree=ntree,
                                                                                                 mtry = 14)}
# remember_me <- rf.model
#rf.model <- randomForest(target ~ .-id, data = sample, mtry = m, ntree = 100, type = 'prob')
yhat <- randomForest:::predict.randomForest(rf.model, type = 'prob')
gini_coeff <- normalized.gini.index(as.numeric(sample$target), 
                                   yhat[,2])
# Now try it on a validation set
set.seed(200)
test_these <- sample(NROW(impute_train), 10000)
validation <- impute_train[test_these,]
yhat <- predict(rf.model, newdata = validation, type = 'prob')
gini_coeff = normalized.gini.index(as.numeric(validation$target), 
                                   yhat[,2])




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
write_csv(to_submit, 'RandomForestSubmission2.csv')
