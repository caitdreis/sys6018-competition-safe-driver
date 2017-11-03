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
  } else{ # So continuous variables
    return(mean(X, na.rm=T))
  }
})

# Now replace all of these NA's with the imputed values
impute_train <- drop_nas_train %>% 
  replace_na(imputation_values)

# Check for multicollinearity
alias(lm(target~.-id, impute_train))
# It looks like we have some perfect multicollinearity (some variables are linear combinations of others)
# We can solve this by removing them: 'ps_ind_09_binTRUE' and 'ps_ind_13_binTRUE'

# Make the target variable a logical
impute_train$target <- as.logical(impute_train$target)
impute_train$ps_ind_09_bin <- NULL
impute_train$ps_ind_13_bin <- NULL

# randomForest function is having trouble with factors with levels over 53, so remove those columns that violate
which(sapply(impute_train, function(X) length(levels(X))) > 50)
# It looks like the column, 'ps_car_11_cat' has this problem
impute_train$ps_car_11_cat <- NULL


# It looks like we don't have to do too much messing with distribution. Things are pretty normally distributed in general
# This is an in-built function for doing Cross-Validation in randomForest
# a <- rfcv(sample[,-c(1,2,10:50)], sample$target, cv.fold=5, scale="log", step=0.5)


set.seed(100)
sample_these <- sample(NROW(impute_train), 10000)
sample <- impute_train[sample_these,]
# We produce a pairwise plot of the predictors to try and decide if we need to do any transformations of the variables
ggpairs(sample[which(lapply(sample, class) %in% c('integer', 'numeric'))])
# For the most part it looks pretty good. There aren't any particularly skewed distributions; most variables are either normal 
# or uniform

# Looking at multicollinearity
vif(lm(target~.-id, sample))
# Not really any columns with a large amount of multicollinearity

# I decided to use Out-Of-Bag error estimation to decide what value of m to use. Above m=15 doesn't
# seem to produce super-strong results. It also takes longer and longer to run the model
oob_estimates <- bind_rows(lapply(X = seq(1, 15), function(m){
  print(m)
  # Use ntree=100 just to get a quick sense of what m-values are better
  rf.portseguro <- randomForest(target ~ .-id, data = sample, mtry = m, ntree = 100)
  yhat <- randomForest:::predict.randomForest(rf.portseguro)
  return(tibble(m = m, 
                num_trees = 100, 
                gini_coeff = normalized.gini.index(as.numeric(sample$target), 
                                                   yhat)))
}))
oob_estimates
# Looks like with a sample size of 10K and ntrees=100, we can get the best gini_coeff with m=12
# With further investigation and adjusting ntrees we find that n=400 gets solid behavior

rf.model <- foreach(ntree=rep(100, 4), .combine=combine, .packages='randomForest', .multicombine=TRUE) %dopar% {randomForest(sample[-c(1,2)],
                                                                                                 sample$target,
                                                                                                 ntree=ntree)}

yhat <- randomForest:::predict.randomForest(rf.model)
gini_coeff = normalized.gini.index(as.numeric(sample$target), 
                                   yhat)




