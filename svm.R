library(tidyverse)
library(e1071)
library(car)

train <- read_csv("train.csv")

# HANDLE MISSING VALUES ====================================================================================

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

# EXPLORATORY ANALYSIS: CATEGORICAL VARIABLES ==============================================================

# Note: includes some ordinal variables that I will treat as categorical

categorical_variables <- c("ps_ind_01", "ps_ind_02_cat", "ps_ind_03", "ps_ind_04_cat", "ps_ind_05_cat",
                           "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
                           "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_14", "ps_ind_15",  "ps_ind_16_bin",  "ps_ind_17_bin",
                           "ps_ind_18_bin", "ps_car_01_cat",  "ps_car_02_cat",  "ps_car_04_cat",
                           "ps_car_06_cat",  "ps_car_07_cat",  "ps_car_08_cat",  "ps_car_09_cat",
                           "ps_car_10_cat",  "ps_car_11_cat", "ps_car_11", "ps_calc_04", "ps_calc_05", "ps_calc_06",
                           "ps_calc_07", "ps_calc_08", "ps_calc_09", "ps_calc_10", "ps_calc_11", "ps_calc_12", "ps_calc_13",
                           "ps_calc_14", "ps_calc_15_bin", "ps_calc_16_bin", "ps_calc_17_bin", "ps_calc_18_bin",
                           "ps_calc_19_bin", "ps_calc_20_bin")

# Function that performs a contingency table chi-square test and returns p-value for every category:
chisq_tests <- function (categories, df, target) {
  names <- character(0)
  p_value <- numeric(0)
  
  for (category in categories) {
    names <- c(names, category)
    
    p_value <- c(p_value, chisq.test(df[[category]], df[[target]])$p.value)
  }
  
  return(data.frame(names, p_value))
}

# Get p-value for contingeny table chi-square test for every category:
train_chisq <- chisq_tests(categorical_variables, train, "target")

train_chisq <- train_chisq %>% 
  mutate(
    significant_.05 = ifelse(p_value < .05, "Yes", "No")) %>% 
  arrange(p_value)

train_chisq

# Significant categorical variables:
sig_cat_vars <- train_chisq %>%
  filter (significant_.05=="Yes") %>% 
  select(names) %>% 
  unlist %>% 
  as.vector

sig_cat_vars

# In decreasing order of p-value:
# [1] "ps_car_11_cat" "ps_ind_05_cat" "ps_car_01_cat" "ps_car_04_cat" "ps_ind_17_bin" "ps_car_07_cat" "ps_car_06_cat"
# [8] "ps_ind_07_bin" "ps_ind_06_bin" "ps_ind_03"     "ps_car_02_cat" "ps_ind_16_bin" "ps_car_09_cat" "ps_ind_04_cat"
# [15] "ps_ind_15"     "ps_car_08_cat" "ps_car_11"     "ps_ind_01"     "ps_ind_02_cat" "ps_ind_08_bin" "ps_ind_09_bin"
# [22] "ps_ind_12_bin" "ps_ind_14"     "ps_ind_18_bin"

# EXPLORATORY ANALYSIS: CONTINUOUS VARIABLES ==============================================================

testglm <- glm(target ~ ps_reg_01 +
                  ps_reg_02 +
                  ps_reg_03 +
                  ps_car_12 +
                  ps_car_13 +
                  ps_car_14 +
                  ps_car_15 +
                  ps_calc_01 +
                  ps_calc_02 +
                  ps_calc_03, data=train, family=binomial("logit"))

vif(testglm)
# No multicollinearity issues.

summary(testglm)
# All variables significant except ps_calc_01, ps_calc_02, ps_calc_03.

# DEAL WITH IMBALANCED DATA ========================================================================

# Only 3% of our 600,000 observations have a target value of 1, so we don't want to cut away any of those data points.
# Instead, we'll create a dataset with a 1:1 ratio of target values.

train_ones <- train %>% 
  filter(target==1)

train_zeros <- train %>% 
  filter(target==0)

# Get the same number of 0s as we can 1s
set.seed(1)
subset_ids <- sample(1:nrow(train_zeros), nrow(train_ones))
train_subsetzeros <- train_zeros[subset_ids,]

# Put them together
new_train <- rbind(train_ones, train_subsetzeros)

# CROSS-VALIDATE SVM MODEL ========================================================================

# The SVM function only works if we build a dataframe containing no unused variables.

# First, we'll try using all of the continuous variables and the top 10 categorical/ordinal variables
# according to sig_cat_vars.

svm_df <- data.frame(target = factor(new_train$target),
                        ps_reg_01 = new_train$ps_reg_01,
                        ps_reg_02 = new_train$ps_reg_02,
                        ps_reg_03 = new_train$ps_reg_03,
                        ps_car_12 = new_train$ps_car_12,
                        ps_car_13 = new_train$ps_car_13,
                        ps_car_14 = new_train$ps_car_14,
                        ps_car_15 = new_train$ps_car_15,
                        ps_ind_05_cat = factor(new_train$ps_ind_05_cat),
                        ps_ind_06_bin = factor(new_train$ps_ind_06_bin),
                        ps_ind_07_bin = factor(new_train$ps_ind_07_bin),
                        ps_ind_17_bin = factor(new_train$ps_ind_17_bin),
                        ps_car_01_cat = factor(new_train$ps_car_01_cat),
                        ps_car_04_cat = factor(new_train$ps_car_04_cat),
                        ps_car_06_cat = factor(new_train$ps_car_06_cat),
                        ps_car_07_cat = factor(new_train$ps_car_07_cat),
                        ps_car_11_cat = factor(new_train$ps_car_11_cat),
                        ps_ind_03 = ordered(new_train$ps_ind_03) # ordinal
                        )

# To cross-validate, we will split the svm_df into a training set and a test set. (K-fold cross-validation
# is not feasible considering the computational demands of SVM.)
# Our testing error is based on the normalized gini index. (Code for this function can be found in gini.R.)

split_ids <- sample(1:nrow(svm_df), nrow(svm_df)/2)

svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]

# 1) Untuned SVM, radial kernel:

radial_untuned_svm <- svm(target ~ ., svm_train, probability=TRUE)

pred <- predict(radial_untuned_svm, svm_test, probability=TRUE)
pred.df <- data.frame(attr(pred, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred.df$X1)
# 0.2480507

# 2) Untuned SVM, linear kernel:

linear_untuned_svm <- svm(target ~ ., svm_train, probability=TRUE, kernel="linear")

lin.pred <- predict(linear_untuned_svm, svm_test, probability=TRUE)
linpred.df <- data.frame(attr(lin.pred, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), linpred.df$X1)
# 0.2405003 - *slightly* worse than the radial kernel, but similar

# 3) How about with 2 more variables from sig_cat_vars - "ps_car_02_cat" and "ps_ind_16_bin"?

svm_df["ps_car_02_cat"] <- factor(new_train$ps_car_02_cat)
svm_df["ps_ind_16_bin"] <- factor(new_train$ps_ind_16_bin)

svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]

radial_untuned_svm_2 <- svm(target ~ ., svm_train, probability=TRUE)

pred3 <- predict(radial_untuned_svm_2, svm_test, probability=TRUE)
pred3.df <- data.frame(attr(pred3, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred3.df$X1)
# 0.2495208 -- an improvement!

# 4) Adding in the next 2 most significant variables: "ps_car_09_cat", "ps_ind_04_cat"

svm_df["ps_car_09_cat"] <- factor(new_train$ps_car_09_cat)
svm_df["ps_ind_04_cat"] <- factor(new_train$ps_ind_04_cat)

svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]

radial_untuned_svm_3 <- svm(target ~ ., svm_train, probability=TRUE)

pred4 <- predict(radial_untuned_svm_3, svm_test, probability=TRUE)
pred4.df <- data.frame(attr(pred4, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred4.df$X1)
# 0.2541848 -- still improving -- probably room for more predictors

# 5) More predictors: "ps_ind_15", "ps_car_08_cat"

svm_df["ps_ind_15"] <- ordered(new_train$ps_ind_15)
svm_df["ps_car_08_cat"] <- factor(new_train$ps_car_08_cat)
svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]

radial_untuned_svm_4 <- svm(target ~ ., svm_train, probability=TRUE)

pred.r4 <- predict(radial_untuned_svm_4, svm_test, probability=TRUE)
predr4.df <- data.frame(attr(pred.r4, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), predr4.df$X1)
# 0.2581669 -- slight improvement, but not as much as before.

# 6) How about one more variable - "ps_car_11"?

svm_df["ps_car_11"] <- ordered(new_train$ps_car_11)
svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]

radial_untuned_svm_5 <- svm(target ~ ., svm_train, probability=TRUE)

pred.r5 <- predict(radial_untuned_svm_5, svm_test, probability=TRUE)
predr5.df <- data.frame(attr(pred.r5, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), predr5.df$X1)
# 0.2584464 -- gini score improved, but only very slightly. We can keep ps_car_11, but we will
# stop adding features at this point, because it may become too computationally intensive once
# we start tuning our model with a larger dataset.

# 7) Now trying different cost parameters. summary(radial_untuned_svm_5) tells us that by default,
# cost = 1. tune() takes too long because it involves 10-fold CV, so we will try different cost values
# and compare the gini index score. Let's start with cost = 0.1.

radial_c0.1 <- svm(target ~ ., svm_train, probability=TRUE, cost=0.1)

pred.c1 <- predict(radial_c0.1, svm_test, probability=TRUE)
pred.c1.df <- data.frame(attr(pred.c1, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred.c1.df$X1)
# 0.2429899 -- not as good.

# 8) How about cost = 2?

radial_c2 <- svm(target ~ ., svm_train, probability=TRUE, cost=2)

pred.c2 <- predict(radial_c2, svm_test, probability=TRUE)
pred.c2.df <- data.frame(attr(pred.c2, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred.c2.df$X1)
# 0.2601684 -- an improvement.

# 9) How about cost = 5?

radial_c5 <- svm(target ~ ., svm_train, probability=TRUE, cost=5)

pred.c5 <- predict(radial_c5, svm_test, probability=TRUE)
pred.c5.df <- data.frame(attr(pred.c5, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred.c5.df$X1)
# 0.2601439 -- slightly worse than before. Looks like cost = 2 is best.

# BUILD FULL MODEL ========================================================================

# Using cost = 2, we'll now train on the 43k observations present in svm_df.

full_model <- svm(target ~ ., svm_df, probability=TRUE, cost=2)

# PREDICT TEST DATA ========================================================================

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

# Variables in test_df need to have the exact same factor levels as in svm_train.
# We need to get rid of five "-1"s in test_df$ps_car_02_cat and one in test_df$ps_car_11.

table(test$ps_car_02_cat) # mode is 1
table(test$ps_car_11) # mode is 3

test$ps_car_02_cat[test$ps_car_02_cat==-1] <- 1
test$ps_car_11[test$ps_car_11==-1] <-3

# Create new df so SVM will run
test_df <- data.frame(
                     ps_reg_01 = test$ps_reg_01,
                     ps_reg_02 = test$ps_reg_02,
                     ps_reg_03 = test$ps_reg_03,
                     ps_car_12 = test$ps_car_12,
                     ps_car_13 = test$ps_car_13,
                     ps_car_14 = test$ps_car_14,
                     ps_car_15 = test$ps_car_15,
                     ps_ind_05_cat = factor(test$ps_ind_05_cat),
                     ps_ind_06_bin = factor(test$ps_ind_06_bin),
                     ps_ind_07_bin = factor(test$ps_ind_07_bin),
                     ps_ind_17_bin = factor(test$ps_ind_17_bin),
                     ps_car_01_cat = factor(test$ps_car_01_cat),
                     ps_car_04_cat = factor(test$ps_car_04_cat),
                     ps_car_06_cat = factor(test$ps_car_06_cat),
                     ps_car_07_cat = factor(test$ps_car_07_cat),
                     ps_car_11_cat = factor(test$ps_car_11_cat),
                     ps_ind_03 = ordered(test$ps_ind_03),
                     ps_car_02_cat = factor(test$ps_car_02_cat),
                     ps_ind_16_bin = factor(test$ps_ind_16_bin),
                     ps_car_09_cat = factor(test$ps_car_09_cat),
                     ps_ind_04_cat = factor(test$ps_ind_04_cat),
                     ps_ind_15 = ordered(test$ps_ind_15),
                     ps_car_08_cat = factor(test$ps_car_08_cat),
                     ps_car_11 = ordered(test$ps_car_11)
)

test_pred <- predict(full_model, test_df, probability=TRUE)

testpred.df <- data.frame(attr(test_pred, "probabilities"))

final_preds <- cbind(test$id, testpred.df$X1)
colnames(final_preds) <- c("id", "target")

write.table(final_preds, file = "svm_tuned.csv", 
            row.names=F, col.names=T, sep=",")
