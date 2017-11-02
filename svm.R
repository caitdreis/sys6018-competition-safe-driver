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

# BUILD SVM MODEL ========================================================================

# The SVM function only works if we build a dataframe containing no unused variables.

# (Using only sig_cat_vars[1:10])

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

split_ids <- sample(1:nrow(svm_df), nrow(svm_df)/2)

svm_train <- svm_df[split_ids,]
svm_test <- svm_df[-split_ids,]


##################################################################################
# QUICK UNTUNED SVM -- KERNEL RADIAL
Sys.time()
quick_svm <- svm(target ~ ., svm_train, probability=TRUE)
Sys.time()

# [1] "2017-11-02 13:21:48 EDT "2017-11-02 13:28:25 EDT" - 7 mins with 21k obs and 17 vars
# # 29k observations and 18 variables: 9 mins

pred <- predict(quick_svm, svm_test, probability=TRUE)

pred.df <- data.frame(attr(pred, "probabilities"))

normalized.gini.index(as.numeric(svm_test$target), pred.df$X1)
# 0.2480507 cool

# kernel = "linear"
# kernel = "polynomial"

# testy stuff ######################
##################################################################################

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
                     ps_ind_03 = ordered(test$ps_ind_03) # ordinal
)

test_pred <- predict(quick_svm, test_df, probability=TRUE)

testpred.df <- data.frame(attr(test_pred, "probabilities"))

final_preds <- cbind(test$id, testpred.df$X1)
names(final_preds) <- c("id", "target")

write.table(final_preds, file = "svm_untuned_01.csv", 
            row.names=F, col.names=T, sep=",")
