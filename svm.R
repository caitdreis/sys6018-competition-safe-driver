library(tidyverse)
library(e1071)

train <- read_csv("train.csv")

# HANDLE MISSING VALUES ====================================================================================

train %>% 
  sapply(function(x) sum(is.na(x)))

# No missing values -- great!

# EXPLORATORY ANALYSIS: CATEGORICAL VARIABLES ==============================================================

categorical_variables <- c("ps_ind_01", "ps_ind_02_cat", "ps_ind_02_cat", "ps_ind_03", "ps_ind_04_cat", "ps_ind_05_cat",
                           "ps_ind_06_bin", "ps_ind_07_bin", "ps_ind_08_bin", "ps_ind_09_bin", "ps_ind_10_bin", "ps_ind_11_bin",
                           "ps_ind_12_bin", "ps_ind_13_bin", "ps_ind_14", "ps_ind_15",  "ps_ind_16_bin",  "ps_ind_17_bin",
                           "ps_ind_18_bin", "ps_car_01_cat",  "ps_car_02_cat",  "ps_car_03_cat",  "ps_car_04_cat",
                           "ps_car_05_cat", "ps_car_06_cat",  "ps_car_07_cat",  "ps_car_08_cat",  "ps_car_09_cat",
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
    p_value <- c(p_value, chisq.test(table(df[[category]], df[[target]]))$p.value)
  }
  
  return(data.frame(names, p_value))
}

# Get p-value for contingeny table chi-square test for every category:
train_chisq <- chisq_tests(categorical_variables, train, "target")

train_chisq2 <- train_chisq %>% 
  mutate(
    significant_.05 = ifelse(p_value < .05, "Yes", "No"),
    significant_.1 = ifelse(p_value < .1, "Yes", "No"))

train_chisq

# Significant categorical variables:


# EXPLORATORY ANALYSIS: CONTINUOUS VARIABLES ==============================================================

target.factor <- as.factor(train$target)

train %>% 
  ggplot() + geom_point(mapping = aes(x=target.factor, y=ps_reg_01), position="jitter", alpha = .08)

qplot(factor(train$ps_reg_01), geom="bar", fill=factor(train$target))
qplot(train$ps_reg_01, geom="histogram", bins = 20, fill=factor(train$target))

train %>% 
  ggplot() + geom_histogram(mapping = aes(ps_reg_01, fill = target.factor), bins = 20)

testglm <- glm(train$target ~ train$ps_reg_01, family=binomial("logit"))
summary(testglm)

testglm2 <- glm(target ~ ps_reg_01 +
                  ps_reg_02 +
                  ps_reg_03 +
                  ps_car_12 +
                  ps_car_13 +
                  ps_car_14 +
                  ps_car_15 +
                  ps_calc_01 +
                  ps_calc_02 +
                  ps_calc_03, data=train, family=binomial("logit"))

vif(testglm2)
# No multicollinearity issues.

summary(testglm2)
# All variables significant except ps_calc_01, ps_calc_02, ps_calc_03.

# BUILD LINEAR AND SVM MODEL ===================================================================

# The SVM function only works if we build a dataframe containing no unused variables.

svm_train <- data.frame(target = factor(train$target),
                        ps_reg_01 = train$ps_reg_01,
                        ps_reg_02 = train$ps_reg_02,
                        ps_reg_03 = train$ps_reg_03,
                        ps_car_12 = train$ps_car_12,
                        ps_car_13 = train$ps_car_13,
                        ps_car_14 = train$ps_car_14,
                        ps_car_15 = train$ps_car_15,
                        ps_ind_01 = factor(train$ps_ind_01),
                        ps_ind_02_cat = factor(train$ps_ind_02_cat),
                        ps_ind_03 = factor(train$ps_ind_03),
                        ps_ind_04_cat = factor(train$ps_ind_04_cat),
                        ps_ind_05_cat = factor(train$ps_ind_05_cat),
                        ps_ind_14 = factor(train$ps_ind_14),
                        ps_car_01_cat = factor(train$ps_car_01_cat),
                        ps_car_04_cat = factor(train$ps_car_04_cat),
                        ps_car_06_cat = factor(train$ps_car_06_cat),
                        ps_car_07_cat = factor(train$ps_car_07_cat),
                        ps_car_09_cat = factor(train$ps_car_09_cat)
                        )

quick_svm <- svm(target ~ ., svm_train)

# kernel = "radial"
# kernel = "linear"
# kernel = "polynomial"
