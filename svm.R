library(tidyverse)
library(e1071)

train <- read_csv("train.csv")

# HANDLE MISSING VALUES ====================================================================================

train %>% 
  sapply(function(x) sum(is.na(x)))

# No missing values -- great!

# EXPLORATORY ANALYSIS: CATEGORICAL VARIABLES ==============================================================

# Function that takes a frequency table between variable and target, returns a
# data frame summarizing counts. "Percent" = percent of '1's in a given level.
get.table <- function(tab) {
  tab.df <- tab %>% 
    matrix(nrow=length(attributes(tab)$dimnames[[1]])) %>% 
    as.data.frame()
  
  colnames(tab.df) <- attributes(tab)$dimnames[[2]]
  
  tab.df <- tab.df %>% 
    mutate(percent = `1`/(`0`+`1`)*100,
           value = attributes(tab)$dimnames[[1]])
  
  return(tab.df)
}

# Look at how levels affect % of target = '1's for all categorical variables:

ps_ind_01.df <- get.table(table(train$ps_ind_01, train$target)) # only slight differences
ps_ind_02.df <- get.table(table(train$ps_ind_02_cat, train$target)) # interesting when value = -1 ***
ps_ind_03.df <- get.table(table(train$ps_ind_03, train$target)) # interesting when value = 0 ***
ps_ind_04.df <- get.table(table(train$ps_ind_04_cat, train$target)) # veeery high when value = -1 ***
ps_ind_05.df <- get.table(table(train$ps_ind_05_cat, train$target)) # some differences **
ps_ind_06.df <- get.table(table(train$ps_ind_06_bin, train$target)) # slight
ps_ind_07.df <- get.table(table(train$ps_ind_07_bin, train$target)) # slight
ps_ind_08.df <- get.table(table(train$ps_ind_08_bin, train$target)) # slight
ps_ind_09.df <- get.table(table(train$ps_ind_09_bin, train$target)) # slight
ps_ind_10.df <- get.table(table(train$ps_ind_10_bin, train$target)) # slight
ps_ind_11.df <- get.table(table(train$ps_ind_11_bin, train$target)) # slight
ps_ind_12.df <- get.table(table(train$ps_ind_12_bin, train$target)) # slight
ps_ind_13.df <- get.table(table(train$ps_ind_13_bin, train$target)) # slight
ps_ind_14.df <- get.table(table(train$ps_ind_14, train$target)) # some differences (not enough data for value=4) *
ps_ind_15.df <- get.table(table(train$ps_ind_15, train$target)) # slight
ps_ind_16.df <- get.table(table(train$ps_ind_16_bin, train$target)) # slight
ps_ind_17.df <- get.table(table(train$ps_ind_17_bin, train$target)) # slight
ps_ind_18.df <- get.table(table(train$ps_ind_18_bin, train$target)) # slight
ps_car_01.df <- get.table(table(train$ps_car_01_cat, train$target)) # interesting when value = -1 ***
ps_car_02.df <- get.table(table(train$ps_car_02_cat, train$target)) # slight
ps_car_03.df <- get.table(table(train$ps_car_03_cat, train$target)) # slight
ps_car_04.df <- get.table(table(train$ps_car_04_cat, train$target)) # some differences?
ps_car_05.df <- get.table(table(train$ps_car_05_cat, train$target)) # slight
ps_car_06.df <- get.table(table(train$ps_car_06_cat, train$target)) # some differences?
ps_car_07.df <- get.table(table(train$ps_car_07_cat, train$target)) # some differences?
ps_car_08.df <- get.table(table(train$ps_car_08_cat, train$target)) # slight
ps_car_09.df <- get.table(table(train$ps_car_09_cat, train$target)) # some differences *
ps_car_10.df <- get.table(table(train$ps_car_10_cat, train$target)) # slight
ps_car_11.df <- get.table(table(train$ps_car_11, train$target)) # slight
ps_calc_04.df <- get.table(table(train$ps_calc_04, train$target)) # slight
ps_calc_05.df <- get.table(table(train$ps_calc_05, train$target)) # slight
ps_calc_06.df <- get.table(table(train$ps_calc_06, train$target)) # slight
ps_calc_07.df <- get.table(table(train$ps_calc_07, train$target)) # diff when value = 9, but not enough data points
ps_calc_08.df <- get.table(table(train$ps_calc_08, train$target)) # slight
ps_calc_09.df <- get.table(table(train$ps_calc_09, train$target)) # slight
ps_calc_10.df <- get.table(table(train$ps_calc_10, train$target)) # slight
ps_calc_11.df <- get.table(table(train$ps_calc_11, train$target)) # slight
ps_calc_12.df <- get.table(table(train$ps_calc_12, train$target)) # slight
ps_calc_13.df <- get.table(table(train$ps_calc_13, train$target)) # slight
ps_calc_14.df <- get.table(table(train$ps_calc_14, train$target)) # slight
ps_calc_15.df <- get.table(table(train$ps_calc_15_bin, train$target)) # almost no differences
ps_calc_16.df <- get.table(table(train$ps_calc_16_bin, train$target)) # almost no differences
ps_calc_17.df <- get.table(table(train$ps_calc_17_bin, train$target)) # almost no differences
ps_calc_18.df <- get.table(table(train$ps_calc_18_bin, train$target)) # almost no differences
ps_calc_19.df <- get.table(table(train$ps_calc_19_bin, train$target)) # almost no differences
ps_calc_20.df <- get.table(table(train$ps_calc_20_bin, train$target)) # almost no differences

ps_calc_20.df %>% 
  ggplot() + geom_point(mapping = aes(x=value, y=percent))

# EXPLORATORY ANALYSIS: CONTINUOUS VARIABLES ==============================================================

# https://stats.stackexchange.com/questions/99736/plotting-a-categorical-response-as-a-function-of-a-continuous-predictor-using-r
# https://onlinecourses.science.psu.edu/stat504/node/159
# https://datascienceplus.com/perform-logistic-regression-in-r/

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

summary(testglm2)
# All variables significant except ps_calc_01, ps_calc_02, ps_calc_03

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

