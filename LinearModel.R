#Kaggle Linear Model

train <- read.csv("train.csv", header=T)

#--------------------- Data Cleaning & Exploratory analysis
#Investigate if there is any data cleaning to 
#Count the NAs across all the columns in the training data, sums the length of NA values
na_count <-sapply(train, function(y) sum(length(which(is.na(y)))))
na_count #shows no NAs across the columns

#however, upon reading through the dataset, values that are listed as -1 are actually unknown
train[train==-1]<-NA #replace all -1 in dataset with NA
na_count <-sapply(train, function(y) sum(length(which(is.na(y))))) #re=search for NAs
na_count #shows multiple columns with NAs

#build function to impute "none" for categorical variables
impute_none <- function(df, colnm){ 
  for(cols in colnm){ #for every column in all the columns
    levels(df[,cols]) <- c(levels(df[,cols]),"none") #identify the column levels
    df[is.na(df[cols]),cols]<- "none" #assigns none to categorical variables that have missing variables
  }
  return(df) #return the new dataframe with imputed variables
}

#list of factors in train data
char_cols <- sapply(train, is.factor) #create a new variable that applies factor to all character columns

#Impute factor variables with "none"
train_impute <- impute_none(train, colnames(train[,char_cols]) ) 

#create a function to assess mode in the target column
getmode <- function(v) {
  unique <- unique(v) #identify the unique values within the data
  unique[which.max(tabulate(match(v, unique)))] #tabulate the unique values
}

mode <- getmode(train) #run the function that we created
mode

#significance testing
fit <- aov(target ~., data=train) #testing significance of variables in the training set
summary(fit) #output for only significant variables shown below



#--------------------- Simple Linear Regression
