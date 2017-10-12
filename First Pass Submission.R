#Sally Gao
#Adrien Mead
#Caitlin Dreisbach

train <- read.csv("train-1.csv", header=TRUE, stringsAsFactors = FALSE)
test <- read.csv("test-1.csv", header=TRUE, stringsAsFactors = FALSE)

#create a function to assess mode in the target column
getmode <- function(v) {
  uniqv <- unique(v) #identify the unique values within the data
  uniqv[which.max(tabulate(match(v, uniqv)))] #tabulate the unique values
}

mode <- getmode(train$target) #run the function that we created
print(mode) #mode is 0

#using subset to only take id and create new target column from testing set
first <- subset(test, select=c("id")) #only take id column from testing data
first$target <- 0 #fill entire column with mode from training data

#write to file
write.table(first, file = "kagglesubmitfirstpass.csv", 
            row.names=F, col.names=T, sep=",")
