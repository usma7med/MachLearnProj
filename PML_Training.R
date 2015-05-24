
# Project R Code
#

library(caret)


setwd("~/Courses/Online Courses/Practical Machine Learning/Project/MachLearnProj")
getwd()

#
# csv file contains #DIV/0 which messes up the reading... makes numbers into factors
#  -- use na.strings to take care of it
data <- read.csv("pml-training.csv",na.strings=c("NA","#DIV/0!")) 
set.seed(3440)


index <- createDataPartition(data$classe, p=0.75, list=FALSE)
training <- data[index,] # will do cross-validation on this
testing <- data[-index,] # hold out sample
rm(data)
#
# Pre-Process 1: check for new zero vars


NAtoZero <- function(df)
{
  df[is.na(df)] <- 0
  return(df)
}

training <- NAtoZero(training)
testing <- NAtoZero(testing)

nsv <- nearZeroVar(training, saveMetrics=TRUE)
rownames(nsv)[nsv$nzv == TRUE] # vars which are almost a constant, these are 95%+ NA's
del_names <- rownames(nsv)[nsv$nzv == TRUE] # do not use these vars
del_names <- c(del_names,"X") # must remove row number variable
training <- training[, !(names(training) %in% del_names)]
testing <- testing[, !(names(testing) %in% del_names)]



# let's try Random Forest
ctrl <- trainControl(method="cv") # default is 10 folds for "cv" method

rf_modfit <- train(classe ~ ., data=training,method="rf", trControl=ctrl, prox=TRUE,tuneGrid=data.frame(mtry=42))
#rf_modfit <- train(classe ~ ., data=training,method="rf", trControl=ctrl, prox=TRUE)

print(rf_modfit)

testpred <- predict(rf_modfit,testing)
table(testpred, testing$classe)
confusionMatrix(testpred, testing$classe)



# let's try bagged Tree

ctrl <- trainControl(method="cv") # default is 10 folds for "cv" method

tb_modfit <- train(classe ~ ., data=training,method="treebag", trControl=ctrl)
print(tb_modfit)

testpred2 <- predict(tb_modfit,testing)
table(testpred2, testing$classe)
confusionMatrix(testpred2, testing$classe)



tdata <- read.csv("pml-testing.csv",na.strings=c("NA","#DIV/0!")) 
tdata <- NAtoZero(tdata)
testdata<- tdata[, !(names(tdata) %in% del_names)]

answers <- predict(rf_modfit,testdata)
answers2 <- predict(tb_modfit, testdata)




pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("./test_output/problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}

pml_write_files(answers)


save.image('pcm_train.Rdata')

