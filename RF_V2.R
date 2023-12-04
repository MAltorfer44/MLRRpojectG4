
# Load necessary libraries
library(ranger)
library(caret)
library(ranger)
library(tidyverse)
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(randomForest)
library(mlr3viz)

rm(list=ls())

load("pre.RData")

# Reduced training data due to performance issues 
#-> limited computational power on laptop




# Set up cross-validation
set.seed(123)
train_control <- trainControl(
  method = "cv",
  number = 5,
  summaryFunction = twoClassSummary,  # For binary classification
  classProbs = TRUE,  # Required for ROC
  verboseIter = TRUE,
  selectionFunction = "best",
  allowParallel = FALSE  # If you want to use parallel processing
  
)

hyperparameter_grid <- expand.grid(
  mtry = c(ncol(train)/2, ncol(train)/3, ncol(train)/4),            # Values to test for the number of variables randomly sampled at each split
  min.node.size = c(1, 5, 9) ,  # Values to test for the minimum node size in trees
  splitrule = c("gini", "extratrees")
)

# Define the model

# only run if you want to. may take a while

# model <- train(DRK_YN ~ .,  #  target variable on all the features
               data = train,  #  dataset
               method = "ranger",
               trControl = train_control,
               tuneGrid = hyperparameter_grid,
               metric = "ROC",
               num.trees = 500)

#save(model, file="model.RData")
load("model.RData")

res_RF = model$results

# arrange best hyperparams (max ROC is the criteria)
# first row is the topcombination
res_RF=res_RF%>%arrange(desc(ROC))

# View the results -> helps to get a feeling for the staility/ importance of hyperparam tuning
print(head(res_RF))

# Fitting mtry = 3, splitrule = extratrees, min.node.size = 9 on full training set



# train another ranger anyway 
#bm_ranger= ranger(DRK_YN ~ .,  #  target variable on all the features
                  data = train,  #  dataset
                  mtry =res_RF$mtry[1], # same as model$bestTune$mtry
                  splitrule = res_RF$splitrule[1],
                  min.node.size = res_RF$min.node.size[1],
                  importance = "impurity",
                  probability = T)

# this library is needed for the pdp and ice -> pdp library cannot handle "ranger" objects
# but it can handle "randomforest" objects 
bm_RandomForest <- randomForest(DRK_YN ~ .,  #  target variable on all the features
                                data = train,  #  dataset
                                # we cannot do the testing in this model
                                # as it can only be an object of class RandomForest
                                # and not of RandomForest.predictions as well
                                # -> the PDP library can't handle the object if i has both classes
                                #xtest = X, 
                                #ytest = Y,
                                mtry =res_RF$mtry[1], # same as model$bestTune$mtry
                                splitrule = res_RF$splitrule[1],
                                min.node.size = res_RF$min.node.size[1],
                                importance = TRUE,
                                probability = T)


#save(bm_ranger, bm_RandomForest,file="bm.RData")

load("bm.RData")

prd = predict(bm_ranger, data = test)
predictions_RF = prd$predictions
#predictions = relevel(predictions, "N")
actual_labels <- test$DRK_YN
prd_RF =as.factor(ifelse(predictions_RF[,1]<0.5,"Y","N"))

# Compute confusion matrix
confusion_matrix_RF <- confusionMatrix(prd_RF, actual_labels)
print(confusion_matrix_RF)




