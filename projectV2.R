library(tidyverse)
library(ggplot2)
library(patchwork)
library(readr)
library(caTools)
library(rpart)
library(mlr3tuning)

rm(list=ls())

# source https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset/data
data0 <- as.data.frame(read_csv("smoke.csv"))


########### data inspection ###################################

# summary stats
# find variable description in source
summary(data0)

dim(data0) # almost one million observations


# print NAs in each column 
sapply(data0, function(y) sum(length(which(is.na(y)))))

# -> No NAs
sum(sapply(data0, function(y) sum(length(which(is.na(y)))))) 





################## data cleaning and preparation ###########################
data=data0
# from characters or numerics to factors
data$sex =as.factor(data$sex)
data$DRK_YN = as.factor(data$DRK_YN)
data$hear_left = as.factor(data$hear_left)
data$hear_right = as.factor(data$hear_right)
data$urine_protein = as.factor(data$urine_protein)
data$SMK_stat_type_cd = as.factor(data$SMK_stat_type_cd)

# reinspect
summary(data)


# check correlation
numeric_columns <- sapply(data, is.numeric)

# Subset the dataframe to include only numeric columns
numeric_data <- data[, numeric_columns]

# Calculate the correlation of numeric variables
correlation_matrix <- cor(numeric_data)

# show a nice plot 
# -> blue positive corr, red negative corr
corrplot::corrplot(correlation_matrix)



# some additional checks and data inspections
# Male more likely to be alcoholic

tab=table(data$DRK_YN, data$sex)

cat("male drinkers:" , tab[2,2]/sum(tab[,2]))
cat("female drinkers:" , tab[2,1]/sum(tab[,1]))

# younger more likely to be alcoholic
ggplot(data)+
  geom_boxplot(aes(age))+
  facet_wrap(~DRK_YN)+
  coord_flip()

df =data %>%
  group_by(DRK_YN) %>%
  summarise(across(everything(), mean, na.rm = TRUE))


# create vectors with the different variable sets

# outcome
outcome = "DRK_YN"

# features that are available or relatively easy to get
feat_available = c("sex","age","height","weight", "sight_left", "sight_right")

# features that just need a small test and no blood/urine sample
feat_maybe = c("waistline","hear_left","hear_right","SBP","DBP")


# features that would need a blood or urine test 
# or require honest self disclosure (smoking variable) 
# -> Do we believe that bad risks will be honest?

# colnames of the whole dataset
data_names=names(data)

# remove subgroups from above
feat_other = setdiff(data_names, c(feat_available, feat_maybe, outcome))

#check if no column was forgotten
length(feat_other)+length(feat_maybe)+
  length(feat_available)+length(outcome)-length(data_names)== 0


############# create train test data ####################

# choose the features we want to work with in the models
data = data%>%
  dplyr::select(all_of(
    c(outcome, feat_available,feat_maybe)))
    # feat_other not included
   

#data=data[,-c(5,12)]
# prevalence is almost 50% -> balanced 
# no need for under or oversampling. Large enough that stratification is not necessary
sum(data[,outcome]=="Y")/nrow(data)

# reproducibility
set.seed(123)

# train percentage
perc=0.05 # no more possible cause my RAM can't handle it

# create a vector with the row index for the train data
train_set = sample(seq_len(nrow(data)), 
                   size = perc*nrow(data), 
                   replace = F # alternatively replace = T
)

train=data[train_set,]

# subtract train to get test (mutually exclusive datasets)
test=data[-train_set,]


############ modelling ########################################



# dummy classifier -> complicated way to compute the prevalence
dummy_class = median(as.numeric(train$DRK_YN))
predict_dummy = rep(dummy_class,length(test$DRK_YN))
confusion_matrix = table(test$DRK_YN, predict_dummy)
dummy_mmce = 1 - (sum(diag(confusion_matrix))/sum(confusion_matrix))
dummy_accuracy = (sum(diag(confusion_matrix))/sum(confusion_matrix))
cat('MMCE',(dummy_mmce),'-', 'Accuracy', dummy_accuracy)


#### logistic reg #############################

# Training model
logistic_model <- glm(DRK_YN ~ age+height+weight+waistline+DBP, 
                      data = train, family = 'binomial')

predict_reg <- predict(logistic_model, 
                       test, type = "response")

summary(logistic_model)


# Depending on our sensitivity specificity preference
# we can increase/decrease as we want
s=0.5
predict_reg01 <- ifelse(predict_reg > s, 1, 0)


confusion_matrix = table(predict_reg01, test$DRK_YN)
confusion_matrix

# calc MMCE and Acc
logreg_mmce = 1 - (sum(diag(confusion_matrix))/sum(confusion_matrix))
logreg_accuracy = (sum(diag(confusion_matrix))/sum(confusion_matrix))
cat('MMCE',(logreg_mmce),'-', 'Accuracy', logreg_accuracy)

# calculate sensitivity and specificity
sens= confusion_matrix[2,2]/sum(confusion_matrix[,2])
sens
spec= sens= confusion_matrix[1,1]/sum(confusion_matrix[,1])
spec


##### CV of LASSO for multicollinearity #########

library(glmnet)

y=as.integer(ifelse(train[,outcome]=="Y",1,0))
x=train[,-which(names(train) == outcome)]

x$sex=ifelse(x$sex=="Male",1,0)
x$hear_left=as.integer(x$hear_left)
x$hear_right=as.integer(x$hear_right)
x$SMK_stat_type_cd = as.integer(x$SMK_stat_type_cd)
x$urine_protein = as.integer(x$urine_protein)
x=as.matrix(x)

fit <- cv.glmnet(x, y, family = "binomial", alpha = 1) 


# Plotting the coefficients
plot(fit, xvar = "lambda", label = TRUE)

# Get lambda.1se value
lambda <- fit$lambda.1se

# Extract coefficients for lambda.1se
coef_min <- coef(fit, s = lambda)

# Plotting the coefficients at lambda.1se
plot(coef_min, xlab = "Coefficient Index", ylab = "Coefficient Value", type = "l", main = paste("Adjusted Coefficients at Lambda = ", lambda))
# We want to have everything above s as a 1


############### MLR #####################



library(mlr3)
library(mlr3pipelines)
# library(mlr3verse)

# keep the test data as  final hold out data to test the model
task_dat = TaskClassif$new(id = "data", backend = train, 
                               target = "DRK_YN", positive = "Y")



task_rf = TaskClassif$new(id = "train", backend = train, 
                          target = "DRK_YN", positive = "Y")

learner = lrn("classif.rpart",
              cp = to_tune(0.00003, 0.3))

# create auto tuner
at = auto_tuner(
  tuner = tnr("random_search"),
  learner = learner,
  resampling = rsmp ("cv", folds=5),
  measure = msr("classif.ce"),
  term_evals = 25)

resampling_outer = rsmp("cv", folds = 10)

# takes a while to run
# rr_tree = resample(task_rf, at, resampling_outer, store_models = TRUE)


results_mlr_tree = as_tibble(extract_inner_tuning_results(rr_tree))

results_mlr_tree=results_mlr_tree%>%
  arrange(classif.ce)

#save(results_mlr_tree,file = "treeRes.RData")


# following the Prof -> not yet sure what it is for

# won't do any damage but not needed as we don't have NAs
imputer = po("imputemedian") # define a pipeline object for imputation

# default cp too high prunes trees after first split
# -> if tree becomes one of our models we'd have to tune this hyperparameter
# MLR way to split
# reproducibility
set.seed(123)
splits = partition(task_dat, ratio = perc)
print(str(splits))
tree = lrn("classif.rpart", cp = 0.001, predict_type = "prob")
tree = as_learner(imputer %>>% tree)

tree$train(task = task_dat, splits$train)
mod = tree$train(task = task_dat, splits$train)

# use the train test split in the training data
prediction = tree$predict(task_dat, splits$test)
# use the separate test data
prediction2 = predict(mod, newdata = test)

# relevel needed for caret to work -> order must be the same
prediction2=relevel(prediction2,"N")
#create confusion matrix
conf_mat_tree=table(prediction2, test$DRK_YN)


# plot of the tree
library(rpart.plot)
rpart.plot(tree$model$classif.rpart$model)

# calculate performance with MLR technique and the train-test data
mes = msrs(c("classif.ce","classif.acc","classif.sensitivity","classif.specificity"))
tree_perf = prediction$score(mes)

prediction$confusion
tree_perf

# compare results to test test data
# -> high variance of trees very evident
# accuracy comparable but sens and spec very different
library(caret) 
confusionMatrix(conf_mat_tree)


#save(train,test,outcome, file="pre.RData")



