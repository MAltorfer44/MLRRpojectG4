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

data$sex =as.factor(data$sex)
data$DRK_YN = as.factor(data$DRK_YN)
data$hear_left = as.factor(data$hear_left)
data$hear_right = as.factor(data$hear_right)
data$urine_protein = as.factor(data$urine_protein)
data$SMK_stat_type_cd = as.factor(data$SMK_stat_type_cd)


set.seed(123)
# train percentage
perc=0.025 # no more possible cause my RAM can't handle it

# create a vector with the row index for the train data
train_set = sample(seq_len(nrow(data)), 
                   size = perc*nrow(data), 
                   replace = F # alternatively replace = T
)

data=data[train_set,]

n <- nrow(data)  # Length of the vector
k <- 30   # Range of random numbers (1 to k)

random_vector <- sample(1:k, n, replace = TRUE)

data$index=random_vector

# containers
df_acc=data.frame(
  all = rep(NA,k),
  sexage = rep(NA,k),
  min = rep(NA,k),
  se = rep(NA,k))
df_sens=data.frame(
  all = rep(NA,k),
  sexage = rep(NA,k),
  min = rep(NA,k),
  se = rep(NA,k)
)

for (i in c(1:k)){
  train=data%>%
    filter(index != i)
  test=data%>%
    filter(index == i)
  lm_all <- glm(DRK_YN ~ ., 
                        data = train, family = 'binomial')
  lm_sa <- glm(DRK_YN ~ sex+age+age*sex, 
                data = train, family = 'binomial')
  lm_min = glm(DRK_YN~sex+age+height+weight+waistline+hear_left+SBP+DBP, 
               data = train, family = 'binomial')
  lm_1se = glm(DRK_YN~sex+age+height+DBP, 
               data = train, family = 'binomial')
 
  
  predict_1 <- predict(lm_all, 
                         test, type = "response")
  predict_2 <- predict(lm_sa, 
                         test, type = "response")
  predict_3 <- predict(lm_min, 
                         test, type = "response")
  predict_4 <- predict(lm_1se, 
                         test, type = "response")
  
  
  # Depending on our sensitivity specificity preference
  # we can increase/decrease as we want
  s=0.5
  predict_01 <- ifelse(predict_1 > s, 1, 0)
  predict_02 <- ifelse(predict_2 > s, 1, 0)
  predict_03 <- ifelse(predict_3 > s, 1, 0)
  predict_04 <- ifelse(predict_4 > s, 1, 0)
  
  
  cm1 = table(predict_01, test$DRK_YN)
  cm2 = table(predict_02, test$DRK_YN)
  cm3 = table(predict_03, test$DRK_YN)
  cm4 = table(predict_04, test$DRK_YN)
  
  # calc Acc and Sens
  df_acc$all[i] = (sum(diag(cm1))/sum(cm1))
  df_acc$sexage[i] = (sum(diag(cm2))/sum(cm2))
  df_acc$min[i]= (sum(diag(cm3))/sum(cm3))
  df_acc$se[i] = (sum(diag(cm4))/sum(cm4))
  
  
  
  
  # calculate sensitivity 
  df_sens$all[i]= cm1[2,2]/sum(cm1[,2])
  df_sens$sexage[i] = cm2[2,2]/sum(cm2[,2])
  df_sens$min[i] = cm3[2,2]/sum(cm3[,2])
  df_sens$se[i] = cm4[2,2]/sum(cm4[,2])
  
  
}

#save(df_acc,df_sens,file = "LogR_BM.RData")
load("LogR_BM.RData")
a=boxplot(df_acc$all, df_acc$sexage, df_acc$min, df_acc$se,
        names = c("All variables", "Sex & age", "Lasso Min", "Lasso 1se"),
        main = "Benchmarking of different feature specifications",
        ylab = "Accuracy")

b=boxplot(df_sens$all, df_sens$sexage, df_sens$min, df_sens$se,
        names = c("All variables", "Sex & age", "Lasso Min", "Lasso 1se"),
        main = "Benchmarking of different feature specifications",
        ylab = "Sensitivity")


