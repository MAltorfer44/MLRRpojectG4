# Load necessary libraries
library(ranger)
library(tidyverse)
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3viz)

rm(list=ls())

load("pre.RData")

gc()
rm(test)






#############################################

task_rf = TaskClassif$new(id = "train", backend = train, 
                          target = "DRK_YN", positive = "Y")

learner = lrn("classif.ranger",
              mtry = to_tune(ncol(train)%/%5, ncol(train)%/%1.5),
              min.node.size = to_tune(1,15),
              sample.fraction = to_tune(0.02,0.2),
              num.trees = to_tune(500,1500))

# create auto tuner
at = auto_tuner(
  tuner = tnr("random_search"),
  learner = learner,
  resampling = rsmp ("cv", folds=4),
  measure = msr("classif.ce"),
  term_evals = 5)

resampling_outer = rsmp("cv", folds = 10)

# takes a while to run
#rr = resample(task_rf, at, resampling_outer, store_models = TRUE)


results_mlr = as_tibble(extract_inner_tuning_results(rr))

results_mlr=results_mlr%>%
  arrange(classif.ce)

#save(results_mlr, file="mlrTuned.RData")
load("mlrTuned.RData")
rf_t = lrn("classif.ranger",mtry=results_mlr$mtry[1], 
           min.node.size=results_mlr$min.node.size[1],
           num.trees = results_mlr$num.trees[1],
           sample.fraction = results_mlr$sample.fraction[1])
imputer=po("imputemedian")
rf_t = as_learner(imputer %>>% rf_t)

mod_RF = rf_t$train(task_rf)

load("pre.RData")

predRF=predict(mod_RF, newdata = test)

# relevel needed for caret to work -> order must be the same
predRF=relevel(predRF,"N")
#create confusion matrix
conf_mat_RF=table(predRF, test$DRK_YN)

#save(conf_mat_RF,file="CM_RF.RData")

library(caret) 
confusionMatrix(conf_mat_RF)


library(iml)

y = train$DRK_YN
x = train[-which(names(train) == "DRK_YN")]

modR = Predictor$new(rf_t, data = x, y = y)


# does not run for me -> not enough computing power
importance = FeatureImp$new(modR, loss = "ce", n.repetitions = 100)
importance$plot()




########## MLR3 RF according to lecture does not work that well for me #################

# issues
# Cannot predict on new data that is not in the task environment
# very slow -> similar to caret ranger

# split into training and test

# adjust hyperparams 
rf = lrn("classif.ranger") 

# 10 fold cv
rdesc = rsmp("cv", folds = 5)

set.seed(123)
res = mlr3::resample(learner = rf, task = task_dat, resampling = rdesc, 
                     store_models = T,
                     store_backends = T)

mes = msrs(c("classif.ce","classif.acc","classif.sensitivity","classif.specificity"))






library(mlr3tuning)
library(paradox)


rf_ps = ParamSet$new(list(
  ParamInt$new("mtry", lower = 1, upper = ncol(train)%/%1.5),
  ParamInt$new("min.node.size", lower = 1, upper = 15),
  ParamInt$new("num.trees", lower = 500, upper = 1500)
))

res_inner = rsmp("cv", folds = 2)
mes_inner = msr("classif.mcc")
terminator = trm("evals", n_evals =1)
tuner = tnr("random_search")

rf_at = AutoTuner$new(
  learner = rf,
  resampling = res_inner,
  measure = mes_inner,
  search_space = rf_ps,
  terminator = terminator,
  tuner = tuner,
  #store_models = TRUE,
  store_tuning_instance = TRUE
)

imputer = po("imputemedian")
rf_at = as_learner(imputer %>>% rf_at)


set.seed(9000)

res_outer = rsmp("cv", folds = 2)

# library(future)
# library(parallel)
# cores = detectCores()-1
# plan("multisession", workers = cores)
# set.seed(123, kind = "L'Ecuyer-CMRG")

task_rf = TaskClassif$new(id = "train", backend = train, 
                          target = "DRK_YN", positive = "Y")

nested_res = mlr3::resample(
  task = task_rf,
  learner = rf_at,
  resampling =  res_outer,
  store_models = T)

extract_inner_tuning_results(nested_res)
plan("sequential")





nested_res$aggregate()
autoplot(nested_res)

nested_res$prediction()
nested_res$help()
