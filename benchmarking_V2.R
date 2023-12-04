# Load necessary libraries
library(ranger)
library(tidyverse)
library(mlr3)
library(mlr3pipelines)
library(mlr3learners)
library(mlr3viz)
rm(list=ls())
####### benchmarking
load("mlrTuned.RData")
load("pre.RData")

# benchmarking is done on new data that was not previously used to tune the RF and tree


# train percentage
perc=0.05 # no more possible cause my RAM can't handle it

# create a vector with the row index for the train data
train_set = sample(seq_len(nrow(test)), 
                   size = perc*nrow(test), 
                   replace = F # alternatively replace = T
)


bm_data=test[train_set,]

rm(test)
rm(train)
gc()
imputer = po("imputemedian")
task_bm = TaskClassif$new(id = "BM", backend = bm_data, 
                          target = "DRK_YN", positive = "Y")


logreg = lrn("classif.log_reg")
logreg = as_learner(imputer %>>% logreg)

tree = lrn("classif.rpart")
tree = as_learner(imputer %>>% tree)

load("treeRes.RData")
# tuned tree
tr = lrn("classif.rpart",
         cp = results_mlr_tree$cp[1])
tr = as_learner(imputer %>>% tr)

baseline = lrn("classif.featureless")
rf_d = lrn("classif.ranger")
rf_d = as_learner(imputer %>>% rf_d)

# iteration  mtry min.node.size sample.fraction num.trees classif.ce
# <int> <int>         <int>           <dbl>     <int>      <dbl>
#   1   2     4     12               0.0310       969      0.294

rf_t = lrn("classif.ranger",mtry=results_mlr$mtry[1], 
           min.node.size=results_mlr$min.node.size[1],
           num.trees = results_mlr$num.trees[1],
           sample.fraction = results_mlr$sample.fraction[1])
rf_t = as_learner(imputer %>>% rf_t)

design_class =  benchmark_grid(
  tasks = task_bm,
  learners = list(logreg, tree, tr, rf_d, rf_t),
  resamplings = rsmp("cv", folds = 20)
)


# causes more issues than it solves problems on my machine
# library(future)
# library(parallel)
# cores = detectCores()-1
# plan("multisession", workers = cores)
# set.seed(123, kind = "L'Ecuyer-CMRG")
# 
# plan("multisession", workers = cores)


# also takes a while to run
bm_class = benchmark(design_class, store_models = F)

#save(bm_class,file="benchmarkMLR.Rdata")
load("benchmarkMLR.Rdata")

# plan("sequential")

mes_class = msrs(c("classif.sensitivity","classif.specificity", "classif.acc", "classif.precision"))

bmr_class = bm_class$aggregate(mes_class)
bmr_class[, c(4, 7:10)]

df_bmr = as.data.table(bmr_class)

  
  
mlr3viz::autoplot(bm_class, measure = msr("classif.acc"))
mlr3viz::autoplot(bm_class, measure = msr("classif.precision"))
