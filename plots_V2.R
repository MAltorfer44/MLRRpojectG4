
library(iml)
library(pdp)



rm(list=ls())

# load necessary objects(data, models) -> save Memory
load("pre.RData")
load("bm.RData")


###################### Plots ##################################$


# variable importance
impVars = bm_ranger$variable.importance%>%sort(decreasing = T)

barplot(impVars, main = "Var importance", col = "grey", las = 2)
varImpPlot(bm_RandomForest)


# partial dependence plot based on RF from RandomForest library


pdp::partial(object=bm_RandomForest, pred.var = c("age", "sex"), plot = TRUE, chull = TRUE)

partialPlot(bm_RandomForest, pred.data = train, x.var = "age")
partial_plot <- pdp::partial(bm_RandomForest, pred.var = "age")






#create prediction object
pred <- Predictor$new(bm_RandomForest, data = train, y=train$DRK_YN)
	
#create feature effect object
eff <- FeatureEffect$new(pred, 
                      feature = "age", 
                         method = "pdp")
plot(eff)






