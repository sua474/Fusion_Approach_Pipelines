library(PLNmodels)
library(ggplot2)
args = commandArgs(trailingOnly=TRUE)


##### Load our dataset #########################################################

abundance <- read.csv(file = args[1],row.names=1,header=TRUE)
taxa_names = list(row.names(abundance))
abundance = t(abundance)
covariates <- read.csv(file="/u1/sua474/Dataset/Desert/Transformed_Covariates.csv",row.names=1,header=TRUE)
print(covariates)
dataset <- prepare_data(abundance,covariates)
print(abundance)
################## TO BE REMOVED WHEN WE HAVE THE COVARIATES ###################
##### Loading the default dataset provided with the implementation #############
#data(trichoptera)
#dataset <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
################################################################################


############################ Training the network on data ######################

network_models <- PLNnetwork(Abundance ~ 1 + offset(log(Offset)), data = dataset)

###############################################################################

##################### Plotting the model ######################################
print(network_models)
#plot(network_models, "diagnostic")
#plot(network_models)
###############################################################################

model_StARS <- getBestModel(network_models, "StARS")
plot(model_StARS)