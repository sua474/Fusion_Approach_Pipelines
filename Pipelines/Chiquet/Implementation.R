library(PLNmodels)
library(ggplot2)
args = commandArgs(trailingOnly=TRUE)

##### Load our dataset #########################################################
dataset <- read.csv(file = args[1],row.names=1,header=TRUE)
taxa_names = list(row.names(dataset))
dataset = t(dataset)

################## TO BE REMOVED WHEN WE HAVE THE COVARIATES ###################
##### Loading the default dataset provided with the implementation #############
data(trichoptera)
dataset <- prepare_data(trichoptera$Abundance, trichoptera$Covariate)
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