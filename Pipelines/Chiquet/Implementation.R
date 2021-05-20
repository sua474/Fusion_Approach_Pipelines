library(PLNmodels)
library(ggplot2)
rm(list = ls())
## Load the data: absolute abundances and covariates
abundance <- read.csv("~/Dropbox/Cyclotron Research/Carbon_Oxidation_and_Fixation_Study/Metagenome/SteveM/Network/abu.species.glom1.csv", row.names = 2)[-1]
## Prepare the dataset
abundance = t(abundance)  # Transpose to have the samples as rows
covariates <- read.csv("~/Dropbox/Cyclotron Research/Carbon_Oxidation_and_Fixation_Study/Metagenome/SteveM/Network/Transformed_Covariates.csv",row.names = 1)
identical(rownames(abundance), rownames(covariates))        # FALSE; there are spaces in the covariate row names
rownames(covariates) <- gsub(" ","", rownames(covariates))  # Replace the spaces
abundance <- abundance[rownames(covariates),]               # Sort abundances by row names in covariates to make row names match
identical(rownames(abundance), rownames(covariates))        # TRUE
dataset <- prepare_data(abundance,covariates)               # Prepare data in proper format for use in PLN model and its variants (PLNmodels package). 
## Generate the networks, select the best, and output the adjacency matrix and node degrees
network_models <- PLNnetwork(Abundance ~ 1 + offset(log(Offset)), data = dataset)
model_StARS <- getBestModel(network_models, "StARS")                      # Network selection
plot(model_StARS, output = "igraph")                                      # Default plotting
model_StARS.adj <- as.matrix(plot(model_StARS, output = "corrplot"))      # Adjacency matrix
model_StARS.deg <- rowSums(model_StARS.adj != 0)                          # Node degrees
write.csv(model_StARS.adj, "~/Dropbox/Cyclotron Research/Carbon_Oxidation_and_Fixation_Study/Metagenome/SteveM/Network/model_StARS.adj.csv")
write.csv(model_StARS.deg, "~/Dropbox/Cyclotron Research/Carbon_Oxidation_and_Fixation_Study/Metagenome/SteveM/Network/model_StARS.deg.csv")