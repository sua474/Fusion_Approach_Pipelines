library(PLNmodels)
library(ggplot2)
library(stringr)

rm(list = ls())
args = commandArgs(trailingOnly=TRUE)


get_file_name<-function(path_to_dir)
{
  fn <- str_split(path_to_dir,"/",simplify = FALSE)
  fn <- fn[[1]][lengths(fn)]
  fn <- str_split(fn, ".csv", simplify = FALSE)
  return (str_trim(fn[[1]][1]))
}


## Load the data: absolute abundances and covariates
abundance <- read.csv(args[1], row.names = 2)[-1]
## Prepare the dataset
abundance = t(abundance)  # Transpose to have the samples as rows
covariates <- read.csv("/u1/sua474/Dataset/Desert/Transformed_Covariates.csv",row.names = 1)
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

dir.create(args[2])                                         # Creates the directory if not there

adj_matrix_name = paste0("StARS_Adjacency_Matrix_",get_file_name(args[1]),".csv")     #Naming the output file
deg_matrix_name = paste0("StARS_Degree_Distribution_",get_file_name(args[1]),".csv")  #Naming the output file

write.csv(model_StARS.adj, paste0(args[2],adj_matrix_name)) # Writing the output
write.csv(model_StARS.deg, paste0(args[2],deg_matrix_name)) # Writing the output
