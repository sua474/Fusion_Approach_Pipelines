library(metaMint)
library(dplyr)
library(GGally)
library(stringr)

args = commandArgs(trailingOnly=TRUE)

dir.create(args[2]) #Creates the output dir

get_file_name<-function(path_to_dir)
{
  fn <- str_split(path_to_dir,"/",simplify = FALSE)
  fn <- fn[[1]][lengths(fn)]
  fn <- str_split(fn, ".csv", simplify = FALSE)
  return (str_trim(fn[[1]][1]))
}

######################### Processing Desert Dataset ########################
dataset <- read.csv(file = args[1],header=TRUE)
dataset <- t(t(dataset))

mclr_dataset <- mclr(dataset)

pcorr <- cggm.pcorr(mclr_dataset,c(as.numeric(args[3]),as.numeric(args[4]),as.numeric(args[5])),'glasso')
adjacency_matrix <- as.data.frame(pcorr$icov)   #Extracts the Adjacency Matrix

#Writes Output to the file
write.csv(adjacency_matrix,paste0(args[2],"Adjacency_Matrix_",get_file_name(args[1]),".csv"),row.names = FALSE) 
