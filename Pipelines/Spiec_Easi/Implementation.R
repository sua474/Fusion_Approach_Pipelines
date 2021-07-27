library(SpiecEasi)
library(igraph)
library(phyloseq)
library(stringr)


args = commandArgs(trailingOnly=TRUE)

print(args[1])
print(args[2])
print(args[3])
print(as.numeric(args[3]))
print(typeof(args[3]))
print(args[4])
print(args[5])

get_file_name<-function(path_to_dir)
{
  fn <- str_split(path_to_dir,"/",simplify = FALSE)
  fn <- fn[[1]][lengths(fn)]
  fn <- str_split(fn, ".csv", simplify = FALSE)
  return (str_trim(fn[[1]][1]))
}

dataset <- read.csv(file = args[1],header=TRUE)
row_names = list(row.names(dataset))
dataset = t(t(dataset))
###################### Model Fitting #############################################
se <- spiec.easi(dataset, method='mb', lambda.min.ratio=as.numeric(args[3]), nlambda=as.integer(args[4]), pulsar.params=list(rep.num=as.integer(args[5])))
adjacency_matrix = as.data.frame(as.matrix(getRefit(se)))
##################################################################################
dir.create(args[2])
write.csv(adjacency_matrix,paste0(args[2],"Adjacency_Matrix_",get_file_name(args[1]),".csv"),row.names = FALSE)
