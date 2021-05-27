library(SPRING)
library(stringr)

args = commandArgs(trailingOnly=TRUE)
dir.create(args[2])                 #Creates the output dir

get_file_name<-function(path_to_dir)
{
  fn <- str_split(path_to_dir,"/",simplify = FALSE)
  fn <- fn[[1]][lengths(fn)]
  fn <- str_split(fn, ".csv", simplify = FALSE)
  return (str_trim(fn[[1]][1]))
}

dataset <- read.csv(file = args[1],row.names=1,header=TRUE)
taxa_names = list(row.names(dataset))
dataset = t(dataset)
#####################################################################

# Apply SPRING on QMP data.
fit.spring <- SPRING(dataset, Rmethod = "original", quantitative = TRUE, lambdaseq = "data-specific", nlambda = 50, rep.num = 10)
# StARS-selected lambda index based on the threshold (default = 0.01)
opt.K <- fit.spring$output$stars$opt.index
# Estimated adjacency matrix from sparse graphical modeling technique ("mb" method) (1 = edge, 0 = no edge)
adj.K <- as.matrix(fit.spring$fit$est$path[[opt.K]])
# Estimated partial correlation coefficient, same as negative precision matrix.
pcor.K <- as.matrix(SpiecEasi::symBeta(fit.spring$output$est$beta[[opt.K]], mode = 'maxabs'))
adjacency_matrix <- as.data.frame(pcor.K)

#Writes Output to the file
write.csv(adjacency_matrix,paste0(args[2],"Adjacency_Matrix_",get_file_name(args[1]),".csv"),row.names = FALSE) 

