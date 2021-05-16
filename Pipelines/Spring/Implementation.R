library(SPRING)
args = commandArgs(trailingOnly=TRUE)
########################## Default Dataset ########################
#data("QMP") # load the data available from this package, containing 106 samples and 91 OTUs.

########################### Desert Dataset ########################
dataset <- read.csv(file = args[1],row.names=1,header=TRUE)
taxa_names = list(row.names(dataset))
dataset = t(dataset)
#####################################################################

# Apply SPRING on QMP data.
fit.spring <- SPRING(dataset, Rmethod = "original", quantitative = TRUE, lambdaseq = "data-specific", nlambda = 50, rep.num = 100)
# With Rmethod = "original", this takes around 23 minutes.
# With Rmethod = "approx", this takes around 2.23 minutes. 
# More details on the comparison of accuracy and speed ("original" vs. "approx")
# are available on the above arXiv reference.

# StARS-selected lambda index based on the threshold (default = 0.01)
opt.K <- fit.spring$output$stars$opt.index
# Estimated adjacency matrix from sparse graphical modeling technique ("mb" method) (1 = edge, 0 = no edge)
adj.K <- as.matrix(fit.spring$fit$est$path[[opt.K]])
# Estimated partial correlation coefficient, same as negative precision matrix.
pcor.K <- as.matrix(SpiecEasi::symBeta(fit.spring$output$est$beta[[opt.K]], mode = 'maxabs'))
