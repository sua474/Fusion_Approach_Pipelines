library(SpiecEasi)
library(igraph)
library(phyloseq)
args = commandArgs(trailingOnly=TRUE)
dataset <- read.csv(file = args[1],row.names=1,header=TRUE)
taxa_names = list(row.names(dataset))

dataset = t(dataset)
###################### Model Fitting #############################################
se <- spiec.easi(dataset, method='mb', lambda.min.ratio=0.001, nlambda=25, pulsar.params=list(rep.num=10))
se.slr     <- adj2igraph(getRefit(se),vertex.attr = taxa_names)
slr.coord <- layout.fruchterman.reingold(se.slr)
vsize    <- rowMeans(clr(dataset, 1))+6

print(se.slr)
##################################################################################

###################### Create Graph #############################################
d <- ncol(dataset)
n <- nrow(dataset)
e <- d

set.seed(10010)
graph <- make_graph('cluster', d, e)
################################################################################

##################### Plot Model ##############################################

huge::huge.roc(se$est$path, graph, verbose=FALSE)
#stars.pr(getOptMerge(se), graph, verbose=FALSE)

par(mfrow=c(1,3))
plot(se.slr, layout=slr.coord, vertex.size=8, vertex.label=NA, main="mb")
