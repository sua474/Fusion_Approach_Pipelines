######################## Installtion Steps ################################
#devtools::install_github("drjingma/metaMint")
#library(metaMint)
#vignette("metaMint")
####################### Library Import ####################################
library(metaMint)
library(dplyr)
library(GGally)
args = commandArgs(trailingOnly=TRUE)
########################## Code Test ######################################
#data(BV, package = "metaMint")
#print(t(OTUs))
#OTUs_mclr <- mclr(t(OTUs))
#print(OTUs_mclr)
############################################################################
######################### Processing Desert Dataset ########################
dataset <- read.csv(file = args[1],row.names=1,header=TRUE)
mclr_dataset <- mclr(t(dataset))
pcorr <- cggm.pcorr(mclr_dataset,c(1.5,0.4,0.3,0.2,0.1,0.05),'glasso')
print(pcorr$icov)

