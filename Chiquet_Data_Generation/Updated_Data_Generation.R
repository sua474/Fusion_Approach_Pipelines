# data generation based on Chiquet et al. (2019)
# p number of taxa
# n number of measurementsa

library(Matrix)
library(sparseMVN)
library(MCMCpack)
library(stats)
library(ramify)
library(pracma)
library(matrixcalc)
library(matlib)
library(mvtnorm)
library(proxyC)


generate_abundance <- function(abundance_data,number_of_draws,taxa,number_of_samples)
{
  
  sum_of_columns <- 1
  abundance_matrix = abundance_data
  yi = -1
  
  for (row in 1:nrow(abundance_data))
  {
    row_vector=-1
    max_val = max(abundance_data[row,])
    taxon_proportion = exp((abundance_data[row,]-max_val))/sum(exp(abundance_data[row,]-max_val)) # taking proportion with the new formula
    draws <- rmultinom(n=number_of_draws, size=sum_of_columns, prob=as.vector(taxon_proportion)) # getting the draws
    draws = t(draws)
    
    for (x in 1:ncol(draws))
    {
      row_vector = c(row_vector, sum(draws[,x]) )
    }
    yi = c(yi,row_vector[2:length(row_vector)])
    
  }
  yi = t(Reshape(as.matrix(yi[2:length(yi)]),taxa,number_of_samples))
  
  return (yi)
}


generate_ground_truth <- function(taxa,u,v,number_of_samples)
{
  #Draw randomly from a uniform distribution, round it to 0 or 1 to make it binary and force symmetry
  G = forceSymmetric(round(array(runif(taxa*taxa,min=0,max=1), dim=c(taxa,taxa))))
  omega_tilda = as.matrix((G*v))
  mask = diag(taxa) * (abs(min(as.array(eigen(omega_tilda,symmetric = TRUE)$vectors)))+u)
  Omega = (omega_tilda + mask)
  Omega_inv = inv(Omega)
  
  ###############################################################################################
  XB = array(rexp(taxa, rate = 1)) #That's a temporary variable made to mimic what they've done in paper
  ###############################################################################################
  ai = round(exp(rmvnorm(number_of_samples,mean=XB,sigma=Omega_inv)))

  return (list("ai" = ai, "ground_truth" = G))
  
}

count_zero_percentage <- function(data)
{
  count = 0
  
  for (i in 1:nrow(data))
  {
    for (j in 1:ncol(data))
    {
      if(data[i,j]==0)
        count = count+1
    }
  }
  
  return ((count/(nrow(data)*ncol(data))))
}

############################################### Run Paramemeters############################

v = 0.3 # As specified in the Chiquet's paper
u = 0.1  # As specified in the Chiquet's paper
taxa = 10
number_of_samples = 30 
number_of_draws = 2000
datasets_to_be_generated =  25
#############################################################################################

for (x in 1:datasets_to_be_generated) 
{
 results = generate_ground_truth(taxa,u,v,number_of_samples) 
 abundance_data = generate_abundance(results$ai,number_of_draws,taxa,number_of_samples)
  
 print(count_zero_percentage(abundance_data))
 write.csv(as.data.frame(as.matrix(results$ground_truth)), file=paste(c("I:/Research_Technician/Dataset/Chiquet_Data_new/Ground_Truth_", x, ".csv"),collapse = ""),row.names=FALSE)
 write.csv(as.data.frame(as.matrix(abundance_data)), file=paste(c("I:/Research_Technician/Dataset/Chiquet_Data_new/Chiquet_", x, ".csv"),collapse = ""),row.names=FALSE)
}