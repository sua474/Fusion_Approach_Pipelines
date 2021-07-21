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

############################################# Helper Functions ###############################

count_zero_percentage_per_taxa <- function(data)
{
  percentage_vector = -1
  for (i in 1:ncol(data))
  {
    count = 0
    for (j in 1:nrow(data))
    {
      if(data[j,i]==0)
        count = count+1
    }
    
    percentage_vector = c(percentage_vector,(count/nrow(data)))
  }
  percentage_vector = percentage_vector[2:length(percentage_vector)]
  return (mean(percentage_vector))
}

normalize_ai <- function(ai)
{
  
  for (i in 1:nrow(ai))
  {
    for(j in 1:ncol(ai))
    {
      if(ai[i,j]>=roundoff_threshold)
      {
        ai[i,j] = 0
      }
    }
  }
  
  return (ai)
}

###################################### Step Functions ########################################

Step_3 <- function(taxon_proportion,number_of_draws,taxa,number_of_samples)
{
  
  sum_of_columns <- 1  # This is as per our logic that sum of each draw should be equal to 1
  yi = -1  # It's initiated at -1 so that it can be discarded later
  
  for (row in 1:nrow(taxon_proportion))
  {
    row_vector=-1
    draws <- rmultinom(n=number_of_draws, size=sum_of_columns, prob=as.vector(taxon_proportion[row,])) # getting the draws # check with Steve for draws
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


Step_2 <- function(ai,roundoff_threshold)
{
  proportions = -1
  
  for(i in 1:nrow(ai))
  {
    taxon_proportion = exp((ai[i,]))/sum(exp(ai[i,]))
    proportions = c(proportions,taxon_proportion)
  }
  
  proportions = t(Reshape(as.matrix(proportions[2:length(proportions)]),ncol(ai),nrow(ai)))
  
return (proportions)

}


Step_1 <- function(taxa,u,v,number_of_samples,B_location)
{
  #Draw randomly from a uniform distribution, round it to 0 or 1 to make it binary and force symmetry
  G = forceSymmetric(round(array(runif(taxa*taxa,min=0,max=1), dim=c(taxa,taxa))))
  omega_tilda = as.matrix((G*v))
  mask = diag(taxa) * (abs(min(as.array(eigen(omega_tilda,symmetric = TRUE)$vectors)))+u)
  Omega = (omega_tilda + mask)
  Omega_inv = inv(Omega)
  
  ############################ Old Method to generate XB and ai ###############################
  #XB = array(rexp(taxa, rate = 1)) #That's a temporary variable made to mimic what they've done in paper
  #ai = round(exp(rmvnorm(number_of_samples,mean=XB,sigma=Omega_inv)))
  #############################################################################################

  ############################### Updated B Matrix Generation #########################
  B_vector <- read.csv(file = B_location,header=FALSE)
  B = array(unlist(B_vector))
  ai = round(exp(rmvnorm(number_of_samples,mean=B,sigma=Omega_inv)))
  
  #######################################################################################  
  t = hist(flatten(normalize_ai(ai)),xlab="ai Values",main="Histogram")
  return (list("ai" = normalize_ai(ai), "ground_truth" = G))
  
}


############################################ Run Paramemeters ##############################

v = 0.3 # As specified in the Chiquet's paper
u = 0.1  # As specified in the Chiquet's paper
taxa = 10
number_of_samples = 30 
number_of_draws = 5000
roundoff_threshold = 100 #rounds off the ai values to zero if they are too large
datasets_to_be_generated =  100

B_location = 'I:/Research_Technician/Development/Chiquet_B_Calculation/Output/B_MR_Absolute.csv' 
#############################################################################################
total = -1
for (x in 1:datasets_to_be_generated) 
{
 results = Step_1(taxa,u,v,number_of_samples,B_location)
 taxon_proportion = Step_2(results$ai,roundoff_threshold)
 abundance_data = Step_3(taxon_proportion,number_of_draws,taxa,number_of_samples)
  
 percentage = count_zero_percentage_per_taxa(abundance_data)
 total = c(total,percentage)
 write.csv(as.data.frame(as.matrix(results$ground_truth)), file=paste(c("I:/Research_Technician/Dataset/Chiquet/Chiquet_100/Ground_Truth_", x, ".csv"),collapse = ""),row.names=FALSE)
 write.csv(as.data.frame(as.matrix(abundance_data)), file=paste(c("I:/Research_Technician/Dataset/Chiquet/Chiquet_100/Chiquet_", x, ".csv"),collapse = ""),row.names=FALSE)
}

paste0("The average zero per taxon for all dataset is: ",mean(total[2:length(total)]))
