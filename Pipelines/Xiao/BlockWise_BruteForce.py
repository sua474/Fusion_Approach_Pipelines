#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[1]:

import argparse
import pandas as pd
import numpy as np
from scipy.linalg import null_space,orth
from sympy import *
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sb
import operator as op
from functools import reduce
import multiprocessing
from multiprocessing import Pool as ThreadPool 
from joblib import Parallel, delayed
from functools import partial
import sys
import difflib
from datetime import datetime
import os
import csv
import time
import math as m
import itertools as it


# # Data Pre-Processing Helper Function

# In[2]:


def drop_zero_indexes(df):
    '''
    Parameter
    df = The loaded datafile to be processed
    
    Description:
    This function deletes all the rows in excel file which are entirely zero (i.e. zero for all features)
    '''
    drop_index = []
    for i in range(0,len(df)):
        if(not df.loc[i].any()):
            drop_index.append(i)
    df.drop(drop_index,inplace=True)
    print("Number of dropped Indexes: "+str(len(drop_index)))
    return df.copy()

def write_max_phi_count(taxon,phi_distribution):
    
    with open(result_dir+"Max_Phi_Distribution_"+str(taxon)+".csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow( [max(phi_distribution)] )
        wr.writerow( [phi_distribution.count(max(phi_distribution))] )

def weighted_frequency(sign_pattern,phi_list,indices):
    '''
    Parameters:
    sign_pattern = The sign pattern
    phi_list =  corresponding phi values
    indices = indexes to operate on
    
    Description:
    takes sum of each individual sign value and return a value with the highest sum
    '''
    weight_positive = 0.0
    weight_negative = 0.0
    weight_zero = 0.0
    
    
    for i in range(0,len(indices)):
        if(sign_pattern[i]==1):
            weight_positive+=phi_list[indices[i]]
        elif(sign_pattern[i]==-1):
            weight_negative+=phi_list[indices[i]]
        elif(sign_pattern[i]==0):
            weight_zero+=phi_list[indices[i]]
    
    if(weight_positive>=weight_negative and weight_positive>=weight_zero):
        return 1
    elif(weight_negative>=weight_positive and weight_negative>=weight_zero):
        return -1
    else:
        return 0

def perform_voting(indices,taxon,phi_distribution,sign_distribution,path_distribution,pair_difference,giveup_percent,np_pd,factor_list):
    '''
    Parameter:
    indices = Index value of phi values above specified threshold
    taxon = Taxon under consideration 
    phi_distribution = Distribution of Phi values for a particular taxon 
    sign_distribution = Distribution of predicted sign values 
    path_distribution = Distribution of predicted path values
    pair_difference = pair difference for a particular taxon
    giveup_percent = the limit of percentage of maximum threshold of phi which can be givenup( above which all 
    values are selected)
    Description:
    Performs voting of sign patterns above a selected specified threshold. Either return a new sign pattern
    or finds a closest match to the calculated best sign pattern and returns it.
    '''
        
    predicted_sign_pattern = []
    pattern = []
    sign_path = [["Matrix Based Sign Satisfaction"]]
    phi = 0 
    
    sign_distribution_df = pd.DataFrame(columns=np.arange(len(sign_distribution[0][0])))
    for i in range(0,len(indices)):
        sign_distribution_df.loc[len(sign_distribution_df)] = sign_distribution[indices[i]][0]
        
    for j in range(0,len(sign_distribution_df.columns)):
        pattern.append(weighted_frequency(sign_distribution_df.iloc[:,j].tolist(),phi_distribution,indices))
    
    predicted_sign_pattern.append(pattern)
    phi,test_sp = get_reduced_intersected_hyperplane_count(predicted_sign_pattern,pair_difference,taxon,False,np_pd,factor_list)         
    phi_j = phi/len(pair_difference.columns)
    giveup_threshold = max(phi_distribution) - (max(phi_distribution)*giveup_percent/100)
        
    if(phi_j>0 and phi_j>=giveup_threshold):
        temp_list = []
        temp = predicted_sign_pattern[0]
        temp = np.asarray(temp)
        temp_list.append(temp)
        return phi_j,sign_path[0],temp_list
    else:
        max_ratio=0
        best_index = 0
        for i in range(0,len(indices)):
            sm=difflib.SequenceMatcher(None,predicted_sign_pattern[0],sign_distribution[indices[i]][0])
            ratio = sm.ratio()
            if(ratio > max_ratio):
                max_ratio = ratio 
                best_index = i
        return phi_distribution[indices[best_index]],path_distribution[indices[best_index]],sign_distribution[indices[best_index]]

def compute_block_lengths(block_indxes,no_of_features):
    '''
    Parameter:
    block_indxes = Indexes of columns to be processed in a block 
    no_of_features = total number of features
    
    Description:
    Computes variable Block length wrt to the given indexes
    '''
        
    block_sizes = []
    ref_index = 0
    for x in block_indxes:
        if(np.sum(block_sizes)+(x-ref_index)<=no_of_features):
            block_sizes.append(x-ref_index)
            ref_index = x
        else:
            break
    if no_of_features > np.sum(block_sizes): 
        block_sizes.append(int(no_of_features - np.sum(block_sizes)))
    
    return block_sizes

# def reduced_pair_difference(pair_difference):
#     '''
#     Parameter:
#     pair_difference = pair difference dataframe
    
#     Description:
#     Reduces the number of common columns of pair differences
#     '''
#     np_pd = pair_difference.to_numpy(dtype='float64',copy=True)
#     pd_sp = np.sign(np_pd)
#     i=0
#     common_count = []
    
#     while(i<pd_sp.shape[1]):
#         counter=1
#         j=i+1
#         while(j<pd_sp.shape[1]):
#             if(np.array_equal(pd_sp[:,i],pd_sp[:,j])):
#                 pd_sp = np.delete(pd_sp,j,1)
#                 counter+=1
#             j+=1
#         common_count.append(counter)
#         i+=1
    
#     return pd_sp,common_count

def reduced_pair_difference(pair_difference):
    '''
    Parameter:
    pair_difference = pair difference dataframe
    
    Description:
    Reduces the number of common columns of pair differences
    '''
    
    np_pd = pair_difference.to_numpy(dtype='float64',copy=True)
    pd_sp = np.sign(np_pd)
    common_count = []
    column_dicitionary = {}

    for i in range (0, pd_sp.shape[1]):
        key = ','.join(map(str, pd_sp[:,i])) 
        if key in column_dicitionary:
            column_dicitionary[key]+= 1
        else:
            column_dicitionary[key] = 1

    first = True
    for x in column_dicitionary.keys():
        l1 = np.array(list(map(float,x.split(","))))
        common_count.append(column_dicitionary[x])
        if(first): 
            l1 = l1[np.newaxis]
            new_matrix = l1.T
            first = False
        else:
            new_matrix = np.insert(new_matrix,new_matrix.shape[1],l1,axis=1)
    
    return new_matrix,common_count

def get_reduced_intersected_hyperplane_count(sign_d,pair_difference,taxon,perform_perturbation,np_pd,factor_list):
    '''
    Parameter:
    sign_d = Heuristic Based Sign Patter 
    pair_difference = Pair Difference Matrix
    
    Description:
    Performs GPU accelerated, matrix based graph creation and path traversal. Return the total number of
    intersected hyperplanes for a particular sign_d and difference matrix
    '''
    
    #print("Received Sign_d: "+str(sign_d))
    
    perturbate_signd = True
    intersection_threshold = 0.0
    mutation_list = [taxon]
    perturbated_sign_d =0 
    
    sign_d = np.asarray(sign_d,dtype='float64')
    sign_d = sign_d.reshape(len(pair_difference),1)
    perturbated_sign_d = sign_d.copy()
    #np_pd,factor_list = reduced_pair_difference(pair_difference)
    
    while(perturbate_signd):
        intersected_hyperplanes = 0.0
        sign_graph = np.multiply(sign_d, np_pd)
        #sign_graph = np.sign(sign_matrix)

        for i in range(0,sign_graph.shape[1]):
            intersected_hyperplanes+=(is_sign_satisfied_matrix(sign_graph[:,i])*factor_list[i])
         
        if(intersected_hyperplanes>intersection_threshold and perform_perturbation==True):
            #print("Old Result: "+str(intersection_threshold)+" New Result: "+str(intersected_hyperplanes))
            perturbated_sign_d = sign_d.copy()
            sign_d,index = get_perturbated_signd(sign_d,mutation_list)
            mutation_list.append(index)
            intersection_threshold = intersected_hyperplanes
        else:
            perturbate_signd = False
            best_sign_d = perturbated_sign_d.reshape(len(pair_difference))
            if(perform_perturbation == False):
                intersection_threshold = intersected_hyperplanes
                
            best_sign_d = np.asarray(best_sign_d.tolist(),dtype="int64")
            temp_sign_d = []
            temp_sign_d.append(best_sign_d)
            #print("Returned Sign_d: "+str(temp_sign_d))

            
    return intersection_threshold,temp_sign_d

def is_sign_satisfied_matrix(sign_path):
    '''
    Parameter:
    sign_path = Numpy array representing the sign path 
    
    Description:
    Takes a numpy array of sign path and checks if the path is sign satisfied or not.
    '''
    
    first_sign = 0
    for i in range(0,sign_path.size):
        if(sign_path[i]!=0 and first_sign==0):
            first_sign = sign_path[i]
        elif(sign_path[i]!=0 and first_sign and (first_sign!=sign_path[i])):
            return 1.0
    return 0.0

def get_samples_for_taxon(ith_taxon,block_length):
    '''
    Parameters:
    ith_taxon = The feature/taxon number
    block_length = Number of Features to be processed
    
    Descrpition:
    Returns all rows of dataframe where the specified taxon's sample value is non-zero. 
    ''' 
    global selected_otu_abundance_df
    active_samples = selected_otu_abundance_df.loc[selected_otu_abundance_df.iloc[:,ith_taxon] != 0].copy()
    return active_samples.iloc[:,:block_length]

def sample_pair_difference_unordered(steady_state_samples):
    '''
    Parameter: 
    steady_state_samples = All rows of dataframe with non-zero entry for a specific taxon
    
    Description:
    Calculates the pari-difference matrix for that particular taxon/feature using built-in pandas functionality.
    This function is faster, but the results of rows are unordered.
    '''
    global discard_threhold
    pair_difference = pd.DataFrame()
    steady_state_samples.reset_index(inplace=True,drop=True)
    
    for i in range (0,len(steady_state_samples)-1):
        differences = steady_state_samples.diff(periods=-1*(i+1))
        differences.dropna(how="all",inplace=True)
        pair_difference = pd.concat([pair_difference,differences])
    pair_difference.reset_index(inplace=True,drop=True)
    pair_difference = pair_difference.transpose()
    pair_difference.reset_index(inplace=True,drop=True)
    
    pair_difference[(pair_difference <= discard_threhold) & (pair_difference >= -1*discard_threhold)] = 0 #discard Smaller Values
    return pair_difference.copy()


def sensitivity_of_jaccobian(cummulative_jaccobian,feature_names,commulative_confidence,result_dir):
    '''
    Parameters:
    cummulative_jaccobian = Column based Jaccobian matrix for all voting iterations.
    feature_names = Name of taxon/feature in the provided data file
    commulative_confidence = All Phi values for top selected results of all iterations of jaccobian
    result_dir = Directory where results will be saved
    
    Description:
    Creates and exports csv and png files of mean and variance heatmaps of column jaccobian matrix
    (each column represents all values of a feature per voting iteration).Also creates plot for
    the values phi for top selected sign pattern 
    '''
    
    # variance_jaccobian = pd.DataFrame(columns=feature_names,index=feature_names)
    # mean_jaccobian = pd.DataFrame(columns=feature_names,index=feature_names)
    # position = 0
    
    # for cols in cummulative_jaccobian.columns:
    #     variance_jaccobian.iloc[position//len(feature_names),position%len(feature_names)] = float(np.var(cummulative_jaccobian[cols].tolist()))
    #     mean_jaccobian.iloc[position//len(feature_names),position%len(feature_names)] = float(np.mean(cummulative_jaccobian[cols].tolist()))
    #     position+=1
    
    # variance_jaccobian = variance_jaccobian.astype('float64')
    # variance_jaccobian.to_csv(result_dir+"Variance_Matrix.csv")
    # plt.figure()
    # plt.axis('tight')
    # plt.title("Variance of Jaccobian Sign Matrix")
    # sb.heatmap(variance_jaccobian,annot=True,center=0,cmap=['white','grey','purple','grey','yellow','orange','red'],cbar=False)
    # plt.savefig(orientation='landscape',fname=result_dir+"Variance_Plot.png",format='png',dpi=600,bbox_inches='tight')
    # plt.show()
    
    # mean_jaccobian = mean_jaccobian.astype('float64')
    # mean_jaccobian.to_csv(result_dir+"Mean_Matrix.csv")
    # plt.figure()
    # plt.axis('tight')
    # plt.title("Mean of Jaccobian Sign Matrix")
    # sb.heatmap(mean_jaccobian,annot=True,center=0,cmap=['blue','purple','grey','orange','red'],cbar=False)
    # plt.savefig(orientation='landscape',fname=result_dir+"Mean_Plot.png",format='png',dpi=600,bbox_inches='tight')
    # plt.show()
    
    # plt.figure()
    # plt.title("Cummulative Phi Values Plot")
    # plt.xticks(rotation=90)
    # plt.plot(feature_names,commulative_confidence,marker='o')
    # plt.savefig(orientation='landscape',fname=result_dir+"Phi_Plot.png",format='png',dpi=600,bbox_inches='tight')
    # plt.show()
    
    with open(result_dir+"Phi_values.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(feature_names)
        wr.writerow(commulative_confidence)
    

def distribution_based_match(phi_distribution,sign_distribution,path_distribution,voting_type,pair_difference,taxon,np_pd,factor_list):
    '''
    Parameters:
    phi_distribution = Disribution of Phi
    sign_distribution = Distrubution of sign for each Phi value
    path_distribution = Distribution of path for each Phi value 
    voting_type = Voting type either 'best'(custom built) or 'max'(original)
    taxon = Taxon/Feature number under consideration
    
    Description:
    Based on user preference, implements a suggested method of sign pattern selection and return the results.
    ''' 
    if(max(phi_distribution)==0):
        temp_list = []
        print("No Solution Found for Taxa: "+str(taxon))
        zero_pattern = np.zeros(len(sign_distribution[0][0]),dtype=int)
        if(len(sign_distribution[0][0])>taxon):
            zero_pattern[taxon] = -1
        temp_list.append(zero_pattern)
        return 0,["No Solution Found for Taxa: "+str(taxon)],temp_list
    elif(voting_type == 'max'):
        top_voted_index = phi_distribution.index(max(phi_distribution))
        return phi_distribution[top_voted_index],path_distribution[top_voted_index],sign_distribution[top_voted_index]
    elif(voting_type == 'best'):
        threshold = max(phi_distribution)
        indices = [i for i, x in enumerate(phi_distribution) if x >= threshold]
        phi,path,pattern = perform_voting(indices,taxon,phi_distribution,sign_distribution,path_distribution,pair_difference,0,np_pd,factor_list)
        return phi,path,pattern


def create_interaction_network(jaccobian,vote_iteration,result_dir):
    '''
    Parameters:
    jaccobian = Jaccobian matrix for particular iteration
    vote_iteration = Voting iteration
    result_dir = directory where results are to be saved
    
    Description:
    Creates a heatmap of jaccobian iteration's predicted sign pattern
    '''
    global selected_otu_abundance_df
    counter=0
    feature_names = selected_otu_abundance_df.columns
    jaccobian_matrix = pd.DataFrame(columns = feature_names, index=feature_names)
    
    for row in jaccobian:
        if(len(row)>=1):
            jaccobian_matrix.loc[feature_names[counter]] = row[0]
        counter+=1
    
    jaccobian_matrix.to_csv(result_dir+"Sign of Jaccobian for Iteration_"+str(vote_iteration)+".csv")
    # plt.figure(figsize=(5,5))
    # plt.title("Sign of Jaccobian for Iteration: "+str(vote_iteration))
    # sb.heatmap(jaccobian_matrix,annot=True, cmap=['Blue','Grey','Red'],center=0,cbar=False)
    # plt.savefig(orientation='landscape',fname=result_dir+"Cummulative_graph_"+str(vote_iteration)+".png",format='png',dpi=600,bbox_inches='tight')
    # plt.show()
    return jaccobian_matrix.copy()

def sign_of_jaccobian(block_length,seed_patterns,voting_type,taxon):
    '''
    Parameter:
    taxon = Taxon number for which processing is being done
    voting_type = Either the one suggested in paper i.e. 'max' or the custom built i.e. 'best' 
    
    Description:
    returns best path with max phi value and the proposed sign pattern after user specified number 
    of iterations
    '''
    seed = []
    pair_difference = pd.DataFrame()
    pair_difference = sample_pair_difference_unordered(get_samples_for_taxon(taxon,block_length))
    np_pd,factor_list = reduced_pair_difference(pair_difference)
    if(len(seed_patterns)>0):
        seed = seed_patterns[taxon][0].tolist()
        #print("Unwrapped Seed: "+str(seed)+" For Taxon: "+str(taxon)+" Type: "+str(type(seed)))
    
    added_block_length = block_length - len(seed)
    print("Added Block Length: "+str(added_block_length)+" For Taxon: "+str(taxon))
    
    global no_of_features
    selected_hyperplanes = []
    iterative_jacobian = []
    sign_distribution = []
    path_distribution = []
    phi_distribution = []
    max_phi = 0.0
    max_path = []
    solution_counter = 0
    
    if(len(pair_difference.columns)>=block_length-1):
        #print("Computing Jaccobian Matrix For Iteration Taxon: "+str(taxon)+" of Iteration: "+str(vote_iteration))
        for solution in it.product([1,-1,0],repeat=added_block_length):
            solution_counter+=1
            solution = tuple(seed) + solution
            #print("Seed: "+str(seed)+" Inference: "+str(solution)+" Taxo: "+str(taxon))
            solution = np.asarray(solution,dtype="int64").reshape(block_length)
            sign_d = []
            sign_d.append(solution) 
            if(len(sign_d[0])>taxon):
                #print("Sign Pattern: "+str(sign_d[0])+" Length: "+str(len(sign_d[0])) + " Taxon: "+str(taxon)+"\n")
                sign_d[0][taxon]=-1
            
            phi,sign_d = get_reduced_intersected_hyperplane_count(sign_d,pair_difference,taxon,False,np_pd,factor_list) 
            phi_j = phi/len(pair_difference.columns)
            phi_distribution.append(phi_j)
            sign_distribution.append(sign_d)
            path_distribution.append(["Matrix Based Sign Satisfaction"])
            #print("\nPhi for Taxon: "+str(taxon)+" of Block Length: "+str(block_length)+" For J: "+str(solution_counter)+" is:"+str(phi_j)) 
        
        ############################################## Write Max Phi Distribution ###################
        # if(block_length == no_of_features):
        #     write_max_phi_count(taxon,phi_distribution)
        ############################################################################################
        
        max_phi,max_path,iterative_jacobian = distribution_based_match(phi_distribution,sign_distribution,path_distribution,voting_type,pair_difference,taxon,np_pd,factor_list)
        return max_phi,max_path.copy(),iterative_jacobian.copy()
    else:
        print("The Combination for the Samples for Taxon: "+str(taxon)+" is lower than N-1")
        temp_list = []
        zero_pattern = np.zeros(block_length,dtype=int)
        if (block_length > taxon):
            zero_pattern[taxon] = -1
        temp_list.append(zero_pattern)
        return -2,["The Combination for the Samples for Taxon: "+str(taxon)+" is lower than N-1"],temp_list
        
def write_parameters(result_dir):
    '''
    Parameter:
    result_dir = directory where results are to be saved
    
    Description:
    Writes all the used parametes for a particular run in a csv file named 'parameters.csv'
    '''
    global block_indxes
    global filename
    global start_time
    global discard_threhold
    global voting_type
	
    parameters = pd.DataFrame(columns=["Parameter Name","Value"])
    parameters.loc[len(parameters)] = ["File Processed",filename]
    parameters.loc[len(parameters)] = ["Block Indexes",block_indxes]
    parameters.loc[len(parameters)] = ["Selection Method:",voting_type]
    parameters.loc[len(parameters)] = ["Discard Threshold:",discard_threhold]
    parameters.loc[len(parameters)] = ["Is Iterative",False]
    parameters.loc[len(parameters)] = ["Start Time",start_time]
    if((datetime.now() - start_time).total_seconds()>2):
        parameters.loc[len(parameters)] = ["End Time",datetime.now()]
    else:
        parameters.loc[len(parameters)] = ["End Time","Running"]
    parameters.to_csv(result_dir+"Parameters.csv",index=False)

######################################################## MAIN FUNCTION ###############################################
# In[9]:
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file1', '-f1', default=False)
    parser.add_argument('--result', '-r', help='Result')
    args = parser.parse_args()

############################### Execution Parameters #######################################################################
    block_indxes = [10,20,30,40,50] #maximum number of brute force iteration
    #result_dir = 'result/test_bw/' #result directory
    discard_threhold = 5.0 #set hyperplane values to 0 if less than +- discard_threshold
    voting_type = 'max'
############################################################################################################################

    #filename = "datafiles/MaizeRoots/Maze_roots_Consolidated_data.csv"
    filename = args.file1
    result_dir = args.result
    selected_otu_abundance_df = pd.read_csv(filename)
    selected_otu_abundance_df = drop_zero_indexes(selected_otu_abundance_df)
    selected_otu_abundance_df.describe()
    
############################## Derived Parameters #########################################################################
    no_of_features = len(selected_otu_abundance_df.columns)
    cores = multiprocessing.cpu_count() if(no_of_features >= multiprocessing.cpu_count()) else no_of_features
    cummulative_jaccobian = pd.DataFrame(columns=np.arange(no_of_features*no_of_features))   
    feature_names = selected_otu_abundance_df.columns
    start_time = datetime.now()
    pool = ThreadPool(cores)
    block_length = 0
    sign_solution_paths = []
    confidence_value = []
    jacobian_i = []
    
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    
    if(os.path.isdir(result_dir)==False):
        print("Can't Find the Specified result direcotry")
        sys.exit("Exiting Program")    
    print("Cores Available: "+str(multiprocessing.cpu_count()))
    print("Using Cores: "+str(cores))
###########################################################################################################################

############################## Main Program Execution #####################################################################
    write_parameters(result_dir)

    for block_size in compute_block_lengths(block_indxes,no_of_features) :
        block_length+=block_size
        taxon = [n for n in range(0,no_of_features)]
    
        taxon_result = pool.map(partial(sign_of_jaccobian,block_length,jacobian_i,voting_type),taxon)
    
        if(block_length==block_size):
            for i in range(0,len(taxon_result)):
                confidence_value.append(taxon_result[i][0])
                sign_solution_paths.append(taxon_result[i][1])
                jacobian_i.append(taxon_result[i][2])  
        else:
            for i in range(0,len(taxon_result)):
                confidence_value[i] = taxon_result[i][0]
                sign_solution_paths[i] = taxon_result[i][1]
                jacobian_i[i] = taxon_result[i][2]

    cummulative_jaccobian.loc[len(cummulative_jaccobian)] = (create_interaction_network(jacobian_i,0,result_dir)).values.flatten()
    sensitivity_of_jaccobian(cummulative_jaccobian,feature_names,confidence_value,result_dir)
    write_parameters(result_dir)
    pool.terminate()


