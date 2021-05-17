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
import copy

####################################### For Xiao VS Ground Truth Match################################
######################################################################################################
def get_exact_match_distribution(ref,indices,sign_distribution):
    
    exact_match_distribution = []

    for i in range(0,len(indices)):
        correct=0
        for j in range(0,len(ref)):
            if(ref[j] == sign_distribution[indices[i]][0][j]):
                correct+=1
        exact_match_distribution.append((correct/len(ref))*100)

    return copy.deepcopy(exact_match_distribution)

def get_without_zero_match_distribution(ref,indices,sign_distribution):
    
    without_zero_match_distribution = []
    ref_length = len(ref) - list(ref).count(0)

    for i in range(0,len(indices)):
        correct=0
        for j in range(0,len(ref)):
            if(int(ref[j])!=0 and int(sign_distribution[indices[i]][0][j])!=0 and ref[j] == sign_distribution[indices[i]][0][j]):
                correct+=1
        without_zero_match_distribution.append((correct/ref_length)*100)

    return copy.deepcopy(without_zero_match_distribution)


def get_always_right_distribution(ref,indices,sign_distribution):
    
    always_right_match_distribution = []

    for i in range(0,len(indices)):
        correct=0
        for j in range(0,len(ref)):
            if(int(ref[j])==0 or ref[j] == sign_distribution[indices[i]][0][j]):
                correct+=1
        always_right_match_distribution.append((correct/len(ref))*100)

    return copy.deepcopy(always_right_match_distribution)

def write_distribution(taxon,accuracy_distribution,file_operator,distribution_name):

    counts = []
    unique_values = []
    
    if(type(accuracy_distribution) is list):
        unique_values = list(set(accuracy_distribution))
        unique_values.sort() 
        for i in range(0,len(unique_values)):
            counts.append(accuracy_distribution.count(unique_values[i]))
    else:
        unique_values.append(accuracy_distribution)
        counts.append(1)
            
    with open(result_dir+"Max_Phi_Distribution_"+str(taxon)+".csv",file_operator, newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #wr.writerow([distribution_name]*len(unique_values))
        wr.writerow(unique_values)
        wr.writerow(counts)

def plot_accuracy_distribution(taxon,phi_distribution,sign_distribution):
    global max_phi_cuttoff
    global ground_truth
    global result_dir
    
    indices = [index for index, element in enumerate(phi_distribution) if element >= max_phi_cuttoff ]
    ref = np.sign(ground_truth.iloc[taxon,:].tolist())
    ref = copy.deepcopy([int(x) for x in ref])

    exact_match_distribution = get_exact_match_distribution(ref,indices,sign_distribution) 
    without_zero_match_distribution = get_without_zero_match_distribution(ref,indices,sign_distribution)
    always_right_match_distribution = get_always_right_distribution(ref,indices,sign_distribution)

    write_distribution(taxon,exact_match_distribution,'w',"Exact_Match")
    write_distribution(taxon,without_zero_match_distribution,'a',"Without_Zero_Match")
    write_distribution(taxon,always_right_match_distribution,'a',"Always_Right_Match")

################################## For Correlation VS Xiao Distribution Match################################
#############################################################################################################

def align_ground_truth_with_correlation(gt_columns,ref_columns,ref_data,corr_threshold):

    updated_reference = []
    global leverage_factor
    
    ##### Make values lower than correlation threhold zero ########
    ##### Adding the "Dont Care" condition using the leverage factor #########
    for i in range(0,len(ref_data)):
        if(ref_data[i]<(corr_threshold-leverage_factor)):
            ref_data[i] = 0.0
        elif(ref_data[i]>(corr_threshold+leverage_factor)):
            ref_data[i] = 1.0
    
    for x in gt_columns:
        found=False
        for i in range(0,len(ref_columns)):
            if(x == ref_columns[i]):
                updated_reference.append(ref_data[i])
                found=True
                break
        if(found==False):
            updated_reference.append(0.0)

    return copy.deepcopy(updated_reference)

def get_AND_metrices(ref,indices,sign_distribution):
    
    AND_metrices = []

    for i in range(0,len(indices)):
        correct=0
        matrix = []
        for j in range(0,len(ref)):
            if(int(ref[j])==1 and (int(sign_distribution[indices[i]][0][j]) == 1 or int(sign_distribution[indices[i]][0][j])== -1) ):
                matrix.append(1)
            else:
                matrix.append(0)
        AND_metrices.append(matrix)
    
    return copy.deepcopy(AND_metrices)

def get_exact_match_AND_matix_distribution(AND_metrices,gt_for_taxon):

    overlap_distribution = []
    gt_for_taxon = [1 if x < 0 else x for x in gt_for_taxon]

    for matrix in AND_metrices:
        correct=0
        for i in range(0,len(gt_for_taxon)):
            if(int(matrix[i]) == int(gt_for_taxon[i])):
                correct+=1
        overlap_distribution.append(correct/len(gt_for_taxon)*100)

    return copy.deepcopy(overlap_distribution)

def get_without_zero_AND_matix_distribution(AND_metrices,gt_for_taxon):
        
    overlap_distribution = []
    gt_for_taxon = [1 if x < 0 else x for x in gt_for_taxon]

    for matrix in AND_metrices:
        correct=0
        for i in range(0,len(gt_for_taxon)):
            if(int(matrix[i])!= 0 and int(gt_for_taxon[i])!=0 and (int(matrix[i]) == int(gt_for_taxon[i])) ):
                correct+=1
        overlap_distribution.append(correct/(len(gt_for_taxon))*100)

    return copy.deepcopy(overlap_distribution)

def get_always_right_AND_matix_distribution(AND_metrices,gt_for_taxon):
    
    overlap_distribution = []
    gt_for_taxon = [1 if x < 0 else x for x in gt_for_taxon]

    for matrix in AND_metrices:
        correct=0
        for i in range(0,len(gt_for_taxon)):
            if(int(gt_for_taxon[i]) ==0 or int(matrix[i]) == int(gt_for_taxon[i])):
                correct+=1
        overlap_distribution.append(correct/(len(gt_for_taxon))*100)

    return copy.deepcopy(overlap_distribution)

def write_lists_to_file(list_of_list,filename):
    
    global result_dir

    with open(result_dir+filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(list_of_list)


def correlation_and_Xiao_overlap_distribution(taxon,phi_distribution,sign_distribution):
    global max_phi_cuttoff
    global ground_truth
    global result_dir
    global corr_dirs
    
    indices = [index for index, element in enumerate(phi_distribution) if element >= max_phi_cuttoff ]
    corr_dirs = correlation_paths.split(";")
    gt_columns = ground_truth.columns.tolist()
    gt_for_taxon = np.sign(ground_truth.iloc[taxon,:].tolist())
    gt_for_taxon = copy.deepcopy([int(x) for x in gt_for_taxon])

    for i in range(0,len(corr_dirs)):
        corr_data = pd.read_csv(corr_dirs[i].replace('\r','')+"/Metric Network.csv")
        corr_data["Unnamed: 0"] = corr_data["Unnamed: 0"].map(str)
        corr_data.set_index("Unnamed: 0",inplace=True)
        ref_columns = corr_data.columns.tolist()
        corr_threshold = float(corr_dirs[i].split('/')[-1].split('_')[-2]) 

        if(gt_columns[taxon] in ref_columns):
            corr_ref = corr_data.loc[gt_columns[taxon]].tolist()
        else:
            corr_ref = [0]*len(ref_columns)

        updated_ref =  np.sign(align_ground_truth_with_correlation(gt_columns,ref_columns,corr_ref,corr_threshold))
        weak_edge_thsld = corr_dirs[i].split("/")[-1].split("_")[-2]
        AND_metrices = get_AND_metrices(updated_ref,indices,sign_distribution) 
        
        EM_AND_distribution = get_exact_match_AND_matix_distribution(AND_metrices,gt_for_taxon)
        WZ_AND_distribution = get_without_zero_AND_matix_distribution(AND_metrices,gt_for_taxon)
        AR_AND_distribution = get_always_right_AND_matix_distribution(AND_metrices,gt_for_taxon)

        write_distribution(taxon, EM_AND_distribution,'a',"EM_AND_Distribution_"+str(weak_edge_thsld))
        write_distribution(taxon, WZ_AND_distribution,'a',"WZ_AND_Distribution_"+str(weak_edge_thsld))
        write_distribution(taxon, AR_AND_distribution,'a',"AR_AND_Distribution_"+str(weak_edge_thsld))


        ############################################## Printing Correlation Distribution #########################
        corr_pattern_print_info = []
        print_sign_distribution = []
        corr_pattern_print_info.append(ref_columns)
        corr_pattern_print_info.append(corr_ref)
        corr_pattern_print_info.append(gt_columns)
        corr_pattern_print_info.append(updated_ref)

        for z in range(0,len(indices)):
            print_sign_distribution.append(sign_distribution[indices[z]][0])

        write_lists_to_file(print_sign_distribution,"Top_Sign_Patterns_of_Taxon_"+str(weak_edge_thsld)+"_"+str(taxon)+".csv")
        write_lists_to_file(AND_metrices,"AND_Matrices_of_Taxon_"+str(weak_edge_thsld)+"_"+str(taxon)+".csv")
        write_lists_to_file(corr_pattern_print_info,"Correlation_GT_of_Taxon_"+str(weak_edge_thsld)+"_"+str(taxon)+".csv")
            

####################################### For Correlation VS Ground Truth Match################################
#############################################################################################################

def get_exact_match_with_correlation(corr_ref_for_taxon,gt_for_taxon):

    correct = 0
    for i in range(0,len(corr_ref_for_taxon)):
        if(corr_ref_for_taxon[i] == gt_for_taxon[i]):
            correct+=1
    
    return (correct/len(corr_ref_for_taxon)*100)

def get_always_right_with_correlation(corr_ref_for_taxon,gt_for_taxon):

    correct = 0
    for i in range(0,len(gt_for_taxon)):
        if (gt_for_taxon[i]==0) or (corr_ref_for_taxon[i] == gt_for_taxon[i]) :
            correct+=1
    
    return (correct/len(corr_ref_for_taxon)*100)

def get_without_zero_with_correlation(corr_ref_for_taxon,gt_for_taxon):
    
    correct = 0
    for i in range(0,len(gt_for_taxon)):
        if gt_for_taxon[i] == 0 and corr_ref_for_taxon[i]==0 and (corr_ref_for_taxon[i] == gt_for_taxon[i]):
            correct+=1

    return (correct/(len(corr_ref_for_taxon))*100)

def correlation_accuracy_distribution(taxon,phi_distribution,sign_distribution):

    global ground_truth
    global result_dir
    global corr_dirs
    
    corr_dirs = correlation_paths.split(";")
    gt_columns = ground_truth.columns.tolist()
    gt_for_taxon = np.sign(ground_truth.iloc[taxon,:].tolist())
    gt_for_taxon = copy.deepcopy([int(x) for x in gt_for_taxon])

    for i in range(0,len(corr_dirs)):
        corr_data = pd.read_csv(corr_dirs[i].replace('\r','')+"/Metric Network.csv")
        corr_data["Unnamed: 0"] = corr_data["Unnamed: 0"].map(str)
        corr_data.set_index("Unnamed: 0",inplace=True)
        ref_columns = corr_data.columns.tolist()
        corr_threshold = float(corr_dirs[i].split('/')[-1].split('_')[-2]) 

        if(gt_columns[taxon] in ref_columns):
            corr_ref = corr_data.loc[gt_columns[taxon]].tolist()
        else:
            corr_ref = [0]*len(ref_columns)

        corr_ref_for_taxon =  np.sign(align_ground_truth_with_correlation(gt_columns,ref_columns,corr_ref,corr_threshold))
        weak_edge_thsld = corr_dirs[i].split("/")[-1].split("_")[-2]
        
        EM_corr = get_exact_match_with_correlation(corr_ref_for_taxon,gt_for_taxon) 
        WZ_corr = get_without_zero_with_correlation(corr_ref_for_taxon,gt_for_taxon)
        AR_corr = get_always_right_with_correlation(corr_ref_for_taxon,gt_for_taxon)
        
        write_distribution(taxon, EM_corr,'a',"Correlation_Accurarcy_Distribution_EM_"+str(weak_edge_thsld))
        write_distribution(taxon, WZ_corr,'a',"Correlation_Accurarcy_Distribution_WZ_"+str(weak_edge_thsld))
        write_distribution(taxon, AR_corr,'a',"Correlation_Accurarcy_Distribution_AR_"+str(weak_edge_thsld))



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
    return copy.deepcopy(pair_difference)

def threshold_ground_truth(ground_truth):
    
    global strong_edge_threshold

    ground_truth[(ground_truth <= strong_edge_threshold) & (ground_truth >= -1*strong_edge_threshold)]=0

    return copy.deepcopy(ground_truth)

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
    return copy.deepcopy(df)


def construct_threshold(data):

    global cuttoff
    global upper_multipler
    global lower_multipler

    for i in range(0,len(data)):
        if(data[i]< 1.0):
            data[i] = 1.0
        elif(data[i] > cuttoff):
            data[i]*=upper_multipler
        else:
            data[i]*=lower_multipler
    
    return copy.deepcopy(data)

def apply_threshold(difference,threshold):

    for i in range(0,len(difference)):
        if(difference[i]<threshold[i]):
            difference[i] = 0.0
    
    return copy.deepcopy(difference)


def sample_pair_difference_with_thresholding(steady_state_samples):

    pair_difference = pd.DataFrame()
    for i in range(0,len(steady_state_samples)):
        #print("i: "+str(i)+" Out of: "+str(len(steady_state_samples)))
        for j in range(i+1,len(steady_state_samples)):
            difference = steady_state_samples.iloc[i,:] - steady_state_samples.iloc[j,:]
            threshold = construct_threshold((steady_state_samples.iloc[j,:] + steady_state_samples.iloc[i,:])/2)
            thresholded_column = apply_threshold(difference.copy(),threshold.copy())
            pair_difference = pd.concat([pair_difference,thresholded_column],axis=1,sort=True)
        #pair_difference.to_csv("PD"+str(i)+".csv")

    return copy.deepcopy(pair_difference)
	
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
    phi,test_sp = get_reduced_intersected_hyperplane_count_Xiao(predicted_sign_pattern,pair_difference,taxon,False,np_pd,factor_list)         
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


def reduced_pair_difference(pair_difference):
    '''
    Parameter:
    pair_difference = pair difference dataframe
    
    Description:
    Reduces the number of common columns of pair differences
    '''
    if(pair_difference.empty):
        return [-2],[-2]
        
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

def get_reduced_intersected_hyperplane_count_Xiao(sign_d,pair_difference,taxon,perform_perturbation,np_pd,factor_list):
    '''
    Parameter:
    sign_d = Heuristic Based Sign Patter 
    pair_difference = Pair Difference Matrix
    
    Description:
    Performs GPU accelerated, matrix based graph creation and path traversal. Return the total number of
    intersected hyperplanes for a particular sign_d and difference matrix
    '''
    
    #print("Received Sign_d: "+str(sign_d)) 
    sign_d = np.asarray(sign_d,dtype='float64')
    sign_d = sign_d.reshape(len(pair_difference),1)
    #np_pd,factor_list = reduced_pair_difference(pair_difference)
    
    cummulative_count = 0.0
    sign_graph = np.multiply(sign_d, np_pd)
    #sign_graph = np.sign(sign_matrix)

    for i in range(0,sign_graph.shape[1]):
        cummulative_count+=is_sign_satisfied_matrix(sign_graph[:,i])*factor_list[i]

    best_sign_d = sign_d.reshape(len(pair_difference))
    best_sign_d = np.asarray(best_sign_d.tolist(),dtype="int64")
    temp_sign_d = []
    temp_sign_d.append(best_sign_d)
    #print("Returned Sign_d: "+str(temp_sign_d))

    return cummulative_count,copy.deepcopy(temp_sign_d)


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


def calculate_similarity(sign_pattern,pair_difference_column):

    similarity_count = 0
    for i in range(sign_pattern.size):
        if(sign_pattern[i]==pair_difference_column[i]):
            similarity_count+=1
    
    return similarity_count/sign_pattern.size


def check_if_equal(list_1, list_2):
    """ Check if both the lists are of same length and if yes then compare
    sorted versions of both the list to check if both of them are equal
    i.e. contain similar elements with same frequency. """
    for i in range(0,len(list_1)):
        if(list_1[i]!=list_2[i]):
            return False

    return True


def align_outputs(gt,data):

    global result_dir

    gt['Unnamed: 0'] = gt.columns.tolist()
    gt.set_index('Unnamed: 0',inplace=True)
    gt = gt[data.columns.tolist()]
    gt = gt.reindex(data.columns.tolist())
    gt.to_csv(result_dir+"GT_Updated.csv",index=False)

    return copy.deepcopy(gt)


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
    
    plt.figure()
    plt.title("Cummulative Phi Values Plot")
    plt.xticks(rotation=90)
    plt.plot(feature_names,commulative_confidence,marker='o')
    plt.savefig(orientation='landscape',fname=result_dir+"Phi_Plot.png",format='png',dpi=600,bbox_inches='tight')
    plt.show()
    
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
    plt.figure(figsize=(5,5))
    plt.title("Sign of Jaccobian for Iteration: "+str(vote_iteration))
    sb.heatmap(jaccobian_matrix,annot=True, cmap=['Blue','Grey','Red'],center=0,cbar=False)
    plt.savefig(orientation='landscape',fname=result_dir+"Cummulative_graph_"+str(vote_iteration)+".png",format='png',dpi=600,bbox_inches='tight')
    plt.show()
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
    #print("Taxon: "+str(taxon)+" Length: "+str(len(pair_difference))+" Columns: "+str(len(pair_difference.columns)))
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

    #####################For Max Accuracy Phi Distribution###########################
    global ground_truth
    ref = np.sign(ground_truth.iloc[taxon,:].tolist()) 
    ref_phi = 0.0
    #################################################################################
    
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
            
            phi,sign_d = get_reduced_intersected_hyperplane_count_Xiao(sign_d,pair_difference,taxon,False,np_pd,factor_list) 
            phi_j = phi/len(pair_difference.columns)

            #####################For Max Accuracy Phi Distribution###########################
            if(check_if_equal(copy.deepcopy(sign_d[0]),copy.deepcopy(ref))):
                ref_phi = phi_j
            #################################################################################

            phi_distribution.append(phi_j)
            sign_distribution.append(sign_d)
            path_distribution.append(["Matrix Based Sign Satisfaction"])
            #print("\nPhi for Taxon: "+str(taxon)+" of Block Length: "+str(block_length)+" For J: "+str(solution_counter)+" is:"+str(phi_j)) 
        
        ################################## For Max Phi Distribution wrt Accuracy######################
        if(block_length == no_of_features):
            plot_accuracy_distribution(taxon,copy.deepcopy(phi_distribution),copy.deepcopy(sign_distribution))
            correlation_and_Xiao_overlap_distribution(taxon,copy.deepcopy(phi_distribution),copy.deepcopy(sign_distribution))
            correlation_accuracy_distribution(taxon,copy.deepcopy(phi_distribution),copy.deepcopy(sign_distribution))
        #############################################################################################

        max_phi,max_path,iterative_jacobian = distribution_based_match(phi_distribution,sign_distribution,path_distribution,voting_type,pair_difference,taxon,np_pd,factor_list)
        return max_phi,copy.deepcopy(max_path),copy.deepcopy(iterative_jacobian),ref_phi
    else:
        print("The Combination for the Samples for Taxon: "+str(taxon)+" is lower than N-1")
        temp_list = []
        zero_pattern = np.zeros(block_length,dtype=int)
        if (block_length > taxon):
            zero_pattern[taxon] = -1
        temp_list.append(zero_pattern)
        return -2,["The Combination for the Samples for Taxon: "+str(taxon)+" is lower than N-1"],temp_list,0


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
    global leverage_factor
    global strong_edge_threshold
	
    parameters = pd.DataFrame(columns=["Parameter Name","Value"])
    parameters.loc[len(parameters)] = ["File Processed",filename]
    parameters.loc[len(parameters)] = ["Block Indexes",block_indxes]
    parameters.loc[len(parameters)] = ["Selection Method:",voting_type]
    parameters.loc[len(parameters)] = ["Discard Threshold:",discard_threhold]
    parameters.loc[len(parameters)] = ["Is Iterative:",False]
    parameters.loc[len(parameters)] = ["Strong Edge Threshold:",strong_edge_threshold]
    parameters.loc[len(parameters)] = ["Dont Care Leverge Factor:",leverage_factor]

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
    parser.add_argument('--file2', '-gt', help='Ground Truth')
    parser.add_argument('--result', '-r', help='Result')
    parser.add_argument('--corr_paths', '-cp', help='Correlation Result Path',default=False)
    args = parser.parse_args()

############################### Execution Parameters #######################################################################
    block_indxes = [10] #maximum number of brute force iteration
    #result_dir = 'result/test_bw/' #result directory

    ##### For Match of Xiao's method against the correlation###### 
    leverage_factor = 0.05 #Correlation leverage factor to extrapolate the don't care region

    ####### Xiao's method thresholds ##########
    strong_edge_threshold = 0.05 # To discard edges from ground truth which are below this value (in both +ve and -ve direction) 
    discard_threhold = 0.005 #set hyperplane values to 0 if less than +- discard_threshold
    voting_type = 'max' # Voting Type

    ######## Other Parameters ####################     
    max_phi_cuttoff = 0.79
    cuttoff = 4.0
    upper_multipler = 2.0
    lower_multipler = 5.0
############################################################################################################################

    filename = args.file1
    ground_truth = pd.read_csv(args.file2)
    result_dir = args.result
    correlation_paths = args.corr_paths

    selected_otu_abundance_df = pd.read_csv(filename)
    selected_otu_abundance_df = drop_zero_indexes(selected_otu_abundance_df)
    selected_otu_abundance_df.describe()
    
############################## Derived Parameters #########################################################################
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    ground_truth = threshold_ground_truth(align_outputs(copy.deepcopy(ground_truth),copy.deepcopy(selected_otu_abundance_df)))
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
    
    ############################ FOR Ground Truth Phi################
    GT_phi = []
    ###############################################################
    
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
                GT_phi.append([i,taxon_result[i][0],taxon_result[i][3],(taxon_result[i][0]-taxon_result[i][3])])
        else:
            for i in range(0,len(taxon_result)):
                confidence_value[i] = taxon_result[i][0]
                sign_solution_paths[i] = taxon_result[i][1]
                jacobian_i[i] = taxon_result[i][2]
                GT_phi[i] = ([i,taxon_result[i][0],taxon_result[i][3],(taxon_result[i][0]-taxon_result[i][3])])

    with open(result_dir+"/Phi_Comparison_of_GT.csv", 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(["Taxon","Max-Phi","Ground_Truth-Phi","Phi-Difference"])
        for i in range(0,len(taxon_result)):
            wr.writerow(GT_phi[i])

    cummulative_jaccobian.loc[len(cummulative_jaccobian)] = (create_interaction_network(jacobian_i,0,result_dir)).values.flatten()
    sensitivity_of_jaccobian(cummulative_jaccobian,feature_names,confidence_value,result_dir)
    write_parameters(result_dir)
    pool.terminate()
