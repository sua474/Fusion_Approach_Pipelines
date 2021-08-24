import pandas as pd
import numpy as np
from pandas.io.parsers import read_csv
import seaborn as sb
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join 
from matplotlib_venn import venn3,venn2
import copy as cp

def read_file(file_path):
    '''
    Read the datafile and performs the value thresholding
    '''

    df =  pd.read_csv(file_path)
    df[df<0] = 1
    df[df>0] = 1
    df = df.astype('int32')
    return cp.deepcopy(df)


def perform_validation(df_1, df_2):

    '''
    Performs validation check on the dimensios of the dataframe
    '''

    if ( df_1.shape[0] == df_2.shape[0] and df_1.shape[1] == df_2.shape[1] ) :
        return True
    else:
        return False


def perform_comparision(file_path_1,file_path_2):

    df_1 = read_file(file_path_1)
    df_2 = read_file(file_path_2)

    if(perform_validation(df_1,df_2)):
        
        return True,{}
    else:
        return False,{}
    

def traverse_directory(traversal_path,algo_names):

    for root, subFolders, files in os.walk(traversal_path):
        for folder in subFolders:
            mypath = "{}/{}".format(root,folder)
            onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

            for file in onlyfiles:
                for i in range(0,len(algo_names)):
                    for j in range(i+1,len(algo_names)):

                        folder_root_1 = root.replace(algo_names[0],algo_names[i])
                        folder_root_2 = root.replace(algo_names[0],algo_names[j])

                        file_path_1 = "{}/{}/{}".format(folder_root_1,folder,file)
                        file_path_2 = "{}/{}/{}".format(folder_root_2,folder,file)
                        
                        perform_comparision(file_path_1,file_path_2)
                        exit()
        
                        
if __name__== "__main__":

    ################################## Execution Parameters ##################################
    output_folder = "/Users/sua474/Desktop/Work/Development/Fusion_Approach_Pipelines/Output" #Path of the output folder with no trailing slash
    algo_names = ["Spiec_Easi_10",'Spring_10','Ma_10']                                        # Name of the Algorithms to be compared

    ################################## Derived Parameters ####################################
    traversal_path = "{}/{}".format(output_folder,algo_names[0])

    ################################## Function Calls ########################################
    traverse_directory(traversal_path,algo_names)
