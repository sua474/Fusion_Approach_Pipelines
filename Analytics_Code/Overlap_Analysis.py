import pandas as pd
import numpy as np
import os
from os import path
import math

def compare_datasets(first_dataset,second_dataset):
    total_elements = len(first_dataset.columns.tolist())*len(first_dataset)
    match_count = 0
    for i in range(0,len(first_dataset.columns.tolist())):
        for j in range(0,len(first_dataset)):
            if(math.ceil(first_dataset.iloc[j,i])!=0 and math.ceil(second_dataset.iloc[j,i])!=0 ):
                match_count+=1
    
    return (match_count/total_elements)

def process_datasets(first_dataset_location,second_dataset_location):
    match_percentage = []

    for root, dirs, files in os.walk(first_dataset_location):
        for f in files:
            first_dir = first_dataset_location.split('/')[1]
            second_dir = second_dataset_location.split('/')[1]
            second_root = root.replace(first_dir,second_dir)
            
            if (path.exists(root+'/'+f) and path.exists(second_root+'/'+f)):
                first_dataset = pd.read_csv(root+'/'+f)
                second_dataset = pd.read_csv(second_root+'/'+f)
                match_percentage.append(compare_datasets(first_dataset,second_dataset))
    
    print("Total Enteries: "+str(len(match_percentage)))
    print("Mean Overlap Percentage: "+str(np.mean(match_percentage)))
    print("Max Overlap Percentage: "+str(np.max(match_percentage)))
    print("Min Overlap Percentage: "+str(np.min(match_percentage)))

if __name__ == "__main__":
    process_datasets("Run_Results/Spiec_Easi","Run_Results/Ma_paper")
