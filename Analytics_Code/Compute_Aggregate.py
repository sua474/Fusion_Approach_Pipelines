import pandas as pd
import os
from os.path import isfile, join 
from os import listdir
import copy as c

def compute_aggregate(traversal_path):

    aggregate_dataset = pd.DataFrame()
    folder_count = 0
    for root, subFolders, files in os.walk(traversal_path):
        for folder in subFolders:
            folder_count+=1
            mypath = "{}/{}".format(root,folder)
            files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
            file_path = "{}/{}/{}".format(root,folder,files[0])
            data = pd.read_csv(file_path)
            if(aggregate_dataset.empty):
                aggregate_dataset = c.copy(data)
            else:
                for i in range(0,data.shape[0]):
                    for j in range(0,data.shape[1]):
                        aggregate_dataset.iloc[i,j] = aggregate_dataset.iloc[i,j] + data.iloc[i,j]


    return aggregate_dataset/folder_count

if __name__ == '__main__':

    output_folder = "Aggregated_Files/Ma_10.csv"
    aggregated_dataset = compute_aggregate('/Users/sua474/Desktop/Work/Development/Fusion_Approach_Pipelines/Results/Ma_10/')
    aggregated_dataset.to_csv(output_folder,index=False)