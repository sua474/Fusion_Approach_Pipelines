import pandas as pd
import os
from os.path import isfile, join 
from os import listdir
import copy as c

def compute_aggregate_output(traversal_path):

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

def compute_aggregate_ground_truth(traversal_path):

    aggregate_dataset = pd.DataFrame()
    file_count = 0
    for root, subFolders, files in os.walk(traversal_path):
        for f in files:
            file_count+=1
            file_path = '{}/{}'.format(root,f)
            print(file_path)
            data = pd.read_csv(file_path)
            if(aggregate_dataset.empty):
                aggregate_dataset = c.copy(data)
            else:
                for i in range(0,data.shape[0]):
                    for j in range(0,data.shape[1]):
                        aggregate_dataset.iloc[i,j] = aggregate_dataset.iloc[i,j] + data.iloc[i,j]

    print('Total Files Processed: {}'.format(file_count))
    return aggregate_dataset/file_count

if __name__ == '__main__':

    output_file = "I:/Research_Technician/Development/Fusion_Approach_Pipelines/Results/Aggregated_Files/Ground_Truth_100T_10IT.csv"
    aggregated_dataset = compute_aggregate_ground_truth('I:/Research_Technician/Dataset/Chiquet/100_Taxa/Ground_Truth_10/')
    aggregated_dataset.to_csv(output_file,index=False)