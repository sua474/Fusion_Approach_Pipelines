import pandas as pd
import sys
import os

class file_reader:

    def __init__(self,file_location,file_name,algorithm_name):
        '''
        Creates a file reader object with the location of file along with its name and the name of algorithm
        they it belongs to
        '''
        self.file_location = file_location
        self.file_name =  file_name
        self.algorithm_name = algorithm_name
    
    def normalize_df(self,df):
        '''
        Simple normalization of the input file data frame so that each file has values approximated to the 
        same range
        '''
        df[df>0] = 1
        df[df<=0] = 0
        return df.copy()

    def get_file(self):
        '''
        Reads the file using the location and name as set at the time of object initialization and then 
        performs the normalization and return it to the calling class (Perform_Matching).
        '''
        loading_path = os.path.join(self.file_location, self.file_name)
        if(os.path.isfile(loading_path)):
            df = pd.read_csv(loading_path)
            df = self.normalize_df(df.copy())
            return df
        else:
            print('File Path: {} Do not Exist'.format(loading_path))
            return False

    def get_simulated_dataset_name(self):
        
        return self.file_location.split('/')[-1]

