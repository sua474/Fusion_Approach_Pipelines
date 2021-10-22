import pandas as pd
import sys
import os

class file_reader:

    def __init__(self,file_location,file_name,algorithm_name):

        self.file_location = file_location
        self.file_name =  file_name
        self.algorithm_name = algorithm_name
    
    def normalize_df(self,df):

        df[df>0] = 1
        df[df<=0] = 0
        return df.copy()

    def get_file(self):
        
        loading_path = self.file_location+'/'+self.file_name
        if(os.path.isfile(loading_path)):
            df = pd.read_csv(loading_path)
            df = self.normalize_df(df.copy())
            return df
        else:
            #print('File Path: {} Do not Exist'.format(loading_path))
            return False

    def get_simulated_dataset_name(self):
        
        return self.file_location.split('/')[-1]

