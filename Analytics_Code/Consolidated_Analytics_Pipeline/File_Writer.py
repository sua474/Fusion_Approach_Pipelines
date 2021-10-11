import pandas as pd 
import os
from os import path

class file_writer:

    def __init__(self,df):

        self.df = df

    def write_csv(self,location):
        
        directory = '/'.join(location.split('/')[0:-1])
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        self.df.to_csv(location,index=False)
        print('Output has been written in at: {}'.format(directory))
        

        
