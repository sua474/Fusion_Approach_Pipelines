import pandas as pd 
import os
from os import path

class file_writer:

    def write_csv(self,location,df):
        '''
        Writes a dataframe to the specified location in the .csv format
        '''
        directory = os.path.dirname(location)
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        df.to_csv(location,index=False)
        print('Output has been written in at: {}'.format(directory))
        

        
