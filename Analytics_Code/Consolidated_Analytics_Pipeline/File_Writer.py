import pandas as pd 
import os
from os import path

class file_writer:

    def write_csv(self,location,df):
        
        directory = '/'.join(location.split('/')[0:-1])
        if(not os.path.exists(directory)):
            os.mkdir(directory)
        df.to_csv(location,index=False)
        print('Output has been written in at: {}'.format(directory))
        

        
