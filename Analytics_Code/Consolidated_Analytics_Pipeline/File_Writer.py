import pandas as pd 
import os
from os import path

class file_writer:

    def __init__(self,df):

        self.df = df

    def write_csv(self,location):

        self.df.to_csv(location,index=False)

        
