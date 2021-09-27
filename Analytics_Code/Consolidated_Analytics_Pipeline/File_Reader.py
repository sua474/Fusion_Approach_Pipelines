import pandas as pd

class file_reader:

    def __init__(self,file_location,file_name,algorithm_name):

        self.file_location = file_location
        self.file_name =  file_name
        self.algorithm_name = algorithm_name

    def get_file(self):
        return pd.read_csv(self.file_location+self.file_name)

    def get_simulated_dataset_name(self):
        return self.file_location.split('/')[-1]

