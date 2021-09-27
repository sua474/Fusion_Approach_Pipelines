import pandas as pd 
import os
from File_Reader import file_reader

class compute_analytics:

    def __init__ (self,algorithms,base_folder,output_filename):

        self.algorithms = algorithms.copy()
        self.base_folder = base_folder
        self.output_filename = output_filename.copy()
        self.data_files = dict.fromkeys(algorithms)
        for algorithm in self.algorithms:
            self.data_files[algorithm] = list()

    def initialize_file_objects(self):

        for algorithm in self.algorithms:
            location = self.base_folder +'/'+ algorithm
            for root, directories, files in os.walk(location):
                for file in files:
                    file_location = root+'/'+file
                    if(file in self.output_filename):
                        self.data_files[algorithm].append(file_reader(file_location,file,algorithm))
    
    def check_validity(self):
        
        for algorithm in self.algorithms:
            print('Algorithm {} Total Files: {}'.format( algorithm, len(self.data_files[algorithm])))
        
        for algorithm in self.algorithms:
            print(algorithm)
            print(self.data_files[algorithm][0].file_location)
          
    
if __name__ == '__main__':
    
    base_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/Taxa_10/IT_10"
    algorithms = ['Spiec_Easi','Xiao','Ma_Paper','Correlation','Spring']
    output_filename = set(['Adjacency_Matrix.csv','Sign of Jaccobian for Iteration_0.csv','Metric Network.csv'])
    
    analytics = compute_analytics(algorithms,base_path,output_filename)
    analytics.initialize_file_objects()
    analytics.check_validity()

    