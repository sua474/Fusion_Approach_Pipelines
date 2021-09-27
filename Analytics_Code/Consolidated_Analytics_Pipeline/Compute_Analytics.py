import pandas as pd 
import os
from File_Reader import file_reader
from Perform_Matching import perform_matching

class compute_analytics:

    def __init__ (self,algorithms,base_folder,output_filename,ground_truth_path):

        self.algorithms = algorithms.copy()
        self.base_folder = base_folder
        self.output_filename = output_filename.copy()
        self.data_files = dict.fromkeys(algorithms)
        self.ground_truth_path = ground_truth_path
        self.ground_truth = dict()
        self.analytics_df = pd.DataFrame(columns = algorithms+['File'])
        
        for algorithm in self.algorithms:
            self.data_files[algorithm] = list()

    def load_input_file_objects(self):

        for algorithm in self.algorithms:
            location = self.base_folder +'/'+ algorithm
            for root, directories, files in os.walk(location):
                for file in files:
                    if(file in self.output_filename):
                        self.data_files[algorithm].append(file_reader(root,file,algorithm))
    
    def load_ground_truth_objects(self):
        
        for root, directories, files in os.walk(self.ground_truth_path):
            for file in files:
                self.ground_truth[file] = file_reader(root,file,'Ground_Truth')
    
    def validate_payload(self,pay_load):
        
        ref_obj = pay_load.pop()
        reference = ref_obj.file_location.split('/')[-1].split('_')[-1]
        
        for obj in pay_load:
            index = obj.file_location.split('/')[-1].split('_')[-1]
            if(index!=reference):
                return False
        
        return reference
            
    def get_analytics(self):
        
        matcher = perform_matching()

        for i in range(0, len(self.ground_truth)):
            pay_load = []
            for algorithm in self.algorithms:
                pay_load.append(self.data_files[algorithm].pop())
            
            index = self.validate_payload(pay_load.copy())
            ground_truth_file_name = 'Ground_Truth_{}.csv'.format(index)
            print('Processing {}'.format(ground_truth_file_name))
            
            matches = matcher.compute_overlap(pay_load,self.ground_truth[ground_truth_file_name])
            matches['File'] = int(index)
            print(matches)
            self.analytics_df.loc[len(self.analytics_df)] = list(matches.values())
                
        return self.analytics_df
    
    def check_validity(self):
        
        for algorithm in self.data_files.keys():
            print('Algorithm {} Total Files: {}'.format( algorithm, len(self.data_files[algorithm])))
        
        print('Ground Truth Total Files: {}'.format(len(self.ground_truth)))
        
        for algorithm in self.data_files.keys():
            print(algorithm)
            print(self.data_files[algorithm][0].file_location)
            
        for x in self.ground_truth.keys():
            print(x)
          
    
if __name__ == '__main__':
    
    base_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/Taxa_10/IT_10"
    algorithms = ['Spiec_Easi','Xiao','Ma_Paper','Correlation','Spring']
    output_filename = set(['Adjacency_Matrix.csv','Sign of Jaccobian for Iteration_0.csv','Metric Network.csv'])
    ground_truth_path = "/u2/sua474/Dataset/Chiquet/Taxa_10/Ground_Truth_10/"

    analytics = compute_analytics(algorithms,base_path,output_filename,ground_truth_path)
    analytics.load_input_file_objects()
    analytics.load_ground_truth_objects()
    df = analytics.get_analytics()
    df.to_csv("Output.csv",index=False)
    #analytics.check_validity()

    