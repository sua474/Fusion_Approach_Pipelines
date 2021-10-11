import pandas as pd 
import os
import sys
from File_Reader import file_reader
from File_Writer import file_writer
from Perform_Matching import perform_matching
from Record_Keeper import record_keeper

class compute_analytics:

    def __init__ (self,algorithms,base_folder,output_filename,ground_truth_path):

        self.algorithms = algorithms.copy()
        self.base_folder = base_folder
        self.output_filename = output_filename.copy()
        self.data_files = dict.fromkeys(algorithms)
        self.ground_truth_path = ground_truth_path
        self.ground_truth = dict()
        self.analytics_df = pd.DataFrame(columns = algorithms+['Data_File_Index'])
        self.aggregated_output = []
        
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
                sys.exit("Files Do Not Belong to the Same Ecological Dataset")
        
        return reference
            
    def compute_analytics(self):
        
        matcher = perform_matching()

        for i in range(0, len(self.ground_truth)):
            pay_load = []
            for algorithm in self.algorithms:
                if(len(self.data_files[algorithm])>=1):
                    pay_load.append(self.data_files[algorithm].pop())
            
            index = self.validate_payload(pay_load.copy())
            ground_truth_file_name = 'Ground_Truth_{}.csv'.format(index)
            print('Processing {}'.format(ground_truth_file_name))
            
            matches = matcher.compute_overlap(pay_load,self.ground_truth[ground_truth_file_name])
            matches['Data_File_Index'] = int(index)
            self.analytics_df.loc[len(self.analytics_df)] = list(matches.values())
    
    def compute_aggregate(self):

        for algorithm in self.algorithms:
            self.aggregated_output.append(self.analytics_df[algorithm].mean()) # Computing average of each column of df
        
    
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
        
    algorithms = ['Spiec_Easi','Xiao','Ma_Paper','Correlation','Spring']
    output_filename = set(['Adjacency_Matrix.csv','Sign of Jaccobian for Iteration_0.csv','Metric Network.csv'])
    record_keeper = record_keeper(algorithms) # Object Initialization
    
    for taxa in ["Taxa_10"]:
        for internal_threshold in ["10","50"]:
            print("Executing Taxa {} and Internal Threshold {}".format(taxa,internal_threshold))
        ######### Input Parameters #####################
            base_path = "/u2/sua474/Fusion_Approach_Pipelines/Output/{}/IT_{}".format(taxa,internal_threshold)
            ground_truth_path = "/u2/sua474/Dataset/Chiquet/{}/Ground_Truth_{}/".format(taxa,internal_threshold)
            result_path = "/u2/sua474/Fusion_Approach_Analytics/Output/Baseline_Result/{}_IT_{}.csv".format(taxa,internal_threshold)
        ################################################
    
        ######## Object Initialization ####################
            analytics = compute_analytics(algorithms,base_path,output_filename,ground_truth_path)
        ######### Start Analytics #########################
            analytics.load_input_file_objects()
            analytics.load_ground_truth_objects()
            analytics.compute_analytics()
            analytics.compute_aggregate()
            record_keeper.add_record(analytics.aggregated_output)
    
    print(record_keeper.aggregated_records)
    
    #file_writer = file_writer(analytics.analytics_df.copy())
    #file_writer.write_csv(result_path)

    