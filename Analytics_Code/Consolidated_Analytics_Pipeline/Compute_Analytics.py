from email.mime import base
import pandas as pd 
import os
import sys
from File_Reader import file_reader
from File_Writer import file_writer
from Perform_Matching import perform_matching
from Record_Keeper import record_keeper

class compute_analytics:

    def __init__ (self,algorithms,base_folder,allowed_filenames,ground_truth_path,analytical_features):
        '''
        Initilizes the object
        '''
        self.algorithms = algorithms.copy()
        self.analytical_features = analytical_features.copy()
        self.base_folder = base_folder
        self.allowed_filenames = allowed_filenames.copy()
        self.data_files = dict.fromkeys(algorithms)
        self.ground_truth_path = ground_truth_path
        self.ground_truth = dict()
        self.analytics_df = pd.DataFrame(columns = analytical_features+['Data_File_Index'] )
        self.aggregated_analytics = pd.DataFrame(columns = analytical_features)
        
        for algorithm in self.algorithms:
            self.data_files[algorithm] = list()

    def load_input_file_objects(self):
        '''
        Loads the outputs of different algorithms into a dicitionary with each algorithm name as its key.
        Note that it just creates and loads the objectsbut not read the file. Reading is done in the 
        Perform_Matching class using the file reader object 
        '''
        for algorithm in self.algorithms:
            location = os.path.join(self.base_folder, algorithm)
            for root, directories, files in os.walk(location):
                file_found = False
                for file in files:
                    if(file in self.allowed_filenames):
                        file_found = True
                        self.data_files[algorithm].append(file_reader(root,file,algorithm))
                if(not file_found):
                    self.data_files[algorithm].append(file_reader(root,"Missing_File",algorithm))
    
    def load_ground_truth_objects(self):
        '''
        Loads the ground truth objects into the dicitionary. Note that it just creates and loads the objects
        but not read the file. Reading is done in the Perform Matching class using the file reader object 
        '''
        assert os.path.exists(self.ground_truth_path), "Ground Truth Path Does Not Exist"
        for root, directories, files in os.walk(self.ground_truth_path):
            for file in files:
                self.ground_truth[file] = file_reader(root,file,'Ground_Truth')
    
    def validate_payload(self,pay_load):
        '''
        The job of this function is to validate if all the outputs from different algorithms belong to the
        same datasets. If so, then it will return the numeriacal index of that dataset. Else it will return
        the error.
        '''
        ref_obj = pay_load.pop()
        reference = os.path.basename(ref_obj.file_location).split('_')[-1]
        for obj in pay_load:
            index = os.path.basename(obj.file_location).split('_')[-1]
            assert index!=reference, sys.exit("Files Do Not Belong to the Same Ecological Dataset")
        
        return reference
            
    def compute_analytics(self):
        '''
        It calls the object of Perform Matching class to compare the algorithm's output wrt the ground truth
        and saves the result for future analysis
        '''
        matcher = perform_matching()

        for i in range(len(self.ground_truth)):
            pay_load = [self.data_files[algorithm].pop() for algorithm in self.algorithms if len(self.data_files[algorithm])>=1]
            
            index = self.validate_payload(pay_load.copy())
            ground_truth_file_name = 'Ground_Truth_{}.csv'.format(index)
            print('Processing {}'.format(ground_truth_file_name))
            
            matches = matcher.compute_overlap(pay_load,self.ground_truth[ground_truth_file_name])
            
            for algorithm in matches.keys(): 
                self.analytics_df.loc[len(self.analytics_df)] = [algorithm]+matches[algorithm]+[int(index)]
    
    def compute_aggregate(self):
        '''
        Computes the aggregate of all analytical feature per different datasets
        '''
        for algorithm in self.algorithms:
            algorithm_df = self.analytics_df.loc[self.analytics_df['Algorithm'] == algorithm]
            mean_dict = algorithm_df[self.analytical_features[1:]].mean()
            self.aggregated_analytics.loc[len(self.aggregated_analytics)] = [algorithm] + list(mean_dict.values)

    
    def check_validity(self):
        '''
        Checks the validity of file. Was just made for sanity check
        '''
        for algorithm in self.data_files.keys():
            print('Algorithm {} Total Files: {}'.format( algorithm, len(self.data_files[algorithm])))
        
        print('Ground Truth Total Files: {}'.format(len(self.ground_truth)))
        
        for algorithm in self.data_files.keys():
            print(algorithm)
            print(self.data_files[algorithm][0].file_location)
            
        for x in self.ground_truth.keys():
            print(x)
          
    
if __name__ == '__main__':
        
    algorithms = ['Spiec_Easi','Xiao','Ma_Paper','Correlation','Spring'] # Names of algorithms (same as output folder name)
    output_filename = set(['Adjacency_Matrix.csv','Sign of Jaccobian for Iteration_0.csv','Metric Network.csv']) # Names of output files of different algorithm, so that it can only read the exact file in the output dir
    analytical_features = ['Algorithm','Accuracy','Penalty'] #Features that are calculated for each output file wrt the ground truth
    record_keeper = record_keeper(algorithms,analytical_features) # Object Initialization
    output_path = "..\..\Output\Analytics_Output\Aggregate.csv" # Path of outputfolder

    for taxa in ["Taxa_10"]: # Number of taxa to run the analytics pipeline for can include all "Taxa_30", "Taxa_50" and "Taxa_100" as well
        for internal_threshold in ["10","50","100"]: # Different internal threshold at which the dataset was created (This threshold is set in the Chiquet data generation code)
            print("Executing Taxa {} and Internal Threshold {}".format(taxa,internal_threshold))
        ######### Input Parameters #####################
            ground_truth_path = "..\..\Dataset\Chiquet_Simulated_Dataset\{}\Ground_Truth_{}".format(taxa,internal_threshold) # Path where ground truth files are
            base_path = "..\..\Output\Algorithm_Output\{}\IT_{}".format(taxa,internal_threshold) #Path where algorithm output is written
            ground_truth_path = os.path.abspath(ground_truth_path) #Creating the complete absolute path
            base_path = os.path.abspath(base_path) #Creating the complete absolute path
        ################################################

        ######## Object Initialization ####################
            analytics = compute_analytics(algorithms,base_path,output_filename,ground_truth_path,analytical_features)
        ######### Start Analytics #########################
            analytics.load_input_file_objects()
            analytics.load_ground_truth_objects()
            analytics.compute_analytics()
            analytics.compute_aggregate()
            record_keeper.add_record(analytics.aggregated_analytics.copy(),taxa,internal_threshold)
            
    print(record_keeper.get_top_performing_algorithms(3))
    file_writer = file_writer()
    file_writer.write_csv(os.path.abspath(output_path),record_keeper.aggregated_records)
    