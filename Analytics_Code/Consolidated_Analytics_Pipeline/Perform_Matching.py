import pandas as pd
from File_Reader import file_reader

class perform_matching:
    
    def dimensionality_check(self,algorithm_df,ground_truth_df):
        
        if (algorithm_df.shape[0] == ground_truth_df.shape[0] and algorithm_df.shape[1] == ground_truth_df.shape[1]):
            return True
        else:
            return False
    
    def compute_exact_match(self,algorithm_df,ground_truth_df):
    
        matches = 0
        total = algorithm_df.shape[0] * algorithm_df.shape[1]
        for i in range(0, algorithm_df.shape[0]):
            for j in range(0, algorithm_df.shape[1]):
                if(algorithm_df.iloc[i,j] == ground_truth_df.iloc[i,j]):
                    matches+=1
        
        return round((matches/total)*100,3)
    
    def compute_overlap(self, algorithms,ground_truth):
        
        output = dict()
        ground_truth_df = ground_truth.get_file()
        
        for algorithm in algorithms:
            algorithm_df = algorithm.get_file()
            if(self.dimensionality_check(algorithm_df,ground_truth_df)):
                match_percentage = self.compute_exact_match(algorithm_df,ground_truth_df)
                output[algorithm.algorithm_name] = match_percentage
            else:
                print('The Algorithm {} and Ground Truth Do not have the same dimensions'.format(algorithm.algorithm_name))
                output[algorithm.algorithm_name] = 0
                
        return output.copy()
                
            
        