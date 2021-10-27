import pandas as pd

class perform_matching:
    
    def dimensionality_check(self,algorithm_df,ground_truth_df):
        
        if (algorithm_df.shape[0] == ground_truth_df.shape[0] and algorithm_df.shape[1] == ground_truth_df.shape[1]):
            return True
        else:
            return False

    def compute_match_statistics(self,algorithm_df,ground_truth_df):

        tp,tn,fp,fn = 0,0,0,0

        for i in range(0, algorithm_df.shape[0]):
            for j in range(0, algorithm_df.shape[1]):
                if(algorithm_df.iloc[i,j] == 1 and ground_truth_df.iloc[i,j] == 1):
                    tp+=1
                elif(algorithm_df.iloc[i,j] == 1 and ground_truth_df.iloc[i,j] == 0):
                    fp+=0
                elif(algorithm_df.iloc[i,j] == 0 and ground_truth_df.iloc[i,j] == 0):
                    tn+=1
                elif(algorithm_df.iloc[i,j] == 0 and ground_truth_df.iloc[i,j] == 1):
                    fn+=1
        
        penalty = round(fp+fn/(tp+tn+fp+fn),3) # Total wrong predictions
        accuracy = round( ((tp + tn) / (tp+tn+fp+fn)),3) # Total Correct Predictions

        return [accuracy,penalty]


    def compute_overlap(self, algorithms,ground_truth):
        
        output = dict()
        ground_truth_df = ground_truth.get_file()
        
        for algorithm in algorithms:
            algorithm_df = algorithm.get_file()
            if(isinstance(algorithm_df,bool)):
                output[algorithm.algorithm_name] = [0]*2 #multiplied by 5 to have equal number of zeros for each analytical value

            elif(self.dimensionality_check(algorithm_df,ground_truth_df)):
                output[algorithm.algorithm_name] = self.compute_match_statistics(algorithm_df,ground_truth_df)
            
            else:
                print('The Algorithm {} and Ground Truth Do not have the same dimensions'.format(algorithm.algorithm_name))
                output[algorithm.algorithm_name] = [0]*2 #multiplied by 5 to have equal number of zeros for each analytical value
                
        return output.copy()
                
            
        