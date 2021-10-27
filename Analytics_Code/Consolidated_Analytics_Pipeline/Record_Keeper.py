import pandas as pd


class record_keeper:

    def __init__(self,algorithms,analytical_features):
        self.aggregated_records = pd.DataFrame(columns=analytical_features+["Taxa","Internal_Threshold"])


    def add_record(self,aggregated_df,taxa,internal_threshold):
        
        aggregated_df['Taxa'] = taxa
        aggregated_df['Internal_Threshold'] = internal_threshold

        self.aggregated_records = self.aggregated_records.append(aggregated_df,ignore_index=True)
    
    def get_top_performing_algorithms(self,number_of_tops):
        #Takes average of each dataframe column and returns the top most algorithms
        result_dict = {}
        top_algos = {}

        for i in range(0,self.aggregated_records.shape[0]):
            algorithm = self.aggregated_records.iloc[i,0] 
            score = self.aggregated_records.iloc[i,1] - self.aggregated_records.iloc[i,2]
            
            if(algorithm in result_dict and result_dict[algorithm] > score):
                result_dict[algorithm] = score
            if(algorithm not in result_dict):
                result_dict[algorithm] = score

        algorithms = list(result_dict.keys())
        scores = list(result_dict.values())
        
        top_indexes = sorted(range(len(scores)), key=lambda k: scores[k])
        top_indexes = top_indexes[-number_of_tops:][::-1]
        
        for index in top_indexes:
            top_algos[algorithms[index]] = scores[index]
        
        return top_algos
        
        
        

