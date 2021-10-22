import pandas as pd


class record_keeper:

    def __init__(self,algorithms):
        self.aggregated_records = pd.DataFrame(columns=['Algorithm','Accuracy','TPR','TNR','FPR','FNR']+["Taxa","Internal_Threshold"])


    def add_record(self,aggregated_df,taxa,internal_threshold):
        
        aggregated_df['Taxa'] = taxa
        aggregated_df['Internal_Threshold'] = internal_threshold

        self.aggregated_records = self.aggregated_records.append(aggregated_df,ignore_index=True)
    
    def get_top_performing_algorithms(self,number_of_tops):
        #Takes average of each dataframe column and returns the top most algorithms
        algorithms = list(self.aggregated_records.columns)
        average = []
        top_algos = []
        for x in algorithms[0:-2]: # Not including the last two columns as they are irrelavant for mean analysis
            average.append(self.aggregated_records[x].mean())
        
        top_indexes = sorted(range(len(average)), key=lambda k: average[k])
        top_indexes = top_indexes[-number_of_tops:][::-1]
        
        for index in top_indexes:
            top_algos.append(algorithms[index])
        
        return top_algos
        
        
        

