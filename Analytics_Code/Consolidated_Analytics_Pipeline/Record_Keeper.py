import pandas as pd


class record_keeper:

    def __init__(self,algorithms):
        self.aggregated_records = pd.DataFrame(columns=algorithms+["Taxa","Internal Threshold"])


    def add_record(self,record_list,taxa,internal_threshold):
        self.aggregated_records.loc[len(self.aggregated_records)] = record_list + [taxa] + [internal_threshold]

