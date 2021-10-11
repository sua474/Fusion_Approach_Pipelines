import pandas as pd


class record_keeper:

    def __init__(self,algorithms):
        self.aggregated_records = pd.DataFrame(columns=algorithms)


    def add_record(self,record_list):
        self.aggregated_records.loc[len(self.aggregated_records)] = record_list

