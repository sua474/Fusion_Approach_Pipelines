import matplotlib.pyplot as plt
from matplotlib_venn import venn3

from itertools import *
import pandas as pd
import copy as c

def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_power_set(algo_names):

	comparators = []
	for group in power_set(algo_names):
		if(len(group)>=2):
			comparators.append(group)

	return comparators

def validate_df(df1,df2,algo1_name,algo2_name):

	if(df1.shape != df2.shape):
		print("Two Dataframes {} and {} are of unequal dimensions".format(algo1_name,algo2_name))
		exit()

	return True

def normalize_df(df1,df2):

	df1[df1>0] = 1
	df1[df1<=0] = 0
	df2[df2>0] = 1
	df2[df2<=0] = 0

	return df1,df2

def overlap(df1,df2,algo1_name,algo2_name):
	
	overlap_df = c.deepcopy(df1)

	validate_df(df1,df2,algo1_name,algo2_name)

	df1,df2 = normalize_df(df1,df2)

	for i in range (0,df1.shape[0]):
		for j in range(0,df1.shape[1]):
			if (df1.iloc[i,j] == df2.iloc[i,j]):
				overlap_df.iloc[i,j] = 1
			elif (df1.iloc[i,j] != df2.iloc[i,j]):
				overlap_df.iloc[i,j] = 0
			else:
				pass
		
	return overlap_df


def compute_overlap(algo_pair):

	global file_location
	algo_1 = algo_pair.pop()
	algo_2 = algo_pair.pop()

	print('Comparing Algorithms {} and {}'.format(algo_1,algo_2))

	algo_1_file = pd.read_csv(file_location+algo_1+".csv")
	algo_2_file = pd.read_csv(file_location+algo_2+".csv")
	
	overlap_df = overlap(algo_1_file,algo_2_file,algo_1,algo_2)

	while(algo_pair):
		algo_3 = algo_pair.pop()
		
		print('Comparing Overlap and {}'.format(algo_3))

		algo_3_file = pd.read_csv(file_location+algo_3+".csv")
		overlap_df = overlap(overlap_df,algo_3_file,'overlap',algo_3)

	return overlap_df

def count_match_vs_mismatch(overlap_df):

	match_count = 0
	mismatch_count = 0

	for i in range(0,overlap_df.shape[0]):
		for j in range(0,overlap_df.shape[1]):
			if(overlap_df.iloc[i,j]==1):
				match_count+=1
			else:
				mismatch_count+=1

	return match_count,mismatch_count

def generate_statistics(sets):
	
	overlap_statistics = []

	for pair in sets:
		pair_list = list(pair)
		overlap_df = compute_overlap(list(pair))
		match,mismatch = count_match_vs_mismatch(overlap_df)
		overlap_statistics.append([pair_list[0],pair_list[1],match,mismatch])

	return overlap_statistics

def plot_statistics(overlap_statistics,algo_names):
	return 0


if __name__=='__main__':

	file_location = "Aggregated_Files/"
	algo_names = ['Spiec_Easi_10', 'Spring_10']
	sets = generate_power_set(algo_names)
	overlap_statistics = generate_statistics(sets)
	print(overlap_statistics,algo_names)