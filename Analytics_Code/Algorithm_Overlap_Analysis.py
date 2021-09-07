import matplotlib.pyplot as plt
from matplotlib_venn import venn3
import seaborn as sns
from itertools import *
import pandas as pd
import copy as c
import numpy as np

def read_file(location):
	return c.copy(pd.read_csv(location))

def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_power_set(algo_names):

	comparators = []
	for group in power_set(algo_names):
		if(len(group)>=1):
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

	algo_1_file = read_file(file_location+algo_1+".csv")
	algo_2_file = read_file(file_location+algo_2+".csv")
	
	overlap_df = overlap(algo_1_file,algo_2_file,algo_1,algo_2)

	while(algo_pair):
		algo_3 = algo_pair.pop()
		
		print('Comparing Overlap and {}'.format(algo_3))

		algo_3_file = read_file(file_location+algo_3+".csv")
		overlap_df = overlap(overlap_df,algo_3_file,'overlap',algo_3)

	return overlap_df

def count_match_vs_mismatch(overlap_df_x,overlap_df_y):

	match_count = 0
	mismatch_count = 0

	validate_df(overlap_df_x,overlap_df_y,"DF X","DF Y")

	for i in range(0,overlap_df_x.shape[0]):
		for j in range(0,overlap_df_x.shape[1]):
			if(overlap_df_x.iloc[i,j]== overlap_df_y.iloc[i,j]):
				match_count+=1
			else:
				mismatch_count+=1

	return (match_count/(match_count+mismatch_count)*100),(mismatch_count/(match_count+mismatch_count)*100)

def generate_statistics(sets):
	
	global file_location
	sets_copy = c.copy(sets)
	overlap_statistics = []

	overlap_df_x = pd.DataFrame()
	overlap_df_y = pd.DataFrame()

	for x in sets:
		if( len(list(x))>1 ):
			overlap_df_x = compute_overlap(list(x))
		else:
			overlap_df_x = read_file(file_location+list(x)[0]+".csv")

		for y in sets_copy:
			#print('Comparing {} with {}'.format(list(x),list(y)))
			if( len(list(y))>1 ):
				overlap_df_y = compute_overlap(list(y))
			else:
				overlap_df_y = read_file(file_location+list(y)[0]+".csv")
			
			match,mismatch = count_match_vs_mismatch(overlap_df_x,overlap_df_y)

			overlap_statistics.append([list(x),list(y),[match,mismatch]])

	return overlap_statistics

def get_string_alias(list):

	str_rep=""

	for val in list:
		str_rep = (str_rep+" "+val).strip()

	return str_rep


def generate_confusion_matrix(overlap_statistics):
	
	vals = list(dict.fromkeys( [get_string_alias(x[0]) for x in overlap_statistics] ))
	confusion_matrix = pd.DataFrame(index = vals, columns=vals)

	for x in overlap_statistics:
		confusion_matrix.loc[get_string_alias(x[0]), get_string_alias(x[1]) ] = x[2][0]

	return (confusion_matrix.astype('float'))

def plot_confusion_matrix(confusion_matrix):
	
	plt.figure()
	#confusion_matrix.to_csv('Confusion_Matrxi.csv')
	mask = np.zeros_like(confusion_matrix)
	mask[np.tril_indices_from(mask)] = True
	sns.heatmap(confusion_matrix, annot=True, fmt=".2f",mask=mask)
	plt.show()


if __name__=='__main__':

	##################### Input variables #################
	file_location = "Aggregated_Files/"
	algo_names = ['Spiec_Easi_10', 'Spring_10']

	#################### Generate Power Sets or combinations of algorithms to be matched ###############
	sets = generate_power_set(algo_names)

	##################### Performs pairwise algorithm matching ##########################################
	overlap_statistics = generate_statistics(sets)

	##################### Generate a cofusion matrix ####################################################
	confusion_matrix = generate_confusion_matrix(overlap_statistics)

	##################### Plot the Confusion Matrix ####################################################
	plot_confusion_matrix(confusion_matrix)
