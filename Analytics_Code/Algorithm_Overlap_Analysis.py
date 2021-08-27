from itertools import *

def power_set(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def generate_power_set(algo_names):

	comparators = []
	for group in power_set(algo_names):
		if(len(group)>=2):
			comparators.append(group)

	return comparators

def compute_overlap(algo_pair):

	print(algo_pair)

	algo_1 = algo_pair.pop()
	algo_2 = algo_pair.pop()

	while(algo_pair):
		print("After")
		print(algo_pair.pop())

def generate_statistics(sets):
	for pair in sets:
		compute_overlap(list(pair))
		

if __name__=='__main__':

	algo_names = ['Ma_10', 'Spiec_Easi_10', 'Spring_10']
	sets = generate_power_set(algo_names)
	generate_statistics(sets)