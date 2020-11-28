# RSA_clustering.py

import sys
import pandas as pd
import numpy as np
from scipy import stats
from timeit import default_timer as timer
from sklearn.manifold import MDS
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
from nltk.corpus import wordnet
import matplotlib.pyplot as plt
import pickle
import gensim
import itertools

from RSA_plot import *
import variables_processing_alt as VP
import Clustering_Auxiliaries as CA

RESOURCE_DIRECTORY = "./Resources/"
MISSING_WORDS = np.array(['chuckled', 'Swedish', 'squawked', 'scalded', 'transfixed', 'gazed', 'death', 'snuggle', 'queasy', 'submersed', 'first', 'yawned', 'scowl', 'grins', 'flunked', 'peppery', 'early', 'Ethiopian', 'sniffed', 'abhor', 'spanked', 'blistered', 'shuddered', 'squealed', 'caressed', 'Chinese', 'nuzzled', 'imbecile', 'counterfeited'])

def main():
	# load words
	all_var = load_instance("all_var_trimed", RESOURCE_DIRECTORY)
	words = all_var.words
	# get matrices
	w2v_dis_mat = all_var.export_variables(np.array(["w2v"]))[ "var_matrices" ][0]
	glove_dis_mat = all_var.export_variables(np.array(["glove"]))["var_matrices"][0]
	arsl_dis_mat = all_var.export_variables(np.array(["Arousal"]))[ "var_matrices" ][0]
	val_dis_mat = all_var.export_variables(np.array(["Valence"]))[ "var_matrices" ][0]
	# wu_dis_mat = assemble_sim_mat(words)
	# path_dis_mat = assemble_sim_mat(words, SFunc = PathSimAvg)

	arousal = all_var.export_variables(np.array(["Arousal"]))["var_arrays"][0]
	valence = all_var.export_variables(np.array(["Valence"]))["var_arrays"][0]
	concreteness = all_var.export_variables(np.array(["Concreteness"]))["var_arrays"][0]
	log_freq =all_var.export_variables(np.array(["Log_Freq"]))["var_arrays"][0]
	ortho_20 =all_var.export_variables(np.array(["Ortho20"]))["var_arrays"][0]
	length = all_var.export_variables(np.array(["Length"]))["var_arrays"][0]


	# clustering
	DIS_MAT = glove_dis_mat
	DIS_MAT_NAME = "glove"
	# word_embeddings = MDS(n_components = 4, dissimilarity = "precomputed")
	# word_transform = word_embeddings.fit_transform(DIS_MAT)
	word_transform, eigvals = CA.PCoA(DIS_MAT, n_dim = 4)

	# clustering_model = AgglomerativeClustering(n_clusters = None, distance_threshold = 0).fit(word_transform.transpose())
	weighted_dist = CA.weighted_euclidian(word_transform, eigvals)
	clustering_model = AgglomerativeClustering(n_clusters = None, distance_threshold = 0, affinity = "precomputed", linkage = "average").fit(weighted_dist)


# evaluating performances
	root, depth, level_dict = CA.build_tree(clustering_model)
	var_array = np.array([arousal, valence, concreteness, log_freq, ortho_20, length])
	var_names = np.array(["Arousal", "Valence", "Concreteness", "Log Freq", "Ortho 20", "Length"])
	var_array = CA.normalize(var_array)
	# evaluate_results = CA.evaluate_clusters(var_array, level_dict, 468, metrics.calinski_harabasz_score)
	evaluate_results = CA.evaluate_clusters(var_array, level_dict, 468, CA.weighted_variance)
	plot_1D(evaluate_results, var_names, start_end = (1,depth - 1), interval = 1, save = True, save_name = "./WV_index.png")
	areas = CA.integration(evaluate_results)
	for index in range(len(areas)):
		print (var_names[index] + "	" + str(areas[index]))

# Word Similarity
###############################################################################
def check_senses(words, pos):
	total_sense = 0
	noun_sense = 0
	non_noun_w = 0
	for word in words:
		all_sense = wordnet.synsets(word)
		n_sense = wordnet.synsets(word, pos = pos)
		if len(all_sense) != len(n_sense):
			n_percentage = (len(n_sense)/len(all_sense))*100
			print (word + ": " + str(n_percentage) + "%")
			non_noun_w += 1
		total_sense += len(all_sense)
		noun_sense += len(n_sense)
	all_percentage = (noun_sense / total_sense)*100
	print ("There are " + str(non_noun_w) + " words that has sense other than the specified POS")
	print ("Overall POS sense percentage: " + str(all_percentage) + "%")

def wuSimAvg(word1,word2):    
	syns1 = set(ss for word in word1 for ss in wordnet.synsets(word,pos = 'n'))
	syns2 = set(ss for word in word2 for ss in wordnet.synsets(word,pos = 'n'))

	sim_value = []
	for syn1 in syns1:
		for syn2 in syns2:
			sim_value.append(wordnet.wup_similarity(syn1,syn2))
	try:
		sim_value_mean = np.nanmean(sim_value)
	except Warning:
		print (word1)
		print (word2)
	
	if sim_value_mean == np.NaN:
		print (word1)
		print (word2)   	
	return sim_value_mean

def wuSimDom(word1, word2):
	syn1 = wordnet.synsets(word1, pos = 'n')[0]
	syn2 = wordnet.synsets(word2, pos = 'n')[0]
	return wordnet.wup_similarity(syn1,syn2)

def PathSimAvg(word1,word2):    
    syns1 = set(ss for word in word1 for ss in wordnet.synsets(word,pos = 'n'))
    syns2 = set(ss for word in word2 for ss in wordnet.synsets(word,pos = 'n'))
    
    sim_value = []
    for syn1 in syns1:
        for syn2 in syns2:
            sim_value.append(wordnet.path_similarity(syn1,syn2))
    try:
    	sim_value_mean = np.nanmean(sim_value)
    except RuntimeWarning:
    	print (word1)
    	print (word2)
    return sim_value_mean

def PathSimDom(word1, word2):
	syn1 = wordnet.synsets(word1, pos = 'n')[0]
	syn2 = wordnet.synsets(word2, pos = 'n')[0]
	return wordnet.path_similarity(syn1,syn2)

def assemble_sim_mat(words, SFunc = wuSimAvg):
	results = np.ones((len(words), len(words)))
	pbar = tqdm(total = (len(words)*len(words) - len(words))//2)
	for row_ind in range(0, len(words)):
		for col_ind in range(row_ind+1, len(words)):
			pbar.update(1)
			results[row_ind, col_ind] = SFunc([words[row_ind]], [words[col_ind]])
			results[col_ind, row_ind] = results[row_ind, col_ind]
	return 1 - results

# Other functions
###############################################################################
def save_instance(instance, var_name, directory):
	file_name = directory + var_name + ".pkl"
	with open (file_name, "wb") as output:
		pickle.dump(instance, output, pickle.HIGHEST_PROTOCOL)
	return

def load_instance(var_name, directory):
	file_name = directory + var_name + ".pkl"
	with open (file_name, "rb") as input_f:
		instance = pickle.load(input_f)
	return instance

if __name__ == "__main__":
	main()