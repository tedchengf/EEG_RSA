import pandas as pd
import numpy as np
import Single_Trial_RSA as STR
import variables_processing as VP
from PPData import *
from scipy import stats
from timeit import default_timer as timer
from CPerm import *
import matplotlib.pyplot as plt
import pickle
from RSA_plot import *

NAMES = ["bin", "Word", "PoS", "Valence", "Arousal", "Concreteness", "Log_Freq", "Val_Cond", "Arsl_Cond", "Freq_Cond", "Conc_Cond", "Length", "Orth", "N2_C", "Orth_F", "N2_F", "V.Mean.Aprrox", "A.Mean.Approx", "OLD20", "SUBTLEX_Freq"]
RESOURCE_DIRECTORY = "./Resources/"
DATA_DIRECTORY = "../DATA/"
DATA = "../DATA/Emsingle1_REP_labeled_cleaned_WithOutliers_2019.04.09.csv"
EEG_START = 'X.100..95'
EEG_END = 'X795.800'
END_COL = 198
VAR_NAME = ['Valence', 'Arousal', 'Concreteness', 'Log_Freq', 'Length', 'Orth_F']
ALL_IV = ['Valence', 'Arousal', 'Concreteness', "w2v"]
ALL_CV = ['Log_Freq', 'Ortho20', 'Length']
# ALL_IV = ['Log_Freq', 'Length', 'Orth_F']

def main():
	# all_data, subjects = load_all_subjects(DATA, EEG_START, EEG_END, np.array(VAR_NAME), end_col = END_COL)	
	all_data, subjects, averaged_result = load_all_subjects(DATA, EEG_START, EEG_END, np.array(VAR_NAME), end_col = END_COL)
	# all_var = load_var("../DATA/words.txt", "../DATA/all_stimuli.txt", NAMES, VAR_NAME)
	all_var = load_instance("all_var", RESOURCE_DIRECTORY)
	ortho_20_data = np.loadtxt("../DATA/old20.txt")
	all_var.update_variable("Ortho20", var_array = ortho_20_data)

	STR_1, STR_2 = create_STR_cond(subjects, all_var)
	all_sub_1 = run_STR(STR_1, "../Program_output/corr_results_1.png")[1]
	all_sub_2 = run_STR(STR_2, "../Program_output/corr_results_2.png")[1]
	print (all_sub_1.shape)

	run_CPerm(all_sub_1[:,:,10:], ALL_IV, "../Program_output/E1_arsl_1/")
	print ("\n\n")
	run_CPerm(all_sub_2[:,:,10:], ALL_IV, "../Program_output/E1_arsl_2/")
	return

	STR_subs = create_STR(subjects, all_var)
	names, subs, all_result = run_STR(STR_subs, "../Program_output/corr_results_1.png")
	subs = np.swapaxes(subs, 0, 1)
	# for index, var_name in enumerate(ALL_IV):
	# 	var_data = subs[index, :, :]
	# 	plot_corr_2D(var_data, var_name, start_end = (-100, 800), interval = 100, clim = (-0.02, 0.02), save = True, save_name = "../Program_output/E1_2D/" + var_name + ".png")
	plot_subjects(subs[1], "Arousal", (4, 6), start_end = (-100, 800), interval = 100, show = True)

	# run_CPerm(all_sub, ALL_IV, "../Program_output/E2_arsl/")

	# raw_corr_matrix = all_var.calculate_variables_correlation(np.array(ALL_IV[:-2] + ALL_CV), corr_type = "raw")
	# raw_corr_matrix_partial = all_var.calculate_variables_correlation(np.array(ALL_IV[:-2]), control_var = np.array(ALL_CV), corr_type = "raw")
	# dsm_corr_matrix = all_var.calculate_variables_correlation(np.array(ALL_IV + ALL_CV))
	# dsm_corr_matrix_partial = all_var.calculate_variables_correlation(np.array(ALL_IV), control_var = np.array(ALL_CV))
	# VP.plot_matrix(raw_corr_matrix, "raw", ALL_IV[:-2] + ALL_CV, save = True)
	# VP.plot_matrix(raw_corr_matrix_partial, "raw_par", ALL_IV[:-2], save = True)
	# VP.plot_matrix(dsm_corr_matrix, "dsm", ALL_IV + ALL_CV, save = True)
	# VP.plot_matrix(dsm_corr_matrix_partial, "dsm_par", ALL_IV, save = True)


	return

# Preprocessing
###############################################################################
def load_all_subjects(data_loc, eeg_start, eeg_end, var_titles, end_col = None, verbose = False):
	if end_col is not None:
		col_subset = np.arange(end_col)
		all_subjects = pd.read_csv(data_loc, usecols = col_subset)
	else:
		all_subjects = pd.read_csv(data_loc)
	all_eeg = all_subjects.loc[:, eeg_start:eeg_end].to_numpy(dtype = float)
	all_var = all_subjects[var_titles].to_numpy(dtype = float)
	all_bin = all_subjects['Bin'].to_numpy(dtype = int)
	# subjects_PPData = __extract_subjects(all_subjects, all_eeg, all_var, all_bin, var_titles, verbose)
	subjects_PPData, averaged_result = __average_subjects(all_subjects, all_eeg, all_var, all_bin, var_titles, verbose)
	return all_subjects, subjects_PPData, averaged_result

def __extract_subjects(all_subjects, all_eeg, all_var, all_bin, var_titles, verbose_val):
	subjects_PPData = []
	subjects_list = all_subjects['Subject'].to_numpy(dtype = int)
	max_sub = np.amax(subjects_list)
	index = 1
	while index <= max_sub:
		# subject not deleted
		if index in subjects_list:
			sub_mask = subjects_list == index
			sub_eeg = all_eeg[sub_mask,:]
			sub_var = all_var[sub_mask,:]
			sub_bin = all_bin[sub_mask]
			sub_PPData = PPData(index, sub_eeg, sub_var, var_titles)
			sub_PPData.run_preprocessing(sub_bin, verbose = verbose_val)
			subjects_PPData.append(sub_PPData)
			print ("subject = " + str(index), end = ": ")
			print (str(sub_PPData.clean_eeg.shape), end = "; ")
			print ("trails = " + str(sub_PPData.bins.shape[0]))
		index = index + 1	
	return subjects_PPData

def __average_subjects(all_subjects, all_eeg, all_var, all_bin, var_titles, verbose_val):
	subjects_PPData = []
	averaged_data = np.zeros((468, 29, 180))
	trial_num = np.zeros((468))
	subjects_list = all_subjects['Subject'].to_numpy(dtype = int)
	max_sub = np.amax(subjects_list)
	index = 1
	while index <= max_sub:
		# subject not deleted
		if index in subjects_list:
			sub_mask = subjects_list == index
			sub_eeg = all_eeg[sub_mask,:]
			sub_var = all_var[sub_mask,:]
			sub_bin = all_bin[sub_mask]
			sub_PPData = PPData(index, sub_eeg, sub_var, var_titles)
			sub_PPData.run_preprocessing(sub_bin, verbose = verbose_val)
			subjects_PPData.append(sub_PPData)
			update_average(sub_PPData.clean_eeg, sub_PPData.bins, averaged_data, trial_num)
			print ("subject = " + str(index), end = ": ")
			print (str(sub_PPData.clean_eeg.shape), end = "; ")
			print ("trails = " + str(sub_PPData.bins.shape[0]))
		index = index + 1	
	
	plt.plot(trial_num)
	plt.show()
	averaged_result = np.zeros((468, 29, 180))
	for index, num_sub in enumerate(trial_num):
		averaged_result[index] = averaged_data[index] / num_sub
	return subjects_PPData, averaged_result

def update_average(sub_eeg, sub_bin, averaged_data, trial_num):
	for eeg_index, curr_bin in enumerate(sub_bin):
		bin_pos = curr_bin - 1
		trial_num[bin_pos] += 1
		averaged_data[bin_pos] += sub_eeg[eeg_index]


# VP
###############################################################################
def load_var(word_loc, var_loc, read_var_names, var_name):
	words = np.loadtxt(word_loc, dtype = np.str)
	original_data = pd.read_csv(var_loc, sep = "	", header = None, names = read_var_names, na_filter=False, keep_default_na=False, na_values=[''])
	all_var = VP.variables(words)
	for name in var_name:
		array = original_data[name].to_numpy(dtype = float)
		all_var.update_variable(name, var_array = array)
	all_var.update_variable("Valence_z_squared", original_data["Valence"].to_numpy(dtype = float), Tfunction = VP.z_squared_transform)
	all_var.semantic_disimilarilty(words, sim_type = "w2v", resource_directory = RESOURCE_DIRECTORY)
	all_var.semantic_disimilarilty(words, sim_type = "glove", resource_directory = RESOURCE_DIRECTORY)
	all_var.update_mask("Arsl_Cond", original_data["Arsl_Cond"].to_numpy(dtype = int), np.array([1,2]))
	all_var.print_attribute_info()
	save_instance(all_var, "all_var", RESOURCE_DIRECTORY)
	return all_var

# RSA
###############################################################################
def create_STR_cond(subjects_PPData, all_var):
	STR_cond_1 = []
	STR_cond_2 = []
	for subject in subjects_PPData:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1) , dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		sub_conds = all_var.create_STR_subject(sub_name, sub_eeg, all_var.variable_names, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, mask_name = "Arsl_Cond")
		STR_cond_1.append(sub_conds[0])
		STR_cond_2.append(sub_conds[1])
	return STR_cond_1, STR_cond_2

def create_STR(subjects_PPData, all_var):
	STR_subs = []
	for subject in subjects_PPData:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		sub = all_var.create_STR_subject(sub_name, sub_eeg, all_var.variable_names, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict)
		STR_subs.append(sub)
	return STR_subs

def run_STR(subjects, save_name):
	start = timer()
	all_subjects = []
	all_corr_results = np.zeros((len(ALL_IV), 91))
	all_subnames = []
	for sub in subjects:
		print ("############################  ", end = "")
		print ("subject " + sub.subname, end = "")
		print ("  ############################")
		all_subnames.append(sub.subname)
		sub.print_attribute_info()	
		corr_result, var_titles = sub.single_trial_RSA(np.array(ALL_IV), CV = np.array(ALL_CV), time_window = 4, step = 2)
		# corr_result, var_titles = sub.single_trial_RSA(np.array(ALL_IV), time_window = 4, step = 2)
		all_subjects.append(corr_result)
		all_corr_results = np.add(all_corr_results, corr_result)
		print ("")
	all_corr_results = all_corr_results/len(subjects)
	end = timer()
	plot_corr(all_corr_results, np.array(ALL_IV), start_end = (-100, 800), interval = 100, save = True, save_name = save_name)
	print ("execution time: " + str(end - start))
	return np.array(all_subnames), np.array(all_subjects), all_corr_results

# Cluster Permutation Test
###############################################################################
def run_CPerm(all_subjects, all_variables, save_directory):
	# changing the all_subjects dim from (subject, variable, corr_value) to 
	# (variable, subject, corr_value)
	all_subjects = np.swapaxes(np.array(all_subjects), 0, 1)
	null_state = np.zeros((all_subjects.shape))
	scale = np.linspace(0, 800, num = 81, dtype = int)
	var_tvals = []	
	var_names = []
	means = []
	stds = []
	for index, var_title in enumerate(all_variables):
		var_CPerm, var_matrix, var_pandas_out = CPerm(var_title ,all_subjects[index, :, :], null_state[index, :, :], 200, pos_label = scale)
		print (var_title)
		print (var_pandas_out)
		var_tvals.append(var_CPerm.Tvals)
		mean, std = var_CPerm.calculate_std_err()
		var_names.append(var_title)
		means.append(mean)
		stds.append(std)
	arsl_mean = means[1]
	arsl_std = stds[1]
	for index, var_name in enumerate(var_names):
		if index == 1:
			pass
		else:
			curr_mean = means[index]
			curr_std = stds[index]
	return

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

# def plot_var(var_name, mean, std, show = False, save = False, save_name = "./var.png"):
# 	plt.plot(mean, label = var_name + " mean", color = "red")
# 	plt.fill_between(np.arange(0,len(mean), step = 1, dtype = int), mean - std, mean + std, alpha = 0.3)
# 	label = np.linspace(-100, 800, num = 10, dtype = int)
# 	x_range = len(mean)
# 	plt.xticks(np.arange(0,x_range, step = 10, dtype = int), label, rotation = "vertical")
# 	plt.legend()
# 	if save == True:
# 		plt.savefig(save_name, format = "png", dpi = 1000)
# 	if show == True:
# 		plt.show()
# 	plt.clf()
# 	return

# def plot_multi_vars(var_names, means, stds, show = False, save = False, save_name = "./var.png"):
# 	for var_index, var_name in enumerate(var_names):
# 		var_mean = means[var_index]
# 		var_std = stds[var_index]
# 		plt.plot(var_mean, label = var_name)
# 		plt.fill_between(np.arange(0, len(var_mean), step = 1, dtype = int), var_mean - var_std, var_mean + var_std, alpha = 0.3)
# 	label = np.linspace(-100, 800, num = 10, dtype = int)
# 	x_range = len(means[0])
# 	plt.xticks(np.arange(0,x_range, step = 10, dtype = int), label, rotation = "vertical")
# 	plt.axis([None, None, -0.03, 0.03])
# 	plt.legend()
# 	if save == True:
# 		plt.savefig(save_name, format = "png", dpi = 1000)
# 	if show == True:
# 		plt.show()
# 	plt.clf()
# 	return

# def plot_corr(all_corr_results, variables_title, show = False, save = False, save_name = "./corr_results.png"):
# 	for var_index in range(len(variables_title)):
# 		var_name = variables_title[var_index]
# 		corr_results = all_corr_results[var_index]
# 		plt.plot(corr_results, label = var_name)
# 	label = np.linspace(-100, 800, num = 10, dtype = int)
# 	x_range = len(all_corr_results[0])
# 	plt.xticks(np.arange(0,x_range, step = 10, dtype = int), label, rotation = "vertical")
# 	plt.axis([None, None, -0.05, 0.05])
# 	plt.legend()
# 	if save == True:
# 		plt.savefig(save_name, format = "png", dpi = 1000)
# 	if show == True:
# 		plt.show()
# 	plt.clf()
# 	return

if __name__ == "__main__":
	main()