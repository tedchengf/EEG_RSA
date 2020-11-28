# multi_cond_RSA.py

import pandas as pd
import numpy as np
import Single_Trial_RSA as STR
from PPData import *
from scipy import stats
from timeit import default_timer as timer
from CPerm import *
import matplotlib.pyplot as plt
import pickle
from RSA_plot import *
import variables_processing_alt as VP

NAMES = ["bin", "Word", "PoS", "Valence", "Arousal", "Concreteness", "Log_Freq", "Val_Cond", "Arsl_Cond", "Freq_Cond", "Conc_Cond", "Length", "Orth", "N2_C", "Orth_F", "N2_F", "V.Mean.Aprrox", "A.Mean.Approx", "OLD20", "SUBTLEX_Freq"]
RESOURCE_DIRECTORY = "./Resources/"
DATA_DIRECTORY = "../DATA/"
DATA = "../DATA/Emsingle2_REP_labeled_cleaned_WithOutliers_2019.04.09.csv"
EEG_START = 'X.100..95'
EEG_END = 'X795.800'
END_COL = 198
VAR_NAME = np.array(['Valence', 'Arousal', 'Concreteness', 'Log_Freq', 'Length', 'Orth_F'])
# ALL_VAR = np.array(['Valence', 'Arousal', 'Concreteness', 'w2v', 'Log_Freq', 'Ortho20', 'Length'])
ALL_VAR = np.array(['Valence', 'Arousal', 'Concreteness', 'w2v', 'Valence_z', 'Log_Freq', 'Ortho20', 'Length'])
# ALL_IV = np.array(['Valence', 'Arousal', 'Concreteness'])
# ALL_IV = np.array(['Valence', 'Arousal', 'Concreteness', 'w2v', 'Valence_z'])
# ALL_CV = np.array(['Log_Freq', 'Ortho20', 'Length'])
ALL_IV = np.array(['Arousal'])
ALL_CV = np.array(['Valence','Concreteness','Log_Freq','Ortho20','Length'])


def main():
	# load eeg
	all_data, subjects = load_all_subjects(DATA, EEG_START, EEG_END, np.array(VAR_NAME), end_col = END_COL)
	
	# load data
	all_var_alt = load_instance("all_var_alt", RESOURCE_DIRECTORY)
	val_z = all_var_alt.export_variables(np.array(["Valence"]))["var_arrays"][0, :]
	all_var_alt.update_variable("Valence_z", var_array = val_z, Tfunction = VP.z_squared_transform)

	val_cond = np.loadtxt(RESOURCE_DIRECTORY + "val_cond.txt", dtype = int)
	neg_neu = val_cond < 3
	neg_pos_1 = val_cond == 1 
	neg_pos_2 = val_cond == 3
	neg_pos = neg_pos_1 + neg_pos_2
	neu_pos = val_cond > 1

	# run correlation
	# mat_1 = all_var_alt.calculate_variables_correlation(ALL_VAR, corr_type = "dsm") 
	# mat_2 = all_var_alt.calculate_variables_correlation(ALL_IV, control_var = ALL_CV, corr_type = "dsm") 
	# mat_1 = all_var_alt.calculate_variables_correlation(ALL_VAR, interpolation = True) 
	# mat_2 = all_var_alt.calculate_variables_correlation(ALL_IV, control_var = ALL_CV, interpolation = True) 
	# # name, mat_1 = all_var_alt.calculate_variables_correlation_interpol(ALL_VAR, partial = False)
	# # name, mat_2 = all_var_alt.calculate_variables_correlation_interpol(ALL_VAR, partial = True)
	# plot_matrix(mat_1, "DSM", var_names = ALL_VAR, clim = (-0.3, 0.3), show = False, save = True)
	# plot_matrix(mat_2, "DSM Partial", var_names = ALL_IV, clim = (-0.3, 0.3), show = False, save = True)

	# run interpol conditional
	# subjects = creat_STR_cond(subjects, "arsl_cond", all_var_alt, ALL_CV)
	# results = []
	# for cond_ind in range(subjects.shape[0]):
	# 	curr_cond = subjects[cond_ind,:,:]
	# 	all_raw_results, all_titles = run_STR_interpol(curr_cond, True, save_name = "corr_result_" + str(cond_ind) + ".png")
	# 	results.append(all_raw_results)
	# for result in results:
	# 	run_CPerm(result[:,:,10:], all_titles, "./")
	# 	print ("\n\n")

	# subjects = creat_STR_cond(subjects, "arsl_cond", all_var_alt, ALL_CV)
	# results = []
	# for cond_ind in range(subjects.shape[0]):
	# 	curr_cond = subjects[cond_ind,:,:]
	# 	all_raw_results, all_titles = run_STR_interpol(curr_cond, True, save_name = "corr_result_" + str(cond_ind) + ".png")
	# 	results.append(all_raw_results)
	# low_result = np.average(results[0], axis = 1)
	# high_result = np.average(results[1], axis = 1)
	# avg_results = np.subtract(low_result, high_result)
	# all_res = np.array([low_result, high_result, avg_results])[:,0,:]
	# plot_corr(all_res, np.array(["low arsl", "high arsl", "avg_diff"]), start_end = (-100, 800), interval = 100, save = True, save_name = "w2v_diff.png")
	# plot_corr(avg_results, np.array(["diff"]), start_end = (-100, 800), interval = 100, save = True, save_name = "w2v_diff.png")
	# run_CPerm(results[:,:,10:], np.array(["diff"]), "./")


	# run interpol
	subjects = create_STR_interpol(subjects, all_var_alt, ALL_CV, None)
	all_raw_results, all_titles = run_STR_interpol(subjects, True, save_name = "./neu_pos_Valence.png")
	all_raw_results = all_raw_results[:,:,40:60]
	run_CPerm(all_raw_results, all_titles, "./")

	# run normal
	# subjects = create_STR(subjects, all_var_alt, True)
	# all_titles, all_subjects, all_corr = run_STR(subjects, True)
	# all_subjects = all_subjects[:,:, 10:]
	# all_subjects = np.swapaxes(np.array(all_subjects), 0, 1)
	# run_CPerm(all_subjects, ALL_IV, "./")
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
	subjects_PPData = __extract_subjects(all_subjects, all_eeg, all_var, all_bin, var_titles, verbose)
	return all_subjects, subjects_PPData

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
		
# RSA
###############################################################################        
def create_STR_interpol(subjects_PPData, all_var, CV, mask):
	STR_subs = []
	pbar = tqdm(total = len(subjects_PPData))
	for subject in subjects_PPData:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		# subs = all_var.create_STR_subject(sub_name, sub_eeg, ALL_IV, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, interpolation = True, max_diff = 0, stim_val_type = "dsm", CV = CV)
		subs = all_var.create_STR_subject_tri(sub_name, sub_eeg, ALL_IV, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, max_diff = 0, decimals = 4, CV = CV, mask = mask)
		STR_subs.append(subs)
		pbar.update(1)
	return np.array(STR_subs)

def creat_STR_cond(subjects_PPData, cond_name, all_var, CV):
	STR_subs = []
	pbar = tqdm(total = len(subjects_PPData))
	for subject in subjects_PPData:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		subs = all_var.create_STR_subject_tri(sub_name, sub_eeg, np.array(["w2v"]), extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict, max_diff = 0, decimals = 4, CV = CV, mask_name = cond_name)
		STR_subs.append(subs)
		pbar.update(1)
	return np.swapaxes(np.array(STR_subs), 0, 1)

def run_STR_interpol(subjects, partial, save_name = "./corr_results.png"):
	all_raw_results = []
	all_results = []
	all_titles = []
	# for index in range(subjects.shape[2]):
	# 	curr_var = subjects[:, 0, index]
	for index in range(subjects.shape[1]):
		curr_var = subjects[:,index]
		sub_result = []
		sub_result_alt = []
		for sub in curr_var:
			print ("############################  ", end = "")
			print ("subject " + sub.subname, end = "")
			print ("  ############################")
			sub.print_attribute_info()
			control_var = sub.variable_names[1:]
			if partial == True:
				corr_result, var_titles = sub.single_trial_RSA([sub.variable_names[0]], CV = control_var, time_window = 5, step = 2)
			else:	
				corr_result, var_titles = sub.single_trial_RSA([sub.variable_names[0]], time_window = 5, step = 2)
			print (corr_result.shape)
			sub_result.append(corr_result)
			sub_result_alt.append(corr_result[0,:])
		sub_result = np.array(sub_result)
		all_raw_results.append(np.array(sub_result_alt))
		all_results.append(np.average(sub_result, axis = 0)[0,:])
		all_titles.append(sub.variable_names[0])
	plot_corr(np.array(all_results), np.array(all_titles), start_end = (-100, 800), interval = 100, axis = [None, None, -0.02, 0.02], save = True, save_name = save_name)
	return np.array(all_raw_results), all_titles

def create_STR(subjects_PPData, all_var, partial):
	STR_subs = []
	pbar = tqdm(total = len(subjects_PPData))
	for subject in subjects_PPData:
		sub_name = subject.subname
		sub_eeg = subject.clean_eeg
		sub_bin = subject.bins.reshape(len(subject.bins), 1)
		cond = np.ones((sub_bin.shape[0],1), dtype = str)
		Cond_sti = np.hstack((cond, sub_bin))
		Cond_dict = dict({"1":1})
		subs = all_var.create_STR_subject(sub_name, sub_eeg, ALL_VAR, extract_type = "cond_sti", cond_sti = Cond_sti, cond_dict = Cond_dict)
		STR_subs.append(subs)
		pbar.update(1)
	return np.array(STR_subs)

def run_STR(subjects, partial):
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
		if partial == True:    
			corr_result, var_titles = sub.single_trial_RSA(ALL_IV, CV = ALL_CV, time_window = 4, step = 2)
		else:
			corr_result, var_titles = sub.single_trial_RSA(ALL_IV, time_window = 4, step = 2)
		all_subjects.append(corr_result)
		all_corr_results = np.add(all_corr_results, corr_result)
		print ("")
	all_corr_results = all_corr_results/len(subjects)
	end = timer()
	plot_corr(all_corr_results, np.array(ALL_IV), start_end = (-100, 800), interval = 100, save = True)
	print ("execution time: " + str(end - start))
	return np.array(all_subnames), np.array(all_subjects), all_corr_results

# Cluster Permutation
###############################################################################

def run_CPerm(all_subjects, all_variables, save_directory):
	# changing the all_subjects dim from (subject, variable, corr_value) to 
	# (variable, subject, corr_value)
	null_state = np.zeros((all_subjects.shape))
	# scale = np.linspace(0, 790, num = 80, dtype = int)
	scale = np.linspace(300, 490, num = 20, dtype = int)
	# scale = np.linspace(500, 790, num = 30, dtype = int)
	var_tvals = []	
	var_names = []
	means = []
	stds = []
	for index, var_title in enumerate(all_variables):
		var_CPerm, var_matrix, var_pandas_out = CPerm(var_title ,all_subjects[index, :, :], null_state[index, :, :], 500, show_progress = True, pos_label = scale)
		print (var_title)
		print (var_pandas_out)
		var_tvals.append(var_CPerm.Tvals)
		mean, std = var_CPerm.calculate_std_err()
		var_names.append(var_title)
		means.append(mean)
		stds.append(std)
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


main()