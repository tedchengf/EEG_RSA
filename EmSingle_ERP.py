# EmSingle_ERP.py

import numpy as np
import pandas as pd
from PPData import *
from scipy import stats
from scipy import sparse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import variables_processing_alt as VP
import Regression_Analysis as RA
from rERP import *
from RSA_plot import *
from tqdm import tqdm
import Auxiliaries as AUX
from mne.stats import spatio_temporal_cluster_1samp_test
from mne import evoked


VAR_NAME = np.array(['Valence', 'Arousal', 'Concreteness', 'Log_Freq', 'Length', 'Ortho20'])
PREDICTORS = np.array(['Intercept', 'Valence', 'Arousal', 'Concreteness', 'Log_Freq', 'Length', 'Ortho20'])
VAR_NAME_ALT = np.array(['Valence', 'Arousal', 'Concreteness', 'Log_Freq', 'Length', 'Orth_F'])
RESOURCE_DIRECTORY = "./Resources/"
DATA = "../DATA/Emsingle2_REP_labeled_cleaned_WithOutliers_2019.04.09.csv"
ALL_CHANNELS = np.array(['Fz', 'Cz', 'Pz', 'C3', 'FP1', 'F7', 'T3', 'P3', 'O1', 'F3', 'FC5', 'CP5', 'P3', 'FC1', 'CP1', 'FPz', 'FC2', 'CP2', 'Oz', 'F4', 'FC6', 'CP6', 'P4', 'C4', 'FP2', 'F8', 'T4', 'P4', 'O2'])
EEG_START = 'X.100..95'
EEG_END = 'X795.800'
END_COL = 198

CHANNEL = "Fz"

neighbors = [{'Fz': ['Fz', 'FPz', 'FP1', 'FP2', 'F3', 'F4', 'FC1', 'FC2']},{'Cz': ['Cz', 'FC1', 'FC2', 'C3', 'C4', 'CP1', 'CP2']},{'Pz': ['Pz', 'CP1', 'CP2', 'P3', 'P4', 'O1', 'O2', 'Oz']},{'C3': ['C3', 'FC5', 'FC1', 'T3', 'Cz', 'CP5', 'CP1']},{'FP1': ['FP1', 'FPz', 'F7', 'F3', 'Fz']},{'F7': ['F7', 'FP1', 'F3', 'FC5', 'T3']},{'T3': ['T3', 'F7', 'FC5', 'C3', 'CP5', 'T5']},{'P3': ['P3', 'CP5', 'CP1', 'T5', 'Pz', 'O1']},{'O1': ['O1', 'T5', 'P3', 'Pz', 'Oz']},{'F3': ['F3', 'FP1', 'F7', 'Fz', 'FC5', 'FC1']},{'FC5': ['FC5', 'F7', 'F3', 'FC1', 'T3', 'C3']},{'CP5': ['CP5', 'T3', 'C3', 'CP1', 'T5', 'P3']},{'P3': ['P3', 'CP5', 'CP1', 'T5', 'Pz', 'O1']},{'FC1': ['FC1', 'F3', 'Fz', 'FC5', 'FC2', 'C3', 'Cz']},{'CP1': ['CP1', 'C3', 'Cz', 'CP5', 'CP2', 'P3', 'Pz']},{'FPz': ['FPz', 'FP1', 'FP2', 'Fz']},{'FC2': ['FC2', 'Fz', 'F4', 'FC1', 'FC6', 'Cz', 'C4']},{'CP2': ['CP2', 'Cz', 'C4', 'CP1', 'CP6', 'Pz', 'P4']},{'Oz': ['Oz', 'Pz', 'O1', 'O2']},{'F4': ['F4', 'FP2', 'Fz', 'F8', 'FC2', 'FC6']},{'FC6': ['FC6', 'F4', 'F8', 'FC2', 'C4', 'T4']},{'CP6': ['CP6', 'C4', 'T4', 'CP2', 'P4', 'T6']},{'P4': ['P4', 'CP2', 'CP6', 'Pz', 'T6', 'O2']},{'C4': ['C4', 'FC2', 'FC6', 'Cz', 'T4', 'CP2', 'CP6']},{'FP2': ['FP2', 'FPz', 'Fz', 'F4', 'F8']},{'F8': ['F8', 'FP2', 'F4', 'FC6', 'T4']},{'T4': ['T4', 'F8', 'FC6', 'C4', 'CP6', 'T6']},{'P4': ['P4', 'CP2', 'CP6', 'Pz', 'T6', 'O2']},{'O2': ['O2', 'Pz', 'P4', 'T6', 'Oz']}]

def main():
	# # load masks
	# low_arsl = load_instance("low_arsl", RESOURCE_DIRECTORY)
	# high_arsl = load_instance("high_arsl", RESOURCE_DIRECTORY)
	# # load subject EEG
	# all_data, subjects = load_all_subjects(DATA, EEG_START, EEG_END, np.array(VAR_NAME), end_col = END_COL)
	# # calculate ERP
	# results = calculate_ERP(subjects, [low_arsl, high_arsl], baseline_index = 20)
	# # plot results
	# chan_low = results[0, np.argwhere(ALL_CHANNELS == CHANNEL)[0,0]]
	# chan_high = results[1, np.argwhere(ALL_CHANNELS == CHANNEL)[0,0]]

	# Tori_low = load_instance("Tori_low", RESOURCE_DIRECTORY)
	# Tori_low = get_index(Tori_low, CHANNEL, ALL_CHANNELS)
	# Tori_high = load_instance("Tori_high", RESOURCE_DIRECTORY)
	# Tori_high = get_index(Tori_high, CHANNEL, ALL_CHANNELS)

	# print (chan_low - Tori_low)
	# print (chan_high - Tori_high)
	# plot_1D(np.array([chan_low - Tori_low, chan_high - Tori_high]), np.array(["low_arsl", "high_arsl"]), axis = [None, None, -3, 8], start_end = (-100, 900), interval = 100, show = True)

###############################################################################
	# load files
	val_cond = np.loadtxt(RESOURCE_DIRECTORY + "val_cond.txt", dtype = int)
	all_data, subjects = load_all_subjects(DATA, EEG_START, EEG_END, VAR_NAME_ALT, end_col = END_COL)
	
	# traditional ERP
	neg_val = val_cond == 1
	neu_val = val_cond == 2
	pos_val = val_cond == 3
	ERP_result = calculate_ERP(subjects, [neg_val, neu_val, pos_val], baseline_index = 20)
	neg_erp = ERP_result[0]
	neu_erp = ERP_result[1]
	pos_erp = ERP_result[2]

	all_var = load_instance("all_var_alt", RESOURCE_DIRECTORY)
	variables = all_var.export_variables(VAR_NAME)["var_arrays"].transpose()
	predictors = np.ones((variables.shape[0], variables.shape[1]+1), dtype = float)
	for var_ind in range(variables.shape[1]):
		# z_scores = stats.zscore(variables[:, var_ind])
		predictors[:, var_ind + 1] = variables[:, var_ind]

	neg_neu = val_cond < 3
	neg_pos_1 = val_cond == 1 
	neg_pos_2 = val_cond == 3
	neg_pos = neg_pos_1 + neg_pos_2
	neu_pos = val_cond > 1

	neg_mask = val_cond == 1
	neu_mask = val_cond == 2
	pos_mask = val_cond == 3

	# rERP result is in the form of (subject, channel, predictor, timepoints)
	weight_mask = np.array([neg_mask, neu_mask])
	neg_neu_results, neg_neu_weighted = conditional_rERP(neg_neu, predictors, subjects, weight_mask = weight_mask)
	# neg_pos_results, neg_pos_weighted = conditional_rERP(neg_pos, predictors, subjects)
	# neu_pos_results, neu_pos_weighted = conditional_rERP(neu_pos, predictors, subjects)
	# all_results, all_results_weighted = conditional_rERP(np.ones(468, dtype = bool), predictors, subjects)

	# neg_neu_beta=RA.recursive_baseline_correction(neg_neu_results, 20)[:,:,1,:]
	# neg_neu_beta=np.array(np.swapaxes(neg_neu_beta,1,2))
	# neg_pos_beta=RA.recursive_baseline_correction(neg_pos_results, 20)[:,:,1,:]
	# neg_pos_beta=np.array(np.swapaxes(neg_pos_beta,1,2))
	# neu_pos_beta=RA.recursive_baseline_correction(neu_pos_results, 20)[:,:,1,:]
	# neu_pos_beta=np.array(np.swapaxes(neu_pos_beta,1,2))
	
	# beta = all_results
	# curr_beta = RA.recursive_baseline_correction(beta, 20)[:,:,2,:]
	# curr_beta = np.array(np.swapaxes(curr_beta, 1, 2))
	
	# neighbor_dict = {}
	# for chan in neighbors:
	# 	neighbor_dict.update(chan)
	# neighbor_mat = AUX.extract_neighbor_matrix(ALL_CHANNELS, neighbor_dict)
	# s_mat = sparse.csr_matrix(neighbor_mat)

	# result_dict = dict({})
	# result_dict.update({"Channel_names": ALL_CHANNELS})
	# result_dict.update({"Neighbor_matrix": neighbor_mat})
	# result_dict.update({"Neg_Neu_beta": neg_neu_beta})
	# result_dict.update({"Neg_Pos_beta": neg_pos_beta})
	# result_dict.update({"Neu_Pos_beta": neu_pos_beta})
	# import scipy.io
	# scipy.io.savemat('rERP_info.mat', result_dict)

	# t_obs, clusters, cluster_pv, HO = spatio_temporal_cluster_1samp_test(curr_beta, out_type = "mask", connectivity = s_mat, n_permutations = 1024)
	# t_obs = np.swapaxes(t_obs, 0,1)
	# clusters = np.swapaxes(np.array(clusters),1,2)
	# plot_cluster_results(t_obs, clusters, cluster_pv, "Arousal all cond")
	

	# results = load_instance("neg_pou_rERP", RESOURCE_DIRECTORY)
	beta = neg_neu_results
	weighted_results = neg_neu_weighted
	curr_beta = average_rERP(beta, ["Intercept", "Valence"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	curr_beta = np.swapaxes(curr_beta, 0, 1)
	rERP_neg = average_rERP(weighted_results[0], ["Intercept", "Valence"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	rERP_neg = np.swapaxes(rERP_neg, 0, 1)
	rERP_neu = average_rERP(weighted_results[1], ["Intercept", "Valence"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	rERP_neu = np.swapaxes(rERP_neu, 0, 1)

	for ind, chan_name in enumerate(["Fz", "Cz", "Pz", "Oz"]):
		# intercept = rERP[ind, 0, :]
		# val = curr_beta[ind, 1, :]
		# weighted_val = rERP[ind, 1, :]
		# mixed = intercept + weighted_val
		# pos = get_index(pos_erp, chan_name, ALL_CHANNELS)
		# neg = get_index(neg_erp, chan_name, ALL_CHANNELS)
		intercept = rERP_neg[ind, 0, :]
		first_est = rERP_neg[ind, 0, :] + rERP_neg[ind, 1, :]
		second_est = rERP_neu[ind, 0, :] + rERP_neu[ind, 1, :]
		title = "neg_neu_est_" + chan_name
		plot_1D(np.array([intercept, first_est, second_est]), np.array(["Intercept","Estimated negative valence ERP", "Estimated neutral valence ERP"]), start_end = (-100, 900), interval = 100, save = True, save_name = title + ".png")

	# for ind, chan_name in enumerate(["Fz", "Cz", "Pz", "Oz"]):
	# 	intercept = rERP[ind, 0, :]
	# 	arousal = curr_beta[ind, 1, :]
	# 	weighted_arousal = rERP[ind, 1, :]
	# 	mixed = intercept + weighted_arousal
	# 	title = "neg_neu_" + chan_name
	# 	# plot_1D(np.array([intercept, arousal, mixed]), np.array(["Baseline (intercept)", "Valence beta (unweighted)", "Estimated Valence rERP"]), start_end = (-100, 900), interval = 100, save = True, save_name = title + ".png")
	# 	plot_1D(np.array([intercept, mixed]), np.array(["Baseline (intercept)", "Valence beta (unweighted)"]), start_end = (-100, 900), interval = 100, save = True, save_name = title + ".png")

	# plot contrast
	# neg_neu_results = load_instance("neg_neu_rERP", RESOURCE_DIRECTORY)
	# neg_pos_results = load_instance("neg_pos_rERP", RESOURCE_DIRECTORY)
	# neu_pos_results = load_instance("neu_pos_rERP", RESOURCE_DIRECTORY)
	# neg_neu = average_rERP(neg_neu_results, ["Arousal"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	# neg_neu = np.swapaxes(neg_neu, 0, 1)
	# neg_pos = average_rERP(neg_pos_results, ["Arousal"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	# neg_pos = np.swapaxes(neg_pos, 0, 1)
	# neu_pos = average_rERP(neu_pos_results, ["Arousal"], ["Fz", "Cz", "Pz", "Oz"], baseline_index = 20)
	# neu_pos = np.swapaxes(neu_pos, 0, 1)

	# for ind, chan_name in enumerate(["Fz", "Cz", "Pz", "Oz"]):
	# 	curr_neg_neu = neg_neu[ind, 0, :]
	# 	curr_neg_pos = neg_pos[ind, 0, :]
	# 	curr_neu_pos = neu_pos[ind, 0, :]
	# 	zero = np.zeros(180)
	# 	title = "contrast_" + chan_name
	# 	plot_1D(np.array([curr_neg_neu, curr_neg_pos, curr_neu_pos, zero]), np.array(["negative neutral model", "negative positive model", "neutral positive model", "zero"]), start_end = (-100, 900), interval = 100, save = True, save_name = title + ".png")

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

# Calculate ERP
###############################################################################
def calculate_ERP(subjects, conditions, baseline_index = None):
	results = []
	for cond_mask in conditions:
		cond_result = []
		for sub in subjects:
			sub_bin = sub.bins - 1
			sub_mask = cond_mask[sub_bin]
			sub_eeg = sub.clean_eeg[sub_mask]
			# average trial by condition
			sub_erp = np.average(sub_eeg, axis = 0)
			if baseline_index is not None:
				for channel_index in range(sub_erp.shape[0]):
					sub_erp[channel_index] = baseline_correction(sub_erp[channel_index], baseline_index)
			cond_result.append(sub_erp)
		# average condition by subject
		cond_ERP = np.average(np.array(cond_result), axis = 0)
		# baseline correction
		# if baseline_index is not None:
		# 	for channel_index in range(cond_ERP.shape[0]):
		# 		channel_ERP = cond_ERP[channel_index]
		# 		baseline = np.sum(channel_ERP[:baseline_index])/baseline_index
		# 		cond_ERP[channel_index] = channel_ERP - baseline
		results.append(cond_ERP)
	return np.array(results)

def baseline_correction(ERP, baseline_index):
	baseline = np.sum(ERP[:baseline_index])/baseline_index
	return ERP - baseline

# rERP
###############################################################################
def average_rERP(rERP, predictors, channels, baseline_index = None):
	all_results = []
	for pred in predictors:
		curr_result = []
		pred_ind = np.argwhere(PREDICTORS == pred)[0,0]
		curr_beta = rERP[:,:,pred_ind,:]
		for chan in channels:
			chan_ind = np.argwhere(ALL_CHANNELS == chan)[0,0]
			chan_beta = curr_beta[:, chan_ind, :]
			if baseline_index is not None:
				# correct basline for each subject
				for sub_ind in range(chan_beta.shape[0]):
					chan_beta[sub_ind] = baseline_correction(chan_beta[sub_ind], baseline_index)
			chan_beta = np.average(chan_beta, axis = 0)
			curr_result.append(chan_beta)
		curr_result = np.array(curr_result)
		all_results.append(curr_result)
	return np.array(all_results)

def conditional_rERP(mask, predictors, subjects, weight_mask = None):
	results = []
	weighted_results = []
	sub_predictors = []
	pbar = tqdm(total = len(subjects))
	for sub in subjects:
		if weight_mask is None:
			sub_x, sub_eeg = get_conditions(mask, predictors, sub)
			sub_x = z_transform(sub_x)	
			sub_regression = RA.subject_regression(np.swapaxes(sub_eeg,0,2), np.swapaxes(sub_x,0,1))
			beta, weighted_beta = sub_regression.regression()
			results.append(np.swapaxes(np.swapaxes(beta,0,1),1,2))
			weighted_results.append(np.swapaxes(np.swapaxes(weighted_beta,0,1),1,2))
			pbar.update(1)
		else:
			sub_x, sub_eeg, curr_weight_mask = get_conditions(mask, predictors, sub, weight_mask = weight_mask)
			sub_x = z_transform(sub_x)	
			sub_regression = RA.subject_regression(np.swapaxes(sub_eeg,0,2), np.swapaxes(sub_x,0,1), weight_mask = curr_weight_mask)
		beta, weighted_beta = sub_regression.regression()
		results.append(np.swapaxes(np.swapaxes(beta,0,1),1,2))
		weighted_results.append(np.swapaxes(np.swapaxes(weighted_beta,1,2),2,3))
		pbar.update(1)
	return np.array(results), np.swapaxes(np.array(weighted_results),0,1)

# def conditional_rERP(mask, predictors, subjects):
# 	results = []
# 	weighted_results = []
# 	sub_predictors = []
# 	pbar = tqdm(total = len(subjects))
# 	for sub in subjects:
# 		sub_x, sub_eeg = get_conditions(mask, predictors, sub)
# 		sub_x = z_transform(sub_x)
# 		sub_beta = rERP(sub_eeg, sub_x)
# 		# print (np.swapaxes(sub_eeg,0,2).shape)
# 		# sub_regression = RR.regression_subject(np.swapaxes(sub_eeg,0,2), np.swapaxes(sub_x,0,1))
# 		# sub_beta = sub_regression.regression()
# 		# sub_beta = np.swapaxes(np.swapaxes(sub_beta,0,1), 1,2)
# 		# print(sub_beta.shape)
# 		averaged_x = np.average(sub_x, axis = 0)
# 		# swap to (channel,time,predictor) to multiply with (1, predictor), then swapping back to (channel, predictor, time)
# 		weighted_beta = np.swapaxes(np.swapaxes(sub_beta,1,2)*averaged_x, 1, 2)
# 		results.append(sub_beta)
# 		weighted_results.append(weighted_beta)
# 		pbar.update(1)
# 	return np.array(results), np.array(weighted_results)

def plot_cluster_results(t_obs, clusters, cluster_pv, group_name):
	print (cluster_pv)
	for ind in range(len(cluster_pv)):
		if cluster_pv[ind] <= 0.1:
			curr_cluster = clusters[ind]
			cluster_tval, channel_mask, sum_tval = mask_2D_results(t_obs, curr_cluster)
			if sum_tval < 0: clim = (0, -5)
			else: clim = (0, 5)
			plot_corr_2D(cluster_tval, "p = " + str(np.round(cluster_pv[ind], decimals = 3)) + "; t = " + str(np.round(sum_tval, decimals = 3)), y_label = ALL_CHANNELS[channel_mask], clim = clim, start_end = (-100, 800), interval = 100, show = False, save = True, save_name = group_name + " cluster num " + str(ind) + ".png")

def mask_2D_results(t_obs, cluster_mask):
	results = []
	sum_tval = 0
	selected_rows = np.zeros(t_obs.shape[0], dtype = bool)
	for row_ind in range(t_obs.shape[0]):
		new_row = np.zeros(t_obs.shape[1])
		for col_ind in range(t_obs.shape[1]):
			if cluster_mask[row_ind, col_ind] == True:
				selected_rows[row_ind] = True
				new_row[col_ind] = t_obs[row_ind, col_ind]
				sum_tval += t_obs[row_ind, col_ind]
			else:
				new_row[col_ind] = np.nan
		if selected_rows[row_ind] == True:
			results.append(new_row)
	return np.array(results), selected_rows, sum_tval

def get_conditions(mask, data, sub, weight_mask = None):
	indices = sub.bins - 1
	mask = mask[indices]
	data = data[indices]
	eeg = sub.clean_eeg
	if weight_mask is None:
		return data[mask], eeg[mask]
	else:
		weight_mask = weight_mask[:,indices]
		return data[mask], eeg[mask], weight_mask[:,mask]

def z_transform(predictors):
	assert len(predictors.shape) == 2
	z_scores = np.ones(predictors.shape)
	for col in range(1, predictors.shape[1]):
		curr_z = stats.zscore(predictors[:, col])
		z_scores[:, col] = curr_z
	return z_scores

# Other functions
###############################################################################
def get_index(data, channel, channel_labels):
	return data[np.argwhere(channel_labels == channel)[0,0]]

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

def plot_1D(Data, names, start_end = None, interval = 100, axis = [None, None, None, None], highlight_intervals = None, show = False, save = False, save_name = "./corr_results.png"):
	assert type(Data) is np.ndarray and len(Data.shape) == 2, "Data must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type(names) is np.ndarray, "names must be an instance of numpy.ndarray"
	assert len(axis) == 4, "axis must have exactly 4 elements"
	matplotlib.rcParams.update({'font.size': 6})
	for var_index in range(len(names)):
		var_name = names[var_index]
		corr_results = Data[var_index]
		plt.plot(corr_results, label = var_name)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	if highlight_intervals is not None:
		assert type(highlight_intervals) is np.ndarray and len(highlight_intervals.shape) == 2, "highlight_intervals must be an instance of numpy.ndarray with exactly 2 dimensions"
		for x_interval in highlight_intervals:
			plt.axvspan(x_interval[0], x_interval[1], color = 'red', alpha = 0.4)
	plt.axis(axis)
	plt.gca().invert_yaxis()
	plt.legend()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

if __name__ == "__main__":
	main()