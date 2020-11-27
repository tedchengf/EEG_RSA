# Single_Trial_RSA.py

import sys
import numpy as np
from numba import njit, jit
from scipy import stats
from tqdm import tqdm

###############################################################################
#							   Auxiliary Functions							  #
###############################################################################

@njit
def abs_diff(x, y):
	return np.absolute(np.subtract(x, y))

def corrcoef(A, B):
	return np.corrcoef(A, B)[0, 1]

def spearmanr(A, B):
	r, p = stats.spearmanr(A, B)
	return r	

###############################################################################
#								  Subject class		   						  #
###############################################################################

class subject:
	def __init__(self, name, eeg, var_names, var_triangulars, triangulars_modifer = None, verbose = False):
		self.subname = name
		self.eeg = eeg.copy()
		self.variable_names = var_names.copy()
		self.variable_triangulars = var_triangulars.copy()
		self.__triangulars_modifier = None
		self.__verbose_flag = verbose
		if len(self.variable_triangulars.shape) == 1:
			self.variable_triangulars = np.array([self.variable_triangulars])
		if triangulars_modifer is None:
			assert self.variable_triangulars.shape[1] == (eeg.shape[0]*eeg.shape[0]-eeg.shape[0])//2, "Trail dimension of eeg and var_triangulars does not match; please make sure the triangulars are extracted from the correct matrix"
		else:
			self.__triangulars_modifier = triangulars_modifer.copy()

	def print_attribute_info(self):
		print ("subject: " + self.subname)
		print ("EEG: "+str(self.eeg.shape))
		print ("variable_names: "+str(self.variable_names))
		print ("variables triangulars: "+str(self.variable_triangulars.shape))

	def single_trial_RSA(self, IV, CV = None, time_window = None, step = None, padding = True, corrfunc = corrcoef):
		# Calculate EEG dissimilarity
		if self.__verbose_flag == True:
			print ("Calculating EEG dissimilarity")
		if time_window is None:
			eeg_sim=self.__eeg_sim_calculation(self.eeg.copy())
		else:
			eeg_sim=self.__eeg_sim_calculation_STW(self.eeg.copy(),time_window, step, padding)
		# Apply modifier
		if self.__triangulars_modifier is not None:
			if self.__verbose_flag == True:
				print ("Applying modifier")
			new_eeg_sim = np.empty((eeg_sim.shape[0], self.variable_triangulars.shape[1]), dtype = float)
			pbar = tqdm(total = new_eeg_sim.shape[1], disable = not self.__verbose_flag)
			for index, indices in enumerate(self.__triangulars_modifier):
				new_eeg_sim[:, index] = np.average(eeg_sim[:, indices], axis = 1)
				pbar.update(1)
			eeg_sim = new_eeg_sim
		# Calculate correlation
		if self.__verbose_flag == True:
			print ("Calculating correlation values")
		if CV is None:
			all_corr, name = self.__correlation_analysis(eeg_sim, IV, corrfunc)
		else:
			all_corr, name = self.__correlation_analysis_partial(eeg_sim, IV, CV, corrfunc)
		return all_corr, name

	def eeg_sim(self, time_window = None, step = None, padding = True, corrfunc = corrcoef):
		# Calculate EEG dissimilarity
		if self.__verbose_flag == True:
			print ("Calculating EEG dissimilarity")
		if time_window is None:
			eeg_sim=self.__eeg_sim_calculation(self.eeg.copy())
		else:
			eeg_sim=self.__eeg_sim_calculation_STW(self.eeg.copy(),time_window, step, padding)
		return eeg_sim

	def conditional_RSA(self,time_window = None, step = None, padding = False):
		# for each condition
		if time_window is None:
			cond_corr = (1 - self.__eeg_sim_calculation(curr_trails))*(-1)
		else:
			cond_corr = (1 - self.__eeg_sim_calculation_STW(time_window, step, padding))*(-1)
		return cond_corr

#------------------------------- Private Functions ---------------------------#

#								  EEG processing	 						  #

	def __eeg_sim_calculation(self, eeg_data):
		all_sim = []
		# looping through time points
		for index in tqdm(range(eeg_data.shape[2]), disable = not self.__verbose_flag):
			curr_eeg = eeg_data[:,:,index].copy()
			# creating a dissimilarity matrix in the shape of (trail, trail)
			dis_mat = 1 - np.corrcoef(curr_eeg)
			# appending the upper triangular of the dissimilarity matrix to 
			# result
			all_sim.append(extract_upper_triangular(dis_mat))
		return np.array(all_sim)

	def __eeg_sim_calculation_STW(self, eeg_data, time_window, step, padding):
		assert step is not None, "step must be specified in the sliding time window approach"
		assert type(time_window) is int and type(step) is int, "time_window and step must be integers"
		# assert time_window % step == 0, "time_window must be divisible by step"
		
		# creating padding
		if padding == True:
			assert (time_window - 1) % 2 == 0 and time_window >= 3, "if padding is set to True, time_window must be an odd integer greater than or equal to 3"
			padding_len = (time_window - 1) // 2
			empty_data = np.zeros((eeg_data.shape[0], eeg_data.shape[1], padding_len))
			# append padding to the front and the end on time axis
			eeg_data = np.append(empty_data, eeg_data, axis = 2)
			eeg_data = np.append(eeg_data, empty_data, axis = 2)
		
		# looping through time windows
		all_tw_sim = []
		max_time = eeg_data.shape[2] - time_window
		pbar = tqdm(total = max_time//step + 1, disable = not self.__verbose_flag)
		index = 0
		while index <= max_time:
			# update the tqdm meter (cosmetic)
			pbar.update(1)
			# tw has the shape of (trail, channel, time window)
			tw = eeg_data[:,:,index:index + time_window].copy()
			# flattening the third dimension into a 2D array
			tw = tw.reshape(eeg_data.shape[0], (eeg_data.shape[1] * time_window))
			# creating a dissimilarity matrix in the shape of (trail, trail)
			dis_mat = 1 - np.corrcoef(tw)
			# appending the upper triangular of the dissimilarity matrix to 
			# result
			all_tw_sim.append(extract_upper_triangular(dis_mat))
			index = index + step

		return np.array(all_tw_sim)

#								 	   RSA	 								  #

	def __correlation_analysis(self, eeg_sim, IV, corrfunc):
		var_indices = self.__extract_var_indices(IV)
		all_corr = []
		for index in tqdm(range(eeg_sim.shape[0]), disable = not self.__verbose_flag):
			curr_corr = []
			for var_index in var_indices:
				curr_corr.append(corrfunc(eeg_sim[index,:], self.variable_triangulars[var_index]))
			all_corr.append(curr_corr)
		return np.array(all_corr).transpose(), np.take(self.variable_names, var_indices)

	def __correlation_analysis_partial(self, eeg_sim, IV, CV, corrfunc):
		# Calculating Variable Residuals
		IV_indices = self.__extract_var_indices(IV)
		CV_indices = self.__extract_var_indices(CV)
		all_var_residuals = []
		all_control_tri = []
		for IV_index in IV_indices:
			var_tri = self.variable_triangulars[IV_index]
			# remove the current IV from the control triangular
			temp_indices = CV_indices[CV_indices != IV_index]
			control_tri = np.take(self.variable_triangulars, temp_indices, axis = 0)
			var_residual = self.__calculate_residual(var_tri, control_tri)
			all_var_residuals.append(var_residual)
			all_control_tri.append(control_tri)
		
		# Calculating Partial Correlation
		all_corr = []
		for index in tqdm(range(eeg_sim.shape[0]), disable = not self.__verbose_flag):
			curr_eeg = eeg_sim[index,:]
			curr_corr = []
			for var_index in range(len(all_var_residuals)):
				var_res = all_var_residuals[var_index]
				control_tri = all_control_tri[var_index]
				curr_eeg_res = self.__calculate_residual(curr_eeg, control_tri)
				curr_corr.append(corrfunc(curr_eeg_res[:,0], var_res[:,0]))
			all_corr.append(curr_corr)

		return np.array(all_corr).transpose(), np.take(self.variable_names, IV_indices)

	def __calculate_residual(self, var_tri, control_tri):
		var = np.reshape(var_tri, (len(var_tri), 1))
		intercept = np.ones((len(var_tri), 1))
		control_mat = control_tri.transpose()
		control_mat = np.hstack((intercept, control_mat))
		# finding the least squared error solution
		lstsq = np.linalg.lstsq(control_mat, var, rcond=None)[0]
		residual = var - control_mat.dot(lstsq)
		return residual

	def __extract_var_indices(self, var_title):
		assert type(var_title) is list or type(var_title) is np.ndarray, "IV and CV must be either list or np.ndarray"
		indices = []
		for variable in var_title:
			# Variable not found in instance
			assert variable in self.variable_names, "Variable " + variable + " not found in the Subject instance."
			var_index = np.where(self.variable_names == variable)[0][0]
			indices.append(var_index)
		return np.array(indices, dtype = int)

###############################################################################
#							   Support functions		   					  #
###############################################################################

def extract_upper_triangular(input_matrix):
	dim = input_matrix.shape[0]
	array_len = (dim * dim - dim) // 2
	array_result = np.empty(array_len, dtype = float)
	__fast_extract(input_matrix, array_result)
	return array_result

@njit
def __fast_extract(matrix, result):
	dim = matrix.shape[0]
	counter = 0
	for row in range(dim):
		for col in range(row + 1, dim):
			result[counter] = matrix[row, col]
			counter = counter + 1
