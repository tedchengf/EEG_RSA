# Regression_Analysis.py

import numpy as np
from scipy import stats
from tqdm import tqdm

###############################################################################
#						   	subject_regression class		   				  #
###############################################################################

# Note: sub_data must be at least 2 dimensional, with the last dimension containing values for regression analysis. predictors must be 2 dimensional, and the last dimension must match with the last dimension of sub_data (dimension of variable * values). weight_mask must be 2 dimensional (with first dimension being conditions and the second being mask for each condition), and the last dimension must match with the last dimension of sub_data
class subject_regression:
	def __init__(self, sub_data, predictors, weight_mask = None):
		assert type(sub_data) is np.ndarray and len(sub_data.shape) >= 2, "sub_data must be an instance of numpy ndarray with at least 2 dimensions"
		assert type(predictors) is np.ndarray and len(predictors.shape) == 2, "predictors must be an instance of numpy ndarray with exactly 2 dimensions"
		assert sub_data.shape[-1] == predictors.shape[-1], str(sub_data.shape) + " does not match " + str(predictors.shape) + " at the last dimension"
		if weight_mask is not None:
			assert weight_mask.shape[-1] == predictors.shape[-1], str(weight_mask.shape) + " does not match " + str(predictors.shape) + " at the last dimension"
			assert len(weight_mask.shape) == 2, "weight_mask must be an instance of numpy ndarray with exactly 2 dimensions"
		
		# Class Variables
		self.sub_data = sub_data
		self.predictors = predictors
		self.betas = None
		self.weighted_betas = None
		self.weight_mask = None

		if weight_mask is not None:
			self.weight_mask = weight_mask

	def regression(self):
		self.betas = np.array(self.__recursive_regression(self.sub_data))
		if self.weight_mask is None:
			average_x = np.average(self.predictors, axis = 1).reshape(1, -1)
			self.weighted_betas = self.betas * average_x
		else:
			weighted_betas = []
			for mask in self.weight_mask:
				curr_average = np.average(self.predictors[:,mask], axis = 1).reshape(1,-1)
				weighted_betas.append(self.betas * curr_average)
			self.weighted_betas = np.array(weighted_betas)
		return self.betas, self.weighted_betas

	def __recursive_regression(self, subproblem):
		# reach the final subproblem
		if len(subproblem.shape) == 1:
			return np.linalg.lstsq(np.swapaxes(self.predictors, 0, 1), subproblem, rcond = None)[0]		
		# not yet reach the final subproblem
		curr_result = []
		for index in range(subproblem.shape[0]):
			subresult = self.__recursive_regression(subproblem[index])
			curr_result.append(subresult)
		return curr_result

###############################################################################
#						   	  group_regression class		   				  #
###############################################################################

# Todo: test weight_mask feature
class group_regression:
	def __init__(self, name, sub_data, predictors, weight_mask = None):
		# Class Variables
		self.name = name
		self.subjects = []
		self.subject_betas = []
		self.subject_weighted_betas = []
		self.group_beta = []
		self.group_weighted_beta = []

		for index in range(len(sub_data)):
			self.subjects.append(subject_regression(sub_data[index], predictors[index], weight_mask = weight_mask))
		self.subjects = np.array(self.subjects)

	def run_regression(self):
		pbar = tqdm(total = len(self.subjects))
		for sub_regression in self.subjects:
			sub_beta, sub_weighted_beta = sub_regression.regression()
			self.subject_betas.append(sub_beta)
			self.subject_weighted_betas.append(sub_weighted_beta)
			pbar.update(1)
		self.subject_betas = np.array(self.subject_betas)
		self.subject_weighted_betas = np.array(self.subject_weighted_betas)
		self.group_beta = np.average(self.subject_betas, axis = 0)
		self.group_weighted_beta = np.average(self.subject_weighted_betas, axis = 0)
		return self.group_beta, self.group_weighted_beta

###############################################################################
#						   		helpful functions			   				  #
###############################################################################

def recursive_baseline_correction(data, baseline_index):
	# reach the final subproblem
	if len(data.shape) == 1:
		return baseline_correction(data, baseline_index)		
	# not yet reach the final subproblem
	curr_result = []
	for index in range(data.shape[0]):
		subresult = recursive_baseline_correction(data[index], baseline_index)
		curr_result.append(subresult)
	return np.array(curr_result)

# Note: the time should be the first dimension for this function
def baseline_correction(ERP, baseline_index):
	baseline = np.sum(ERP[:baseline_index])/baseline_index
	return ERP - baseline

# Note: the predictors are in the shape of (variable, values)
def z_transform(predictors):
	assert len(predictors.shape) == 2
	z_scores = []
	for row in predictors:
		z_scores.append(stats.zscore(row))
	return np.array(z_scores)

# Note: the predictors are in the shape of (variable, values), and the intercept is added to the front row
def add_intercept(predictors):
	assert len(predictors.shape) == 2
	intercept = np.ones((1, predictors.shape[1]), dtype = float)
	predictors = np.vstack((intercept, predictors))
	return predictors

