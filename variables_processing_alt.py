# variables_processing.py

import gensim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import Single_Trial_RSA as STR
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from numba import njit, jit
from scipy import spatial
from scipy import stats
from tqdm import tqdm

###############################################################################
#							  Auxiliary Functions							  #
###############################################################################

def nearst_neighbor_1D(stimuli_array, max_diff = 0, range_type = "percentage", stim_val = "average"):
	assert type(stimuli_array) is np.ndarray and len(stimuli_array.shape) == 1, "The input stimuli_array must be an instance of numpy.ndarray with exactly 1 dimension"
	assert range_type in ("raw", "percentage"), "The input range_type must be one from (raw, percentage)"
	assert stim_val in ("central", "average"), "The input stim_val must be one from (cenetral, average)"
	unique_val = np.unique(stimuli_array)
	unique_dict = dict()
	# building dictionary
	for val in unique_val:
		unique_dict.update({val: []})
	# updating dictionary
	for index, elem in enumerate(stimuli_array):
		unique_dict[elem].append(index)
	if max_diff == 0:
		NN_dict = dict()
		for val in unique_val:
			final_ind = unique_dict[val]
			final_mask = np.zeros(stimuli_array.shape, dtype = bool)
			final_mask[final_ind] = True
			NN_dict.update({val: (final_mask, np.array(final_ind))})
		return NN_dict
	else:
		val_range = unique_val[-1] - unique_val[0]
		if range_type == "raw":
			assert max_diff <= val_range, "The input max_diff cannot be greater than the range of stimuli_array under raw range_type "
			diff = max_diff
		else:
			assert 0 <= max_diff and max_diff <= 100, "The input max_diff cannot be smaller than 0 or greater than 100 under percentage range_type "
			diff = val_range * (max_diff / 100)
		NN_dict = dict()
		for val in unique_val:
			left_ind = np.searchsorted(unique_val, val - diff, side = 'left')
			right_ind = np.searchsorted(unique_val, val + diff, side = 'right')
			# No neighbor in range
			if left_ind == right_ind:
				final_ind = np.array(unique_dict[val]) 
				final_mask = np.zeros(stimuli_array.shape, dtype = bool)
				final_mask[final_ind] = True
				NN_dict.update({val: (final_mask, np.array(final_ind))})
			# More than 1 neighbor in range
			else:
				matched_ind = []
				all_val = 0
				for index in range(left_ind, right_ind):
					# Do not allow array to run out of index
					if (index < len(unique_val)):
						curr_val = unique_val[index]
						matched_ind.append(unique_dict[curr_val])
						all_val = all_val + curr_val*len(unique_dict[curr_val])
				final_ind = np.concatenate(matched_ind)
				final_mask = np.zeros(stimuli_array.shape, dtype = bool)
				final_mask[final_ind] = True
				if stim_val == "average":
					final_val = all_val / len(final_ind) 
				else:
					final_val = val
				NN_dict.update({final_val: (final_mask, np.array(final_ind))})
		return NN_dict

@njit
def abs_diff(x, y):
	return np.absolute(np.subtract(x, y))

def eculidian_dist(x, y):
	return np.square(np.subtract(x, y))

def z_squared_transform(var_array):
	z_scores = stats.zscore(var_array)
	return np.square(z_scores)

def spearmanr(A, B):
	r, p = stats.spearmanr(A, b=B)
	return r	

def matrix_iteration(data_array, target_matrix, function, skip_diagonal=True):
	mat_dim = target_matrix.shape[0]
	if skip_diagonal == True:
		for r in range(mat_dim):
			for c in range(r+1, mat_dim):
				target_matrix[r,c] = function(data_array[r], data_array[c])
				target_matrix[c,r] = target_matrix[r,c]
	else:
		for r in range(mat_dim):
			for c in range(r, mat_dim):
				target_matrix[r,c] = function(data_array[r], data_array[c])
				target_matrix[c,r] = target_matrix[r,c]

###############################################################################
#						   variable_processing class		   				  #
###############################################################################

class variables:
	def __init__(self, words):
		self.words = np.array(words, dtype = str)
		self.variable_names = None
		self.variable_arrays = None
		self.variable_matrices = None
		self.variable_triangulars = None
		self.masks_dict = None
		self.__word_dict = None

	def print_attribute_info(self):
		if self.variable_names is None:
			print ("uninitialized")
		else:
			print ("variables: " + str(self.variable_names))
			print ("variable arrays: " + str(self.variable_arrays.shape))
			print ("variable matrices: " + str(self.variable_matrices.shape))
			print ("variable triangulars: " + str(self.variable_triangulars.shape))
		if self.masks_dict is not None:
			print ("Masks:")
			for key in self.masks_dict:
				print ("  " + key + ": ")
				for label in self.masks_dict[key]:
					print ("    " + str(label))

	def update_variable(self, var_name, var_array = None, var_matrix = None, Tfunction = None, Dfunction = abs_diff):
		assert var_array is not None or var_matrix is not None, "Either var_array or var_matrix has to be specified"
		assert type(var_name) is str, "var_name must be a string"

		if var_array is not None:
			assert type(var_array) is np.ndarray and len(var_array.shape) == 1, "The input var_array must be an instance of numpy.ndarray with exactly 1 dimension"
			assert var_array.shape[0] == self.words.shape[0], "The dimension of var_array does not match the dimension of words"
			if Tfunction is not None:
				var_array = Tfunction(var_array)
			var_matrix = self.__matrix_calculation(var_array, Dfunction)
		else:
			assert type(var_matrix) is np.ndarray and len(var_matrix.shape) == 2, "The input var_matrix must be an instance of numpy.ndarray with exactly 2 dimensions"
			assert var_matrix.shape[0] == self.words.shape[0], "The row/col of var_matrix does not match the dimension of words"
			var_array = np.empty((var_matrix.shape[0]))
			var_array[:] = np.NaN 

		var_triangular = extract_upper_triangular(var_matrix)
		self.__update_var(var_name, var_array, var_matrix, var_triangular)

	def update_mask(self, mask_name, mask_source):
		if type(mask_source) is np.ndarray:
			assert type(mask_source) is np.ndarray and len(mask_source.shape) == 1, "The mask_source must be an instance of numpy.ndarray with exactly 1 dimension"
			assert mask_source.shape[0] == self.words.shape[0], "The dimension of mask_source does not match the dimension of words"
			mask_labels = np.unique(mask_source)
			curr_mask = dict()
			for label in mask_labels:
				label_mask = mask_source == label
				curr_mask.update({label: label_mask})
			if self.masks_dict is None:
				self.masks_dict = {mask_name: curr_mask}
			else:
				self.masks_dict.update({mask_name: curr_mask})

	def __extract_mask(self, mask):
		assert type(mask) is np.ndarray and len(mask.shape) == 1, "The mask must be an instance of numpy.ndarray with exactly 1 dimension"
		assert mask.shape[0] == self.words.shape[0], "The dimension of mask does not match the dimension of words"
		mask_labels = np.unique(mask)
		# print(mask_labels)
		all_conditions = dict()
		for label in mask_labels:
			if label == False: continue
			label_mask = mask == label
			all_conditions.update({label: label_mask})
			# print (all_conditions)
		return all_conditions

	def export_variables(self, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None):
		var_indices, var_arrays, var_matrices, var_triangulars = self.__extract_var(var_names)
		indices = self.__determine_indices(extract_type, cond_sti, cond_dict, words)
		if mask is not None:
			assert type(mask) is str or type(mask) is np.ndarray, "mask must be either a string or an instance of numpy.ndarray"
			if type(mask) is np.ndarray:
				curr_dict = self.__extract_mask(mask)
			if type(mask) is str:
				assert self.masks_dict is not None, "masks_dict uninitialized; please use update_mask() to create mask_dict for this instance"
				assert mask in self.masks_dict, "mask not found"
				curr_dict = self.masks_dict[mask]
			result = []
			for label in curr_dict:
				empty_conditions = []
				label_mask = curr_dict[label]
				label_indices = self.__masking_indices(indices, label_mask)
				# empty condition for this subject
				if not label_indices:
					empty_conditions.append(label)
					continue
				curr_var_arrays, curr_var_matrices, curr_var_triangulars = self.__extract_by_index(label_indices, var_arrays, var_matrices, var_triangulars)
				label_dict = {"label": label, "label_mask": label_mask, "var_names": var_names, "indices": label_indices, "var_arrays": curr_var_arrays, "var_matrices": curr_var_matrices, "var_triangulars": curr_var_triangulars}
				result.append(label_dict)
			return result, empty_conditions
		else:
			var_arrays, var_matrices, var_triangulars = self.__extract_by_index(indices, var_arrays, var_matrices, var_triangulars)
			result_dict = {"var_names": var_names, "indices": indices, "var_arrays": var_arrays, "var_matrices": var_matrices, "var_triangulars": var_triangulars}
			return result_dict

	# def create_STR_subject(self, sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm", CV = None):
	# 	assert type(eeg) is np.ndarray and len(eeg.shape) == 3, "eeg must be an instance of numpy.ndarray with exactly 3 dimensions"
	# 	assert stim_val_type in ("dsm","raw")
	# 	if CV is not None: assert type(CV) is np.ndarray and len(CV.shape) == 1, "When specified, CV must be an instance of numpy.ndarray with exactly 1 dimension"
	# 	if interpolation == False:
	# 		result = self.export_variables(var_names, extract_type = extract_type, cond_sti = cond_sti, cond_dict = cond_dict, words = words)
	# 		subject = STR.subject(sub_name, eeg, result["var_names"], result["var_triangulars"])
	# 		return subject
	# 	else:
	# 		result = self.export_variables(var_names)
	# 		indices = self.__determine_indices(extract_type, cond_sti, cond_dict, words)
	# 		subjects = []
	# 		CV_dict = None
	# 		if CV is not None:
	# 			CV_dict = self.export_variables(CV)
	# 		for var_ind in range(len(result['var_names'])):
	# 			var_name = result["var_names"][var_ind]
	# 			var_arr = result["var_arrays"][var_ind]
	# 			var_matrix = result["var_matrices"][var_ind]
	# 			subjects.append(self.__interpolate_EEG(sub_name, eeg, indices, var_name, var_arr, var_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict = CV_dict))
	# 		return subjects

	def create_STR_subject(self, sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask_name = None, interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm", CV = None):
		assert type(eeg) is np.ndarray and len(eeg.shape) == 3, "eeg must be an instance of numpy.ndarray with exactly 3 dimensions"
		assert stim_val_type in ("dsm","raw")
		if mask_name is not None:
			var_dicts = self.export_variables(var_names, extract_type = extract_type, cond_sti = cond_sti, cond_dict = cond_dict, words = words, mask = mask_name)[0]
			unmasked_indices = self.__determine_indices(extract_type, cond_sti, cond_dict, words)
			subjects = dict()
			for condition in var_dicts:
				cond_sti = np.array(condition["indices"]) + 1
				cond_mask = condition["label_mask"]
				cond_eeg = eeg[cond_mask[unmasked_indices]]
				cond_sub = self.create_STR_subject(sub_name + " " + str(condition["label"]), cond_eeg, var_names, extract_type = "cond_sti", cond_sti = cond_sti, interpolation = interpolation, max_diff = max_diff, range_type = range_type, stim_val = stim_val, stim_val_type = stim_val_type, CV = CV)
				subjects.update({condition["label"]: cond_sub})
			return subjects
		if CV is not None: assert type(CV) is np.ndarray and len(CV.shape) == 1, "When specified, CV must be an instance of numpy.ndarray with exactly 1 dimension"
		if interpolation == False:
			result = self.export_variables(var_names, extract_type = extract_type, cond_sti = cond_sti, cond_dict = cond_dict, words = words)
			subject = STR.subject(sub_name, eeg, result["var_names"], result["var_triangulars"])
			return subject
		else:
			result = self.export_variables(var_names)
			indices = self.__determine_indices(extract_type, cond_sti, cond_dict, words)
			subjects = []
			CV_dict = None
			if CV is not None:
				CV_dict = self.export_variables(CV)
			for var_ind in range(len(result['var_names'])):
				var_name = result["var_names"][var_ind]
				var_arr = result["var_arrays"][var_ind]
				var_matrix = result["var_matrices"][var_ind]
				subjects.append(self.__interpolate_EEG(sub_name, eeg, indices, var_name, var_arr, var_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict = CV_dict))
			return subjects

	def create_STR_subject_tri(self, sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None, decimals = None, CV = None, max_diff = 0, range_type = "percentage", stim_val = "average"):
		if mask is None:
			IV_dict = self.export_variables(var_names, extract_type = extract_type, cond_sti = cond_sti, cond_dict = cond_dict, words = words)
			CV_dict = None
			if CV is not None:
				CV_dict = self.export_variables(CV, extract_type = extract_type,cond_sti = cond_sti, cond_dict = cond_dict, words = words)
			subjects = self.__interpolate_EEG_tri(sub_name, eeg, IV_dict, CV_dict, max_diff, range_type, stim_val, decimals = decimals)
			return subjects 
		else:
			indices = self.__determine_indices(extract_type, cond_sti, cond_dict, words)
			label_IV_dict = self.export_variables(var_names, extract_type = extract_type, cond_sti = cond_sti, cond_dict = cond_dict, words = words, mask = mask)[0]
			CV_dict = None
			if CV is not None:
				CV_dict = self.export_variables(CV, extract_type = extract_type,cond_sti = cond_sti, cond_dict = cond_dict, words = words, mask = mask)[0]
			conditions = []
			for index in range(len(label_IV_dict)):
				IV_dict = label_IV_dict[index]
				curr_label = IV_dict['label']
				label_mask = IV_dict["label_mask"]
				label_eeg = eeg[label_mask[indices]]
				if CV_dict is not None: curr_CV_dict = CV_dict[index]
				curr_subs = self.__interpolate_EEG_tri(sub_name, label_eeg, IV_dict, curr_CV_dict, max_diff, range_type, stim_val, decimals = decimals)
				conditions.append(curr_subs)
			return np.array(conditions)

	def calculate_variables_correlation(self, var_names, control_var = None, corr_type = "dsm", interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm"):
		assert corr_type in ("dsm", "raw"), "invalid corr_type: must be one of the following two: (dsm, raw)"
		var_indices, var_arrays, var_matrices, var_triangulars = self.__extract_var(var_names)
		if interpolation == False:
			if corr_type == "raw":
				if control_var is None:
					return np.corrcoef(var_arrays)
				else:
					return self.__var_partial_correlation(var_names, control_var, self.variable_arrays)
			else:
				if control_var is None:
					# return np.corrcoef(var_triangulars)
					target_matrix = np.ones((var_triangulars.shape[0], var_triangulars.shape[0]))
					matrix_iteration(var_triangulars, target_matrix, spearmanr)
					return target_matrix
				else:		
					return self.__var_partial_correlation(var_names, control_var, self.variable_triangulars)
		else:
			return self.__var_correlation_interpolation(var_names, control_var, max_diff, range_type, stim_val, stim_val_type)

	# TODO: Implement control_var option
	def calculate_variables_correlation_interpol(self, var_names, partial = True, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm"):
		result = self.export_variables(var_names) 
		var = []
		control = []
		for var_ind in range(len(result['var_names'])):
			var_name = result["var_names"][var_ind]
			print (var_name)
			var_arr = result["var_arrays"][var_ind]
			interpolation_dict = nearst_neighbor_1D(var_arr, max_diff = 0, range_type = "percentage", stim_val = "average")
			new_arr = np.array(list(interpolation_dict.keys()))
			final_ind = []
			# interpolate eeg data
			for key in new_arr:
				final_ind.append(np.array(interpolation_dict[key][1]))
			if stim_val_type == "dsm":
				new_var_names, new_tris = self.__assemble_tri_mat(final_ind,np.array([var_name]), result["var_matrices"][var_ind])
				other_var_names, other_tris = self.__assemble_tri_mat(final_ind, np.delete(result["var_names"], var_ind), np.delete(result["var_matrices"], var_ind, axis = 0))
			else:
				new_var_names, new_tris = self.__assemble_tri_arr(final_ind,np.array([var_name]), result["var_arrays"][var_ind])
				other_var_names, other_tris = self.__assemble_tri_arr(final_ind, np.delete(result["var_names"], var_ind), np.delete(result["var_arrays"], var_ind, axis = 0))
			var.append(new_tris)
			control.append(other_tris)
		return result['var_names'], self.__var_partial_1vA(var, control, partial)

	def check_missing_words(self, words, sim_type = "w2v", resource_directory = "./"):
		assert sim_type in ("w2v", "lsa", "glove", "random_walk"), "invalid sim_type; must be one of the following four: (w2v, lsa, glove, random_walk)"

		if sim_type == "w2v":
			print ("loading: " + resource_directory + "GoogleNews-vectors-negative300.bin")
			vocab = gensim.models.KeyedVectors.load_word2vec_format(resource_directory + "GoogleNews-vectors-negative300.bin", binary=True).vocab
		elif sim_type == "lsa":
			print ("loading: " + resource_directory + "wiki_en_Nov2019")
			vocab = gensim.models.LsiModel.load(resource_directory + "wiki_en_Nov2019").id2word.values()
		elif sim_type == "glove":
			print ("loading: " + resource_directory + "gensim_glove_vectors.txt")
			vocab = gensim.models.KeyedVectors.load_word2vec_format(resource_directory + "gensim_glove_vectors.txt", binary=False).vocab
		
		bad_words = []
		bad_words_indices = []
		pbar = tqdm(total = len(words))
		for index, word in enumerate(words):
			pbar.update(1)
			if word not in vocab:
				bad_words.append(word)
				bad_words_indices.append(index)
		percentage = (len(bad_words) / len(words)) * 100
		if len(bad_words) == 0:
			print ("\nNo missing words")
		else:
			print ("\nFound " + str(len(bad_words)) + " missing words; occupying " + str(percentage) + "%" + " of all words")
		return bad_words, bad_words_indices

	def semantic_disimilarilty(self, words, sim_type = "w2v", update = True, resource_directory = "./"):
		assert sim_type in ("w2v", "lsa", "glove", "random_walk"), "invalid sim_type; must be one of the following four: (w2v, lsa, glove, random_walk)"
		if sim_type == "w2v":
			var_matrix = self.__w2v_similarity(words, resource_directory + "GoogleNews-vectors-negative300.bin")
		elif sim_type == "lsa":
			var_matrix = self.__LSA_similarity(words, resource_directory + "wiki_en_Nov2019")
		elif sim_type == "glove":
			var_matrix = self.__glove_similarity(words, resource_directory + "gensim_glove_vectors.txt")

		var_array = np.empty((var_matrix.shape[0]))
		var_array[:] = np.NaN 		
		var_triangular = extract_upper_triangular(var_matrix)
		if update == True:
			self.__update_var(sim_type, var_array, var_matrix, var_triangular)
		return var_matrix

	def impute_missing_values(self, var_names, impute_target = "tri", update = True):
		assert impute_target in ("tri","arr"), "invalid impute_target: must be one from (tri, arr)"
		var_indices, var_arrays, var_matrices, var_triangulars = self.__extract_var(var_names)
		if impute_target == "tri":
			dataset = var_triangulars
		else:
			dataset = var_arrays
		imp = sklearn.impute.IterativeImputer(imputation_order='random')
		result = imp.fit_transform(dataset)
		if update == True:
			if impute_target == "tri":
				self.variable_triangulars[var_indices] = result
			else:
				self.variable_arrays[var_indices] = result
		return result

	def delete_trials(self, indices = None, words = None):
		assert indices is not None or words is not None, "indices or words cannot be unspecfied at the same time"
		assert indices is None or words is None, "indices and words cannot be both specified at the same time"
		assert self.variable_names is not None, "Variable uninitialized"
		if words is not None:
			indices = self.__find_words_indices(words)
		# initialize class variables
		self.variable_triangulars = None
		self.masks_dict = None
		self.__word_dict = None
		original_ind = np.arange(len(self.words), dtype = int)
		self.words = np.delete(self.words, indices)
		self.variable_arrays = np.delete(self.variable_arrays, indices, axis=1)
		mat_ind = np.delete(original_ind, indices)
		new_mat = []
		new_tri = []
		for mat in self.variable_matrices:
			curr_mat = mat[np.ix_(mat_ind, mat_ind)]
			new_mat.append(curr_mat)
			new_tri.append(extract_upper_triangular(curr_mat))
		self.variable_matrices = np.array(new_mat)
		self.variable_triangulars = np.array(new_tri)
		return

#------------------------------- Private Functions ---------------------------#

	def __matrix_calculation(self, vector, Dfunction):
		dim = len(vector)
		comp_vector = vector.copy()
		matrix = []
		matrix = Dfunction(comp_vector[0], comp_vector)
		# comparing all elements of the vector with one element at a time
		# the first element is seperated from the loop purely because a list
		# has to be defined to use np.vstack
		for index in range(1, dim):
			line = Dfunction(comp_vector[index], comp_vector)
			# functions the same as append
			matrix = np.vstack((matrix, line))
		return matrix

	def __w2v_similarity(self, words, corpus):
		print ("loading: " + corpus)
		wv_model = gensim.models.KeyedVectors.load_word2vec_format(corpus, binary=True)
		vocab = wv_model.vocab.keys()
		results = np.ones((len(words), len(words)))
		bad_words_count = 0
		pbar = tqdm(total = (len(words)*len(words) - len(words))//2)
		for row_ind in range(0, len(words)):
			for col_ind in range(row_ind+1, len(words)):
				pbar.update(1)
				try:
					results[row_ind, col_ind] = wv_model.similarity(words[row_ind], words[col_ind])
				except KeyError:
					results[row_ind, col_ind] = np.NaN
					bad_words_count += 1
				results[col_ind, row_ind] = results[row_ind, col_ind]
		if bad_words_count > 0:
			percentage = (bad_words_count / ((len(words)*len(words) - len(words))//2)) * 100
			print (str(bad_words_count) + " word pairs not found, occupying " + str(percentage) + "%" + " of total word pairs")
		return 1 - results

	def __glove_similarity(self, words, corpus):
		print ("loading: " + corpus)
		glove_model = gensim.models.KeyedVectors.load_word2vec_format(corpus, binary=False)
		vocab = glove_model.vocab.keys()
		results = np.ones((len(words), len(words)))
		bad_words_count = 0
		pbar = tqdm(total = (len(words)*len(words) - len(words))//2)
		for row_ind in range(0, len(words)):
			for col_ind in range(row_ind+1, len(words)):
				pbar.update(1)
				try:
					results[row_ind, col_ind] = glove_model.similarity(words[row_ind], words[col_ind])
				except KeyError:
					results[row_ind, col_ind] = np.NaN
					bad_words_count += 1
				results[col_ind, row_ind] = results[row_ind, col_ind]
		if bad_words_count > 0:
			percentage = (bad_words_count / ((len(words)*len(words) - len(words))//2)) * 100
			print (str(bad_words_count) + " word pairs not found, occupying " + str(percentage) + "%" + " of total word pairs")
		return 1 - results

	def __LSA_similarity(self, words, model_path):
		print ("loading: " + model_path)
		lsa_model = gensim.models.LsiModel.load(model_path)
		results = np.ones((len(words), len(words)))
		bad_words_count = 0
		pbar = tqdm(total = (len(words)*len(words) - len(words))//2)
		for row_ind in range(0, len(words)):
			for col_ind in range(row_ind+1, len(words)):
				pbar.update(1)
				# print (words[row_ind] + " vs. " + words[col_ind])
				try: 
					results[row_ind, col_ind] = self.__LSA_wordsim(words[row_ind], words[col_ind], lsa_model.id2word, lsa_model)
				except RuntimeError:
					results[row_ind, col_ind] = np.NaN
					bad_words_count += 1
				results[col_ind, row_ind] = results[row_ind, col_ind]
		if bad_words_count > 0:
			percentage = (bad_words_count / ((len(words)*len(words) - len(words))//2)) * 100
			print (str(bad_words_count) + " word pairs not found, occupying " + str(percentage) + "%" + " of total word pairs")		
		return 1 - results

	# Source: https://github.com/a-paxton/Gensim-LSI-Word-Similarities
	def __LSA_wordsim(self,word1,word2,target_dictionary,target_lsi_model):
		# look up each word in the LSA/LSI model of choice
		vec_bow1 = target_dictionary.doc2bow([word1])
		vec_bow2 = target_dictionary.doc2bow([word2])

		# make sure both words are actually in the topic space
		if len(vec_bow1) > 0 and len(vec_bow2) > 0:
			# if they are, go ahead and find their values in the "num_topic"-dimensional space created from gensim.models.LsiModel
			vec_lsi1 = pd.DataFrame(target_lsi_model[vec_bow1],columns=['dim','val'])
			vec_lsi2 = pd.DataFrame(target_lsi_model[vec_bow2],columns=['dim','val'])
			return -1*spatial.distance.cosine(vec_lsi1['val'],vec_lsi2['val'])+1 # snippet from Rick Dale 
		# if the word isn't in the topic space, kick back an error
		else:
			raise RuntimeError('Word pair not found in topic space: '+str(word1)+','+str(word2)+'.')

	def __var_partial_correlation(self, var_names, control_names, input_array):
		all_var_indices = self.__extract_var(var_names)[0]
		all_con_indices = self.__extract_var(control_names)[0]

		partial_corr_matrix = np.ones((all_var_indices.shape[0], all_var_indices.shape[0]))
		for i in range(all_var_indices.shape[0]):
			curr_var_index = all_var_indices[i]
			curr_var_array = input_array[curr_var_index]
			curr_con_indices = all_con_indices[all_con_indices != curr_var_index]
			for j in range(i + 1, all_var_indices.shape[0]):
				next_var_index = all_var_indices[j]
				next_var_array = input_array[next_var_index]
				con_indices = curr_con_indices[curr_con_indices != next_var_index]
				con_array = input_array[con_indices, :]
				curr_res = self.__calculate_residue(curr_var_array, con_array)
				next_res = self.__calculate_residue(next_var_array, con_array)
				corr_value = np.corrcoef(curr_res[:,0], next_res[:,0])[0, 1]
				partial_corr_matrix[i,j] = corr_value
				partial_corr_matrix[j,i] = corr_value
		return partial_corr_matrix

	def __var_correlation_interpolation(self, IV, CV, max_diff, range_type, stim_val, stim_val_type):
		var_dict = self.export_variables(IV)
		indices = np.arange(var_dict["var_arrays"][0].shape[0])
		CV_dict = None
		if CV is not None:
			CV_dict = self.export_variables(CV)
		corr_result = []
		for var_ind in range(len(var_dict['var_names'])):
			curr_result = [1]
			var_name = var_dict["var_names"][var_ind]
			var_arr = var_dict["var_arrays"][var_ind]
			var_matrix = var_dict["var_matrices"][var_ind]
			final_ind, new_names, curr_tri = self.__variable_interpolation(indices, var_name, var_arr, var_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict = CV_dict)
			if stim_val_type == "dsm":
				other_var_names, other_tris = self.__assemble_tri_mat(final_ind, np.delete(var_dict["var_names"], var_ind), np.delete(var_dict["var_matrices"], var_ind, axis = 0))
			else:
				other_var_names, other_tris = self.__assemble_tri_arr(final_ind, np.delete(var_dict["var_names"], var_ind), np.delete(var_dict["var_arrays"], var_ind,axis = 0))
			if len(other_tris.shape) == 1:
				other_tris = np.array([other_tris])
			if CV is None:
				for tri in other_tris:
					curr_result.append(np.corrcoef(curr_tri, tri)[0,1])
			else:
				control_tris = curr_tri[1:,:]
				curr_tri = curr_tri[0, :]
				for tri in other_tris:
					A_res = self.__calculate_residue(curr_tri, control_tris)
					B_res = self.__calculate_residue(tri, control_tris)
					curr_result.append(np.corrcoef(A_res[:, 0], B_res[:, 0])[0,1])
			corr_result.append(np.roll(np.array(curr_result), var_ind))
		return np.array(corr_result)

	def __var_partial_1vA(self, var, control):
		corr_result = []
		for index in range(len(var)):
			curr_var = var[index]
			curr_other_var = np.delete(var, index)
			curr_result = [1]
			for c_index in range(len(curr_control)):
				temp_IV = curr_control[c_index]
				if partial == True:
					A_res = self.__calculate_residue(np.array(curr_var)[0,:], np.delete(curr_control, c_index, 0))
					B_res = self.__calculate_residue(np.array(temp_IV), np.delete(curr_control, c_index, 0))
					corr_value = np.corrcoef(A_res[:, 0], B_res[:, 0])[0,1]
				else:
					corr_value = np.corrcoef(np.array(curr_var)[0, :], temp_IV)[0,1]
				curr_result.append(corr_value)
			corr_result.append(np.roll(np.array(curr_result), index))
		return np.array(corr_result)

	def __calculate_residue(self, var_tri, control_tri):
		var = np.reshape(var_tri, (len(var_tri), 1))
		intercept = np.ones((len(var_tri), 1))
		control_mat = control_tri.transpose()
		control_mat = np.hstack((intercept, control_mat))
		# finding the least squared error solution
		lstsq = np.linalg.lstsq(control_mat, var, rcond=None)[0]
		residual = var - control_mat.dot(lstsq)
		return residual

	def __update_var(self, var_name, var_array, var_matrix, var_triangular):
		if self.variable_names is None:
			self.variable_names = np.array(var_name)
			self.variable_arrays = np.array([var_array])
			self.variable_matrices = np.array([var_matrix])
			self.variable_triangulars = np.array([var_triangular])
		else:
			self.variable_names = np.append(self.variable_names, var_name)
			self.variable_arrays = np.vstack((self.variable_arrays, np.array([var_array])))
			self.variable_matrices = np.vstack((self.variable_matrices, np.array([var_matrix])))
			self.variable_triangulars = np.vstack((self.variable_triangulars, np.array([var_triangular])))

	# todo: handle exception with only 1 var name
	def __extract_var(self, var_names):
		assert self.variable_names is not None, "Variables are not initalized"
		assert type(var_names) is np.ndarray and len(var_names.shape) == 1, "The input names must be an instance of numpy.ndarray with exactly 1 dimension"
		var_indices = []
		for name in var_names:
			assert name in self.variable_names, "Variable " + name + " not found in the instance"
			var_indices.append(np.where(self.variable_names == name)[0][0])
		var_arrays = self.variable_arrays[var_indices, :]
		var_matrices = self.variable_matrices[var_indices, :, :]
		var_triangulars = self.variable_triangulars[var_indices, :]
		return np.array(var_indices), var_arrays, var_matrices, var_triangulars

	def __determine_indices(self, extract_type, cond_sti, cond_dict, words):
		assert extract_type in ("cond_sti", "words", "None"), "invalid extract_type: must be one from (cond_sti, words, None)"
		if extract_type == "cond_sti":
			assert cond_sti is not None, "cond_sti uninitialized"
			assert type(cond_sti) is np.ndarray and len(cond_sti.shape) <= 2, "The cond_sti must be an instance of numpy.ndarray with at most 2 dimensions"
			if len(cond_sti.shape) == 2:
				assert cond_dict is not None and type(cond_dict) is dict, "cond_dict not properly initialized"
				indices = self.__find_cond_sti_indices(cond_sti, cond_dict)
			else:
				indices = cond_sti - 1
		elif extract_type == "words":
			assert words is not None, "words uninitialized"
			indices = self.__find_words_indices(words)
		else:
			indices = np.arange(self.words.shape[0])
		return indices

	def __find_words_indices(self, words):
		if self.__word_dict is None:
			self.__word_dict = dict()
			for index, w in enumerate(self.words):
				self.__word_dict.update({w:index})
		indices = []
		for w in words:
			indices.append(self.__word_dict[w])
		return np.array(indices)

	def __find_cond_sti_indices(self, cond_sti, cond_dict):
		indices = []
		for index in range(cond_sti.shape[0]):
			cond = cond_sti[index, 0]
			sti = int(cond_sti[index, 1])
			cond_start = int(cond_dict[cond])
			indices.append(cond_start + sti - 2)
		return np.array(indices)

	def __masking_indices(self, indices, mask):
		# The indices match the original order
		if np.array_equal(indices,  np.arange(self.words.shape[0])):
			return indices[mask]
		masked_indices = []
		for index in indices:
			if mask[index] == True:
				masked_indices.append(index)
		return masked_indices

	def __extract_by_index(self, indices, var_arrays, var_matrices, var_triangulars):
		if np.array_equal(indices,  np.arange(self.words.shape[0])):
			return var_arrays, var_matrices, var_triangulars
		var_arrays = var_arrays[:, indices]
		new_matrices = np.empty((var_matrices.shape[0], len(indices), len(indices)))
		new_var_triangulars = np.empty((var_triangulars.shape[0],(len(indices)*len(indices) - len(indices))//2))
		# for each variable
		for index in range(var_matrices.shape[0]):
			new_matrices[index] = var_matrices[index][np.ix_(indices,indices)]
			new_var_triangulars[index] = extract_upper_triangular(new_matrices[index])
		var_matrices = new_matrices
		var_triangulars = new_var_triangulars
		return var_arrays, var_matrices, var_triangulars

	def __avg_triangular(self, result):
		var_names = result[0]['var_names']
		all_var_matrices = self.__extract_var(var_names)[2]
		if len(all_var_matrices.shape) == 2:
			all_var_matrices = np.array([all_var_matrices])
		triangulars = []
		label_indices = []
		for label_dict in result:
			label_indices.append(label_dict["indices"])
		# for each matrix
		for index in range(all_var_matrices.shape[0]):
			curr_mat = all_var_matrices[index, :, :]
			curr_tri = []
			for row in range(len(label_indices)):
				for col in range(row + 1, len(label_indices)):
					row_indices = label_indices[row]
					col_indices = label_indices[col]
					curr_tri.append(np.average(curr_mat[np.ix_(row_indices, col_indices)]))
			triangulars.append(curr_tri)
		return np.array(triangulars)

	def __interpolate_EEG(self, sub_name, EEG, indices, IV_name, IV_array, IV_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict = None):
		final_masks, new_var_names, new_var_tris = self.__variable_interpolation(indices, IV_name, IV_array, IV_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict)
		sub_EEG = []
		for mask in final_masks:
			sub_EEG.append(np.average(EEG[mask], axis = 0))
		return (STR.subject(sub_name + " " + IV_name, np.array(sub_EEG), new_var_names, new_var_tris))

	def __interpolate_EEG_tri(self, sub_name, eeg, IV_dict, CV_dict, max_diff, range_type, stim_val, decimals = None):
		subjects = []
		for IV_ind in range(len(IV_dict["var_names"])):
			IV_name = IV_dict["var_names"][IV_ind]
			IV_tri = IV_dict["var_triangulars"][IV_ind]
			if decimals is not None:
				IV_tri = np.around(IV_tri, decimals = decimals)
			interpolation_dict = nearst_neighbor_1D(IV_tri, max_diff = max_diff, range_type = range_type, stim_val = stim_val)
			# unique triangular value, arranged from smallest to largest
			new_var_names = np.array([IV_name])
			new_tris = np.array([list(interpolation_dict.keys())])
			final_ind = np.array(list(interpolation_dict.values()))[:, 1]
			if CV_dict is not None:
				other_var_names, other_tris = self.__assemble_control_interpolation(IV_name, CV_dict, final_ind, "tri")
				new_var_names = np.concatenate((new_var_names, other_var_names))
				new_tris = np.concatenate((new_tris, other_tris))
			curr_subject = STR.subject(sub_name + " " + IV_name, eeg, new_var_names, new_tris, triangulars_modifer = final_ind)
			subjects.append(curr_subject)
		return subjects

	def __variable_interpolation_tri(self, IV_dict, CV_dict, max_diff, range_type, stim_val, decimals = None):
		final_inds = []
		var_names = []
		tris = []
		for IV_ind in range(len(IV_dict["var_names"])):
			IV_name = IV_dict["var_names"][IV_ind]
			IV_tri = IV_dict["var_triangulars"][IV_ind]
			if decimals is not None:
				IV_tri = np.around(IV_tri, decimals = decimals)
			interpolation_dict = nearst_neighbor_1D(IV_tri, max_diff = max_diff, range_type = range_type, stim_val = stim_val)
			# unique triangular value, arranged from smallest to largest
			new_var_names = np.array([IV_name])
			new_tris = np.array([list(interpolation_dict.keys())])
			final_ind = np.array(list(interpolation_dict.values()))[:, 1]
			if CV_dict is not None:
				other_var_names, other_tris = self.__assemble_control_interpolation(IV_name, CV_dict, final_ind, "tri")
				new_var_names = np.concatenate((new_var_names, other_var_names))
				new_tris = np.concatenate((new_tris, other_tris))
			final_inds.append(final_ind)
			var_names.append(new_var_names)
			tris.append(new_tris)
		return np.array(final_inds), np.array(var_names), np.array(tris)

	def __variable_interpolation(self, indices, IV_name, IV_array, IV_matrix, max_diff, range_type, stim_val, stim_val_type, CV_dict = None):
		interpolation_dict = nearst_neighbor_1D(IV_array, max_diff = max_diff, range_type = range_type, stim_val = stim_val)
		new_IV_arr = np.array(list(interpolation_dict.keys()))
		final_masks = []
		final_ind = []
		for key in new_IV_arr:
			curr_mask = interpolation_dict[key][0][indices]
			# remove empty groups
			if np.any(curr_mask) != False:
				final_masks.append(curr_mask)
				final_ind.append(np.array(interpolation_dict[key][1]))
		if stim_val_type == "dsm":
			new_var_names, new_tris = self.__assemble_tri_mat(final_ind,np.array([IV_name]), IV_matrix)
		else:
			new_var_names, new_tris = self.__assemble_tri_arr(final_ind,np.array([IV_name]), IV_array)
		if CV_dict is not None:
			other_var_names, other_tris = self.__assemble_control_interpolation(IV_name, CV_dict, final_ind, stim_val_type)
			new_var_names = np.concatenate((new_var_names, other_var_names))
			new_tris = np.concatenate((new_tris, other_tris))
		return final_masks, new_var_names, new_tris

	def __assemble_control_interpolation(self, IV_name, CV_dict, final_ind, stim_val_type):
		var_names = CV_dict["var_names"]
		var_arrays = CV_dict["var_arrays"]
		var_matrices = CV_dict["var_matrices"]
		var_triangulars = CV_dict["var_triangulars"]
		if IV_name in var_names:
			IV_ind = np.argwhere(var_names == IV_name)[0,0]
			var_names = np.delete(var_names, IV_ind)
			var_arrays = np.delete(var_arrays, IV_ind, axis = 0)
			var_matrices = np.delete(var_matrices, IV_ind, axis = 0)
			var_triangulars = np.delete(var_triangulars, IV_ind, axis = 0)
		if stim_val_type == "dsm":
			return self.__assemble_tri_mat(final_ind, var_names, var_matrices)
		elif stim_val_type == "raw":
			return self.__assemble_tri_arr(final_ind, var_names, var_arrays)
		else:
			return self.__assemble_tri(final_ind, var_names, var_triangulars)

	def __assemble_tri_arr(self, arr_indices, names, arrays):
		new_names = []
		tris = []
		assert type(names) is np.ndarray
		if len(arrays.shape) == 1:
			arrays = np.array([arrays])
		for index, var_name in enumerate(names):
			original_var_arr = arrays[index]
			new_var_arr = []
			for indices in arr_indices:
				curr_val = np.average(original_var_arr[indices])
				new_var_arr.append(curr_val)
			new_mat = self.__matrix_calculation(np.array(new_var_arr), abs_diff)
			new_tri = extract_upper_triangular(new_mat)
			new_names.append(var_name)
			tris.append(new_tri)
		return np.array(new_names), np.array(tris)

	def __assemble_tri_mat(self, arr_indices, names, matrices):
		new_names = []
		tris = []
		assert type(names) is np.ndarray
		# for each variable
		if len(matrices.shape) == 2:
			matrices = np.array([matrices])
		for index, var_name in enumerate(names):
			original_var_mat = matrices[index]
			new_tri = []
			# obtain sub matrix and calculate average
			for row in range(len(arr_indices)):
				for col in range(row + 1, len(arr_indices)):
					row_indices = arr_indices[row]
					col_indices = arr_indices[col]
					new_tri.append(np.average(original_var_mat[np.ix_(row_indices, col_indices)]))
			new_names.append(var_name)
			tris.append(new_tri)
		return np.array(new_names), np.array(tris)

	def __assemble_tri(self, arr_indices, names, triangulars):
		new_names = []
		final_tris = []
		if len(triangulars.shape) == 1:
			triangulars = np.array([triangulars])
		for index, var_name in enumerate(names):
			original_tri = triangulars[index]
			new_tri = []
			for indices in arr_indices:
				curr_val = np.average(original_tri[indices])
				new_tri.append(curr_val)
			new_names.append(var_name)
			final_tris.append(new_tri)
		return np.array(new_names), np.array(final_tris)

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