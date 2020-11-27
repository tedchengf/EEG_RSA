# Auxiliaries.py
import pickle
import numpy as np
from numba import njit, jit

# Save/Load
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

# Matrix Operations
###############################################################################
def check_symmetry(matrix):
	mat_dim = matrix.shape[0]
	asymmetric_pairs = []
	for r in range(mat_dim):
		for c in range(mat_dim):
			if matrix[r,c] != matrix[c,r]:
				asymmetric_pairs.append(np.array([(r,c), (c,r)]))
	if not asymmetric_pairs: return True
	else: return np.array(asymmetric_pairs)

def extract_neighbor_matrix(key_names, neighbor_dictionary):
	neighbor_matrix = np.zeros((len(key_names), len(key_names)), dtype = int)
	for r in range(len(key_names)):
		curr_key = key_names[r]
		curr_neighbor = neighbor_dictionary[curr_key]
		for c in range(len(key_names)):
			if c == r or key_names[c] in curr_neighbor:
				neighbor_matrix[r,c] = 1
	return neighbor_matrix

@jit
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

# Numerical Operations
###############################################################################
def corrcoef(A, B):
	return np.corrcoef(A, B)[0, 1]