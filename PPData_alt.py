# preprocessing.py

import numpy as np
import matplotlib.pyplot as plt

def read_file(file):
	data = []
	with open (file, "r") as eeg_file:
		for line in eeg_file:
			data.append(line.split())	
	data = np.array(data)
	return data

def string_to_float(data):
	data = data.astype(float)
	return data

def average_trails(trails):
	total_trails = trails.shape[0]
	temp = trails.transpose()
	result = []
	for row in temp:
		result.append((np.sum(row))/total_trails)
	base_line = (sum(result[:25]))/25
	result = result - base_line
	return result

def plot_erp(erp1, erp2, name1, name2):
	plt.plot(erp1, label = name1)
	plt.plot(erp2, label = name2)
	plt.legend()
	plt.show()
	return

# print_line
# input: list, string output file name
# output: none
# print the list to the output file
def print_line(input_list, output):
	with open (output, "a") as output_file:
		for item in input_list:
			output_file.write(str(item) + "\t")
		output_file.write("\n")
	return

###############################################################################
#								   PPData Class								  #
###############################################################################

class PPData:
	
	def __init__(self, name):
		# Initialize all class variables
		self.subname = name
		self.raw_eeg = None
		self.clean_eeg = None
		self.raw_variables = None
		self.clean_variables = None
		self.variables_title = None
		self.bad_trails = None
		self.bins = None 				# bins start from 1
		self.indices = None 			# indices start from 0

	def load_eeg_from_file(self, file, mask = None):
		raw = read_file(file)
		raw = string_to_float(raw)
		raw = self.__reshape_eegdata(raw, mask = mask)
		self.raw_eeg = raw
		return raw

	def load_variables_from_file(self, file):
		variables = read_file(file)
		self.variables_title = variables[0]
		self.raw_variables = string_to_float(variables[1:,:])
		return self.variables_title, self.raw_variables

	def load_variables(self, title, raw_variables):
		self.variables_title = title.copy()
		self.raw_variables = raw_variables.copy()
		return

	def clear_artifacts(self):
		assert self.raw_eeg is not None, "eeg uninitialized"
		self.bad_trails, self.bins = self.__detect_artifacts(self.raw_eeg)
		self.indices = self.bins - 1
		self.clean_eeg = self.__clear_artifacts(self.raw_eeg)
		if self.clean_variables is not None:
			self.clean_variables = self.__clear_artifacts(self.raw_variables)
		return self.clean_eeg, self.bins

	def print_class_info(self):
		print ("Subject: " + str(self.subname))
		print ("EEG info: ")
		print ("  " + "raw_eeg: " + str(self.raw_eeg.shape))
		print ("  " + "clean_eeg: " + str(self.clean_eeg.shape))
		print ("Variables info: ")
		print ("  " + "raw_variables: " + str(self.raw_variables.shape))
		print ("  " + "clean_variables: " + str(self.clean_variables.shape))
		print ("Bad trails info: ")
		print ("  " + "bad_trails_num: " + str(len(self.bad_trails)))
		print ("  " + "bad_trails: " + str(self.bad_trails))
		return

	def calculate_erp(self, variable, data_range, channel_index):
		if variable not in self.variables_title:
			print ("fail to find the variable")
			return None
		
		index = np.where(self.variables_title == variable)[0]
		var_list = self.clean_variables[:,index].flatten()
		index_list = var_list.argsort()
		
		eeg_small = self.clean_eeg[index_list[:data_range]][:,channel_index,:].copy()
		eeg_large = self.clean_eeg[index_list[-data_range:]][:,channel_index,:].copy()
		# print (var_list[index_list[:data_range]])
		# print (var_list[index_list[-data_range:]])

		average_small = average_trails(eeg_small)
		average_large = average_trails(eeg_large)

		return average_large, average_small

	def regroup_data(self, group_size):
		eeg_regrouped = self.__regroup_eeg(group_size, self.clean_eeg)
		variables_regrouped = self.__regroup_variables(group_size, self.clean_variables)
		return eeg_regrouped, variables_regrouped
#								Private Functions							  #

	def __regroup_eeg(self, group_size, data):
		data = data.copy()
		groups = data.shape[0]//group_size
		# for each time point
		time_array = []
		for time_index in range(data.shape[2]):
			# for each channel
			channel_array = []
			for channel_index in range(data.shape[1]):
				# obtain regrouped trail data
				trail_array = []
				for group_index in range(groups):
					lower_lim = group_index * group_size
					upper_lim = lower_lim + group_size
					group_average = np.average(data[lower_lim : upper_lim, channel_index, time_index])
					trail_array.append(group_average)
				channel_array.append(trail_array)
			time_array.append(channel_array)
		regrouped_data = np.swapaxes(np.array(time_array), 0, 2)
		return regrouped_data

	def __regroup_variables(self, group_size, data):
		data = data.copy()
		groups = data.shape[0]//group_size
		variables_array = []
		for variable_index in range(data.shape[1]):
			trail_array = []
			for group_index in range(groups):
				lower_lim = group_index * group_size
				upper_lim = lower_lim + group_size
				group_average = np.average(data[lower_lim : upper_lim, variable_index])
				trail_array.append(group_average)
			variables_array.append(trail_array)
		regrouped_data = np.swapaxes(np.array(variables_array), 0, 1)
		return regrouped_data

	def __reshape_eegdata(self, data, mask = None):
		data = data[32:]
		data = np.reshape(data, (960,32,256))
		data = np.delete(data, [0,1,2], 1)
		if mask is not None:
			data = data[mask, :, :]
		return data

	def __detect_artifacts(self, eeg_data):
		trail_index = 0
		rejected_trails = []
		accepted_trials = []
		while trail_index < len(eeg_data):
			trail = eeg_data[trail_index]
			row = trail.shape[0]
			col = trail.shape[1]
			step = int(col/row)
			test = True
			row_index = 0
			col_index = 0
			while test == True and row_index < row:
				if trail[row_index][col_index] == 0:
					test = True
				else:
					test = False
				row_index = row_index + 1
				col_index = col_index + step
			if test == True:
				rejected_trails.append(trail_index)
			else:
				accepted_trials.append(trail_index)
			trail_index = trail_index + 1
		accepted_trials = np.add(1, accepted_trials)
		return np.array(rejected_trails), accepted_trials

	def __clear_artifacts(self, data):
		return np.delete(data, self.bad_trails, axis = 0)