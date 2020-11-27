# RSA_plot.py

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_matrix(matrix, title, var_names = None, clim = None, show = False, save = False):
	print (title + " dim: " + str(matrix.shape))
	matplotlib.rcParams.update({'font.size': 7})
	plt.figure()
	if var_names is not None:
		extent_max = 2*(matrix.shape[0])
	else:
		extent_max = matrix.shape[0]
	plt.imshow(matrix, interpolation='none', extent = [0, extent_max, 0, extent_max]);
	plt.title(title)
	if clim is not None:
		plt.clim(clim[0],clim[1])
	plt.colorbar()
	if var_names is not None:
		locs = np.arange(1, extent_max, step=2)
		plt.xticks(locs, var_names, rotation = 30)
		plt.yticks(np.flip(locs), var_names, rotation = 30)
	if save == True:
		plt.savefig(title + ".png", format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_corr(all_corr_results, var_names, start_end = None, interval = 100, axis = [None, None, None, None], show = False, save = False, save_name = "./corr_results.png"):
	assert type(all_corr_results) is np.ndarray and len(all_corr_results.shape) == 2, "all_corr_results must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert len(axis) == 4, "axis must have exactly 4 elements"
	matplotlib.rcParams.update({'font.size': 6})
	for var_index in range(len(var_names)):
		var_name = var_names[var_index]
		corr_results = all_corr_results[var_index]
		plt.plot(corr_results, label = var_name)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = all_corr_results.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	plt.axis(axis)
	plt.legend()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_1D(Data, names, title = None, start_end = None, interval = 100, axis = [None, None, None, None], highlight_intervals = None, show = False, save_name = None):
	assert type(Data) is np.ndarray and len(Data.shape) == 2, "Data must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type(names) is np.ndarray, "names must be an instance of numpy.ndarray"
	assert len(axis) == 4, "axis must have exactly 4 elements"
	matplotlib.rcParams.update({'font.size': 6})
	for var_index in range(len(names)):
		var_name = names[var_index]
		corr_results = Data[var_index]
		plt.plot(corr_results, label = var_name)
	if title is not None:
		plt.title(title)
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
	plt.legend()
	if save_name is not None:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_subjects(Data, title, frame, start_end = None, interval = 100, axis = [None, None, None, None], show = False, save = False, save_name = "./all_sub_corr.png"):
	matplotlib.rcParams.update({'font.size': 3})
	fig, ax = plt.subplots(nrows=frame[0], ncols=frame[1])
	for sub_index in range(Data.shape[0]):
		row_ind = sub_index // frame[1]
		col_ind = sub_index % frame[1]
		ax[row_ind, col_ind].plot(Data[sub_index, :])
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round((end-start) / x_range))
		ax[row_ind, col_ind].xaxis.set_ticklabels(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	if show == True:
		plt.show()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000)

def plot_corr_2D(Data, title, y_label = None, start_end = None, interval = 100, clim = None, show = False, save = False, save_name = "2D_corr.png"):
	assert type(Data) is np.ndarray and len(Data.shape) == 2, "Data must be an instance of numpy.ndarray with exactly 2 dimensions"
	current_cmap = plt.cm.get_cmap()
	current_cmap.set_bad(color='grey')
	plt.figure(figsize = (10,5))
	plt.imshow(Data, interpolation="none", aspect='auto')
	plt.title(title)
	plt.colorbar()
	if clim is not None:
		plt.clim(clim)
	if y_label is not None:
		assert type(y_label) is np.ndarray, "y_label must be an instance of numpy.ndarray"
		assert y_label.shape[0] == Data.shape[0], "The first dimension of Data and y_label does not match"
		plt.yticks(np.arange(0, len(y_label), step = 1, dtype = int), y_label)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = Data.shape[1]
		step = int(round(x_range / (len(label) - 1)))
		plt.xticks(np.arange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_variability(var_names, means, stds, start_end = None, interval = 100, axis = [None, None, None, None], show = False, save = False, save_name = "./var.png"):
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert type (means) is np.ndarray and len(means.shape) == 2, "means must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert type (stds) is np.ndarray and len(stds.shape) == 2, "stds must be an instance of numpy.ndarray with exactly 2 dimensions"
	assert len(axis) == 4, "axis must have exactly 4 elements"	
	for var_index, var_name in enumerate(var_names):
		var_mean = means[var_index]
		var_std = stds[var_index]
		plt.plot(var_mean, label = var_name)
		plt.fill_between(np.arange(0, len(var_mean), step = 1, dtype = int), var_mean - var_std, var_mean + var_std, alpha = 0.3)
	if start_end is not None:
		start = int(start_end[0])
		end = int(start_end[1])
		label = np.linspace(start, end, (end-start)//interval + 1, dtype = int)
		x_range = means.shape[1]
		step = int(round((end-start) / x_range))
		plt.xticks(np.arrange(0, x_range, step = step, dtype = int), label, rotation = "vertical")
	plt.axis(axis)
	plt.legend()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000)
	if show == True:
		plt.show()
	plt.clf()

def plot_density(var_names, var_arrays, normalize = True, hist = False, show = False, save = False, save_name = "./var.png"):
	assert type(var_names) is np.ndarray, "var_names must be an instance of numpy.ndarray"
	assert type(var_arrays) is np.ndarray and len(var_arrays.shape) == 2, "var_arrays must be an instance of numpy.ndarray with exactly 2 dimensions"
	for var_name, var_array in zip(var_names, var_arrays):
		if normalize == True:
			arr_sum = np.sum(var_array)
			var_array = np.divide(var_array, arr_sum)
		sns.distplot(var_array, label = var_name, hist = hist, norm_hist=True)
	plt.legend()
	if save == True:
		plt.savefig(save_name, format = "png", dpi = 1000, transparent = True)
	if show == True:
		plt.show()
	plt.clf()

def plot_dendrogram_with_matrix(linkage_matrix_1, linkage_matrix_2, matrix, clim = None,  **kwargs):
	from scipy.cluster.hierarchy import dendrogram
	# Initialize figure
	fig = plt.figure(figsize=(10,10))
	# Plot first dendrogram
	# ax1 = fig.add_axes([0.15,0.1,0.15,0.6])
	ax1 = fig.add_axes([0.1,0.1,0.15,0.65])
	d1 = dendrogram(linkage_matrix_1, **kwargs, orientation='left')
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.axis("off")
	# Plot second dendrogram
	# ax1 = fig.add_axes([0.3,0.7,0.6,0.15])
	ax2 = fig.add_axes([0.25,0.75,0.65,0.15])
	d2 = dendrogram(linkage_matrix_2, **kwargs)
	ax2.set_xticks([])
	ax2.set_yticks([])
	ax2.axis("off")
	# plot matrix
	# axmatrix = fig.add_axes([0.3,0.1,0.6,0.6])
	axmatrix = fig.add_axes([0.25,0.1,0.65,0.65])
	idx1 = d1['leaves']
	idx2 = d2['leaves']
	matrix = matrix[idx1,:]
	matrix = matrix[:,idx2]
	im = axmatrix.matshow(matrix, aspect='auto', origin='lower', cmap = "plasma")
	axmatrix.set_xticks([])
	axmatrix.set_yticks([])
	# plot colorbar
	if clim is not None:
		im.set_clim(clim[0],clim[1])
	axcolor = fig.add_axes([0.93,0.1,0.02,0.65])
	plt.colorbar(im, cax=axcolor)
	return fig

def plot_dendrogram_with_variable_matrix(linkage_matrix, matrix, var_names, clim = None,  **kwargs):
	from scipy.cluster.hierarchy import dendrogram
	# Initialize figure
	fig = plt.figure(figsize=(10,10))
	# Plot top dendrogram
	ax1 = fig.add_axes([0.1,0.75,0.8,0.15])
	d1 = dendrogram(linkage_matrix, **kwargs)
	ax1.set_xticks([])
	ax1.set_yticks([])
	ax1.axis("off")
	# plot matrix
	axmatrix = fig.add_axes([0.1,0.1,0.8,0.65])
	idx1 = d1['leaves']
	matrix = matrix[:,idx1]
	im = axmatrix.matshow(matrix, aspect='auto', cmap="plasma")
	axmatrix.set_xticks([])
	axmatrix.set_yticks(np.arange(matrix.shape[0], dtype = int))
	axmatrix.set_yticklabels(var_names)
	# plot colorbar
	if clim is not None:
		im.set_clim(clim[0],clim[1])
	axcolor = fig.add_axes([0.93,0.1,0.02,0.65])
	plt.colorbar(im, cax=axcolor)
	return fig
