# variables_processing_alt.py
A class for storing variables. It is recommended to store all variables together into one instance, since many functions operate on multiple variables simultanously.
- **Class Attributes** 
  - *words*: a numpy array of all the stimuli words used in all the trials. This class assumes that all trials use different words. This attribute must be specified upon initialization.
  - *variable_names*: a numpy array of all the variable names. Empty when initialized, updated by **`update_variable()`** function. ***Do not change this attribute manually***; they are used to find the indexes of *variable_arrays*, *variable_matrices*, and *variable_triangulars*.
  - *variable_arrays*: a 2D numpy array of all the variable values. Empty when initialized, updated by **`update_variable()`** function. Dimension: (variable, values).
  - *variable_matrices*: a 3D numpy array of all the variable dissimilarity matrices. Empty when initialized, updated by **`update_variable()`** function. Dimension: (variable, values, values).
  - *variable_triangulars*: a 2D numpy array of all the flatten triangular arrays obtained from the variable dissimilarity matrices. Empty when initialized, updated by **`update_variable()`** function. Dimension: (variable, triangular values).
  - *masks_dict*: a python dictionary that contains all the conditions. Empty when initialized, updated by **`update_mask()`** function.  Structure: {Condition name : {label name : mask numpy array}}. ***Depreciated***.
  - *__word_dict*: Private variable. a python dictionary that contains all the indices of the words.  Empty when initialized, automatically updated by __find_words_indices() function.
<br/><br/>

- **Public Functions**
  - **`print_attribute_info()`**: Print the basic information of the class attributes.
  - **`update_variable()`**: Add a variable to the class.
  - **`update_mask()`**: Add a mask to the class. ***Depreciated***.
  - **`export_variables()`**: Export variables according to the specified format.
  - **`create_STR_subject()`**: Create a **Single_Trial_RSA** instance.
  - **`create_STR_subject_tri()`**: Temporary function. Create a **Single_Trial_RSA** instance, applying all specifications to the interpolated triangular data.
  - **`calculate_variables_correlation()`**: Calculate the pairwise (partial) correlation of specified variables.
  - **`calculate_variables_correlation_interpol()`**: Temporary function. Calculate the pairwise (partial) correlation of specified variables, with interpolation specifications avaliable.
  - **`check_missing_words()`**: Check whether a list of specified words are included in a specified corpus
  - **`semantic_disimilarilty()`**: Calculate pairwise semantic dissimilarity of a list of specified words using a specified corpus
  - **`impute_missing_values()`**: To be tested. Impute missing value of a variable from other variables
  - **`delete_trials()`**: Delete specified trials from the dataset
<br/><br/>

- **Other Functions**
  - `**nearst_neighbor_1D()**`: A nearest neighbor algorithm that group values from a 1D array
  - `**matrix_iteration()**`: A template function that can apply the embeded function iteratively to all pairwise values

## Public Functions

<code>**print_attribute_info**()</code>
<br/> Print the status of the following attributes: *variable_names*, *variable_arrays*, *variable_matrices*, *variable_triangulars*, *masks_dict*
<br/><br/>

<code>**update_variable**(var_name, var_array = None, var_matrix = None, Tfunction = None, Dfunction = abs_diff)</code>
<br/> Update a variable to the class. Note that at least one of the two parameters **var_array, var_matrix** must be defined 
- **Parameters**
  - **var_name: *str*** <br/>
    The name of the variable
  - **var_array: *None* or *numpy ndarray* with shape = (x,)** <br/>
    The 1D variable array. *Optional*.
  - **var_matrix: *None* or *numpy ndarray* with shape = (x, x)** <br/>
    The 2D dissimilarity matrix. *Optional*.
  - **Tfunction: *None* or *function*** <br/>
    The function for transforming the **var_array**. Ignored unless **var_array** is specified. Tfunction should be formulated as `Tfunction(x)` where *x* is a numpy 1D array.  *Optional*.
  - **Dfunction: *None* or *function*** <br/>
    The function for calculating the pairwise dissimilarity value. Ignored unless **var_array** is specified. By default, Dfunction calculates the absolute difference. Dfunction should be formulated as `Dfunction(x,y)` where *x, y* are numerical values.  *Optional*.
- **Returns**
  - **None**
<br/><br/>

<code>**update_mask**(mask_name, mask_source)</code>
<br/> Update a mask to the class. ***Depreciated***.
<br/><br/>

<code>**export_variables**(var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None)</code>
<br/> export variables with their corresponding 1D variable array, 2D dissimilarity matrix, and 1D array of collapsed upper triangular values above the diagonal.
- **Parameters**
  - **var_names: *numpy ndarray* with shape = (x,)** <br/>
    The array of the names of the variables to be extracted.
  - **extract_type: *"None"*, *"cond_sti"*, or *"words"*** <br/>
    The type of indexing options. Default = "None", and all variable values will be exported in the original order. **"cond_sti"** orders the variable values through condition-stimuli pairs. **"words"** orders the variable values through their corresponding words. <br/>
    ***Note:*** While the **"cond_sti"** option assumes the specified trials to be a subset of the trials stored in the instance,  the **"words"** option does not make this assumption and exports all trials found in the interaction of specified words and the stored words. Therefore the **"words"** option is particularly useful when the user delete some trials from the current instance. <br/>
    ***Note***: This parameter can be specified together with **mask**. The function will only extract variable values presented both in the mask and in the specified indexing options.
  - **cond_sti: *None* or *numpy ndarray* with shape = (x,) or (x,2)** <br/>
    The condition-stimuli pair. Ignored unless `extract = "cond_sti"`. If **cond_sti** is 1D, the function will intrepret it as the bin number assigned to trials (starting from 1). If **cond_sti** is 2D, the left column will indicate the condition and the right column will indicate the stimulus ID relative to that condition (starting from 1). For example, `[1,1]` indicates the first stimulus in condition 1, and `[3, 10]` indicates the 10th stimulus in condition 3. *Optional*.
  - **cond_dict: *None* or *dict* with the format {condition name: starting position}** <br/>
    The dictionary that defines the starting position of all conditions provided in **cond_sti**. Ignored unless `extract_type = "cond_sti"` and **cond_sti** is a 2D array. The function assumes that the variables values stored in the instance contain all possible trials, and that their order matches the order of all the original trials. For example, suppose the trials are arranged in the following condition: `[1,1,1,2,2,3,3,3]`. Condition 1 starts at trial 1, condition 2 starts at trial 4, and condition 3 starts at trial 6. Therefore, the **cond_dict** should be `({1: 1}, {2: 4}, {3: 6})`. *Optional*.
  - **words: *None* or *numpy ndarray* with shape = (x,)** <br/>
    The 1D array of words. Note that this parameter need not be a subset of the stored words in the instance. The program will only export words in the intersection of the specified array and the stored array.
  - **mask: *None*, *str*, or *numpy ndarray* with shape = (x,)** <br/>
    The condition mask that will be applied to the 1D variable arrays. If **mask** is ***str***, the function will find the corresponding mask stored in the instance (***Depreciated***). If **mask** is ***numpy ndarray***, the dimension of the mask must be identical to the dimension of class attributes *words*. If `dtype = bool`, the trials labeled `False` will be ignored. Otherwise, for each unique character in the mask array, the function will divide the variable values into its own subset. For example, suppose `mask = [1,1,1,2,2,3,3,3]`, then the function will divide the dataset into the three subsets using the three masks: `[1,1,1,0,0,0,0,0]`, `[0,0,0,1,1,0,0,0]`, and `[0,0,0,0,0,1,1,1]`. The three subsets will be exported seperately; refer to the return section for more details. *Optional*. <br/>
    ***Note***: This parameter can be specified together with **extract_type** options. The function will only extract variable values presented both in the mask and in the specified indexing options.
- **Returns**
  - **Results: *dict* or *list*** <br/>
    If **mask** is `None` or creates only one subset of variable values, then the **Results** will be a ***dict*** with the following keys: `"var_names", "indices", "var_arrays", "var_matrices", "var_triangulars"`. <br/>
    If **mask** creates n seperate subsets (n > 1), then the **Results** will be a ***list*** of n ***dict***, with each dict having the following keys: `"label", "label_mask" ,"var_names", "indices", "var_arrays", "var_matrices", "var_triangulars"`
  - **empty_conditions: *list*** <br/>
    A list of conditions that have no trials under the indexing and mask constraints. Only appears when **mask** is specified.
  - **Missing_ind: *None* or *numpy ndarray*** <br/>
    The indexes not found in the current instance. The indexes are defined relatively to the specified **word** parameter, and will be ***np ndarray*** only when `extract_type = "words".
 
<code>**create_STR_subject**(sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None, interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm", CV = None)</code>
<br/> Format the eeg data and the variable values and create one or multiple **Single_Trial_RSA** instance(s)
- **Parameters**
  - **sub_name: *str*** <br/>
    The name of the current subject. 
  - **eeg: *numpy ndarray* with shape = (x,y,z)** <br/>
    The eeg data of the current subject, where *x* is the trial dimension, *y* is the channel dimension, and *z* is the time dimension. 
  - **var_names, extract_type, cond_sti, cond_dict, words, mask** <br/>
    Please refer to **`export_variables`** for details.  <br/>
    ***Note***: If the trials contained in the **eeg** is not a subset of the trials of the variable values, please use `extract_type = "words"`.   <br/>
    ***Note***: While the indexing options are relative to the input **eeg** data, **mask** is relative to the variable data stored in the instance. This function ensure that the trial dimension of **eeg** and other variable values will match.
  - **interpolation: *bool*** <br/>
    Allow interpolation. Default = **False**.  <br/>
    ***Note***: the interpolation option is compatible with indexing options and **mask**.  <br/>
    ***Note***: The interpolation will be applied seperately for each variable specified in **var_names**. If more than 1 variable is specified, the function will return a seperate **Single_Trial_RSA** instance for each variable. 
  - **max_diff: *int*** <br/>
    Specify the maximum absolute difference between a value and a cluster center for that value to be included into the cluster. Default = **0**. Ignored unless `interpolation = True`. If `max_diff = 0`, only identical values will be included in the same cluster.
  - **range_type: *"percentage"* or *"raw"*** <br/>
    The type of value for **max_diff**. Default = **percentage**. Ignored unless `interpolation = True`. If set to **"percentage"**, the maximum difference will be the set percentage of the overall range of the dataset. If set to **"raw"**, the maximum difference will be the value specified for **max_diff**.
  - **stim_val: *"average"* or *"central"*** <br/>
    The type of value that will become the value of a cluster. Default = **"average"**. Ignored unless `interpolation = True`. If set to **"average"**, the cluster value will be the average of all values in the cluster. If set to **central**, the cluster value will be the value at the center of the cluster.
  - **stim_val_type: *"dsm"* or *"raw"*** <br/>
    The type of stimuli value to be considered. Default = **dsm**. Ignored unless `interpolation = True`. If set to **"dsm"**, the function interpolate the sub-matrix specified by the indexes of values contained in a cluster by averaging the sub-matrix of the corresponding dissimilarity matrix. This option requires `stim_val = "average"`. If set to **"raw"**, the function will interpolate the 1D variable array, and the variable dissimilarity matrix and the associated flatten upper triangular value will be recalculated based on the new 1D array.
  - **CV: *None* or *numpy ndarray* with shape = (x,)** <br/>
    The control variables for **Single_Trial_RSA**. Default = **None**. If **CV** is a ***numpy ndarray***, then it contains the name of the variables to be exported to the **Single_Trial_RSA** instance. If a variable specified in **var_names** is also specified in **CV**, the variable will only be exported once. If `interpolation = True`, then all the variable specified in **CV** will be interpolated according to the interpolation policy created for a particular variable specified in **var_names**. <br/>
    ***Note*** This function only export relevant variables to the **Single_Trial_RSA** instance, but the user needs to specify the control variable in the **Single_Trial_RSA** when performing RSA. This parameter is intended to let the function apply interpolation policy created from other variables to the control variables.
