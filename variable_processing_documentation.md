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

- **Public Class Functions**
  - **`print_attribute_info()`**: Print the basic information of the class attributes.
  - **`update_variable()`**: Add a variable to the class.
  - **`update_mask()`**: Add a mask to the class. ***Depreciated***.
  - **`export_variables()`**: Export variables according to the specified format.
  - **`create_STR_subject()`**: Create a **Single_Trial_RSA** instance.
  - **`create_STR_subject_tri()`**: Temporary function. Create a **Single_Trial_RSA** instance, applying all specifications to the interpolated triangular data.
  - **`calculate_variables_correlation()`**: Calculate the pairwise (partial) correlation of specified variables.
  - **`calculate_variables_correlation_interpol()`**: Temporary function. Calculate the pairwise (partial) correlation of specified variables, with interpolation specifications avaliable. ***Depreciated***.
  - **`check_missing_words()`**: Check whether a list of specified words are included in a specified corpus
  - **`semantic_disimilarilty()`**: Calculate pairwise semantic dissimilarity of a list of specified words using a specified corpus
  - **`impute_missing_values()`**: To be tested. Impute missing value of a variable from other variables
  - **`delete_trials()`**: Delete particular trials from the instance
<br/><br/>

- **Other Functions**
  - **`nearst_neighbor_1D()`**: A nearest neighbor algorithm that group values from a 1D array
  - **`matrix_iteration()`**: A template function that can apply the embeded function iteratively to all pairwise values
  - **`abs_diff()`, `eculidian_dist()`**: Two dissimilarity functions in the form of **Dfunction** for **`update_variable()`**.
  - **`z_squared_transform()`, `z_absolute_transform()`**: Two transformation functions in the form of **Tfunction** for **`update_variable()`**.

## Public Class Functions

<code>**print_attribute_info**()</code>
<br/> Print the status of the following attributes: *variable_names*, *variable_arrays*, *variable_matrices*, *variable_triangulars*, *masks_dict*
<br/><br/>

<code>**update_variable**(var_name, var_array = None, var_matrix = None, Tfunction = None, Dfunction = abs_diff)</code>
<br/> Update a variable to the class. Note that at least one of the two parameters **var_array, var_matrix** must be defined.
- **Parameters**
  - **var_name: *str*** <br/>
    The name of the variable
  - **var_array: *None* or *numpy ndarray* with shape = (n,)** <br/>
    The 1D variable array. *Optional*.
  - **var_matrix: *None* or *numpy ndarray* with shape = (n, n)** <br/>
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
  - **var_names: *numpy ndarray* with shape = (n,)** <br/>
    The array of the names of the variables to be extracted.
  - **extract_type: "None", "cond_sti", or "words"** <br/>
    The type of indexing options. Default = "None", and all variable values will be exported in the original order. **"cond_sti"** orders the variable values through condition-stimuli pairs. **"words"** orders the variable values through their corresponding words. <br/>
    ***Note:*** While the **"cond_sti"** option assumes the specified trials to be a subset of the trials stored in the instance,  the **"words"** option does not make this assumption and exports all trials found in the interaction of specified words and the stored words. Therefore the **"words"** option is particularly useful when the user delete some trials from the current instance. <br/>
    ***Note***: This parameter can be specified together with **mask**. The function will only extract variable values presented both in the mask and in the specified indexing options.
  - **cond_sti: *None* or *numpy ndarray* with shape = (n,) or (n,2)** <br/>
    The condition-stimuli pair. Ignored unless `extract_type = "cond_sti"`. If **cond_sti** is 1D, the function will intrepret it as the bin number assigned to trials (starting from 1). If **cond_sti** is 2D, the left column will indicate the condition and the right column will indicate the stimulus ID relative to that condition (starting from 1). For example, `[1,1]` indicates the first stimulus in condition 1, and `[3, 10]` indicates the 10th stimulus in condition 3. *Optional*.
  - **cond_dict: *None* or *dict* with the format {condition name: starting position}** <br/>
    The dictionary that defines the starting position of all conditions provided in **cond_sti**. Ignored unless `extract_type = "cond_sti"` and **cond_sti** is a 2D array. The function assumes that the variables values stored in the instance contain all possible trials, and that their order matches the order of all the original trials. For example, suppose the trials are arranged in the following condition: `[1,1,1,2,2,3,3,3]`. Condition 1 starts at trial 1, condition 2 starts at trial 4, and condition 3 starts at trial 6. Therefore, the **cond_dict** should be `({1: 1}, {2: 4}, {3: 6})`. *Optional*.
  - **words: *None* or *numpy ndarray* with shape = (n,)** <br/>
    The 1D array of words. Note that this parameter need not be a subset of the stored words in the instance. The program will only export words in the intersection of the specified array and the stored array.
  - **mask: *None*, *str*, or *numpy ndarray* with shape = (n,)** <br/>
    The condition mask that will be applied to the 1D variable arrays. If **mask** is ***str***, the function will find the corresponding mask stored in the instance (***Depreciated***). If **mask** is ***numpy ndarray***, the dimension of the mask must be identical to the dimension of class attributes *words*. If `dtype = bool`, the trials labeled `False` or `0` will be ignored. Otherwise, for each unique character in the mask array, the function will divide the variable values into its own subset. For example, suppose `mask = [1,1,1,2,2,3,3,3]`, then the function will divide the dataset into the three subsets using the three masks: `[1,1,1,0,0,0,0,0]`, `[0,0,0,1,1,0,0,0]`, and `[0,0,0,0,0,1,1,1]`. The three subsets will be exported seperately; refer to the return section for more details. *Optional*. <br/>
    ***Note***: This parameter can be specified together with **extract_type** options. The function will only extract variable values presented both in the mask and in the specified indexing options.
- **Returns**
  - **Results: *dict* or *list*** <br/>
    If **mask** is `None` or creates only one subset of variable values, then the **Results** will be a ***dict*** with the following keys: `"var_names", "indices", "var_arrays", "var_matrices", "var_triangulars"`. <br/>
    If **mask** creates n seperate subsets (n > 1), then the **Results** will be a ***list*** of n ***dict***, with each dict having the following keys: `"label", "label_mask" ,"var_names", "indices", "var_arrays", "var_matrices", "var_triangulars"`
  - **empty_conditions: *list*** <br/>
    A list of conditions that have no trials under the indexing and mask constraints. Only appears when **mask** is specified.
  - **Missing_ind: *None* or *numpy ndarray*** <br/>
    The indexes not found in the current instance. The indexes are defined relatively to the specified **word** parameter, and will be ***np ndarray*** only when `extract_type = "words".
<br/><br/> 
 
<code>**create_STR_subject**(sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None, interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm", CV = None)</code>
<br/> Format the eeg data and the variable values and create one or multiple **Single_Trial_RSA** instance(s)
- **Parameters**
  - **sub_name: *str*** <br/>
    The name of the current subject. 
  - **eeg: *numpy ndarray* with shape = (n,a,b)** <br/>
    The eeg data of the current subject, where *n* is the trial dimension, *a* is the channel dimension, and *b* is the time dimension. 
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
  - **range_type: "percentage" or "raw"** <br/>
    The type of value for **max_diff**. Default = **percentage**. Ignored unless `interpolation = True`. If set to **"percentage"**, the maximum difference will be the set percentage of the overall range of the dataset. If set to **"raw"**, the maximum difference will be the value specified for **max_diff**.
  - **stim_val: "average" or "central"** <br/>
    The type of value that will become the value of a cluster. Default = **"average"**. Ignored unless `interpolation = True`. If set to **"average"**, the cluster value will be the average of all values in the cluster. If set to **central**, the cluster value will be the value at the center of the cluster.
  - **stim_val_type: "dsm" or "raw"** <br/>
    The type of stimuli value to be considered. Default = **"dsm"**. Ignored unless `interpolation = True`. If set to **"dsm"**, the function interpolate the sub-matrix specified by the indexes of values contained in a cluster by averaging the sub-matrix of the corresponding dissimilarity matrix. This option requires `stim_val = "average"`. If set to **"raw"**, the function will interpolate the 1D variable array, and the variable dissimilarity matrix and the associated flatten upper triangular value will be recalculated based on the new 1D array.
  - **CV: *None* or *numpy ndarray* with shape = (n,)** <br/>
    The control variables for ***Single_Trial_RSA instance***. Default = **None**. If **CV** is a ***numpy ndarray***, then it contains the name of the variables to be exported to the ***Single_Trial_RSA instance***. If a variable specified in **var_names** is also specified in **CV**, the variable will only be exported once. If `interpolation = True`, then all the variable specified in **CV** will be interpolated according to the interpolation policy created for a particular variable specified in **var_names**. <br/>
    ***Note*** This function only export relevant variables to the ***Single_Trial_RSA instance***, but the user needs to specify the control variable in the ***Single_Trial_RSA instance*** when performing RSA. This parameter is intended to let the function apply interpolation policy created from other variables to the control variables.
- **Returns**
  - **Subjects: *Single_Trial_RSA instance*, *numpy ndarray* or *dict*** <br/>
    By default, the function will return one ***Single_Trial_RSA instance***. <br/>
    If `interpolation = True` and more than one variable is specified in **var_names**, then the **Subjects** will be a ***numpy ndarray*** of ***Single_Trial_RSA instance***, with the length of the array equivalent to the number of variable specified in **var_names**. <br/>
    If **mask** is defined, then the **Subjects** will be a ***dict***, with the keys being the unique characters in **mask**. The value associated with a key is either a ***Single_Trial_RSA instance*** or a ***numpy ndarray*** of ***Single_Trial_RSA instance***, depending on the interpolation settings. For each key, the ***Single_Trial_RSA instance*** will only contain the specific trials associated with a unique character (condition) in the **mask**
<br/><br/> 

<code>**create_STR_subject_tri**(sub_name, eeg, var_names, extract_type = "None", cond_sti = None, cond_dict = None, words = None, mask = None, interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm", CV = None)</code>
<br/> Format the eeg data and the variable values and create one or multiple **Single_Trial_RSA** instance(s). Similar to **`create_STR_subject()`**, except that the interpolation is applied on the flattened upper triangular arrays of the variables' dissimilarity matrices.
<br/><br/> 

<code>**calculate_variables_correlation**(var_names, control_var = None, corr_type = "dsm", interpolation = False, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm")</code>
- **Parameters**
  - **var_names: *numpy ndarray* with shape = (n,)** <br/>
    The array of the names of the variables to be extracted.
  - **control_var: *None* or *numpy ndarray* with shape = (n,)** <br/>
    The control variables. Default = ***None***. If **control_var** is a ***numpy ndarray***, then it contains the name of the variables to be controlled for in partial correlation. If a variable specified in **var_names** is also specified in **control_var**, the variable will not be included in the control variable if it is currently the target variable. For example, if `var_names = ["a", "b", "c"]` and `control_var = ["a", "b", "c"]`, then while calculating the correlation between `a,b`, only `c` will be taken as the control. While calculating the correlation between `a,c`, only `b` will be taken as the control, and so on and so forth. *Optional*.
  - **corr_type: "dsm" or "raw"** <br/>
    The type of data used for correlation. Default = **"dsm"**. If `corr_type = "dsm"`, then pearson correlation will be applied to the flatten upper triangular array of the dissimilarity matrix. If `corr_type = "raw"`, then pearson correlation will be applied to the 1D variable array.
  - **interpolation, max_diff, range_type, stim_val, stim_val_type** <br/>
    Please refer to **`create_STR_subject()`** for details. <br/>
- **Returns**
  - **correlation_matrix: *numpy ndarray* with shape = (n, n)** <br/>
    The pairwise correlation of specified variables. <br/>
    ***Note***: If `interpolation = True`, the pairwise correlation matrix will not be symmetrical. In this matrix, the x-axis represents the target variable that is used to create the interpolation policy, and the y-axis represents the control variable that will be interpolated according to the policy created by the target variable. For example, while `[a, b]` represents "the correlation between interpolated a and b according to the interpolation policy of variable a", `[b, a]` represents "the correlation between interpolated b and a according to the interpolation policy of variable b".
<br/><br/> 

<code>**calculate_variables_correlation_interpol**(var_names, partial = True, max_diff = 0, range_type = "percentage", stim_val = "average", stim_val_type = "dsm")</code>
<br/> Variable correlation with interpolation options. ***Depreciated***, with its functionality completely replaced by **`calculate_variables_correlation()`**
<br/><br/> 

<code>**check_missing_words**(words, sim_type = "w2v", resource_directory = "./")</code>
<br/> Check the words in the specified list that are missing from a given corpus
- **Parameters**
  - **words: *numpy ndarray* with shape = (n,)** <br/>
    The 1D array of words to check.
  - **sim_type: "w2v", "lsa", "glove"** <br/>
    The type of corpus to load. <br/>. Default = **"w2v"**
    If `sim_type = "w2v"`, then the function expect to find the corpus `"GoogleNews-vectors-negative300.bin"` in the specified directory. <br/>
    If `sim_type = "lsa"`, then the function expect to find the corpus `"wiki_en_Nov2019"` in the specified directory. <br/>
    If `sim_type = "glove"`, then the function expect to find the corpus `"gensim_glove_vectors.txt"` in the specified directory. <br/>
  - **resource_directory** <br/>
    The resource directory that stores the corpus. Default = **"./"**
- **Returns**
  - **bad_words: *numpy ndarray* with shape = (n,)** <br/>
    The array of missing words
  - **bad_words_indices: *numpy ndarray* with shape = (n,)** <br/>
    The indexes of the missing words, relative to the specified array **words**
<br/><br/> 

<code>**semantic_disimilarilty**(words, update = True, sim_type = "w2v", resource_directory = "./")</code>
<br/> Calculate the 2D semantic dissimilarity matrix based on the words. The dissimilarity score is calculated using cosine similarity. Note that if a word word is not found in the corpus, all associated matrix values will be `numpy.NaN`.
- **Parameters**
  - **words: *numpy ndarray* with shape = (n,)** <br/>
    The 1D array of words.
  - **update: *bool*** <br/>
    Indicate whether the user wants to update the dissimilarity matrix directly to the instance. Default = **True**. If `update = True`, then the variables will be updated with the name `"w2v"`, `"lsa"`, or `"glove"` according to **sim_type**. The associated 1D array will be an empty array.
  - **sim_type, resource_directory** <br/>
    Please refer to **`check_missing_words()`** for details. <br/>
- **Returns**
  - **var_matrix: *numpy ndarray* with shape = (n, n)** <br/>
    The 2D semantic dissimilarity matrix.
<br/><br/> 

<code>**impute_missing_values**(var_names, impute_target = "tri", update = True)</code>
<br/> Utilizes `sklearn.impute.IterativeImputer()` with `mputation_order="random"`. ***Note***: this function has not been thoroughly tested.
- **Parameters**
  - **var_names: *numpy ndarray* with shape = (n,)** <br/>
    The array of the names of the variables to be included in the imputation process.
  - **impute_target: "tri" or "arr"** <br/>
    The type of array used for imputation. Default = **"tri"**. If `impute_target = "tri"`, then the flattened arrays of the upper triangular of the dissimilarity matrices will be used. If `impute_target = "arr"`, then the original 1D variable arrays will be used.
  - **update: *bool*** <br/>
    Indicate whether the user wants to update the results of imputation to the targeted array.
- **Returns**
  - **result: *numpy ndarray* with shape = (n, b)** <br/>
    The imputation results, with the first dimension being variables and the second dimension being the dimension of the targeted type of array.
<br/><br/> 

<code>**delete_trials**(indices = None, words = None)</code>
<br/> Delete specified trials from the instance. All the class attributes will be updated accordingly, except that *masks_dict* will be set to ***None***. At least one of the two optional parameters must be defined.
- **Parameters**
  - **indices: *None* or *numpy ndarray* with shape = (n,)** <br/>
    The indices of the trials to be deleted. Default = **None**. *Optional*.
  - **words : *None* or *numpy ndarray* with shape = (n,)** <br/>
    The words of the trials to be deleted. Default = **None**. *Optional*.
- **Returns**
  - **indices: *numpy ndarray* with shape = (n,)** <br/>
    The indices of the deleted trials.
  - **Missing_ind : *numpy ndarray* with shape = (n,)** <br/>
    The indices of trials that are specified but not found in the instances. Only defined if **words** is not **None**, and the indices are relative to the **words**. 


## Other Functions
<code>**nearst_neighbor_1D**(stimuli_array, max_diff = 0, range_type = "percentage", stim_val = "average")</code>
<br/> A basic 1D nearst neighbor algorithm. Note that this implementation allows "overlapping", so one value can be in more than one cluster.
- **Parameters**
  - **stimuli_array: *numpy ndarray* with shape = (n,)** <br/>
    The target array to perform the algorithm
  - **max_diff: *int*** <br/>
    Specify the maximum absolute difference between a value and a cluster center for that value to be included into the cluster. Default = **0**. Ignored unless `interpolation = True`. If `max_diff = 0`, only identical values will be included in the same cluster.
  - **range_type: "percentage" or "raw"** <br/>
    The type of value for **max_diff**. Default = **percentage**. Ignored unless `interpolation = True`. If set to **"percentage"**, the maximum difference will be the set percentage of the overall range of the dataset. If set to **"raw"**, the maximum difference will be the value specified for **max_diff**.
  - **stim_val: "average" or "central"** <br/>
    The type of value that will become the value of a cluster. Default = **"average"**. Ignored unless `interpolation = True`. If set to **"average"**, the cluster value will be the average of all values in the cluster. If set to **central**, the cluster value will be the value at the center of the cluster.
- **Returns**
  - **NN_dict: *dict*** <br/>
  The nearest-neighbor dictionary. The keys will be the value in the center of the clusters, and the values will be `(mask, index)`. The `mask` will be a boolian ***numpy ndarray*** with `True` indicating the values that belong to the current cluster. The `index` will be a numerical ***numpy ndarray*** that contains the indexes of values that belong to the current cluster. Both the `mask` and the `index` are defined relatively to **stimuli_array**.
<br/><br/>

<code>**matrix_iteration**(data_array, target_matrix, function, skip_diagonal=True)</code>
<br/> A function that provides a template for iterative pair-wise comparison operation. Note that this function does not return anything, but modify the **target_matrix**.
- **Parameters**
  - **data_array: *numpy ndarray* with shape = (n,)** <br/>
    A 1D array to form the matrix
  - **target_matrix: *numpy ndarray* with shape = (n, n)** <br/>
    The 2D array in which the result will be stored. Note that **target_matrix** must be a square matrix, and the two dimensions must agree with the first dimension of **data_array**
  - **function: *function*** <br/>
    A function that specifies the pairwise operation. **function** should appear in the from of `function(x,y)` where `x,y` are the two items in the **data_array** . The value returned will be stored in the corresponding position in the **target_matrix**.
  - **skip_diagonal: *bool*** <br/>
    A boolian value specifying whether the calculation will skip the diagonal values. Default = **True**.
<br/><br/>
