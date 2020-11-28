# EEG_RSA
Toolboxes for my RSA analysis, with associate classes and functions. There are two kind of files here: independent toolboxes as classes or standalone functions, and main functions that requires the toolboxes to function.

# Author
Feng Cheng, Tufts University, Research Assistant at NeuroCognition of Language Lab

## Environment File: 
**RSA_env.yml**

## Toolboxes:
#### variables_processing_alt.py
- **Description** A class for storing variables. It is recommended to store all variables together into one instance, since many functions operate on multiple variables simultanously.
- **Status**: In Progress. Lack comments; need to rearrange some functions for simpler interactions.
- **Functionality** (refer to individual functions for details)
  - Calculate Representational Disimilarity Matrix (RDM) using a customizable function
  - Manage the variables through masking/re-ordering
  - Export variables and associated RDMs
  - Prepare upper-triangular RDM for RSA 
  - Calculate correlation (Partial/Non-partial) between variables values and their RDMs
  - Impute missing values with iterative imputation **(not tested yet)**
  - Calculate cosine similarity from a given word embedding corpus
#### Single_Trial_RSA.py
- **Description** A class for performing representational similarity analysis (RSA).
- **Status**: Finished. Lack comments.
- **Functionality** (refer to individual functions for details)
  - Calculate RSA with partial/non-partial correlation. Can also define timewindow and step increment for RSA.
#### Regression_Analysis.py
- **Description** Two classes for performing liner regression.
- **Status**: Finished.
- **Functionality** (refer to individual functions for details)
  - Recursively compute linear regression on all dimensions
  - A group regression wrapper for regressional ERP analysis
#### RSA_plot.py
- **Description** A group of functions that uses matplotlib for plotting
- **Status**: In Progress. Lack comments; need to rearrange some functions for simpler interactions.
- **Functionality** (refer to individual functions for details)
  - Customized functions for plotting 1D/2D correlation data 
  - Customized functions for plotting correlation matrix
  - Customized functions for plotting dendrograms
#### PPData_alt.py
- **Description** A class for simple preprocessing of EEG and variable data. Suitable for importing several dataset in our lab.
- **Status**: Finished.
- **Functionality** (refer to individual functions for details)
  - Import data from txt/csv files
  - Clear pre-labeled artifact trials
  - Reassign bin labels to EEG data
#### Clustering_Auxiliaries.py
- **Description** A group of functions for clustering/dimension analysis
- **Status**: Finished. Lack comments.
- **Functionality** (refer to individual functions for details)
  - Perform Principal Coordinate Analysis (PCoA, a.k.a. classical MDS) on a given representational dissimilarity matrix (RDM)
  - Analyze the principal coordinates by correlating results with variables
  - Recover full tree structure from the result of sklearn AgglomerativeClustering function. The tree is arranged as a binary tree, and each parent cluster is formed by merging the two children sub-clusters. A dictionary coding the clusters at each layer of the tree in the form of mask is also returned for easy evaluation (if at some level a cluster reaches its terminal stage and contains only 1 member, the cluster label of the cluster will be inhereited in the deeper layer). 
  - Plot dendrograms (some obtained from scipy examples)
  - Evaluate performances of a group of hierarchical cluster
#### CPerm.py
- **Description** A class for performing 1D cluster permutation test
- **Status**: Finished.
- **Functionality** (refer to individual functions for details)
  - 1D cluster permutation
#### Auxiliaries.py
- **Description** A group of simple auxiliary functions
- **Status**: Finished.
