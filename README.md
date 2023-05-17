# SPIN-AI
Spatially Infromed Artificial Intelligence. A deep learning method for identifying Spatially Predictive Genes from spatial transcriptomics data.

1. Prerequisites

SPIN-AI was built using the following software versions:

python 3.10.0

pandas 1.4.3

numpy 1.23.1

tensorflow 2.9.0 

keras 2.9.0

matplotlib 3.5.2

For running the attributions script, we recommend a virtual environment with:

python 3.7.5

tensorflow 1.15.0

pandas 1.3.5

numpy 1.21.6

matplotlib 3.5.3

2. Input Data Format

For each spatial transcriptomics experiment, a directory should be created as [experiment_name]/Processed_Data/

Within the directory should be a spot by counts matrix (counts.csv) and a coordinates matrix (coords.csv) with spots as rows and coordinates as columns.

3. Workflow

**build_models.py**: builds the SPIN-AI models and performs hyperparameter tuning. This script will create a CV folder within the experiment directory that contains performance metrics, predictions, and models for each cross-validation fold and for each hyperparameter.

Parameters: <(string)experiment_name>

**Output**: Outputs are deposited in the [experiment_name]/CV/ directory

[number_of_layers_learningRate]/predictions.csv: test fold predictions from a model trained with hyperparameters in the directory name

[number_of_layers_learningRate]/Models/: saved models from each iteration of cross-validation

cv_results.csv: CV error for each hyperparameter set

fold_ids.csv: CV fold assignments for each spot

pred.png: visualization of prediction error across different spots in the spatial transcriptomic sample

**compute_attributions.py**: computes feature attributions using DeepLift on the models with the best hyperparameter combination as determined in the previous step. The DeepLift reference is set to a matrix of 0s, but other references can be manually specified.

Parameters: <(string)experiment_name>

**Output**: Outputs are deposited in the [experiment_name]/CV/ directory

/Attribution/: Directory that contains all DeepLift attributions for all input genes for spots from each CV fold. Attributions are stored as a csv file for each fold.

**get_spgs.py**: calculates SPGs from the attribution files. SPG criteria can be controlled using parameters.

Parameters: <(string)experiment_name>  <(float)meanImportance_threshold> <(float [0-1])percent_non_zero_importance_threshold>

**Output**: Outputs are deposited in the [experiment_name]/CV/ directory

/Attribution/spgTable.csv: table containing the mean importance, MNI, and PNI scores for all genes and whether or not the gene is an SPG.
