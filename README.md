# SPIN-AI
Spatially Infromed Artificial Intelligence. A deep learning method for identifying Spatially Predictive Genes from spatial transcriptomics data.

1. Prerequisites
2. Input Data Format

For each spatial transcriptomics experiment, a directory should be created as [experiment_name]/Processed_Data/

Within the directory should be a spot by counts matrix (counts.csv) and a coordinates matrix (coords.csv) with spots as rows and coordinates as columns.

3. Workflow

build_models.py: builds the SPIN-AI models and performs hyperparameter tuning. This script will create a CV folder within the experiment directory that contains performance metrics, predictions, and models for each cross-validation fold and for each hyperparameter.

Parameters: (string)experiment_name

compute_attributions.py: computes feature attributions using DeepLift on the models with the best hyperparameter combination as determined in the previous step. The DeepLift reference is set to a matrix of 0s, but other references can be manually specified.

Parameters: (string)experiment_name

visualization.py: visualizes a spatial transcriptomics slide colored by the distance of each spot to their original location.

get_spgs.py: calculates SPGs from the attribution files. SPG criteria can be controlled using parameters.

Parameters: (string)experiment_name  (float)meanImportance_threshold (float [0-1])percent_non_zero_importance_threshold
