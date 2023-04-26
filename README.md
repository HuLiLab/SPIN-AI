# SPIN-AI
Spatially Infromed Artificial Intelligence. A deep learning method for identifying Spatially Predictive Genes from spatial transcriptomics data.

1. Prerequisites
2. Input Data Format

For each spatial transcriptomics experiment, a directory should be created as [experiment_name]/Processed_Data/

Within the directory should be a spot by counts matrix and a coordinates matrix with spots as rows and coordinates as columns.

3. Workflow

build_models.py: builds the SPIN-AI models and performs hyperparameter tuning. The __main__ section can be altered to suit the experiment directory structure. This script will create a CV folder within the experiment directory that contains performance metrics, predictions, and models for each cross-validation fold and for each hyperparameter.

compute_attributions.py: computes feature attributions using DeepLift on the models with the best hyperparameter combination as determined in the previous step. The DeepLift reference is set to a matrix of 0s, but other references can be manually specified.

visualization.py: visualizes a spatial transcriptomics slide colored by the distance of each spot to their original location.

get_spgs.py: calculates SPGs from the attribution files. SPG criteria can be controlled using parameters.
