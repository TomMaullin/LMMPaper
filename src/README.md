# src folder

This folder contains the following four subfolders:

 - `lib`: This folder contains all code which is shared between the examples and simulations (i.e. general helper functions, the FS algorithms, test data generation,... etc).
 - `SATExample`: This folder contains backend code specifically used for the SAT Scores Example.
 - `Simulations`: This folder contains backend code specifically used for the parameter estimation and T-statistic simulations.
 - `TwinExample`: This folder contains backend code specifically used for the Twin Study Example.

 A full breakdown of the files in each of these folders is given below.

## lib

The `lib` folder contains the following files:

 - `npMatrix2d.py`: Helper functions for 2d numpy array operations.
 - `npMatrix3d.py`: Helper functions for 3d numpy array operations.
 - `genTestDat.py`: Functions for generating test data
 - `est2d.py`: Parameter estimation methods for inference on one model.
 - `est3d.py`: Parameter estimation methods for inference on multiple models.

Throughout these files, the conventional suffixes `2d` and `3d` are often employed. A function of a file with the suffix `2d` contains code which is designed to perform parameter estimation for a single model. The suffix 2D refers to the fact that for one model, the matrices `X`, `Y` and `Z` are treated as 2-dimensional throughout the code. A function or file with the suffix `3d` will contain code designed to perform parameter estimation for multiple models concurrently. As X,Y and Z are all 3 dimensional (an extra dimension has been added for "model number"), all arrays considered in these pieces of code are 3-dimensional, hence the suffix. The 3D code is used only to speed up the ground truth computation for the T-statistic degrees of freedom. Other than the ground truth degrees of freedom estimate for the T-statistics, to ensure fair comparison, none of the results presented in the paper were derived using the 3D code.

## SATExample

The `SATExample` folder contains the following file:

 - `SATsExample.r`: Code for running the SATs example using `lmer`. Further details can be found in `results/SATsExample.ipynb`.

## Simulations

The `Simulations` folder contains the following files:

 - `Boxplots.r`: Script used to generate degrees of freedom boxplots using simulation results.
 - `LMMPaperSim.py`: Python functions for setting up simulations and running the Fisher Scoring methods on the simulated data. 
 - `LMMPaperSim.r`: R functions for running `lmer` on the simulated data.
 - `LMMPaperSim.sh`: Bash scripts that may be adapted to run the simulations on a cluster set-up.

Further details can be found in `results/ParameterEstimationSimulations.ipynb` and `results/TStatisticSimulations.ipynb`.

## TwinExample

The `TwinExample` folder contains the following files:

 - `ACE.py`: Functions used to perform parameter estimation for the ACE model.
 - `hcp2blocks.m`: Function which categorizes family units by type. This code is taken from (`andersonwinkler/HCP`)[https://github.com/andersonwinkler/HCP].

Further details can be found in `results/TwinsExample.ipynb`.