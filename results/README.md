# results folder

This folder contains 4 Python notebooks. These notebooks detail how the code used in this repository may be used to run;

 - `ParameterEstimationSimulations.ipynb`: The parameter estimation simulations.
 - `TStatisticSimulations.ipynb`: The T-statistic simulations.
 - `SATsExample.ipynb`: The SAT score example.
 - `TwinsExample.ipynb`: The twin study example.

Through the notebooks, all Python code related to this project can be run. In addition, the notebooks also provide explicit detail on how to:

 - Obtain the [Wu-Minn HCP dataset](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms) for the Twins example.
 - Preprocess the Twins dataset using the `MatLab` function `hcp2blocks.m`, borrowed from [`andersonwinkler/HCP`](https://github.com/andersonwinkler/HCP) and found in `src\TwinExample`. 
 - Run the `R` functions included in this repository to obtain the `lmer` results used for comparison.
 - Run the `bash` scripts used for simulation cluster submission.

If anything about the notebooks is unclear, please leave an open issue on the GitHub repository.
