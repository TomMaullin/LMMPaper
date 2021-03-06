# A Fisher Scoring approach for crossed multiple-factor Linear Mixed Models: Examples and Simulations

This repository contains all code for the examples and simulations presented in the paper `A Fisher Scoring approach for crossed multiple-factor Linear Mixed Models`. Python notebooks are provided to ensure all results can be reproduced.
  
## Table of contents
   * [How to cite?](#how-to-cite)
   * [Contents overview](#contents-overview)
   * [Reproducing results](#reproducing-results)

## How to cite?

A BibTex citation for this repository can be found in the [CITATION](CITATION) file. Please replace the commit field with the current commit tag when citing the repository.

# Contents overview

In this repository there are 4 folders:

 - `results`: This folder contains 4 Python notebooks. These notebooks detail how the code used in this repository may be used to run;
   - The parameter estimation simulations.
   - The T statistic simulations.
   - The SAT score example.
   - The twin study example.
 - `data`: This folder contains the data for the SAT score example given in the paper. Any simulated data generated by the notebooks will also be saved here.
 - `src`: This folder contains all functions and code employed to run the Fisher Scoring algorithms.

To understand the code in this repository, it is strongly recommended that you begin with the Python notebooks in the `results` folder to get an understanding of how the code can be run. Following this, in each folder a `README.md` file has been provided. These files give a brief overview of the contents of each folder, and should help you to understand the code and it's layout.

## Reproducing results

The Python notebooks in the `results` folder can be used to reproduce the results of the SAT score example and twin study example exactly. The data for the SAT score example is from [the longitudinal evaluation of school change and performance (LESCP) dataset](https://www2.ed.gov/offices/OUS/PES/esed/lescp_highlights.html); a dataset which is publicly available and free to use (for example, it can be found as one of the example datasets in the HLM software package). An extract of this data, which was used to perform the analysis described in the paper, is found in the `data` folder. The data for the Twin study example is from the Wu-Minn Human Connectome Project (HCP) dataset [(Van Essen et al. (2013))](https://pubmed.ncbi.nlm.nih.gov/23684880/) and is not publicly available. To run the Python notebook for this example you must first obtain the data from the [Wu-Minn HCP cohort website](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms). The notebook will explain how the analysis can be run once the data has been obtained.

The Python notebooks in the `results` folder also demonstrate how the simulations can be run individually and how simulation results can be collated. As a large number of simulations were run for the paper, these notebooks act only as a guide and sketch outline for how to run the same volume of simulations. The notebooks provide all detail necessary to run individual simualtion instances locally but it is suggested that to run a larger volume of simulations, a cluster should be used for computation, with custom scripts for job submission. Full discussion and guidance on how this can be done is given in the notebooks.
