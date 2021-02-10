# A Comparative Study of Gamma Markov Chains for Temporal Non-Negative Matrix Factorization

This repository contains the code used to run the experimental work of the article "A Comparative Study of Gamma Markov Chains for Temporal Non-Negative Matrix Factorization", published in IEEE Transactions on Signal Processing, 2021.
arXiv : https://arxiv.org/abs/2006.12843

## Contents

### Codes

The code is run through **main.py**, which itself calls five *MAP_*.py* scripts, which correspond to the five models considered in the experiments. The script *config.py* contains several global parameters of the experiments.

### Datasets

## Dependencies
Implemented in Python 3.7
- Numpy
- Scipy
- Sys

## Instructions to run the experiments (Section IV.C)

Experiments are run in the following way
> python main.py [path_to_dataset] [factorization_rank] [seed]
The code then produces a .txt file containing a LaTeX-compliant table of results.

The experiments of the article can be reproduced using the following prompts
> python main.py data_NIPS.npy 3 20200928
> python main.py data_lastfm.npy 5 20200928
> python main.py data_ICEWS.npy 5 20200928