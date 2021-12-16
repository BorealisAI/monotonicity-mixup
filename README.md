# Enforcing monotonicity in neural networks

This repo contains the code for the paper: 
#### [Not Too Close and Not Too Far: Enforcing Monotonicity Requires Penalizing The Right Points](https://openreview.net/forum?id=xdFqKVlDHnY) 
by Joao Monteiro<sup>1</sup>, Mohamed Osama Ahmed<sup>2</sup>, Hossein Hajimirsadeghi<sup>2</sup>, and Greg Mori<sup>2</sup>

1. Institut National de la Recherche Scientifique
2. Borealis AI

## Running experiments

We provide scripts to easily launch experiments once requirememnts are installed.

Examples:

```
./submit_all_reg.sh blogData cmn_MLP
```

Or, for experiments with synthetic data:

```
./synth_train_all_reg.sh cmn_MLP
```

### Data preparation

Data needs to be prepared in advance and placed under ./exp/data/

We provide scripts to prepare data and to generate the data required for synthetic experiments under ./data_utils/

Raw data for a subset of the datasets we consider can be found at:

- COMPAS: https://github.com/gnobitab/CertifiedMonotonicNetwork/blob/main/compas/compas_scores_two_years.csv
- BlogFeedback: https://archive.ics.uci.edu/ml/datasets/BlogFeedback#.

## Citation:

```
@inproceedings{
monteiro2021not,
title={Not Too Close and Not Too Far:  Enforcing Monotonicity Requires Penalizing The Right Points},
author={Joao Monteiro and Mohamed Osama Ahmed and Hossein Hajimirsadeghi and Greg Mori},
booktitle={eXplainable AI approaches for debugging and diagnosis.},
year={2021},
url={https://openreview.net/forum?id=xdFqKVlDHnY}
}
```