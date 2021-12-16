#!/bin/bash

# Script to plot results for each regularizer across a list of datasets.
# Usage: ./plot_all_reg.sh model

model=$1

source activate pytorch

for dataset in "compas" "blogData" "loanData"; do
    python plot_curves.py --dataset $dataset --model $model
done

echo "DONE."
