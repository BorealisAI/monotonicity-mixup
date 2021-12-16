#!/bin/bash

# Script to run a single training run for each regularizer and save per epoch performance metrics.
# Usage: ./log_all_reg.sh dataset model. This will perform the training run locally.

dataset=$1
model=$2

cd exp

source activate pytorch

for regularization_strategy in "none" "cmn" "train" "mixup" "mixup_random" "cmn_train" "cmn_mixup" "cmn_mixup_train"; do
    python train.py --reg_mode $regularization_strategy --dataset $dataset --model $model --log_only --no_cp --quiet
done

echo "DONE."
