#!/bin/bash

# Usage: ./submit_all_reg.sh dataset model

dataset=$1
model=$2

for regularization_strategy in "none" "cmn" "train" "mixup_random"; do
    python experiment_launcher.py --reg_mode ${regularization_strategy} --dataset $dataset --model $model
    sleep 2
done
