#!/bin/bash

# Usage: ./synth_train_all_reg.sh model

model=$1

cd exp

source activate pytorch

for regularization_strategy in "none" "cmn" "train" "mixup_random"; do
    python synth_train.py --reg_mode ${regularization_strategy} --model $model --quiet --verify > synt_${regularization_strategy}.out
done
