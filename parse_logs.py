# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#  Script to parse slurm log files and generate summary into a .csv

# Usage example: python parse_logs.py --dataset compas --model cmn_MLP

import glob
import argparse
import pandas as pd

# Template of slurm stdout files with results.
# The 3 fields are: reg_mode, dataset, model
LOG_NAME_TEMPLATE = "monotonicity_{}_{}_{}.*.out"

# Output file name template
OUT_NAME_TEMPLATE = "results/monotonicity_{}_{}.csv"

# Set of regularization options.
REGULARIZATION_MODES = {
    "none",
    "mixup",
    "mixup_random",
    "cmn",
    "train",
    "cmn_train",
    "cmn_mixup",
    "cmn_mixup_train",
}

FIELDS = {  # Result fields to look for within log files.
    "Validation accuracy",
    "Test accuracy",
    "Robust Validation accuracy",
    "Robust Test accuracy",
    "Validation RMSE",
    "Test RMSE",
    "Robust Validation RMSE",
    "Robust Test RMSE",
    "Fraction of non-monotonic random points",
    "Fraction of non-monotonic train points",
    "Fraction of non-monotonic test points",
    "Fraction of non-monotonic mixup points",
}

# Options
parser = argparse.ArgumentParser(
    description="Parsing results of trained monotonic models."
)
parser.add_argument(
    "--dataset",
    choices=["compas", "blogData", "loanData", "xrayData", "syntheticData"],
    default="compas",
    help="Dataset.",
)
parser.add_argument(
    "--model",
    choices=["base_MLP", "cmn_MLP"],
    default="cmn_MLP",
    help="Model architecture.",
)
args = parser.parse_args()


# Dictionary to gather results
# First key indicates the regularization approach and
# the second one indicates the field
results = {k: {} for k in REGULARIZATION_MODES}

# Iterate over regularization modes

for reg_mode in REGULARIZATION_MODES:
    # There should be a single file per reg_mode/dataset/model combination.
    # If multiple files exist, only the first one is considered.
    log_file = glob.glob(LOG_NAME_TEMPLATE.format(reg_mode, args.dataset, args.model))

    if len(log_file) < 1:
        continue  # No file found for the given reg_mode/dataset/model combination.

    # Reads data of first file only.
    with open(log_file[0], "r") as f:
        file_data = f.readlines()

    for line in file_data:
        try:
            k, v = line.split(": ")
            if k in FIELDS:
                results[reg_mode][k] = v.strip()
        except ValueError:
            continue  # Lines that cannot be split by ': ' do not correspond to evaluation metrics.

# Creates data frame out of results
df = pd.DataFrame(results)

# Dumps results to csv
df.to_csv(OUT_NAME_TEMPLATE.format(args.dataset, args.model))
