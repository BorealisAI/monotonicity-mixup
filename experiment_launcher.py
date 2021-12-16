# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

## Script to write temporary shell script to submit job with slurm.

# Usage example: python experiment_launcher.py --reg_mode mixup_random --dataset compas --job-submission-command sbatch --command-template-file ./slurm_template.txt
# Local execution example:  python experiment_launcher.py --reg_mode mixup_random --dataset compas --job-submission-command none --command-template-file ./base_template.txt

import os
import subprocess
import argparse

TEMP_SUBMISSION_SCRIPT = "./tmp"

# Training options
parser = argparse.ArgumentParser(description="Training monotonic models.")
parser.add_argument(
    "--reg_mode",
    choices=[
        "none",
        "none_adv",
        "mixup",
        "mixup_random",
        "cmn",
        "train",
        "cmn_train",
        "cmn_adv",
        "cmn_mixup",
        "cmn_mixup_train",
    ],
    default="none",
    help="Regularization strategy.",
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
parser.add_argument(
    "--command-template-file",
    type=str,
    default="./base_template.txt",
    help="File containing training command.",
)
parser.add_argument(
    "--job-submission-command",
    type=str,
    default="bash",
    help="File containing training command.",
)
args = parser.parse_args()

if args.job_submission_command == "none" or args.job_submission_command == "None":
    args.job_submission_command = ""

# Reads in the submission template.
with open(args.command_template_file, "r") as f:
    submission_script_str = f.read()

# Fills in the command line options.
submission_script_str = submission_script_str.format(
    args.reg_mode, args.dataset, args.model, args.reg_mode, args.dataset, args.model
)

# Writes submission script to temporary file.
with open(TEMP_SUBMISSION_SCRIPT, "w") as f:
    f.write(submission_script_str)

# Submits job.
cmd = f"{args.job_submission_command} {TEMP_SUBMISSION_SCRIPT}"
p = subprocess.Popen(cmd, stderr=subprocess.PIPE, shell=True)
(out, err) = p.communicate()
p.wait()

# Cleans up
os.remove(TEMP_SUBMISSION_SCRIPT)
