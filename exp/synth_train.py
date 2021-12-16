# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import sys

sys.path.append("../")
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import constants
import data_utils.syntheticData_loader as syntheticData_data
from utils.networks import *
from utils.common import *
from utils.train_test_func import *
from utils.monotonicity_check import monotonicity_check

import random

random.seed(42)

DATASET_ID = "syntheticData"
# Those should have the same length
NUM_DIMENSIONS = (100, 200, 400, 500)
NUM_MONOTONIC_DIMENSIONS = (20, 40, 80, 100)

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
    "--model",
    choices=["base_MLP", "cmn_MLP"],
    default="cmn_MLP",
    help="Model architecture.",
)
parser.add_argument(
    "--verify",
    action="store_true",
    default=False,
    help="Enables monotonicity data-dependent verification.",
)
parser.add_argument(
    "--quiet", action="store_true", default=False, help="Disables logging on stdout.",
)
parser.add_argument(
    "--no_cp", action="store_true", default=False, help="Disables model checkpointing.",
)
parser.add_argument(
    "--log_only",
    action="store_true",
    default=False,
    help="Save per epoch performance.",
)
parser.add_argument(
    "--dry_run",
    action="store_true",
    default=False,
    help="Runs a single training iteration and stops.",
)
args = parser.parse_args()

# Printing out args for logging.
args_dict = dict(vars(args))
for k, v in args_dict.items():
    print(f"{k}: {v}")

# Printing out config for logging
for constant in dir(constants):
    if "__" not in constant:
        value = getattr(constants, constant)
        if isinstance(value, dict):
            # Printing out data-dependent config
            for k, v in value[DATASET_ID].items():
                print(f"{k}: {v}")
        else:
            print(f"{constant}: {value}")

# Setting up device usage
CUDA = True if not constants.NO_CUDA and torch.cuda.is_available() else False

if CUDA:
    DEVICE = torch.device(f"cuda:0")
else:
    DEVICE = torch.device("cpu")

# Data preparation

# Get data loading function corresponding to DATASET_ID
for atrib in list(globals().keys()):
    if DATASET_ID in atrib:
        data_load_func = atrib


if __name__ == "__main__":

    if constants.CONFIG[DATASET_ID]["TASK"] == "classification":
        criterion = nn.BCEWithLogitsLoss()
        TEST_METRIC = "accuracy"
        TEST_PERF_BETTER_HIGH = (
            True  # Flag to indicate the test performance is better when higher.
        )
    elif constants.CONFIG[DATASET_ID]["TASK"] == "regression":
        criterion = nn.MSELoss()
        TEST_METRIC = "RMSE"
        TEST_PERF_BETTER_HIGH = (
            False  # Flag to indicate the test performance is better when lower.
        )
    else:
        raise ValueError(
            "Unknown task. Check the task field in CONFIG within constants.py."
        )

    for i in range(len(NUM_DIMENSIONS)):
        dimensions = NUM_DIMENSIONS[i]
        monotonic_dimensions = NUM_MONOTONIC_DIMENSIONS[i]
        # Generate data
        X_train, y_train, X_val, y_val, X_test, y_test = globals()[
            data_load_func
        ].load_data(
            path_to_data=constants.CONFIG[DATASET_ID]["DATA_PATH"],
            n_dimensions=dimensions,
            n_monotonic_dimensions=monotonic_dimensions,
            valid_split=constants.CONFIG[DATASET_ID]["VALIDATION_DATA_FRACTION"],
        )

        data_train = Data.TensorDataset(X_train, y_train)

        data_train_loader = Data.DataLoader(
            dataset=data_train,
            batch_size=constants.CONFIG[DATASET_ID]["BATCH_SIZE"],
            shuffle=True,
            num_workers=constants.NUM_WORKERS,
        )

        val_perf_list = []
        test_perf_list = []
        non_monotonic_frac_random_list = []
        non_monotonic_frac_train_list = []
        non_monotonic_frac_test_list = []
        non_monotonic_frac_mixup_list = []

        for run in range(constants.CONFIG[DATASET_ID]["NUM_RUNS"]):

            # Training setup
            net = globals()[args.model](
                mono_feature=monotonic_dimensions,
                non_mono_feature=dimensions - monotonic_dimensions,
                sub_num=constants.CONFIG[DATASET_ID]["SUB_NUM"],
                mono_hidden_num=constants.CONFIG[DATASET_ID]["MONO_HIDDEN_NUM"],
                non_mono_hidden_num=constants.CONFIG[DATASET_ID]["NON_MONO_HIDDEN_NUM"],
                bottleneck=dimensions
                // 2,  # Grows the bottleneck as the number of dimensions grows
                use_batch_norm=constants.CONFIG[DATASET_ID]["USE_BATCH_NORM"],
            )

            net = net.to(DEVICE)

            optimizer = torch.optim.Adam(
                net.parameters(), lr=constants.CONFIG[DATASET_ID]["LR"]
            )

            cp_full_path = os.path.join(
                constants.CP_PATH,
                f"{args.model}_{args.reg_mode}_{DATASET_ID}_BN:{constants.CONFIG[DATASET_ID]['USE_BATCH_NORM']}_L1:{constants.CONFIG[DATASET_ID]['L1_PENALTY_COEF']}_RUN:0.pt",
            )

            logs_full_path = os.path.join(
                constants.CP_PATH,
                f"{args.model}_{args.reg_mode}_{DATASET_ID}_BN:{constants.CONFIG[DATASET_ID]['USE_BATCH_NORM']}_L1:{constants.CONFIG[DATASET_ID]['L1_PENALTY_COEF']}_PERFLOG.pt",
            )

            val_perf = 0.0 if TEST_PERF_BETTER_HIGH else float("inf")
            test_perf = 0.0 if TEST_PERF_BETTER_HIGH else float("inf")
            min_grad_curr = float("inf")

            for epoch in range(constants.CONFIG[DATASET_ID]["NUM_EPOCHS"]):
                train(  # Trains for an epoch
                    net,
                    optimizer,
                    criterion,
                    data_train_loader,
                    constants.CONFIG[DATASET_ID][
                        "L1_PENALTY_COEF"
                    ],  # Only has effect if set to value > 0.
                    constants.CONFIG[DATASET_ID]["CMN_REGULARIZATION_LAMBDA"],
                    constants.CONFIG[DATASET_ID]["MIXUP_REGULARIZATION_LAMBDA"],
                    monotonic_dimensions,
                    constants.CONFIG[DATASET_ID]["REGULARIZATION_BUDGET"],
                    args.reg_mode,
                    None if args.quiet else epoch,
                    adversarial_training=(
                        args.reg_mode == "cmn_adv" or args.reg_mode == "none_adv"
                    ),
                )

                if args.dry_run:
                    exit(1)

                val_perf_cur = test(  # Evaluates on validation data
                    net,
                    X_val,
                    y_val,
                    monotonic_dimensions,
                    None if args.quiet else f"Val. {TEST_METRIC}",
                    regression=isinstance(
                        criterion, nn.MSELoss
                    ),  # Flag for RMSE evaluation instead of accuracy
                )

                # Logs results for model with best val. performance in the current run
                if (TEST_PERF_BETTER_HIGH and val_perf_cur > val_perf) or (
                    not TEST_PERF_BETTER_HIGH and val_perf_cur < val_perf
                ):

                    val_perf = val_perf_cur

                    test_perf = test(  # Evaluates on test data only for best models on validation data.
                        net,
                        X_test,
                        y_test,
                        monotonic_dimensions,
                        None if args.quiet else f"Test {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    val_robust_perf = adversarial_test(  # Robust evaluation on validation data
                        net,
                        X_val,
                        y_val,
                        monotonic_dimensions,
                        None if args.quiet else f"Robust Val. {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    test_robust_perf = adversarial_test(  # Robust evaluation on validation data
                        net,
                        X_test,
                        y_test,
                        monotonic_dimensions,
                        None if args.quiet else f"Robust Test {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    if not args.no_cp:
                        torch.save(net.state_dict(), cp_full_path)

                    if not args.quiet:
                        print(
                            f"Best (Val. {TEST_METRIC}, Test {TEST_METRIC} ): {val_perf:.3f}, {test_perf:.3f}"
                        )
                        print(
                            f"Best (Robust Val. {TEST_METRIC}, Robust Test {TEST_METRIC} ): {val_robust_perf:.3f}, {test_robust_perf:.3f}"
                        )

            # Logging best performances at each run for computing confidence intervals
            val_perf_list.append(val_perf)
            test_perf_list.append(test_perf)

            # Data-dependent monotonicity checks
            if args.verify:
                if not args.no_cp:
                    net.load_state_dict(torch.load(cp_full_path, map_location=DEVICE))
                non_monotonic_frac_random = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[DATASET_ID][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=monotonic_dimensions,
                    n_features=X_train.shape[1],
                    data_x=None,
                    mixup=False,
                )
                non_monotonic_frac_train = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[DATASET_ID][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=monotonic_dimensions,
                    data_x=X_train,
                    mixup=False,
                )
                non_monotonic_frac_test = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[DATASET_ID][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=monotonic_dimensions,
                    data_x=X_test,
                    mixup=False,
                )
                non_monotonic_frac_mixup = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[DATASET_ID][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=monotonic_dimensions,
                    data_x=X_train,
                    mixup=True,
                )
                non_monotonic_frac_random_list.append(100.0 * non_monotonic_frac_random)
                non_monotonic_frac_train_list.append(100.0 * non_monotonic_frac_train)
                non_monotonic_frac_test_list.append(100.0 * non_monotonic_frac_test)
                non_monotonic_frac_mixup_list.append(100.0 * non_monotonic_frac_mixup)

        print(
            f"\nRESULTS FOR {monotonic_dimensions} MONOTONIC DIMENSIONS OUT OF {dimensions} DIMENSIONS:\n"
        )
        val_perf_mean, val_perf_ci95 = make_confidence_interval(val_perf_list)
        test_perf_mean, test_perf_ci95 = make_confidence_interval(test_perf_list)

        if isinstance(criterion, nn.MSELoss):  # Regression case.
            print(f"Validation {TEST_METRIC}: {val_perf_mean:.3f}+-{val_perf_ci95:.3f}")
            print(f"Test {TEST_METRIC}: {test_perf_mean:.3f}+-{test_perf_ci95:.3f}")
        else:  # Classification case.
            print(
                f"Validation {TEST_METRIC}: {val_perf_mean:.1f}%+-{val_perf_ci95:.1f}%"
            )
            print(f"Test {TEST_METRIC}: {test_perf_mean:.1f}%+-{test_perf_ci95:.1f}%")

        # Verification results
        if args.verify:
            (
                non_monotonic_frac_random_mean,
                non_monotonic_frac_random_ci95,
            ) = make_confidence_interval(non_monotonic_frac_random_list)
            (
                non_monotonic_frac_train_mean,
                non_monotonic_frac_train_ci95,
            ) = make_confidence_interval(non_monotonic_frac_train_list)
            (
                non_monotonic_frac_test_mean,
                non_monotonic_frac_test_ci95,
            ) = make_confidence_interval(non_monotonic_frac_test_list)
            (
                non_monotonic_frac_mixup_mean,
                non_monotonic_frac_mixup_ci95,
            ) = make_confidence_interval(non_monotonic_frac_mixup_list)

            print(
                f"Fraction of non-monotonic random points: {non_monotonic_frac_random_mean:.2f}%+-{non_monotonic_frac_random_ci95:.2f}%"
            )
            print(
                f"Fraction of non-monotonic train points: {non_monotonic_frac_train_mean:.2f}%+-{non_monotonic_frac_train_ci95:.2f}%"
            )
            print(
                f"Fraction of non-monotonic test points: {non_monotonic_frac_test_mean:.2f}%+-{non_monotonic_frac_test_ci95:.2f}%"
            )
            print(
                f"Fraction of non-monotonic mixup points: {non_monotonic_frac_mixup_mean:.2f}%+-{non_monotonic_frac_mixup_ci95:.2f}%"
            )

