# Copyright (c) 2021-present, Royal Bank of Canada.
# Copyright (c) 2021-present, https://github.com/gnobitab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the implementation from CMN:
# https://github.com/gnobitab/CertifiedMonotonicNetwork
####################################################################################

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
import data_utils.compas_loader as compas_data
import data_utils.blogData_loader as blogData_data
import data_utils.loanData_loader as loanData_data
import data_utils.xrayData_loader as xrayData_data
import data_utils.syntheticData_loader as syntheticData_data
from utils.networks import *
from utils.certify import *
from utils.common import *
from utils.train_test_func import *
from utils.monotonicity_check import monotonicity_check

import random

random.seed(42)


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
    "--certify",
    action="store_true",
    default=False,
    help="Enables monotonicity certification.",
)
parser.add_argument(
    "--verify",
    action="store_true",
    default=False,
    help="Enables monotonicity data-dependent verification.",
)
parser.add_argument(
    "--plot_pdp", action="store_true", default=False, help="Enables outputing PDP.",
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
            for k, v in value[args.dataset].items():
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

# Get data loading function corresponding to args.dataset
for atrib in list(globals().keys()):
    if args.dataset in atrib:
        data_load_func = atrib

# Get data
X_train, y_train, X_val, y_val, X_test, y_test = globals()[data_load_func].load_data(
    path_to_data=constants.CONFIG[args.dataset]["DATA_PATH"],
    valid_split=constants.CONFIG[args.dataset]["VALIDATION_DATA_FRACTION"],
)

NUMBER_OF_FEATURES = X_train.shape[1]

data_train = Data.TensorDataset(X_train, y_train)

data_train_loader = Data.DataLoader(
    dataset=data_train,
    batch_size=constants.CONFIG[args.dataset]["BATCH_SIZE"],
    shuffle=True,
    num_workers=constants.NUM_WORKERS,
)

if __name__ == "__main__":

    # Lists to gather per run performance
    # Used afterwards to compute performance confidence intervals
    val_perf_list = []
    test_perf_list = []
    val_robust_perf_list = []
    test_robust_perf_list = []
    certification_min_grad_list = []
    certification_results_list = []
    non_monotonic_frac_random_list = []
    non_monotonic_frac_train_list = []
    non_monotonic_frac_test_list = []
    non_monotonic_frac_mixup_list = []

    if constants.CONFIG[args.dataset]["TASK"] == "classification":
        criterion = nn.BCEWithLogitsLoss()
        TEST_METRIC = "accuracy"
        TEST_PERF_BETTER_HIGH = (
            True  # Flag to indicate the test performance is better when higher.
        )
    elif constants.CONFIG[args.dataset]["TASK"] == "regression":
        criterion = nn.MSELoss()
        TEST_METRIC = "RMSE"
        TEST_PERF_BETTER_HIGH = (
            False  # Flag to indicate the test performance is better when lower.
        )
    else:
        raise ValueError(
            "Unknown task. Check the task field in CONFIG within constants.py."
        )

    # performance_log is only populated if args.log_only is set to True.
    performance_log = {"non_monotonic_frac": [], f"{TEST_METRIC}": []}

    number_of_training_runs = (
        1 if args.log_only else constants.CONFIG[args.dataset]["NUM_RUNS"]
    )

    for run in range(number_of_training_runs):
        # Training setup
        net = globals()[args.model](
            mono_feature=constants.CONFIG[args.dataset]["MONO_FEATURE"],
            non_mono_feature=NUMBER_OF_FEATURES
            - constants.CONFIG[args.dataset]["MONO_FEATURE"],
            sub_num=constants.CONFIG[args.dataset]["SUB_NUM"],
            mono_hidden_num=constants.CONFIG[args.dataset]["MONO_HIDDEN_NUM"],
            non_mono_hidden_num=constants.CONFIG[args.dataset]["NON_MONO_HIDDEN_NUM"],
            bottleneck=constants.CONFIG[args.dataset]["BOTTLENECK"],
            use_batch_norm=constants.CONFIG[args.dataset]["USE_BATCH_NORM"],
        )

        if run == 0:
            param_amount = 0
            for p in net.named_parameters():
                print(p[0], p[1].numel())
                param_amount += p[1].numel()
            print("total param amount:", param_amount)

        net = net.to(DEVICE)

        optimizer = torch.optim.Adam(
            net.parameters(), lr=constants.CONFIG[args.dataset]["LR"]
        )

        cp_full_path = os.path.join(
            constants.CP_PATH,
            f"{args.model}_{args.reg_mode}_{args.dataset}_BN:{constants.CONFIG[args.dataset]['USE_BATCH_NORM']}_L1:{constants.CONFIG[args.dataset]['L1_PENALTY_COEF']}_RUN:{run}.pt",
        )

        logs_full_path = os.path.join(
            constants.CP_PATH,
            f"{args.model}_{args.reg_mode}_{args.dataset}_BN:{constants.CONFIG[args.dataset]['USE_BATCH_NORM']}_L1:{constants.CONFIG[args.dataset]['L1_PENALTY_COEF']}_PERFLOG.pt",
        )

        val_perf = 0.0 if TEST_PERF_BETTER_HIGH else float("inf")
        test_perf = 0.0 if TEST_PERF_BETTER_HIGH else float("inf")
        min_grad_curr = float("inf")

        for epoch in range(constants.CONFIG[args.dataset]["NUM_EPOCHS"]):
            train(  # Trains for an epoch
                net,
                optimizer,
                criterion,
                data_train_loader,
                constants.CONFIG[args.dataset][
                    "L1_PENALTY_COEF"
                ],  # Only has effect if set to value > 0.
                constants.CONFIG[args.dataset]["CMN_REGULARIZATION_LAMBDA"],
                constants.CONFIG[args.dataset]["MIXUP_REGULARIZATION_LAMBDA"],
                constants.CONFIG[args.dataset]["MONO_FEATURE"],
                constants.CONFIG[args.dataset]["REGULARIZATION_BUDGET"],
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
                constants.CONFIG[args.dataset]["MONO_FEATURE"],
                None if args.quiet else f"Val. {TEST_METRIC}",
                regression=isinstance(
                    criterion, nn.MSELoss
                ),  # Flag for RMSE evaluation instead of accuracy
            )

            if args.log_only:
                non_monotonic_frac_val = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[args.dataset][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    data_x=X_val,
                    mixup=False,
                )

                performance_log[f"{TEST_METRIC}"].append(val_perf_cur)
                performance_log["non_monotonic_frac"].append(non_monotonic_frac_val)

            else:

                # Logs results for model with best val. performance in the current run
                if (TEST_PERF_BETTER_HIGH and val_perf_cur > val_perf) or (
                    not TEST_PERF_BETTER_HIGH and val_perf_cur < val_perf
                ):

                    val_perf = val_perf_cur

                    test_perf = test(  # Evaluates on test data only for best models on validation data.
                        net,
                        X_test,
                        y_test,
                        constants.CONFIG[args.dataset]["MONO_FEATURE"],
                        None if args.quiet else f"Test {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    val_robust_perf = adversarial_test(  # Robust evaluation on validation data
                        net,
                        X_val,
                        y_val,
                        constants.CONFIG[args.dataset]["MONO_FEATURE"],
                        None if args.quiet else f"Robust Val. {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    test_robust_perf = adversarial_test(  # Robust evaluation on validation data
                        net,
                        X_test,
                        y_test,
                        constants.CONFIG[args.dataset]["MONO_FEATURE"],
                        None if args.quiet else f"Robust Test {TEST_METRIC}",
                        regression=isinstance(
                            criterion, nn.MSELoss
                        ),  # Flag for RMSE evaluation instead of accuracy
                    )

                    if not args.no_cp:
                        torch.save(net.state_dict(), cp_full_path)

                    if not args.quiet:
                        print(
                            f"Run {run} - best (Val. {TEST_METRIC}, Test {TEST_METRIC} ): {val_perf:.3f}, {test_perf:.3f}"
                        )
                        print(
                            f"Run {run} - best (Robust Val. {TEST_METRIC}, Robust Test {TEST_METRIC} ): {val_robust_perf:.3f}, {test_robust_perf:.3f}"
                        )

        if args.log_only:
            torch.save(performance_log, logs_full_path)
        else:

            # Logging best performances at each run for computing confidence intervals
            val_perf_list.append(val_perf)
            test_perf_list.append(test_perf)
            val_robust_perf_list.append(val_robust_perf)
            test_robust_perf_list.append(test_robust_perf)

            # Overall monotonicity certification
            if args.certify:
                if not args.no_cp:
                    net.load_state_dict(torch.load(cp_full_path, map_location=DEVICE))
                mono_flag, min_gradient = certify_neural_network(
                    net,
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    budget=constants.CONFIG[args.dataset]["CERTIFICATION_BUDGET"],
                )

                certification_min_grad_list.append(min_gradient)
                certification_results_list.append(1.0 if min_gradient > 0 else 0.0)

                if mono_flag:
                    print(f"Run {run}: Certified Monotonic. ")
                else:
                    print(f"Run {run}: Not Monotonic")

            # Data-dependent monotonicity checks
            if args.verify:
                if not args.no_cp:
                    net.load_state_dict(torch.load(cp_full_path, map_location=DEVICE))
                non_monotonic_frac_random = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[args.dataset][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    n_features=NUMBER_OF_FEATURES,
                    data_x=None,
                    mixup=False,
                )
                non_monotonic_frac_train = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[args.dataset][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    data_x=X_train,
                    mixup=False,
                )
                non_monotonic_frac_test = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[args.dataset][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    data_x=X_test,
                    mixup=False,
                )
                non_monotonic_frac_mixup = monotonicity_check(
                    net,
                    number_of_points=constants.CONFIG[args.dataset][
                        "MONOTONICITY_CHECK_BUDGET"
                    ],
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    data_x=X_train,
                    mixup=True,
                )
                non_monotonic_frac_random_list.append(100.0 * non_monotonic_frac_random)
                non_monotonic_frac_train_list.append(100.0 * non_monotonic_frac_train)
                non_monotonic_frac_test_list.append(100.0 * non_monotonic_frac_test)
                non_monotonic_frac_mixup_list.append(100.0 * non_monotonic_frac_mixup)

            # Partial dependency plots
            if args.plot_pdp:
                pdp_full_path = os.path.join(
                    constants.PDP_PATH,
                    f"{args.model}_{args.reg_mode}_{args.dataset}_BN:{constants.CONFIG[args.dataset]['USE_BATCH_NORM']}_L1:{constants.CONFIG[args.dataset]['L1_PENALTY_COEF']}_RUN:{run}.jpg",
                )
                plot_pdp(
                    model=net,
                    data=X_test,
                    n_monotonic_features=constants.CONFIG[args.dataset]["MONO_FEATURE"],
                    out_path=pdp_full_path,
                )

    if not args.log_only:

        # Output final results in terms of 95% confidence intervals

        # Performances
        val_perf_mean, val_perf_ci95 = make_confidence_interval(val_perf_list)
        test_perf_mean, test_perf_ci95 = make_confidence_interval(test_perf_list)

        val_robust_perf_mean, val_robust_perf_ci95 = make_confidence_interval(
            val_robust_perf_list
        )
        test_robust_perf_mean, test_robust_perf_ci95 = make_confidence_interval(
            test_robust_perf_list
        )

        if isinstance(criterion, nn.MSELoss):  # Regression case.
            print(f"Validation {TEST_METRIC}: {val_perf_mean:.3f}+-{val_perf_ci95:.3f}")
            print(f"Test {TEST_METRIC}: {test_perf_mean:.3f}+-{test_perf_ci95:.3f}")
            print(
                f"Robust Validation {TEST_METRIC}: {val_robust_perf_mean:.3f}+-{val_robust_perf_ci95:.3f}"
            )
            print(
                f"Robust Test {TEST_METRIC}: {test_robust_perf_mean:.3f}+-{test_robust_perf_ci95:.3f}"
            )
        else:  # Classification case.
            print(
                f"Validation {TEST_METRIC}: {val_perf_mean:.1f}%+-{val_perf_ci95:.1f}%"
            )
            print(f"Test {TEST_METRIC}: {test_perf_mean:.1f}%+-{test_perf_ci95:.1f}%")
            print(
                f"Robust Validation {TEST_METRIC}: {val_robust_perf_mean:.1f}%+-{val_robust_perf_ci95:.1f}%"
            )
            print(
                f"Robust Test {TEST_METRIC}: {test_robust_perf_mean:.1f}%+-{test_robust_perf_ci95:.1f}%"
            )

        # Certification results
        if args.certify:
            (
                certification_min_grad_mean,
                certification_min_grad_ci95,
            ) = make_confidence_interval(certification_min_grad_list)
            certification_results_mean = np.mean(certification_results_list)

            print(
                f"Minimum gradient: {certification_min_grad_mean:.4f}+-{certification_min_grad_ci95:.4f}"
            )
            print(f"Average certification results: {certification_results_mean:.4f}")

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

