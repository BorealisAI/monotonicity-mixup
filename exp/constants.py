# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# Data-dependent training and model config:
# IMPORTANT: the field 'task' in the CONFIG dictionary should take values from ["classification", "regression"]
# Model configurations within config only have effect for cmn_MLP networks.

CONFIG = {
    "compas": {
        "TASK": "classification",
        "DATA_PATH": "./data/compas/compas_scores_two_years.csv",
        "MONO_FEATURE": 4,
        "NUM_EPOCHS": 100,
        "NUM_RUNS": 20,  # Number of independent training runs.
        "BATCH_SIZE": 256,
        "LR": 5e-3,
        "L1_PENALTY_COEF": 0.0,  # Only has effect if set to value > 0.
        "VALIDATION_DATA_FRACTION": 0.2,  # Should be in [0.0, 1.0]. Indicates the fraction of the training sample used for validation.
        "CERTIFICATION_BUDGET": 10,  # Number of attempts for finding non-monotonic points in the input space.
        "MONOTONICITY_CHECK_BUDGET": 10000,  # Number of points to check for monotonicity against.
        "SUB_NUM": 1,
        "MONO_HIDDEN_NUM": 100,
        "NON_MONO_HIDDEN_NUM": 100,
        "BOTTLENECK": 10,
        "USE_BATCH_NORM": False,
        "REGULARIZATION_BUDGET": 1024,  # Common regularization config
        "MIXUP_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
        "CMN_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
    },
    "blogData": {
        "TASK": "regression",
        "DATA_PATH": "./data/blogData",
        "MONO_FEATURE": 8,
        "NUM_EPOCHS": 50,
        "NUM_RUNS": 20,  # Number of independent training runs.
        "BATCH_SIZE": 256,
        "LR": 5e-3,
        "L1_PENALTY_COEF": 0.0,  # Only has effect if set to value > 0.
        "VALIDATION_DATA_FRACTION": 0.2,  # Should be in [0.0, 1.0]. Indicates the fraction of the training sample used for validation.
        "CERTIFICATION_BUDGET": 10,  # Number of attempts for finding non-monotonic points in the input space.
        "MONOTONICITY_CHECK_BUDGET": 10000,  # Number of points to check for monotonicity against.
        "SUB_NUM": 1,
        "MONO_HIDDEN_NUM": 100,
        "NON_MONO_HIDDEN_NUM": 100,
        "BOTTLENECK": 100,
        "USE_BATCH_NORM": False,
        "REGULARIZATION_BUDGET": 1024,  # Common regularization config
        "MIXUP_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
        "CMN_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
    },
    "loanData": {
        "TASK": "regression",
        "DATA_PATH": "./data/loanData/loans_full_schema.csv",
        "MONO_FEATURE": 11,
        "NUM_EPOCHS": 100,
        "NUM_RUNS": 20,  # Number of independent training runs.
        "BATCH_SIZE": 256,
        "LR": 5e-3,
        "L1_PENALTY_COEF": 0.0,  # Only has effect if set to value > 0.
        "VALIDATION_DATA_FRACTION": 0.3,  # Should be in [0.0, 1.0]. Indicates the fraction of the training sample used for validation.
        "CERTIFICATION_BUDGET": 10,  # Number of attempts for finding non-monotonic points in the input space.
        "MONOTONICITY_CHECK_BUDGET": 10000,  # Number of points to check for monotonicity against.
        "SUB_NUM": 1,
        "MONO_HIDDEN_NUM": 100,
        "NON_MONO_HIDDEN_NUM": 100,
        "BOTTLENECK": 10,
        "USE_BATCH_NORM": False,
        "REGULARIZATION_BUDGET": 1024,  # Common regularization config
        "MIXUP_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
        "CMN_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
    },
    "xrayData": {
        "TASK": "classification",
        "DATA_PATH": "./data/xrayData",
        "MONO_FEATURE": 2,
        "NUM_EPOCHS": 200,
        "NUM_RUNS": 20,  # Number of independent training runs.
        "BATCH_SIZE": 256,
        "LR": 5e-3,
        "L1_PENALTY_COEF": 0.0,  # Only has effect if set to value > 0.
        "VALIDATION_DATA_FRACTION": 0.3,  # Should be in [0.0, 1.0]. Indicates the fraction of the training sample used for validation.
        "CERTIFICATION_BUDGET": 10,  # Number of attempts for finding non-monotonic points in the input space.
        "MONOTONICITY_CHECK_BUDGET": 10000,  # Number of points to check for monotonicity against.
        "SUB_NUM": 1,
        "MONO_HIDDEN_NUM": 100,
        "NON_MONO_HIDDEN_NUM": 100,
        "BOTTLENECK": 100,
        "USE_BATCH_NORM": False,
        "REGULARIZATION_BUDGET": 1024,  # Common regularization config
        "MIXUP_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
        "CMN_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
    },
    "syntheticData": {
        "TASK": "regression",
        "DATA_PATH": "./data/synthData/data_{}_{}.pt",  # f'./data/synthData/data_{n_dimensions}_{n_monotonic_dimensions}.pt'"
        "MONO_FEATURE": -1,  # Unused in thise case since data is generated artificially.
        "NUM_EPOCHS": 50,
        "NUM_RUNS": 20,  # Number of independent training runs.
        "BATCH_SIZE": 256,
        "LR": 5e-3,
        "L1_PENALTY_COEF": 0.0,  # Only has effect if set to value > 0.
        "VALIDATION_DATA_FRACTION": 0.2,  # Should be in [0.0, 1.0]. Indicates the fraction of the training sample used for validation.
        "CERTIFICATION_BUDGET": 10,  # Number of attempts for finding non-monotonic points in the input space.
        "MONOTONICITY_CHECK_BUDGET": 10000,  # Number of points to check for monotonicity against.
        "SUB_NUM": 1,
        "MONO_HIDDEN_NUM": 100,
        "NON_MONO_HIDDEN_NUM": 100,
        "BOTTLENECK": 100,
        "USE_BATCH_NORM": False,
        "REGULARIZATION_BUDGET": 1024,  # Common regularization config
        "MIXUP_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
        "CMN_REGULARIZATION_LAMBDA": 1e4,  # Mixup regularization config
    },
}


NO_CUDA = False  # Disables GPU use if True.
NUM_WORKERS = 4

# Path to save checkpoints
CP_PATH = "./checkpoint"

# Path to save plots
PDP_PATH = "./pdps"
