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

## Prepares the blog data in torch tensors format.
## Dataset information can be found in https://archive.ics.uci.edu/ml/datasets/BlogFeedback#.

import os
import glob
import random
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

random.seed(10)  # Fixing seed for fixed data splits across experiments

TRAIN_DATA_FILE = "blogData_train.csv"
TEST_DATA_GLOB = "blogData_test-*.csv"

# List of indices of monotonic features. Models are expected to be
# monotonicaly non-decreasing with respect to those.
MONOTONIC_INDICES = (50, 51, 52, 53, 55, 56, 57, 58)


def _features_normalization(train_data, test_data):
    """Normalizes train and test features using stats from the train partition.

    Args:
        train_data (np.array): Batch of train data with shape [TRAIN_DATA_SIZE, N_DIMENSIONS].
        test_data (np.array): Batch of test data with shape [TEST_DATA_SIZE, N_DIMENSIONS].

    Returns:
        Tuple: Pair corresponding to normalized train and test data.
    """

    scaler = MinMaxScaler()
    scaler.fit(train_data)

    return scaler.transform(train_data), scaler.transform(test_data)


def _remove_outliers(data, target, threshold=0.9):
    """Removes points where the target value is greater the points as frequent as 'threshold'.

    Args:
        data (np.array): Batch of data with shape [DATA_SIZE, N_DIMENSIONS].
        target (np.array): Batch of targets with shape [DATA_SIZE, 1].
            IMPORTANT: Targets are assumed to be lower bounded in 0.
        threshold (float, optional): Value in [0, 1] indicating where to set the
            maximum target value. Defaults to 0.9.

    Returns:
        Tuple: Pair corresponding to data and targets where data points with target
            values higher than the threshold fraction of the population are removed.
    """

    assert threshold >= 0.0 and threshold <= 1.0, "Threshold should be in [0.0, 1.0]."

    sorted_target = np.sort(target, axis=0)

    cutoff_value = sorted_target[int(len(target) * threshold), 0]
    included_indices = (target <= cutoff_value).squeeze(1)

    return data[included_indices, :], target[included_indices, :]


def _move_monotonic_features(data):
    """Moves monotonic features to the beginning of the data matrix. 

    Args:
        data (np.array): Batch of data with shape [DATA_SIZE, N_DIMENSIONS].

    Returns:
        np.array: Data metrics with rearranged columns.
    """

    monotonic_x = data[:, MONOTONIC_INDICES]
    non_monotonic_x = np.delete(data, axis=1, obj=MONOTONIC_INDICES)

    return np.concatenate((monotonic_x, non_monotonic_x), axis=1)


def load_data(
    path_to_data="./data/blogData", valid_split=0.2,
):
    """Loads Blog feedback data.

    Args:
        path_to_data (str, optional): Path to data folder containing csv files. Defaults to 'data/blogData'.
        valid_split (float, optional): value in [0,1] indicating fraction training sample to be used
            for validation. Defaults to 0.2.

    Returns:
        list: Tensors with training and testing data.
    """

    assert (
        valid_split >= 0.0 and valid_split <= 1.0
    ), "valid_split should be in [0.0, 1.0]."

    # Load train data. Train data is in a single csv file.
    train_data_path = os.path.join(path_to_data, TRAIN_DATA_FILE)
    train_data = np.genfromtxt(train_data_path, delimiter=",", dtype=None,)

    train_x, train_y = train_data[:, :-1], train_data[:, -1:]

    # Filter out data points with too high of a target value.
    train_x, train_y = _remove_outliers(train_x, train_y)

    # Load test data. Test data is made available in 60 separate csv files.
    test_data_files = glob.glob(os.path.join(path_to_data, TEST_DATA_GLOB))
    test_data_list = []
    for test_data_file in test_data_files:
        test_data_list.append(np.genfromtxt(test_data_file, delimiter=",", dtype=None,))
    test_data = np.concatenate(test_data_list, axis=0)

    test_x, test_y = test_data[:, :-1], test_data[:, -1:]

    # Filter out data points with too high of a target value.
    test_x, test_y = _remove_outliers(test_x, test_y)

    # Scale features and target values.
    train_x, test_x = _features_normalization(train_x, test_x)
    train_y, test_y = _features_normalization(train_y, test_y)

    # Move monotonic features to the start columns.
    train_x = _move_monotonic_features(train_x)
    test_x = _move_monotonic_features(test_x)

    # Random train/validation splitting

    random_indices = list(range(train_x.shape[0]))
    random.shuffle(random_indices)

    train_data_size = int(train_x.shape[0] * (1.0 - valid_split))
    train_indices, validation_indices = (
        random_indices[:train_data_size],
        random_indices[train_data_size:],
    )

    X_train = train_x[train_indices, :]
    y_train = train_y[train_indices, :]
    X_val = train_x[validation_indices, :]
    y_val = train_y[validation_indices, :]

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(test_x).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(test_y).float()
    y_val = torch.tensor(y_val).float()

    return X_train, y_train, X_val, y_val, X_test, y_test
