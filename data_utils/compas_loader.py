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

import numpy as np
import pandas as pd
import torch


def generate_one_hot_mat(mat):
    upper_bound = np.max(mat)
    mat_one_hot = np.zeros((mat.shape[0], int(upper_bound + 1)))

    for j in range(mat.shape[0]):
        mat_one_hot[j, int(mat[j])] = 1.0

    return mat_one_hot


def generate_normalize_numerical_mat(mat):
    mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))

    return mat


def normalize_data_ours(data_train, data_test):
    ### in this function, we normalize all the data to [0, 1], and bring education_num, capital gain, hours per week to the first three columns, norm to [0, 1]
    n_train = data_train.shape[0]
    n_test = data_test.shape[0]
    data_feature = np.concatenate((data_train, data_test), axis=0)

    data_feature_normalized = np.zeros((n_train + n_test, 1))
    class_list = [5, 6]
    mono_list = [0, 1, 2, 3]
    ### store the class variables
    start_index = []
    cat_length = []
    ### Normalize Mono Features
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            if i == mono_list[0]:
                mat = data_feature[:, i]
                mat = mat[:, np.newaxis]
                data_feature_normalized = generate_normalize_numerical_mat(mat)
            else:
                mat = data_feature[:, i]
                mat = generate_normalize_numerical_mat(mat)
                mat = mat[:, np.newaxis]
                data_feature_normalized = np.concatenate(
                    (data_feature_normalized, mat), axis=1
                )
        else:
            continue
    ### Normalize non-mono features and turn class labels to one-hot vectors
    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            continue
        else:
            mat = data_feature[:, i]
            mat = generate_normalize_numerical_mat(mat)
            mat = mat[:, np.newaxis]
            data_feature_normalized = np.concatenate(
                (data_feature_normalized, mat), axis=1
            )

    for i in range(data_feature.shape[1]):
        if i in mono_list:
            continue
        elif i in class_list:
            mat = data_feature[:, i]
            mat = generate_one_hot_mat(mat)
            start_index.append(data_feature_normalized.shape[1])
            cat_length.append(mat.shape[1])
            data_feature_normalized = np.concatenate(
                (data_feature_normalized, mat), axis=1
            )
        else:
            continue

    data_train = data_feature_normalized[:n_train, :]
    data_test = data_feature_normalized[n_train:, :]

    assert data_test.shape[0] == n_test
    assert data_train.shape[0] == n_train

    return data_train, data_test, start_index, cat_length


def load_data(
    path_to_data="./data/compas/compas_scores_two_years.csv", valid_split=0.2,
):
    """Loads COMPAS data.

    Args:
        path_to_data (str, optional): Path to data. Defaults to 'data/compas_prepped.json'.
        valid_split (float, optional): value in [0,1] indicating fraction training sample to be used
            for validation. Defaults to 0.2.

    Returns:
        list: Tensors with training data.
    """

    assert (
        valid_split >= 0.0 and valid_split <= 1.0
    ), "valid_split should be in [0.0, 1.0]"

    data = pd.read_csv(path_to_data)
    # Data cleaning as performed by propublica
    data = data[data["days_b_screening_arrest"] <= 30]
    data = data[data["days_b_screening_arrest"] >= -30]
    data = data[data["is_recid"] != -1]
    data = data[data["c_charge_degree"] <= "O"]
    data = data[data["score_text"] != "N/A"]

    n = data.shape[0]
    n_train = int(n * 0.8)
    n_test = n - n_train

    replace_data = [
        [
            "African-American",
            "Hispanic",
            "Asian",
            "Caucasian",
            "Native American",
            "Other",
        ],
        ["Male", "Female"],
    ]

    for row in replace_data:
        data = data.replace(row, range(len(row)))

    data = (
        np.array(
            pd.concat(
                [
                    data[
                        [
                            "priors_count",
                            "juv_fel_count",
                            "juv_misd_count",
                            "juv_other_count",
                            "age",
                            "race",
                            "sex",
                        ]
                    ],
                    data[["two_year_recid"]],
                ],
                axis=1,
            ).values
        )
        + 0
    )

    # Shuffle for train/test splitting
    np.random.seed(seed=78712)
    np.random.shuffle(data)

    data_train = data[:n_train, :]
    data_test = data[n_train:, :]

    X_train = data_train[:, :7].astype(np.float64)
    y_train = data_train[:, 7].astype(np.uint8)

    X_test = data_test[:, :7].astype(np.float64)
    y_test = data_test[:, 7].astype(np.uint8)

    X_train, X_test, _, _ = normalize_data_ours(X_train, X_test)

    n = X_train.shape[0]
    n = int((1.0 - valid_split) * n)  # Split for training/validation data.
    X_val = X_train[n:, :]
    y_val = y_train[n:]
    X_train = X_train[:n, :]
    y_train = y_train[:n]

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float().unsqueeze(1)
    y_test = torch.tensor(y_test).float().unsqueeze(1)
    y_val = torch.tensor(y_val).float().unsqueeze(1)

    return X_train, y_train, X_val, y_val, X_test, y_test

